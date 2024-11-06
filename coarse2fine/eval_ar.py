import os
import math
import h5py
import random
import os.path as osp
import torch, torchvision
import numpy as np
from tqdm import tqdm
from copy import deepcopy
# import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import matplotlib.pyplot as plt
from robodata.pytorch_util import dict_apply

from train_util import load_realworld_image_dataset, load_shape_meta, load_sep_vae_model, load_obs_encoder
from vqvae import build_vae_disc
from autoreg import build_vae_var
from robodata.normalizer import LinearNormalizer

### rotation transformer
from robodata.rotation_transformer import RotationTransformer
rotation_transformer = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d', from_convention=None, to_convention=None)

def build(args):
    """
    @func: build everything
    """
    ### device load
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] device : {device}')

    ### load obs encoder
    shape_meta=load_shape_meta()
    obs_encoder = load_obs_encoder(shape_meta)

    ### load vae and var
    vae, var = build_vae_var(
        device=device,
        patch_nums=(1,2,3,4),
        ## VAE
        V=512, 
        Cvae=8, 
        ch=2, 
        num_actions=16,
        dropout=0.05,
        beta=0.25,
        using_znorm=True,
        quant_conv_ks=3,
        quant_resi=0.5,
        share_quant_resi=4,
        ## VAR
        obs_encoder = obs_encoder,  # 🔥 🔥 🔥
        depth=16,                   # 🔥 🔥 🔥
        n_obs_steps=1,              # 🔥 🔥 🔥
        embed_dim=160,              # 🔥 🔥 🔥
        shared_aln=False,           # whether to use shared adaln
        attn_l2_norm=True,          # whether to use L2 normalized attention
        init_adaln=0.5,             # for var
        init_adaln_gamma=1e-3,      # for var
        init_head=0.02,             # for var
        init_std=-1,                # for var
    )
    
    ### load var
    var_wo_ddp=torch.load(args.ckpt, map_location='cpu')['trainer']['ema_var_wo_ddp']
    var.load_state_dict(var_wo_ddp, strict=True)
    var.eval()
    for p in var.parameters(): p.requires_grad_(False)
    
    ### load vae
    vae_local=torch.load(args.ckpt, map_location='cpu')['trainer']['vae_local']
    vae.load_state_dict(vae_local, strict=True)
    # load_sep_vae_model(vae, 'can')
    vae.eval()                                             
    for p in vae.parameters(): p.requires_grad_(False)     
    
    ### load norm
    normalizer = LinearNormalizer()
    norm_local=torch.load(args.ckpt, map_location='cpu')['trainer']['var_norm']
    normalizer.load_state_dict(norm_local)
    
    ### carry
    var.to(device)
    var.eval()
    vae.to(device)
    vae.eval()
    normalizer.to(device)
    normalizer.eval()
    del var_wo_ddp, vae_local, norm_local
    print(f'[INFO] VAE/VAR Finished')
    
    ### random seed
    seed=42
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    
    print(f'[INFO] Policy Initialization Finished')

    return var, vae, normalizer


def choose_config_data(args):
    """
    @func: choose the config data for test
    """
    ### load the meta data and separate
    shape_meta=load_shape_meta()
    rgb_keys = list()
    lowdim_keys = list()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)

    ### load the data
    data=list() 
    with h5py.File(args.data_path) as file:
        demos = file['data']
        for i in tqdm(args.traj_indices, desc="Loading the specific trajectories in hdf5 "):
            ### demo
            demo = demos[f'demo_{i}']
            ### observations
            obs = demo['obs']
            obs_dict = dict()
            for key in rgb_keys:
                # move channel last to channel first
                # T,H,W,C
                # convert uint8 image to float32
                obs_dict[key] = np.moveaxis(obs[key],-1,1
                    ).astype(np.float32) / 255.
                # T,C,H,W
            for key in lowdim_keys:
                obs_dict[key] = obs[key][:].astype(np.float32)

            ### actions
            raw_actions = demo['actions'][:].astype(np.float32) # 🔥 🔥 🔥
            ## 🔥 Transform 🔥
            pos = raw_actions[...,:3]           # pos
            quat = raw_actions[...,3:7]         # quat | xyzw
            quat = quat[..., [3, 0, 1, 2]]      # xyzw(ori) -> wxyz(pt3d)
            gripper = raw_actions[...,7:]       # gripper | [0,1]
            gripper = gripper*2-1               # [0,1](ori) -> [-1,1]
            assert (max(gripper) == 1.0) and (min(gripper) == -1.0)   # make sure it's belong to [-1,1]
            rot = rotation_transformer.forward(quat)
            actions = np.concatenate([
                pos, rot, gripper
            ], axis=-1).astype(np.float32)      # 3 + 6 + 1 = 10 dimension

            data.append({'obs': obs_dict, 'actions': actions})

    return data

def var_run(args, var, vae, normalizer, data):
    """
    :func: run the autoregressive model
    """
    ### get the current device
    device = next(var.parameters()).device
    obs_keys = ['agentview_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos']

    ### preprocess the actions & observations
    slice_size = args.n_action_steps
    actual_max_length = max(len(d['actions']) for d in data)
    margin_max_length = math.ceil(actual_max_length / slice_size) * slice_size
    def pad_data(data, max_len):
        pad_length = max_len - data.shape[0]
        if data.ndim == 2:      # T x C 
            return np.vstack((data, np.tile(data[-1], (pad_length, 1))))
        elif data.ndim == 4:    #  T x C x H x W 
            return np.concatenate((data, np.tile(data[-1:], (pad_length, 1, 1, 1))), axis=0)
        else:
            raise ValueError("Unsupported data dimensions for padding")
    for d in data:
        d['actions'] = pad_data(d['actions'], margin_max_length)
        d['obs'] = {key: pad_data(d['obs'][key], margin_max_length) for key in obs_keys}
    actions = np.stack([d['actions'] for d in data], axis=0)
    obs = {key: np.stack([d['obs'][key] for d in data], axis=0) for key in obs_keys}
    B,L,C = actions.shape                                            # BLC
    actions_run = np.split(actions, L // slice_size, axis=1)         # BLC
    
    ### run
    act_repo_raw=[]
    act_repo_vae=[]
    with torch.no_grad():
        for idx, action_raw in enumerate(actions_run):
            # get obs
            current_idx = args.n_action_steps * idx
            obs_tmp = {key: obs[key][:, current_idx:current_idx + args.n_obs_steps] for key in obs_keys}
            # get data
            obs_dict = dict_apply(obs_tmp, lambda x: torch.from_numpy(x).to(device=device))
            nobs = normalizer.normalize(obs_dict)                                                               # -> [B,T,...]
            # predict
            action_pred = var.autoregressive_infer_cfg(nobs=nobs, vae_proxy=vae)                                # -> B1LC | 🔥 🔥 🔥
            action_pred = action_pred.view(action_pred.shape[0],action_pred.shape[2],action_pred.shape[3])      # -> BLC
            # unnormalize prediction
            action_pred = normalizer['action'].unnormalize(action_pred).to('cpu').numpy()
            # get action
            start = args.n_obs_steps - 1
            end = start + args.n_action_steps
            action_pred = action_pred[:,start:end]
             # store
            act_repo_raw.append(action_raw)
            act_repo_vae.append(action_pred)

    np.save(f"{args.save_path}/act_raw.npy", np.concatenate(act_repo_raw, axis=1))
    np.save(f"{args.save_path}/act_vae.npy", np.concatenate(act_repo_vae, axis=1))

def vis_act_raw2vae_linechart(args, raw_pth, vae_pth, save_pth, B_idx, save_png_name, is_rotation_6d=True):    
    """
    @fun: 🚗 ACTION 🚗
    """
    matrix_raw = np.load(raw_pth)
    matrix_vae = np.load(vae_pth)
    if is_rotation_6d : 
        dimensions = ['x','y','z','r1','r2','r3','r4','r5','r6','gripper']  # Rotation-6D | 🔥 🔥 🔥
        act_dim = 10
    else:
        dimensions = ['x','y','z','rx','ry','rz','gripper']                 # Axis-Angle | 💦 💦 💦
        act_dim = 7
    print('the martix has shape : ', matrix_raw.shape )
    assert matrix_raw.shape == matrix_vae.shape , "Shape mismatch between raw and VAE matrices"
    fig, axs = plt.subplots(act_dim, 1, figsize=(10, 20)) # 🔥 🔥 🔥
    x_ticks = range(0, matrix_raw.shape[1], args.n_action_steps)
    loss = np.mean((matrix_raw[B_idx, :, :] - matrix_vae[B_idx, :, :])**2)
    for i in range(act_dim): # 🔥 🔥 🔥
        axs[i].plot(matrix_raw[B_idx, :, i], label='raw', color='blue')
        axs[i].plot(matrix_vae[B_idx, :, i], label='vae', color='red')
        axs[i].legend()
        axs[i].set_title(dimensions[i])
        axs[i].set_xlabel('Action')
        axs[i].set_ylabel('Value')
        for x_tick in x_ticks:
            axs[i].axvline(x=x_tick, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    loss_str = "{:.5f}".format(loss)
    img_path = f"{save_pth}/{save_png_name}_{loss_str}.png"
    plt.savefig(img_path, dpi=300)
    print(f"Image saved to {img_path}")


### 📖 LIST 📖
import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--data_path", type=str, default="/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl/bowl.hdf5")
    parser.add_argument("--save_path", type=str, default="/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/tmp/eval/bowl_ar_zscore")
    parser.add_argument("--ckpt", type=str, default="/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/local_output/act_ar_bowl-img-ly16-b64g1-im1-em160_v512_110316/ar-ep_4000-accmean_29.68-acctail_32.00.pth")
    parser.add_argument("--vocab_num", type=int, default=512)
    parser.add_argument("--traj_indices", type=int, nargs='+', default=[23,35,48,51,60,70,82,93,104,122])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--n_action_steps", type=int, default=8)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    ### params
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True) # create the folder if not existing

    ### get data
    var, vae, normalizer = build(args)
    data = choose_config_data(args)

    ### run
    var_run(args, var, vae, normalizer, data)
    
    ### visualization
    root_pth = args.save_path
    save_pth = root_pth + '/images'
    raw_pth = root_pth + '/act_raw.npy'
    vae_pth = root_pth + '/act_vae.npy'
    os.makedirs(save_pth, exist_ok=True) 
    for B_idx in range(len(args.traj_indices)):
        traj_idx = args.traj_indices[B_idx]
        save_png_name = f"act_{traj_idx}"
        vis_act_raw2vae_linechart(args, raw_pth, vae_pth, save_pth, B_idx, save_png_name)

    