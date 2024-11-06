import os
import os.path as osp
import torch, torchvision
from copy import deepcopy
import random
import numpy as np
from tqdm import tqdm
# import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import matplotlib.pyplot as plt
from robodata.pytorch_util import dict_apply
import h5py

from train_util import load_realworld_image_dataset, load_shape_meta, load_sep_vae_model
from vqvae import build_vae_disc

### transformer
from robodata.rotation_transformer import RotationTransformer
rotation_transformer = RotationTransformer(from_rep='quaternion', 
                                            to_rep='rotation_6d',
                                            from_convention=None,
                                            to_convention=None)

# ALL_CH_SEP=-1
def build(args, device, patch_nums):
    """
    @func: build the model and data normalizer
    """

    ### build vae, var
    vae = build_vae_disc(
        device=device,
        ## encoder | decoder
        V=args.vocab_num, 
        Cvae=8,             # 🔥 🔥 🔥 
        ch=2,               # 🔥 🔥 🔥 
        # action_dim=7,     # 🔥 🔥 🔥 
        num_actions=16,
        dropout=0.0,
        ## quant
        beta=0.25,
        using_znorm=True,   # 🔥 🔥 🔥
        quant_conv_ks=3,
        quant_resi=0.5,
        share_quant_resi=4,
        patch_nums=patch_nums,
        ## initialization
        vae_init=-0.5,
        vocab_init=-1,
    )

    ### vae
    vae = load_sep_vae_model(vae, 'bowl')
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    print(f'[INFO] vae finished')

    ### normalizer
    shape_meta = load_shape_meta()
    _, _, normalizer = load_realworld_image_dataset(args, shape_meta, 1)
    normalizer.to(device)
    print(f'[INFO] normalizer finished')
    
    return normalizer, vae

def choose_config_data(args):
    """
    @func: choose the config data for test
    """
    data=list() 
    with h5py.File(args.data_path) as file:
        demos = file['data']
        for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
            
            if i in args.traj_indices:

                demos = file['data']
                demo = demos[f'demo_{i}']
            
                obs = demo['obs']
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

                data.append({'obs': obs, 'actions': actions})

    return data

def vae_run(args, vae, normalizer, data):
    """
    :func: 
    """
    ### preprocess the actions
    import math
    actual_max_length = max(len(d['actions']) for d in data)
    margin_max_length = math.ceil(actual_max_length / args.slice_size) * args.slice_size
    padded_actions = []
    for d in data:
        padded_act = np.vstack((d['actions'], np.tile(d['actions'][-1], (margin_max_length - len(d['actions']), 1))))
        padded_actions.append(padded_act)
    padded_actions = np.array(padded_actions)                               # BLC
    B,L,C = padded_actions.shape                                            # BLC
    actions_run = np.split(padded_actions, L // args.slice_size, axis=1)    # BLC
    ### run
    act_repo_raw=[]
    act_repo_vae=[]
    with torch.no_grad():
        for action_raw in actions_run:
            ## vae
            action_vae = deepcopy(action_raw)                                   # [B,num_actions,action_dim] ｜  [B,16,10]
            action_vae = normalizer['action'].normalize(action_vae)             #  BL7
            action_vae = action_vae.view(B,1,args.slice_size,C).contiguous()    # [B,1,num_actions,action_dim] | [B,1,16,10]
            rec_action_vae = vae.inp_to_action(action_vae)
            rec_action_vae = normalizer['action'].unnormalize(rec_action_vae)   # [B,1,num_actions,action_dim] | [B,1,16,10]
            rec_action_vae = rec_action_vae.to('cpu').numpy().squeeze(axis=1)   # [B,num_actions,action_dim] | [B,16,10]
            act_repo_raw.append(action_raw)
            act_repo_vae.append(rec_action_vae)
    np.save(f"{args.save_path}/act_raw.npy", np.concatenate(act_repo_raw, axis=1))
    np.save(f"{args.save_path}/act_vae.npy", np.concatenate(act_repo_vae, axis=1))


### 🚗 ACTION 🚗
import numpy as np
import matplotlib.pyplot as plt
def vis_act_raw2vae_linechart(raw_pth, vae_pth, save_pth, B_idx, save_png_name, is_rotation_6d=True):    
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
    x_ticks = range(0, matrix_raw.shape[1], 16)
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
    parser.add_argument("--data_path", type=str, default="/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl.hdf5")
    parser.add_argument("--save_path", type=str, default="/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/tmp/eval/bowl_vae_wo_r_norm")
    parser.add_argument("--vocab_num", type=int, default=512)
    parser.add_argument("--traj_indices", type=int, nargs='+', default=[23,35,48,51,60,70,82,93,104,122])
    parser.add_argument("--slice_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    ### params
    args = get_args()
    patch_nums = (1, 2, 3, 4)
    device = 'cuda:0'
    os.makedirs(args.save_path, exist_ok=True) # create the folder if not existing

    ### get data
    normalizer, vae = build(args, device, patch_nums)
    data = choose_config_data(args)

    ### run
    vae_run(args, vae, normalizer, data)
    
    ### visualization
    root_pth = args.save_path
    save_pth = root_pth + '/images'
    raw_pth = root_pth + '/act_raw.npy'
    vae_pth = root_pth + '/act_vae.npy'
    os.makedirs(save_pth, exist_ok=True) 
    for B_idx in range(len(args.traj_indices)):
        traj_idx = args.traj_indices[B_idx]
        save_png_name = f"act_{traj_idx}"
        vis_act_raw2vae_linechart(raw_pth, vae_pth, save_pth, B_idx, save_png_name)