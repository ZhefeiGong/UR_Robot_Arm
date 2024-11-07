import os
import math
import h5py
import random
import os.path as osp
import torch, torchvision
import numpy as np
from tqdm import tqdm
from copy import deepcopy
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import matplotlib.pyplot as plt
from autoreg import build_vae_var
from utils.pytorch_util import dict_apply
from utils.inference_util import load_shape_meta, load_obs_encoder
from utils.normalizer import LinearNormalizer

### rotation
from utils.rotation_transformer import RotationTransformer
rotation_transformer = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d', from_convention=None, to_convention=None)

### build everything
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
    var_local=torch.load(args.ckpt, map_location='cpu')['trainer']['ema_var_wo_ddp']
    var.load_state_dict(var_local, strict=True)
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
    del var_local, vae_local, norm_local
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

def random_data():
    obs=dict()
    obs['agentview_image'] = np.random.randn(1, 1, 3, 120, 160)
    obs['robot0_eef_pos'] = np.random.randn(1, 1, 3)
    obs['robot0_eef_quat'] = np.random.randn(1, 1, 4)
    obs['robot0_eye_in_hand_image'] = np.random.randn(1, 1, 3, 120, 160)
    obs['robot0_gripper_qpos'] = np.random.randn(1, 1, 1)
    print(obs)
    return obs

def run(args, var, vae, normalizer, data):
    ### device load
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] device : {device}')
    ### run
    with torch.no_grad():
        # get data
        obs_dict = dict_apply(data, lambda x: torch.from_numpy(x).to(device=device))
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
        print(action_pred.shape)
        # get original action
        action_calc = undo_transform_action(action_pred)
        print(action_calc.shape)

def undo_transform_action(action):
    rot_dim = action.shape[-1]-3-1
    pos=action[...,:3]
    rot=action[...,3:3+rot_dim]
    gripper=action[...,[-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos,rot,gripper], axis=-1)
    return uaction

### 📖 LIST 📖
import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--ckpt", type=str, default="/home/robot/UR_Robot_Arm/coarse2fine/ckpt/ar-ep_4000-accmean_29.68-acctail_32.00.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--n_action_steps", type=int, default=8)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    ### params
    args = get_args()

    ### get data
    var, vae, normalizer = build(args)
    data = random_data()

    ### run
    run(args, var, vae, normalizer, data)
    