#!/usr/bin/env python

import os
import cv2
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
def build(ckpt_pth):
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
    var_local=torch.load(ckpt_pth, map_location='cpu')['trainer']['ema_var_wo_ddp']
    var.load_state_dict(var_local, strict=True)
    var.eval()
    for p in var.parameters(): p.requires_grad_(False)
    
    ### load vae
    vae_local=torch.load(ckpt_pth, map_location='cpu')['trainer']['vae_local']
    vae.load_state_dict(vae_local, strict=True)
    # load_sep_vae_model(vae, 'can')
    vae.eval()                                             
    for p in vae.parameters(): p.requires_grad_(False)     
    
    ### load norm
    normalizer = LinearNormalizer()
    norm_local=torch.load(ckpt_pth, map_location='cpu')['trainer']['var_norm']
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
    return obs

def given_data():
    obs=dict()
    obs['agentview_image'] = image_format_amend("/home/robot/UR_Robot_Arm/coarse2fine/data/scene1.jpg")[None,None,...]
    obs['robot0_eye_in_hand_image'] = image_format_amend("/home/robot/UR_Robot_Arm/coarse2fine/data/wrist1.jpg")[None,None,...]
    obs['robot0_eef_pos'] = np.array([-0.36229283359785835,-0.4150547746810219,0.4619846199476397])[None,None,...]
    obs['robot0_eef_quat'] = np.array([0.4246134752330147, 0.9042700809168371, -0.01994586432104633, 0.04001474610297329,])[None,None,...]
    obs['robot0_gripper_qpos'] = np.array([0.0])[None,None,...]
    return obs

def infer(var, vae, normalizer, data, n_obs_steps, n_action_steps, is_verbose=False):
    ### device load
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if is_verbose: print(f'[INFO] device : {device}')
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
        start = n_obs_steps - 1
        end = start + n_action_steps
        action_pred = action_pred[:,start:end]
        # get original action
        action_quat = undo_transform_action(action_pred) # xyz | wxyz | gripper
        action_quat = action_quat[...,[0,1,2,4,5,6,3,7]] # xyz | xyzw | gripper
        # whether to show
        if is_verbose:
            print('[INFO] action pred shape : ', action_pred.shape)
            print('[INFO] action quat shape : ', action_quat.shape)
    return action_quat

def undo_transform_action(action):
    rot_dim = action.shape[-1]-3-1
    pos=action[...,:3]
    rot=action[...,3:3+rot_dim]
    gripper=action[...,[-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos,rot,gripper], axis=-1)
    return uaction

def image_format_amend(img_pth, RESIZE_WIDTH = 160, RESIZE_HEIGHT = 120):
    image = cv2.imread(img_pth)                                 # BGR
    image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))    # BGR | (480，640，3) -> (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # RGB               # 
    image_arr = np.array(image_rgb)                             # BHWC | [0,255]
    return image_arr

def obs_format_amend(obs):
    rgb_keys = ["agentview_image", "robot0_eye_in_hand_image"]
    lowdim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    for key in rgb_keys:
        # move channel last to channel first
        # B,T,H,W,C -> B,T,C,H,W
        # convert uint8 image to float32
        obs[key] = np.moveaxis(obs[key],-1,2).astype(np.float32) / 255.
    for key in lowdim_keys:
        obs[key] = obs[key][:].astype(np.float32)
    return obs


def print_2d_arr(info,actions):
    """
    @fun :
    """
    print(info)
    for row in actions: 
        print('[', end="")
        for v in row:
            print(f"{v:.7f}", end="")
            print(' ', end="")
        print(']')
    return

### 📖 LIST 📖
import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--ckpt_pth", type=str, default="/home/robot/UR_Robot_Arm/coarse2fine/ckpt/ar-ep_4000-accmean_29.68-acctail_32.00.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--n_action_steps", type=int, default=8)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    ### params
    args = get_args()

    ### build model
    var, vae, normalizer = build(ckpt_pth = args.ckpt_pth)

    ### get data
    # data = random_data()
    data = obs_format_amend(given_data())

    ### run
    actions = infer(var, vae, normalizer, data, args.n_obs_steps, args.n_action_steps, True).reshape(-1,8) 

    print([-0.36229283359785835,-0.4150547746810219,0.4619846199476397,0.4246134752330147,0.9042700809168371,-0.01994586432104633,0.04001474610297329,0.0])
    print_2d_arr('[INFO] robot action | raw : ', actions)


"""
[-0.3622928 -0.4150547 0.4619846 0.4246134 0.9042700 -0.0199458 0.0400147 0.0]
"""

"""
[-0.3641393 -0.4166141 0.4482937 0.4243543 0.9043197 -0.0202377 0.0414696 -1.0199691 ]
[-0.3681723 -0.4092316 0.4593124 0.4250602 0.9040313 -0.0213388 0.0399486 -0.9833924 ]
[-0.3617996 -0.4100017 0.4545057 0.4242024 0.9044400 -0.0206207 0.0401906 -1.0038750 ]
[-0.3588532 -0.4029518 0.4525110 0.4245056 0.9042506 -0.0196520 0.0417083 -1.0002413 ]
[-0.3504641 -0.4056237 0.4555535 0.4248687 0.9041038 -0.0208324 0.0406069 -0.9953830 ]
[-0.3521069 -0.3949804 0.4535229 0.4246165 0.9042028 -0.0196591 0.0416135 -0.9916106 ]
[-0.3598781 -0.3744820 0.4584403 0.4256662 0.9037449 -0.0198403 0.0407423 -0.9989278 ]
[-0.3756610 -0.3590542 0.4582780 0.4255756 0.9037163 -0.0190873 0.0426372 -0.9981372 ]
"""


"""
[-0.3641393 -0.4166141 0.4482937 0.4243543 0.9043197 -0.0202377 0.0414696 -1.0199691 ]
[-0.3681722 -0.4092316 0.4593124 0.4250603 0.9040312 -0.0213388 0.0399486 -0.9833924 ]
[-0.3617996 -0.4100017 0.4545057 0.4242025 0.9044400 -0.0206207 0.0401906 -1.0038750 ]
[-0.3588532 -0.4029518 0.4525110 0.4245056 0.9042506 -0.0196520 0.0417083 -1.0002413 ]
[-0.3504641 -0.4056237 0.4555535 0.4248687 0.9041038 -0.0208324 0.0406069 -0.9953830 ]
[-0.3521069 -0.3949804 0.4535229 0.4246165 0.9042028 -0.0196591 0.0416135 -0.9916106 ]
[-0.3598782 -0.3744820 0.4584403 0.4256662 0.9037449 -0.0198403 0.0407423 -0.9989278 ]
[-0.3756610 -0.3590542 0.4582780 0.4255756 0.9037163 -0.0190873 0.0426372 -0.9981372 ]
"""
