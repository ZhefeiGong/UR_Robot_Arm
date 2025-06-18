#!/usr/bin/env python

import os
import cv2
import math
import h5py
import json
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
from carp_utils.pytorch_util import dict_apply
from carp_utils.inference_util import load_shape_meta, load_obs_encoder
from carp_utils.normalizer import LinearNormalizer

### rotation
from carp_utils.rotation_transformer import RotationTransformer
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
    obs['robot0_eef_pos'] = np.array([-0.5370786233477692,-0.14595360977698918,0.49753826739803164])[None,None,...]
    obs['robot0_eef_quat'] = np.array([-0.7295055052319397, -0.6835752261703961, 0.022747529127961453, 0.0054016590929929905,])[None,None,...]
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

# def obs_format_amend(obs):
#     rgb_keys = ["agentview_image", "robot0_eye_in_hand_image"]
#     lowdim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
#     for key in rgb_keys:
#         # 🔥 Move channel last to channel first 🔥
#         # B,T,H,W,C -> B,T,C,H,W | convert uint8 image to float32
#         obs[key] = np.moveaxis(obs[key],-1,2).astype(np.float32) / 255.
#     for key in lowdim_keys:
#         obs[key] = obs[key][:].astype(np.float32)
#         # 🔥 Quat to Rotation6d 🔥
#         if key == "robot0_eef_quat":
#             obs[key] = obs[key][..., [3, 0, 1, 2]]                    # xyzw(ori) -> wxyz(pt3d)
#             obs[key] = rotation_transformer.forward(obs[key])         # quat -> rotation6d
#         # 🔥 Grip to (-1,1) 🔥
#         elif key == "robot0_gripper_qpos":
#             obs[key] = obs[key]*2-1                                       # [0,1](ori) -> [-1,1]
#             # assert (max(obs[key]) == 1.0) and (min(obs[key]) == -1.0)     # make sure it's belong to [-1,1]
#     return obs

def obs_format_amend(obs):
    rgb_keys = ["agentview_image", "robot0_eye_in_hand_image"]
    lowdim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    for key in rgb_keys:
        # 🔥 Move channel last to channel first 🔥
        # B,T,H,W,C -> B,T,C,H,W | convert uint8 image to float32
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
    parser.add_argument("--ckpt_pth", type=str, default="/home/robot/UR_Robot_Arm/coarse2fine/ckpt/cup/carp/ar-ep_4300-accmean_63.67-acctail_60.26.pth")
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

""" CARP(tcp)
[-0.5366486 -0.1485423 0.4910749 0.7283340 0.6848071 -0.0230980 -0.0059470 -0.9945959 ]
[-0.5374855 -0.1495975 0.4921175 0.7279218 0.6852531 -0.0229232 -0.0056904 -1.0048740 ]
[-0.5368797 -0.1485909 0.4929864 0.7281961 0.6849653 -0.0228434 -0.0055634 -0.9933511 ]
[-0.5384305 -0.1483620 0.4955364 0.7282633 0.6848956 -0.0226892 -0.0059980 -1.0031811 ]
[-0.5362377 -0.1490296 0.4961967 0.7285010 0.6846497 -0.0225014 -0.0058830 -1.0084103 ]
[-0.5360029 -0.1489360 0.4973463 0.7282117 0.6849485 -0.0227599 -0.0059352 -1.0003974 ]
[-0.5349273 -0.1489806 0.4981977 0.7280952 0.6850785 -0.0226401 -0.0056909 -1.0012531 ]
[-0.5348154 -0.1493642 0.4976016 0.7283698 0.6847838 -0.0226614 -0.0059288 -1.0041410 ]
"""

""" CARP(local)
[-0.5366486 -0.1485424 0.4910749 0.7283340 0.6848071 -0.0230980 -0.0059470 -0.9945958 ]
[-0.5374855 -0.1495975 0.4921175 0.7279218 0.6852531 -0.0229232 -0.0056904 -1.0048740 ]
[-0.5368798 -0.1485908 0.4929864 0.7281961 0.6849653 -0.0228434 -0.0055634 -0.9933510 ]
[-0.5384306 -0.1483620 0.4955364 0.7282633 0.6848956 -0.0226892 -0.0059980 -1.0031810 ]
[-0.5362377 -0.1490296 0.4961967 0.7285010 0.6846497 -0.0225014 -0.0058830 -1.0084105 ]
[-0.5360029 -0.1489360 0.4973463 0.7282117 0.6849485 -0.0227599 -0.0059352 -1.0003976 ]
[-0.5349274 -0.1489806 0.4981977 0.7280952 0.6850785 -0.0226401 -0.0056909 -1.0012531 ]
[-0.5348154 -0.1493642 0.4976016 0.7283698 0.6847838 -0.0226614 -0.0059287 -1.0041411 ]
"""
