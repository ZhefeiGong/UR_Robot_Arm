#!/usr/bin/env python

import os
import cv2
import math
import h5py
import random
import os.path as osp
import torch, torchvision
import dill
import hydra
import numpy as np
from tqdm import tqdm
from copy import deepcopy
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import matplotlib.pyplot as plt
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

### rotation
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
rotation_transformer = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d', from_convention=None, to_convention=None)

def build(ckpt_pth, is_verbose=False):
    """
    :func: run the diffusion policy
    """
    ### device load
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if is_verbose: print(f'[INFO] device : {device}')
    # load checkpoint
    payload = torch.load(open(ckpt_pth, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    if is_verbose: print('[INFO] Loading Finish')
    ## get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema: policy = workspace.ema_model
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    if is_verbose: print('[INFO] Got Policy')
    ## return
    return policy

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

def infer(policy, data, n_obs_steps, n_action_steps, is_verbose=False):
    ### device load
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if is_verbose: print(f'[INFO] device : {device}')
    ### run
    with torch.no_grad():
        # get data
        obs_dict = dict_apply(data, lambda x: torch.from_numpy(x).to(device=device))
        # predict
        action_dict = policy.predict_action(obs_dict)
        np_action_dict = dict_apply(action_dict,lambda x: x.detach().to('cpu').numpy())
        action_pred = np_action_dict['action_pred']
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
    image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))    # BGR | (480,640,3) -> (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
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
    parser.add_argument("--ckpt_pth", type=str, default="/home/robot/UR_Robot_Arm/coarse2fine/ckpt/dp-bowl-epoch=0600-val_loss=0.097.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--n_action_steps", type=int, default=8)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    ### params
    is_verbose=True
    args = get_args()

    ### build model
    policy = build(args.ckpt_pth, is_verbose)

    ### get data
    # data = random_data()
    data = obs_format_amend(given_data())

    ### run
    actions = infer(policy, data, args.n_obs_steps, args.n_action_steps, is_verbose).reshape(-1,8) 

    print([-0.36229283359785835,-0.4150547746810219,0.4619846199476397,0.4246134752330147,0.9042700809168371,-0.01994586432104633,0.04001474610297329,0.0])
    print_2d_arr('[INFO] robot action | raw : ', actions)

"""
[-0.3622928 -0.4150547 0.4619846 0.4246134 0.9042700 -0.0199458 0.0400147 0.0]
"""

""" CARP
[-0.3612558 -0.4116496 0.4635365 0.4243457 0.9043154 -0.0204264 0.0415581 -1.0199691 ]
[-0.3689657 -0.4083814 0.4725467 0.4250512 0.9040264 -0.0215426 0.0400445 -0.9833924 ]
[-0.3609210 -0.4115919 0.4641589 0.4241956 0.9044364 -0.0207756 0.0402632 -1.0038750 ]
[-0.3586527 -0.4072773 0.4648821 0.4244998 0.9042478 -0.0197792 0.0417681 -1.0002413 ]
[-0.3513922 -0.4171208 0.4682400 0.4248604 0.9040995 -0.0210168 0.0406935 -0.9953830 ]
[-0.3491384 -0.4118931 0.4661402 0.4246114 0.9042003 -0.0197700 0.0416655 -0.9916106 ]
[-0.3537332 -0.3945919 0.4685365 0.4256597 0.9037417 -0.0199844 0.0408102 -0.9989278 ]
[-0.3633390 -0.3803273 0.4645104 0.4255653 0.9037117 -0.0193043 0.0427394 -0.9981372 ]
"""

""" DP
[-0.3595971 -0.4172549 0.4616775 0.4205656 0.9061850 -0.0194748 0.0396745 -0.9998913 ]
[-0.3612032 -0.4173669 0.4616720 0.4205073 0.9062397 -0.0183049 0.0395990 -0.9969572 ]
[-0.3608074 -0.4053078 0.4602814 0.4203532 0.9062744 -0.0188517 0.0401803 -0.9983850 ]
[-0.3603871 -0.3771339 0.4595951 0.4201910 0.9064227 -0.0182169 0.0388006 -0.9971131 ]
[-0.3719346 -0.3472206 0.4601459 0.4194533 0.9067644 -0.0185321 0.0386504 -0.9957903 ]
[-0.4056949 -0.3289000 0.4611574 0.4190427 0.9069399 -0.0183661 0.0390607 -0.9944122 ]
[-0.4294128 -0.3244735 0.4611254 0.4185379 0.9072424 -0.0181380 0.0375248 -0.9954804 ]
[-0.4326633 -0.3242779 0.4385154 0.4199780 0.9065505 -0.0185647 0.0379485 -0.9969504 ]
"""
