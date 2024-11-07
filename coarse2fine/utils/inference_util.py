import json
import os
import torch
import wandb
import torch.nn as nn

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn

from utils.pytorch_util import replace_submodules
from utils.robomimic_config_util import get_robomimic_config
import utils.crop_randomizer as dmvc

def load_shape_meta():
    """
    :func:
    """
    shape_meta = {
        "action": {
            "shape": [10]
        },
        "obs": {
            "agentview_image": {
                # "shape": [3, 84, 84],
                "shape": [3, 120, 160],
                "type": "rgb"
            },
            "robot0_eef_pos": {
                "shape": [3]
            },
            "robot0_eef_quat": {
                "shape": [4]
            },
            "robot0_eye_in_hand_image": {
                # "shape": [3, 84, 84],
                "shape": [3, 120, 160],
                "type": "rgb"
            },
            "robot0_gripper_qpos": {
                # "shape": [2]
                "shape": [1]
            }
        }
    }
    return shape_meta



def load_obs_encoder(shape_meta):
    """
    :func:
    """
    # initialize the shape meta
    
    # crop_shape=(76, 76) # sim
    crop_shape=(112, 112) # 🔥 real 🔥

    obs_encoder_group_norm=True
    eval_fixed_crop=True
    # parse shape_meta
    action_shape = shape_meta['action']['shape']
    assert len(action_shape) == 1
    action_dim = action_shape[0]
    obs_shape_meta = shape_meta['obs']
    obs_config = {
        'low_dim': [],
        'rgb': [],
        'depth': [],
        'scan': []
    }
    obs_key_shapes = dict()
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        obs_key_shapes[key] = list(shape)
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            obs_config['rgb'].append(key)
        elif type == 'low_dim':
            obs_config['low_dim'].append(key)
        else:
            raise RuntimeError(f"Unsupported obs type: {type}")
    # get raw robomimic config
    config = get_robomimic_config(
        algo_name='bc_rnn',
        hdf5_type='image',
        task_name='square',
        dataset_type='ph')
    with config.unlocked():
        # set config with shape_meta
        config.observation.modalities.obs = obs_config
        # set random crop parameter
        ch, cw = crop_shape
        for key, modality in config.observation.encoder.items():
            if modality.obs_randomizer_class == 'CropRandomizer':
                modality.obs_randomizer_kwargs.crop_height = ch
                modality.obs_randomizer_kwargs.crop_width = cw
    # init global state
    ObsUtils.initialize_obs_utils_with_config(config)
    # load model
    policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )
    obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs'] # 🔥 🔥 🔥
    # replace batch norm with group norm
    if obs_encoder_group_norm:
        replace_submodules(
            root_module=obs_encoder,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features//16, 
                num_channels=x.num_features)
        )
        # obs_encoder.obs_nets['agentview_image'].nets[0].nets
    # obs_encoder.obs_randomizers['agentview_image']
    if eval_fixed_crop:
        replace_submodules(
            root_module=obs_encoder,
            predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
            func=lambda x: dmvc.CropRandomizer(
                input_shape=x.input_shape,
                crop_height=x.crop_height,
                crop_width=x.crop_width,
                num_crops=x.num_crops,
                pos_enc=x.pos_enc
            )
        )
    # already initialized
    return obs_encoder
