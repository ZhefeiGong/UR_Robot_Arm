#!/usr/bin/env python

import sys
import cv2
import rospy
import math
from copy import deepcopy
import numpy as np
import geometry_msgs.msg as geometry_msgs
import ur5e_ctrl_jeff.msg
import motion_commander
from motion_commander import MotionCommander
from joint_listener import JointStateListener
from PIL import Image

from utils import ask_confirmation, action_to_command

# from utils import euler_to_quaternion, quaternion_to_euler, format_state_array
# from utils import cartesian_linear_mapping, curtail_duplicate_action
# import json
# from std_srvs.srv import Trigger, TriggerRequest

from wrist_camera import WristSubscriber
from scene_camera import SceneSubscriber

from cap_client import CAPClient

if sys.version_info[0] < 3:
    """ 
    @func : compatibility for python2 and python3 
    """
    input = raw_input

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

def preprocess_image(scene_image, wrist_image, resize_width=160, resize_height=120):
    """
    Preprocess scene and wrist images by resizing, rotating, and formatting as numpy arrays.
    Args:
        scene_image (PIL.Image): Image from the scene camera.
        wrist_image (PIL.Image): Image from the wrist camera.
        resize_width (int): Width to resize the image.
        resize_height (int): Height to resize the image.
    Returns:
        tuple: Preprocessed scene and wrist images as numpy arrays with shape (H, W, C).
    """
    # Preprocess scene image
    scene_image = scene_image.convert("RGB")
    scene_image = scene_image.transpose(Image.ROTATE_180)               # Rotate by 180 degrees if necessary
    scene_image = scene_image.resize((resize_width, resize_height))     # Resize to desired dimensions
    scene_image = np.array(scene_image)                                 # Convert to numpy array with shape (H, W, C)
    scene_image = cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB)          # to RGb
    
    # Preprocess wrist image
    wrist_image = wrist_image.convert("RGB")
    wrist_image = wrist_image.resize((resize_width, resize_height))     # Resize to desired dimensions
    wrist_image = np.array(wrist_image)                                 # Convert to numpy array with shape (H, W, C)
    wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)          # to RGB

    return scene_image, wrist_image


def run():
    """
    @func : run the whole process
    """

    ### initialization
    rospy.init_node("inference")

    ### nodes
    motion_client = MotionCommander()
    joint_subscriber = JointStateListener()
    scene_image_subscriber = SceneSubscriber()
    wrist_image_subscriber = WristSubscriber()
    policy_client = CAPClient()

    ### params
    scene_pth = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/cap/scene.jpg"
    wrist_pth = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/cap/wrist.jpg"
    is_ask_confirmation = True
    is_image_save = True

    ### run
    while True:
        
        if is_ask_confirmation: 
            ask_confirmation(prompt="we'll catch the current images of the scene and wrist...")
        
        # Capture and preprocess images
        scene_cur_image = Image.fromarray(scene_image_subscriber.get_current_image())
        wrist_cur_image = Image.fromarray(wrist_image_subscriber.get_current_image())
        scene_img_arr, wrist_img_arr = preprocess_image(scene_cur_image, wrist_cur_image)

        # Save images if required
        if is_image_save:
            Image.fromarray(scene_img_arr).save(scene_pth) # [H,W,3]
            Image.fromarray(wrist_img_arr).save(wrist_pth) # [H,W,3]
        
        ### get the robot state
        if is_ask_confirmation: ask_confirmation(prompt="we'll catch the current state of the robot...")
        robot_state = motion_client.get_state() # [1,7] | xyz,xyzw,g
        print_2d_arr('[INFO] robot state | quat : ', robot_state)
        obs=dict()
        obs['robot0_eef_pos'] = np.array(robot_state[0][:3])[None,None,...]         # [3,] -> [1,1,3,]
        obs['robot0_eef_quat'] = np.array(robot_state[0][3:7])[None,None,...]       # [4,] -> [1,1,4,]
        obs['robot0_gripper_qpos'] = np.array(robot_state[0][-1:])[None,None,...]   # [1,] -> [1,1,1,]
        obs['robot0_eye_in_hand_image'] = wrist_img_arr[None,None,...]              # [H,W,3] -> [1,1,H,W,3]
        obs['agentview_image'] = scene_img_arr[None,None,...]                       # [H,W,3] -> [1,1,H,W,3]

        # ### standard
        # def image_format_amend(img_pth, RESIZE_WIDTH = 160, RESIZE_HEIGHT = 120):
        #     image = cv2.imread(img_pth)                                 # BGR
        #     image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))    # BGR | (480，640，3) -> (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
        #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # RGB               # 
        #     image_arr = np.array(image_rgb)                             # BHWC | [0,255]
        #     return image_arr
        # def given_data():
        #     obs=dict()
        #     obs['agentview_image'] = image_format_amend("/home/robot/UR_Robot_Arm/coarse2fine/data/scene1.jpg")[None,None,...]
        #     obs['robot0_eye_in_hand_image'] = image_format_amend("/home/robot/UR_Robot_Arm/coarse2fine/data/wrist1.jpg")[None,None,...]
        #     obs['robot0_eef_pos'] = np.array([-0.36229283359785835,-0.4150547746810219,0.4619846199476397])[None,None,...]
        #     obs['robot0_eef_quat'] = np.array([0.4246134752330147, 0.9042700809168371, -0.01994586432104633, 0.04001474610297329,])[None,None,...]
        #     obs['robot0_gripper_qpos'] = np.array([0.0])[None,None,...]
        #     return obs
        # obs = given_data()

        ### get the actions
        if is_ask_confirmation: ask_confirmation(prompt="we'll send the observation from model...")
        actions_pred = policy_client.predict_traj(obs)
        actions_pred = np.array(actions_pred).reshape(-1,8)         # [H,D] | pos(3) + rot(4) + grip(1)
        print_2d_arr('[INFO] robot action | raw : ', actions_pred)
        
        ### get the command
        if is_ask_confirmation: ask_confirmation(prompt="we'll converse the instruction...")
        # action_arrary_quaternion = euler_to_quaternion(robot_actions)
        pose_list, grip_list, duration_list = action_to_command(actions_pred, first_duration=5, duration=5)
        
        # pose_list=[pose_list[0]]
        # grip_list=[grip_list[0]]
        # duration_list=[duration_list[0]]
        
        print('[INFO] robot action | pose : \n',pose_list)
        print('[INFO] robot action | gripper : \n',grip_list)
        print('[INFO] robot action | duration : \n',duration_list)
        
        ### run
        if is_ask_confirmation: ask_confirmation(prompt="we'll execute the trajectory...")
        motion_client.execute_arm_gripper_trajectory(pose_list, grip_list, duration_list, is_ask_conf=False)
        
        
if __name__ == "__main__":
    run()
