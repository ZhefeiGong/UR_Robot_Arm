#!/usr/bin/env python

import sys
import cv2
import rospy
import math
from copy import deepcopy
import numpy as np

from PIL import Image
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface 
from rtde_io import RTDEIOInterface as RTDEIO
import time
import rospy
import time
import numpy as np
from std_msgs.msg import Float64MultiArray

### !!! import the package before importing the others !!! ###
import tele_ctrl_jeff
from robotiq_gripper import RobotiqGripper
# from space_mouse import SpaceMouse
from utils import axis_to_euler, axis_to_quat, quat_to_axis, euler_to_axis, ask_confirmation, preprocess_image, print_2d_arr
from wrist_camera import WristSubscriber
from scene_camera import SceneSubscriber
from policy_client import PolicyClient

# define robot parameters
ROBOT_HOST = "192.168.2.6"
GP_OPEN = 0
GP_CLOSE = 1
GP_CRITERIA = 0.5
IS_VERBOSE = True
IS_CHECK = True
IS_SAVE = True
MOVE_SPEED = 0.015
IS_CARP = True

"""BOWL - 131
blue -> red | grey
green -> red | grey
"""

"""CUP- 134
red
blue
green
"""

"""TIGER - 58
red -> grey
red -> green
"""

def image_format_amend(img_pth, RESIZE_WIDTH = 160, RESIZE_HEIGHT = 120):
    image = cv2.imread(img_pth)                                 # BGR
    image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))    # BGR | (480,640,3) -> (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # RGB               # 
    image_arr = np.array(image_rgb)                             # BHWC | [0,255]
    return image_arr
def given_data():
    obs=dict()
    obs['agentview_image'] = image_format_amend("/home/robot/UR_Robot_Arm/coarse2fine/data/scene1.jpg")[None,None,...]
    obs['robot0_eye_in_hand_image'] = image_format_amend("/home/robot/UR_Robot_Arm/coarse2fine/data/wrist1.jpg")[None,None,...]
    obs['robot0_eef_pos'] = np.array([-0.36229283359785835,-0.4150547746810219,0.4619846199476397])[None,None,...]
    obs['robot0_eef_quat'] = np.array([0.4246134752330147, 0.9042700809168371, -0.01994586432104633, 0.04001474610297329,])[None,None,...]
    obs['robot0_gripper_qpos'] = np.array([0.0])[None,None,...]
    return obs

# py3
if sys.version_info[0] < 3: 
    """ 
    @func : compatibility for python2 and python3 
    """
    input = raw_input

# run
def run():
    """
    @func : run the whole process
    """

    ### initialization
    rospy.init_node("inference_node")

    ### instances
    scene_image_subscriber = SceneSubscriber()
    wrist_image_subscriber = WristSubscriber()
    policy_client = PolicyClient()

    ## RTDE
    print("[INFO] Starting RTDE...")
    rtde_ctl = RTDEControlInterface(ROBOT_HOST)
    rtde_rcv = RTDEReceiveInterface(ROBOT_HOST)
    rtde_io = RTDEIO(ROBOT_HOST)
    
    ## Gripper
    print("[INFO] Creating gripper...")
    gripper = RobotiqGripper()
    print("[INFO] Connecting to gripper...")
    gripper.connect(ROBOT_HOST, 63352)
    print("[INFO] Activating gripper...")
    gripper.activate()
    gripper_status = GP_OPEN

    ### params
    if IS_CARP:
        scene_pth = "/home/robot/UR_Robot_Arm/tele_ws/src/tele_ctrl_jeff/img/carp/scene.jpg"
        wrist_pth = "/home/robot/UR_Robot_Arm/tele_ws/src/tele_ctrl_jeff/img/carp/wrist.jpg"
    else:
        scene_pth = "/home/robot/UR_Robot_Arm/tele_ws/src/tele_ctrl_jeff/img/dp/scene.jpg"
        wrist_pth = "/home/robot/UR_Robot_Arm/tele_ws/src/tele_ctrl_jeff/img/dp/wrist.jpg"

    ### run
    while True:
        
        if IS_VERBOSE: 
            ask_confirmation(prompt="we'll catch the current images of the scene and wrist...")
        
        # Capture and preprocess images
        scene_cur_image = Image.fromarray(scene_image_subscriber.get_current_image())
        wrist_cur_image = Image.fromarray(wrist_image_subscriber.get_current_image())
        scene_img_arr, wrist_img_arr = preprocess_image(scene_cur_image, wrist_cur_image)

        # Save images if required
        if IS_SAVE:
            Image.fromarray(scene_img_arr).save(scene_pth) # [H,W,3]
            Image.fromarray(wrist_img_arr).save(wrist_pth) # [H,W,3]
        
        ### get the robot state
        if IS_VERBOSE: ask_confirmation(prompt="we'll catch the current state of the robot...")
        actual_pose = rtde_rcv.getActualTCPPose()
        cart = np.array(actual_pose[:3])
        axis_angle = np.array(actual_pose[3:])
        euler = axis_to_euler(axis_angle)
        quat = axis_to_quat(axis_angle)
        print_2d_arr('[INFO] robot state | raw : ', np.array([[*cart,*quat]]))

        ### build obs
        obs=dict()
        obs['agentview_image'] = scene_img_arr[None,None,...]                       # [H,W,3] -> [1,1,H,W,3]
        obs['robot0_eye_in_hand_image'] = wrist_img_arr[None,None,...]              # [H,W,3] -> [1,1,H,W,3]
        obs['robot0_eef_pos'] = np.array(cart)[None,None,...]                       # [3,] -> [1,1,3,]
        obs['robot0_eef_quat'] = np.array(quat)[None,None,...]                      # [4,] -> [1,1,4,]
        obs['robot0_gripper_qpos'] = np.array([gripper_status])[None,None,...]      # [1,] -> [1,1,1,]
        # obs = given_data()

        ### get the actions
        if IS_VERBOSE: ask_confirmation(prompt="we'll send the observation from model...")
        # start_time =time.time()
        actions_pred = policy_client.predict_traj(obs)
        # end_time =time.time()
        # print(f'[INFO] inference time:{(end_time-start_time):.6f} s')
        actions_pred = np.array(actions_pred).reshape(-1,8)                         # [H,D] | pos(3) + rot(4) + grip(1)
        print_2d_arr('[INFO] robot action | pred : ', actions_pred)

        ### run
        if IS_CHECK: ask_confirmation(prompt="we'll execute the trajectory...")
        print('[INFO] robot action | exe')
        for action in actions_pred:
            ## move
            next_state = [*action[:3], *quat_to_axis(action[3:7])]
            print_2d_arr(None, [next_state])
            # if IS_VERBOSE: ask_confirmation(prompt="we'll execute the trajectory...")      
            rtde_ctl.moveL(next_state, speed=MOVE_SPEED)
            
            ## gripper
            if action[-1] > GP_CRITERIA and gripper_status == GP_OPEN:
                gripper_status = GP_CLOSE
                gripper.move(gripper.get_closed_position(), 255, 255)
            if action[-1] < GP_CRITERIA and gripper_status == GP_CLOSE:
                gripper_status = GP_OPEN
                gripper.move(gripper.get_open_position(), 255, 255)
            print(gripper_status)


if __name__ == "__main__":
    run()


"""Garbage Repo
# if IS_VERBOSE: ask_confirmation(prompt="we'll execute the trajectory...")
# temp = [-0.2689094, -0.4695275, 0.3553323, 1.3320397, 2.7912169, -0.0922271]
# axis = temp[3:]
# print(axis)
# print(axis_to_euler(axis))
# print(axis_to_quat(axis))
# print(euler_to_axis(axis_to_euler(axis)))
# print(quat_to_axis(axis_to_quat(axis)))     
# rtde_ctl.moveL(temp, speed=MOVE_SPEED)
"""

"""Garbage for test
def image_format_amend(img_pth, RESIZE_WIDTH = 160, RESIZE_HEIGHT = 120):
    image = cv2.imread(img_pth)                                 # BGR
    image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))    # BGR | (480,640,3) -> (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # RGB               # 
    image_arr = np.array(image_rgb)                             # BHWC | [0,255]
    return image_arr
def given_data():
    obs=dict()
    obs['agentview_image'] = image_format_amend("/home/robot/UR_Robot_Arm/coarse2fine/data/scene1.jpg")[None,None,...]
    obs['robot0_eye_in_hand_image'] = image_format_amend("/home/robot/UR_Robot_Arm/coarse2fine/data/wrist1.jpg")[None,None,...]
    obs['robot0_eef_pos'] = np.array([-0.36229283359785835,-0.4150547746810219,0.4619846199476397])[None,None,...]
    obs['robot0_eef_quat'] = np.array([0.4246134752330147, 0.9042700809168371, -0.01994586432104633, 0.04001474610297329,])[None,None,...]
    obs['robot0_gripper_qpos'] = np.array([0.0])[None,None,...]
    return obs
"""
