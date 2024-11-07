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
import numpy as np
from std_msgs.msg import Float64MultiArray

### !!! import the package before importing the others !!! ###
import tele_ctrl_jeff
from robotiq_gripper import RobotiqGripper
# from space_mouse import SpaceMouse
from utils import axis_to_euler, axis_to_quat, quat_to_axis, euler_to_axis, ask_confirmation, preprocess_image, print_2d_arr
from wrist_camera import WristSubscriber
from scene_camera import SceneSubscriber
from cap_client import CAPClient

# define robot parameters
ROBOT_HOST = "192.168.2.4"
GP_OPEN = 0
GP_CLOSE = 1
GP_CRITERIA = 0.5
IS_VERBOSE = True
IS_SAVE = True
MOVE_SPEED = 0.01

if sys.version_info[0] < 3:
    """ 
    @func : compatibility for python2 and python3 
    """
    input = raw_input

def run():
    """
    @func : run the whole process
    """

    ### initialization
    rospy.init_node("inference_node")

    ### instances
    scene_image_subscriber = SceneSubscriber()
    wrist_image_subscriber = WristSubscriber()
    policy_client = CAPClient()

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
    scene_pth = "/home/robot/UR_Robot_Arm/tele_ws/src/tele_ctrl_jeff/img/cap/scene.jpg"
    wrist_pth = "/home/robot/UR_Robot_Arm/tele_ws/src/tele_ctrl_jeff/img/cap/wrist.jpg"

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

        obs=dict()
        obs['robot0_eef_pos'] = np.array(cart)[None,None,...]                       # [3,] -> [1,1,3,]
        obs['robot0_eef_quat'] = np.array(quat)[None,None,...]                      # [4,] -> [1,1,4,]
        obs['robot0_gripper_qpos'] = np.array([gripper_status])[None,None,...]      # [1,] -> [1,1,1,]
        obs['robot0_eye_in_hand_image'] = wrist_img_arr[None,None,...]              # [H,W,3] -> [1,1,H,W,3]
        obs['agentview_image'] = scene_img_arr[None,None,...]                       # [H,W,3] -> [1,1,H,W,3]

        ### get the actions
        if IS_VERBOSE: ask_confirmation(prompt="we'll send the observation from model...")
        actions_pred = policy_client.predict_traj(obs)
        actions_pred = np.array(actions_pred).reshape(-1,8)                         # [H,D] | pos(3) + rot(4) + grip(1)
        print_2d_arr('[INFO] robot action | pred : ', actions_pred)

        ### run
        if IS_VERBOSE: ask_confirmation(prompt="we'll execute the trajectory...")
        print('[INFO] robot action | exe')
        for action in actions_pred:
            ## move
            next_state = [*action[:3], *quat_to_axis(action[3:7])]
            print_2d_arr(None, [next_state])
            # if IS_VERBOSE: ask_confirmation(prompt="we'll execute the trajectory...")      
            rtde_ctl.moveL(next_state, speed=MOVE_SPEED)
            
            ## gripper
            if action[-1] > GP_CRITERIA:
                gripper_status = GP_CLOSE
                gripper.move(gripper.get_closed_position(), 255, 255)
            elif gripper_status == GP_CLOSE:
                gripper_status = GP_OPEN
                gripper.move(gripper.get_open_position(), 255, 255)
        
        
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