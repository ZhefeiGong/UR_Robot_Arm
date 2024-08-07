#!/usr/bin/env python

import sys
import rospy
import numpy as np
import geometry_msgs.msg as geometry_msgs

import ur5e_ctrl_jeff.msg

import motion_commander
from motion_commander import MotionCommander

from realsense_camera import RealsenseCamera
from wrist_camera import WristCamera

from vla_client import VLAClient
from vla_client import load_image, get_completeTraj, get_trajNdArray

from utils import capture_image, ask_confirmation, generate_initial_img
from utils import euler_to_quaternion, quaternion_to_euler, format_state_array, action_to_command
from utils import cartesian_linear_mapping


# Compatibility for python2 and python3
if sys.version_info[0] < 3:
    input = raw_input

def run():
    """
    run the whole process

    """
  
    # initialization
    rospy.init_node("inference")
    motion_client = MotionCommander()
    
    # get the initial image
    ask_confirmation(prompt="we'll start the client to recieve the msg from VLA")
    vla_client = VLAClient(host="192.168.2.5", port=5050)

    # [[x_min,x_max],[y...],[z...],[rx...],[ry...],[rz...]]

    cart_vla = np.array([[0.18, 0.68],
                         [-0.27, 0.38],
                         [-0.20, 0.20],
                         [-1.0,1.0],
                         [-0.51,0.40],
                         [0.77,2.60]])

    cart_ur5e = np.array([[-0.80, 0.00],
                          [-0.80, 0.00],
                          [0.35, 0.95],
                          [-1.0,1.0],
                          [-0.51,0.40],
                          [0.77,2.60]])
    
    # run
    while True:
      
      # get the initial image
      ask_confirmation(prompt="we'll capture the image of the scene and wrist...")
      img_path_scene = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/scene/test.jpg"
      img_path_wrist = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/wrist/test.jpg"
      img_path_initial = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/init.jpg"
      generate_initial_img(img_path_scene, img_path_wrist, img_path_initial)
      
      # get the initial state
      ask_confirmation(prompt="we'll recieve the state from ur5e...")
      # print('[INFO] robot : ', motion_client.get_arm_cartesian_state())
      robot_state = motion_client.get_state()
      print('[INFO] robot state | quat : \n', robot_state)
      robot_state_euler = quaternion_to_euler(robot_state)
      # print('[INFO] robot state | euler | ur5e : \n', robot_state_euler)
      robot_state_euler_vla = cartesian_linear_mapping(robot_state_euler, cart_ur5e, cart_vla)
      # print('[INFO] robot state | euler | vla: \n', robot_state_euler_vla)
      robot_state_euler_str= format_state_array(robot_state_euler_vla)
      print('[INFO] robot state | euler | vla | str: \n', robot_state_euler_str)
      
      # get the actions
      ask_confirmation(prompt="we'll recieve the action prediction from VLA...")
      initialImg = load_image(img_path_initial)
      infer_param = {
          "initialImg" : initialImg,
          # "finalImg" : finalImg,
          "instruction" : "sweep the green cloth to the left side of the table", # 
          "template" : "12:37:15", # "class_id : index : num_action"
          "reward" : 0,
          "prompt_img" : False,
          "last_actions" : "",
          "maximumLength" : 1024,
          "robot_state" : robot_state_euler_str,
      }
      
      response = vla_client.infer_traj(infer_param)
      action_arrary_vla = get_trajNdArray(get_completeTraj(response['traj']))
      action_arrary = cartesian_linear_mapping(action_arrary_vla, cart_vla, cart_ur5e)
      print(action_arrary)

      ask_confirmation(prompt="we'll converse the instruction...")
      action_arrary_quaternion = euler_to_quaternion(action_arrary)
      pose_list, grip_list, duration_list = action_to_command(action_arrary_quaternion, first_duration=2)

      print(pose_list)
      print(grip_list)
      print(duration_list)

      ask_confirmation(prompt="we'll execute the trajectory...")
      motion_client.execute_arm_gripper_trajectory(pose_list, grip_list, duration_list)
      
if __name__ == "__main__":

    run()

"""

Left : 
position: 
  x: 0.7475911898993305
  y: -0.5743821463293052
  z: 0.11661759649524651
orientation: 
  x: -0.48674986917500396
  y: 0.8077668386818404
  z: -0.261060603773449
  w: 0.20599674837604703

Forward : 
position: 
  x: -0.6506659758656678
  y: -0.6831609515964959
  z: 0.12006975657531037
orientation: 
  x: -0.27438078876285177
  y: -0.9013413588315229
  z: 0.3300476095421987
  w: 0.0580302770379361

Right:
position: 
  x: -0.7686532009130691
  y: 0.5470185098037601
  z: 0.11999203731955846
orientation: 
  x: -0.7976858364614958
  y: -0.5014361063984752
  z: 0.21140990333237297
  w: 0.2599326648993115

Back:
position: 
  x: 0.6524083783606978
  y: 0.6814611455488356
  z: 0.11983193523695325
orientation: 
  x: -0.9010403076688259
  y: 0.2755634300749633
  z: -0.05830160341065199
  w: 0.3298364488669124

Original:
position: 
  x: -0.00017823228760290258
  y: -0.23209021588849815
  z: 1.0798836533883538
orientation: 
  x: -0.00021111763133863402
  y: -0.7066607816230179
  z: 0.707552436110109
  w: 0.0002128378286350897

Manip:
position: 
  x: -0.42380384169870894
  y: -0.6873218873293475
  z: 0.5792064915753267
orientation: 
  x: 0.2880332113164632
  y: 0.9333942609689806
  z: -0.20724109068518082
  w: 0.05350843952605036


"""
