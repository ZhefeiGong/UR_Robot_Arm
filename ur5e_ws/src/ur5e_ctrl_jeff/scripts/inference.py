#!/usr/bin/env python

import sys
import cv2
import rospy
import numpy as np
import geometry_msgs.msg as geometry_msgs

import ur5e_ctrl_jeff.msg

import motion_commander
from motion_commander import MotionCommander
from PIL import Image

from vla_client import VLAClient
from vla_client import load_image, get_completeTraj, get_trajNdArray

from utils import capture_image, ask_confirmation, generate_initial_img
from utils import euler_to_quaternion, quaternion_to_euler, format_state_array, action_to_command
from utils import cartesian_linear_mapping

from wrist_camera import WristSubscriber
from scene_camera import SceneSubscriber

# Compatibility for python2 and python3
if sys.version_info[0] < 3:
    input = raw_input

# def main():
#     rospy.init_node('image_subscriber_node', anonymous=True)
#     image_subscriber = SceneSubscriber()
#     rate = rospy.Rate(10)  # 10Hz
#     while not rospy.is_shutdown():
#         current_image = image_subscriber.get_current_image()
#         print(type(current_image))
#         if current_image is not None:
#             cv2.imshow("Current Image", current_image)
#             cv2.waitKey(1)
#         rate.sleep()
#     cv2.destroyAllWindows()

def run():
    """
    run the whole process

    """
    
    ### initialization
    rospy.init_node("inference")
    motion_client = MotionCommander()
    scene_image_subscriber = SceneSubscriber()
    wrist_image_subscriber = WristSubscriber()

    ### get the initial image
    ask_confirmation(prompt="we'll start the client to recieve the msg from VLA")
    vla_client = VLAClient(host="192.168.2.3", port=5050)
    
    ### run
    while True:
      
      # get the initial image
      # ask_confirmation(prompt="we'll capture the image of the scene and wrist...")
      img_path_scene = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/scene/test.jpg"
      img_path_wrist = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/wrist/test.jpg"
      scene_current_image = Image.fromarray(scene_image_subscriber.get_current_image())
      wrist_current_image = Image.fromarray(wrist_image_subscriber.get_current_image())
      img_path_initial = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/init.jpg"
      generate_initial_img( image_scene_path=img_path_scene, 
                            image_wrist_path=img_path_wrist, 
                            img_path_initial=img_path_initial, 
                            scene_current_image=scene_current_image, 
                            wrist_current_image=wrist_current_image,
                            is_path=False)
      
      # get the initial state
      # ask_confirmation(prompt="we'll recieve the state from ur5e...")
      robot_state = motion_client.get_state()
      print('[INFO] robot state | quat : \n', robot_state)
      robot_state_euler = quaternion_to_euler(robot_state)
      print('[INFO] robot state | euler : \n', robot_state_euler)
      robot_state_euler_str= format_state_array(robot_state_euler)
      print('[INFO] robot state | euler | str: \n', robot_state_euler_str)
      
      # get the actions
      # ask_confirmation(prompt="we'll recieve the action prediction from VLA...")
      initialImg = load_image(img_path_initial)
      infer_param = {
          "initialImg" : initialImg,
          # "finalImg" : finalImg,
          "instruction" : "Sweep the green cloth to the left side of the table", # 
          "template" : "12:36:5", # "class_id : index : num_action"
          "reward" : 0,
          "prompt_img" : False,
          "last_actions" : "",
          "maximumLength" : 1024,
          "robot_state" : robot_state_euler_str,
      }
      
      response = vla_client.infer_traj(infer_param)
      action_arrary = get_trajNdArray(get_completeTraj(response['traj']))
      print(action_arrary)

      ask_confirmation(prompt="we'll converse the instruction...")
      action_arrary_quaternion = euler_to_quaternion(action_arrary)
      pose_list, grip_list, duration_list = action_to_command(action_arrary_quaternion, first_duration=5, duration=5)
      print(pose_list)
      print(grip_list)
      print(duration_list)

      ask_confirmation(prompt="we'll execute the trajectory...")
      motion_client.execute_arm_gripper_trajectory(pose_list, grip_list, duration_list, is_ask_conf=False)
      
      
if __name__ == "__main__":

    run()



"""

🌟 category :
1. Take the tiger out of the red bowl and put it in the grey bowl.
2. Sweep the green cloth to the left side of the table.
3. Pick up the blue cup and put it into the brown cup.
4. Put the ranch bottle into the pot.    

"""