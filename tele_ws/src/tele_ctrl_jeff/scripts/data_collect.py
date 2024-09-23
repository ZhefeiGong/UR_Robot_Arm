#!/usr/bin/env python

import os
import cv2
import json
import time
import rospy
import argparse

import numpy as np
from wrist_camera import WristCamera
from scene_camera import SceneCamera
from utils import capture_image, action_to_command, ask_confirmation


CLASS = "Cloth"
SLEEP = 0.5
ID = 54
TASK = "Sweep the green cloth to the left side of the table"
"""
    🌟 category :
    1. Take the tiger out of the red bowl and put it in the grey bowl.
    2. Sweep the green cloth to the left side of the table.
    3. Pick up the blue cup and put it into the brown cup.
    4. Put the ranch bottle into the pot.
"""

COMMAND = "move"
"""
    🌟 category :
    1. listen
    2. move
    3. collect
"""


class NumpyEncoder(json.JSONEncoder):
    """
    
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def ensure_folder_exists(folder_path):
    """
    Check if the folder exists
    
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return

def save_data(file_path, new_data):
    """
    
    """

    if os.path.exists(file_path):
        existing_data = np.loadtxt(file_path, delimiter=',')
        if existing_data.ndim == 1:
            existing_data = existing_data[np.newaxis, :]
        data_to_save = np.vstack((existing_data, new_data))
    else:
        data_to_save = new_data
    
    np.savetxt(file_path, data_to_save, delimiter=',')

def collect_to_pose(args):
    """
    
    """

    rospy.init_node("collecting")

    time_sleep = SLEEP

    camera_w = WristCamera()
    camera_s = SceneCamera()

    camera_w.start()
    camera_s.start()
    motion_client = MotionCommander()

    root_path = f"/home/robot/DATASET/{CLASS}/"+"traj"+str(args['collect_id'])+"/"
    root_img_path_scene = root_path + "image/scene/"
    root_img_path_wrist = root_path + "image/wrist/"
    ensure_folder_exists(root_path)
    ensure_folder_exists(root_img_path_scene)
    ensure_folder_exists(root_img_path_wrist)
    
    data_list = []

    mode = "NONE"
    img_count = 1

    if camera_w.wait_for_ready() and camera_s.wait_for_ready():
        
        rospy.loginfo(" ---- begin ---- ")

        window_name = "window"
        cv2.namedWindow(window_name)
        blank_image = np.zeros((500,500,3),np.uint8)

        while True : 
            
            cv2.imshow(window_name,blank_image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('c'):
                mode = "collecting"
            elif key == ord('q'):
                mode = "quit"

            ###
            if mode == "collecting":
                
                img_path_scene = root_img_path_scene + 'scene' + str(img_count) + '.jpg'
                img_path_wrist = root_img_path_wrist + 'wrist' + str(img_count) + '.jpg'

                capture_image(camera_w,img_path_wrist)
                capture_image(camera_s,img_path_scene)

                data = {
                    "imgw" : None,
                    "imgs" : None,
                    "task" : None,
                    "pose" : None,
                }
                data['imgs'] = '/image/scene/scene' + str(img_count) + '.jpg'
                data['imgw'] = '/image/wrist/wrist' + str(img_count) + '.jpg'
                data['task'] = args['task']

                robot_state = motion_client.get_state()
                data['pose'] = np.array(robot_state[0])
                data_list.append(data)

                img_count += 1
                time.sleep(time_sleep)

            ###
            elif mode == "quit":

                json_path = root_path + "data.json"
                with open(json_path, 'w') as json_file:
                    json_str = json.dumps(data_list, cls=NumpyEncoder, indent=4, ensure_ascii=False)
                    json_file.write(json_str)

                break
            
    else:
        rospy.loginfo("[ERROR] wait for the camera to time out")

    camera_w.stop()
    camera_s.stop()
    cv2.destroyAllWindows()

def listen_to_pose(args):
    """
    
    """

    rospy.init_node("listening")
    motion_client = MotionCommander()
    rospy.loginfo(" ---- begin ---- ")

    robot_state = motion_client.get_state()[0]
    if args['is_gripper_open'] : 
        robot_state[-1]=0.0
    else :
        robot_state[-1]=1.0
    save_data(args['save_traj_path'], robot_state)

    rospy.loginfo(f"save to {args['save_traj_path']}")

def move_to_pose(args):
    """
    
    """

    rospy.init_node("moving")
    if os.path.exists(args['save_traj_path']):
        traj_arrary = np.loadtxt(args['save_traj_path'], delimiter=',')
    motion_client = MotionCommander()
    rospy.loginfo(" ---- begin ---- ")

    pose_list, grip_list, duration_list = action_to_command(traj_arrary, first_duration=8, duration=8)

    rospy.loginfo(pose_list)
    rospy.loginfo(grip_list)
    rospy.loginfo(duration_list)

    ask_confirmation(prompt="we'll execute the trajectory...")
    motion_client.execute_arm_gripper_trajectory(pose_list, grip_list, duration_list)


if __name__ == "__main__":

    ###
    args = {
        'collect_id': ID,
        'is_gripper_open': True,
        'save_traj_path': f"/home/robot/DATASET/{CLASS}/pose.csv",
        'task' : TASK, 
        'command': COMMAND , 
    }
    
    ###
    if args['command'] == "collect":
        collect_to_pose(args)
    elif args['command'] == "listen":
        listen_to_pose(args)
    elif args['command'] == "move":
        move_to_pose(args)
    else:
        rospy.logwarn("the command is not found")
