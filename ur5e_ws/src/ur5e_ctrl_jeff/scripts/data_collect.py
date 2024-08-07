#!/usr/bin/env python

import json
import rospy
import cv2
import os

import ur5e_ctrl_jeff.msg

import numpy as np

from wrist_camera import WristCamera
from scene_camera import SceneCamera

from utils import capture_image, action_to_command, ask_confirmation
from motion_commander import MotionCommander

import time

save_traj_path = "/home/robot/DATASET/Cloth/pose.csv"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def collect_to_pose(id=0):

    rospy.init_node("collecting")

    camera_w = WristCamera()
    camera_s = SceneCamera()

    camera_w.start()
    camera_s.start()
    motion_client = MotionCommander()

    root_path = "/home/robot/DATASET/Cloth/"+"traj"+str(id)+"/"
    root_img_path_scene = root_path + "image/scene/"
    root_img_path_wrist = root_path + "image/wrist/"
    data_list = []

    mode = "NONE"
    img_count = 1

    if camera_w.wait_for_ready() and camera_s.wait_for_ready():
        
        print(" ---- begin ---- ")

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
                    "pose" : None,
                }
                data['imgs'] = '/image/scene/scene' + str(img_count) + '.jpg'
                data['imgw'] = '/image/wrist/wrist' + str(img_count) + '.jpg'

                robot_state = motion_client.get_state()
                data['pose'] = np.array(robot_state[0])
                data_list.append(data)

                img_count += 1
                time.sleep(0.5)

            ###
            elif mode == "quit":

                json_path = root_path + "data.json"
                with open(json_path, 'w') as json_file:
                    json_str = json.dumps(data_list, cls=NumpyEncoder, indent=4, ensure_ascii=False)
                    json_file.write(json_str)

                break
        
    else:
        print("[ERROR] 等待相机准备超时")

    camera_w.stop()
    camera_s.stop()
    cv2.destroyAllWindows()

def save_data(file_path, new_data):
    if os.path.exists(file_path):
        existing_data = np.loadtxt(file_path, delimiter=',')
        if existing_data.ndim == 1:
            existing_data = existing_data[np.newaxis, :]
        data_to_save = np.vstack((existing_data, new_data))
    else:
        data_to_save = new_data
    
    np.savetxt(file_path, data_to_save, delimiter=',')

def listen_to_pose(is_open:bool=True):

    rospy.init_node("listening")
    motion_client = MotionCommander()
    pose_list = []

    print(" ---- begin ---- ")

    # window_name = "window"
    # cv2.namedWindow(window_name)
    # blank_image = np.zeros((500,500,3),np.uint8)
    # while True : 
    #     cv2.imshow(window_name,blank_image)
    #     key = cv2.waitKey(10) & 0xFF
    #     if key == ord('c'):
    #         robot_state = motion_client.get_state()
    #         print(robot_state[0])
    #         pose_list.append(robot_state[0])
    #     elif key == ord('q'):
    #         array = np.vstack(pose_list)
    #         np.save(save_traj_path, array)
    #         print("save to ", save_traj_path)
    #         break

    robot_state = motion_client.get_state()
    if is_open : 
        robot_state[0][-1]=0.0
    else :
        robot_state[0][-1]=1.0
    save_data(save_traj_path, robot_state[0])
    print("save to ", save_traj_path)

def move_to_pose():

    rospy.init_node("moving")
    if os.path.exists(save_traj_path):
        traj_arrary = np.loadtxt(save_traj_path, delimiter=',')
    motion_client = MotionCommander()

    pose_list, grip_list, duration_list = action_to_command(traj_arrary, first_duration=10, duration=10)

    print(pose_list)
    print(grip_list)
    print(duration_list)

    ask_confirmation(prompt="we'll execute the trajectory...")
    for i in range(len(traj_arrary)):
        motion_client.execute_arm_gripper_trajectory([pose_list[i]], [grip_list[i]], [duration_list[i]])


if __name__ == "__main__":

    # collect_to_pose(id=40)

    # is_open = True
    # listen_to_pose(is_open)

    # move_to_pose()
