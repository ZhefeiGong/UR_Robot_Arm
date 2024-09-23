#!/usr/bin/env python

import os
import cv2
import json
import time
import rospy
import argparse
from PIL import Image
import numpy as np
from std_msgs.msg import Float64MultiArray

import tele_ctrl_jeff
from wrist_camera import WristSubscriber
from scene_camera import SceneSubscriber

ROOTPATH = "/home/robot/DATASET"
NAME = "test"
SLEEP = 1.0
ID = 0
TASK = "Sweep the green cloth to the left side of the table"
"""
    🌟 category :
    1. Take the tiger out of the red bowl and put it in the grey bowl.
    2. Sweep the green cloth to the left side of the table.
    3. Pick up the blue cup and put it into the brown cup.
    4. Put the ranch bottle into the pot.
"""

def preprocess_image(scene_Image, wrist_Image):
    """
    @func : 
    """
    # scene
    scene_Image = scene_Image.convert("RGB")
    b,g,r = scene_Image.split()
    scene_Image = Image.merge("RGB", (r,g,b))
    scene_Image = scene_Image.transpose(Image.ROTATE_180)
    # wrist
    wrist_Image = wrist_Image.convert("RGB")
    b,g,r = wrist_Image.split()
    wrist_Image = Image.merge("RGB", (r,g,b))
    wrist_Image = wrist_Image.transpose(Image.ROTATE_180)

    return scene_Image, wrist_Image

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class RobotStateSubcriber:
    def __init__(self):
        rospy.Subscriber('robot/pose', Float64MultiArray, self.callback)
        self.state = None
    def callback(self, msg):
        self.state = msg.data
    def get_current_state(self):
        return self.state

def ensure_folder_exists(folder_path, args):
    """
    Check if the folder exists
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        count= args['id']
        while True:
            folder_path_tmp = folder_path + str(count)
            if not os.path.exists(folder_path_tmp):
                os.makedirs(folder_path_tmp)
                return folder_path_tmp
            else:
                count += 1

def run(args):
    """
    
    """
    rospy.init_node("data_collect")

    time_sleep = args['time_sleep']
    
    scene_image_subscriber = SceneSubscriber()
    wrist_image_subscriber = WristSubscriber()
    robot_state_subscriber = RobotStateSubcriber()
    
    data_list = []
    img_count = 1
    rospy.loginfo("---------------[BEGIN]---------------")
    window_name = "window"
    cv2.namedWindow(window_name)
    mode = "NONE"
    blank_image = np.zeros((500,500,3),np.uint8)

    while True : 
        
        cv2.imshow(window_name,blank_image)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('b'):
            root_path = args['root_path'] + '/' + args['name']
            root_path = ensure_folder_exists(root_path, args)
            root_img_path_scene = root_path +'/'+ "image/scene/"
            root_img_path_wrist = root_path +'/'+ "image/wrist/"
            ensure_folder_exists(root_img_path_scene, args)
            ensure_folder_exists(root_img_path_wrist, args)
            rospy.loginfo(f"[INFO] collect to : {root_path}")
            data_list = []
            img_count = 1
            mode = "collecting"
        elif key == ord('e'):
            json_path = root_path + "/" +"data.json"
            with open(json_path, 'w') as json_file:
                json_str = json.dumps(data_list, cls=NumpyEncoder, indent=4, ensure_ascii=False)
                json_file.write(json_str)
            rospy.loginfo(f"[INFO] save to : {json_path}")
            mode = "NONE"
        elif key == ord('q'):
            rospy.loginfo("[INFO] exit...")
            mode = "quit"

        ### collectiong
        if mode == "collecting":
            img_path_scene = root_img_path_scene + 'scene' + str(img_count) + '.jpg'
            img_path_wrist = root_img_path_wrist + 'wrist' + str(img_count) + '.jpg'
            scene_cur_image = Image.fromarray(scene_image_subscriber.get_current_image())
            wrist_cur_image = Image.fromarray(wrist_image_subscriber.get_current_image())
            scene_cur_image,wrist_cur_image = preprocess_image(scene_cur_image,wrist_cur_image)
            scene_cur_image.save(img_path_scene)
            wrist_cur_image.save(img_path_wrist)
            
            data = {
                "imgw" : None,
                "imgs" : None,
                "task" : None,
                "pose" : None,
            }
            data['imgs'] = img_path_scene
            data['imgw'] = img_path_wrist
            data['task'] = args['task']

            robot_state = robot_state_subscriber.get_current_state()
            data['pose'] = np.array(robot_state)

            data_list.append(data)
            img_count += 1
            time.sleep(time_sleep)

        ### quit
        elif mode == "quit":
            break
    
    cv2.destroyAllWindows()



if __name__ == "__main__":


    args = {
        'id': ID,
        'root_path': ROOTPATH,
        'task' : TASK, 
        'name' : NAME,
        'time_sleep' : SLEEP,
    }

    run(args)

