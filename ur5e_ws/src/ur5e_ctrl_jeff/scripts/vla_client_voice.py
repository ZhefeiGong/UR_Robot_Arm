#!/usr/bin/env python

import socket
import json
import time
from time import sleep
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import traceback


class RequestClient:
    """
    Communicate with the server through socket
    """
    def __init__(self, host, port):
        """ init """
        self.host = host
        self.port = port
    
    def predict(self, files):
        """ predict """
        url = f"http://{self.host}:{self.port}/predict"
        response = requests.post(url, files=files, timeout=1000*60)
        print(response)
        return response.json()

class VLAClient:
    """
    Client for VLA inference
    """
    def __init__(self, host="172.16.78.10", port=00000):
        """
        initialization
        """
        self.client = RequestClient(host, port)
        print(f"[INFO] Connecting to {host}:{port}")
    def predict_traj(self, goal, robot_obs, scene_pth, wrist_pth, max_retries=3, retry_delay=1):
        """
        inference
        """
        # init
        attempts = 0
        # data construction
        image_scene_data = np.array(Image.open(scene_pth).convert("RGB")).tobytes()
        image_wrist_data = np.array(Image.open(wrist_pth).convert("RGB")).tobytes()
        goal_data = goal
        robot_obs_data = robot_obs
        payload = {"instruction": goal_data, "robot_obs": robot_obs_data}
        files = {
            "json": json.dumps(payload),
            "img_static": ("img_stat.txt", image_scene_data, "text/plain"),
            "img_gripper": ("img_grip.txt", image_wrist_data, "text/plain"),
        }
        # inference
        while attempts < max_retries:
            response = None
            try:
                response = self.client.predict(files)
                return response
            except Exception as e:
                print(f"[ERROR] response: {response} :  {e}")
                traceback.print_exc()
                time.sleep(retry_delay)  # wait before retrying
            finally:
                attempts += 1
        # after trying several times, we had better reconnect again
        print("[ERROR] Maximum retries reached, operation failed")
        return False
    def close(self):
        """
        close the socket
        """
        self.sock.close()

if __name__ == "__main__":
    
    # goal = "pick up a bottle for me"
    # path_static = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/calvin_scene.jpg"
    # path_gripper = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/calvin_wrist.jpg"
    # image_static = Image.open(path_static).convert("RGB")
    # img_static = np.array(image_static)
    # image_gripper = Image.open(path_gripper).convert("RGB")
    # img_gripper = np.array(image_gripper)
    # robot_obs_data = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
    # img_static_data = img_static.tobytes()
    # img_gripper_data = img_gripper.tobytes()
    # url = "http://172.16.78.10:39017/predict"
    # payload = {"instruction": goal, "robot_obs": robot_obs_data}
    # files = {
    #     "json": json.dumps(payload),
    #     "img_static": ("img_stat.txt", img_static_data, "text/plain"),
    #     "img_gripper": ("img_grip.txt", img_gripper_data, "text/plain"),
    # }
    # action = requests.post(url, files=files)
    # print(action.json())
    # print(action)
    
    goal = "pick up a bottle for me"
    scene_pth = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/calvin_scene.jpg"
    wrist_pth = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/calvin_wrist.jpg"
    robot_obs = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
    vla_client = VLAClient(host="172.16.78.10", port=5050)
    actions = vla_client.predict_traj(goal=goal,
                                    robot_obs=robot_obs,
                                    scene_pth=scene_pth,
                                    wrist_pth=wrist_pth)
    print(actions)

