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

def load_image(image_file):
    """
    change the image to a list aiming for transmission

    """
        
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    # Convert the image to a numpy Array
    np_image = np.array(image)

    # Convert the numpy array to a List
    image_list = np_image.tolist()

    return image_list

class RequestClient:
    """
    Communicate with the server through socket
    
    """

    def __init__(self, host, port):
        """ init """
        self.host = host
        self.port = port

    def inference(self, data):
        """ inference """
        url = f"http://{self.host}:{self.port}/inference"
        response = requests.post(url, data=data, timeout=1000*60)
        return response.text
    
    def load_model(self, model_path):
        """ load the model """
        url = f"http://{self.host}:{self.port}/load"
        response = requests.post(url, data=model_path, timeout=5000*60)
        return response.text

    def check_status(self, data):
        """ check the status """
        url = f"http://{self.host}:{self.port}/check"
        response = requests.post(url, data=data)
        return response.text

class VLAClient:
    """
    Client for VLA inference

    """

    def __init__(self, host="10.0.2.11", port=30466):
        """
        initialization
        """

        self.client = RequestClient(host, port)
        print(f"[INFO] Connecting to {host}:{port}")
    
    def infer_traj(self, data, max_retries=3, retry_delay=1):
        """
        inference
        """

        # init
        attempts = 0

        # inference
        while attempts < max_retries:
            response = None
            try:
                data = json.dumps(data).encode("utf-8")
                response = self.client.inference(data)
                return json.loads(response)
            except Exception as e:
                print(f"[ERROR] response: {response} :  {e}")
                traceback.print_exc()
                time.sleep(retry_delay)  # wait before retrying
            finally:
                attempts += 1

        # after trying several times, we had better reconnect again
        print("[ERROR] Maximum retries reached, operation failed")
        return False

    def load_model(self, data):
        """
        load the model
        """

        data = json.dumps(data).encode("utf-8")
        response = self.client.load_model(data)

        print(f"[RESPONCE] Model loaded: {response}")

        if response == "success":
            return True
        else:
            return False

    def check_status(self, data):
        """
        check the status of the server
        """

        data = json.dumps(data).encode("utf-8")
        response = self.client.check_status(data)

        print(f"[RESPONCE] Status: {response}")

        return json.loads(response)

    def close(self):
        """
        close the socket
        """

        self.sock.close()

def get_completeTraj(traj):
    """ """

    # check
    if traj == "" or traj == None:
        return ""

    # find the [ and ]
    first_open_index = traj.find("[")
    last_close_index = traj.rfind("]")

    # can not find
    if first_open_index == -1 or last_close_index == -1:
        return ""

    # from first left [ to first right ]
    return traj[first_open_index:last_close_index + 1]

def get_trajNdArray(traj: str) -> np.ndarray:
    """ """
    
    # remove the "
    traj = traj.replace('"', "")
    
    # find the first left [
    while traj[0] == "[":
        # cut the first right ]
        traj = traj[1:-1]
    
    # split the actions
    split_strings = traj.split("][")

    # split each substring, convert the first five to float and the last one to int
    nested_array = np.array([[float(x) for i, x in enumerate(s.split(","))] for s in split_strings])
    
    return nested_array

if __name__ == "__main__":

    load_param = {
        "clip_direct_load" : False,
        "dino_vision_tower" : "dinov2-large",
        "model_path" : "/liujinxin/code/Reflect/test_checkpoints/13b/dino/finetune_13_12_no_1_5_base_dino_large_7500_c_1_6000/checkpoint-8000"
    }

    
    initialImg = load_image("/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/init.jpg") # 
    finalImg = load_image("/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/init.jpg") # 

    # initialImg = load_image("/Users/zhefeigong/Downloads/workspace/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/init.png") # 
    # finalImg = load_image("/Users/zhefeigong/Downloads/workspace/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/init.png") # 
    
    infer_param = {
        "initialImg" : initialImg,
        # "finalImg" : finalImg,
        "instruction" : "go push the red block left",
        "template" : "12:37:15", # "0:0:*" for random dialogue
        "reward" : 0,
        "prompt_img" : False, 
        "last_actions" : "",
        "maximumLength" : 220,
        "robot_state" : "[0.0259, -0.2313, 0.5713, 3.0905, -0.0291, 1.5001, 1]",
    }
    
    # Template : """{instruction}"" following ""{initialImg}"", describe the next ""{step}"" actions.","{actions}"
    
    client = VLAClient(host="192.168.2.5", port=5050)
    
    # result2laod = client.load_model(load_param)
    # print(result2laod)
    
    result2infer = client.infer_traj(infer_param)
    print(get_completeTraj(result2infer['traj']))
    print(get_trajNdArray(get_completeTraj(result2infer['traj'])))
    
    