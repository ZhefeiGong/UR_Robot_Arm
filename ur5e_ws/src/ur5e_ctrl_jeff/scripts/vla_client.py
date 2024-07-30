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
    
    """

    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    # Convert the image to a numpy array
    np_image = np.array(image)

    # Convert the numpy array to a list
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
    
    def inference_traj(self, data, max_retries=3, retry_delay=1):
        """
        inference
        """

        # init
        attempts = 0

        # inference
        while attempts < max_retries:
            response = None
            try:
                # self.sock.settimeout(5.0) # set a timeout for socket operations (optional)
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

if __name__ == "__main__":

    # 30487 -> LLM1(traj)
    # 30466 -> LLM2(reward)
    
    client = VLAClient(port=30487)

    for i in range(5):
        
        initialImg = load_image("/liujinxin/code/calvin/dataset/task_D_D/images/0.jpg")

        data = {
            "initialImg": initialImg,
            "initialRobotState": [0.0018, -0.0498, 0.5481, 3.0264, -0.0792, 1.4298, 1],
            "instruction": "move the door to the left side",
            "experiences": None,
            "reward": 100,
        }
        
        response = client.send(data)
        traj = response["traj"]

        if traj is not None:
            print("Received traj: ", traj)
        else:
            break

        sleep(5)
    
    client.close()
