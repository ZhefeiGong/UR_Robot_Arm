#!/usr/bin/env python

import socket
import json
import time
from time import sleep
import requests
import numpy as np
import traceback

class RequestClient:
    """
    Communicate with the server through HTTP requests
    """
    def __init__(self, host="127.0.0.1", port="5000"):
        """ Initialize client """
        self.host = host
        self.port = port
    
    def predict(self, payload):
        """ Send prediction request """
        url = f"http://{self.host}:{self.port}/predict"
        response = requests.post(url, json=payload, timeout=1000*60)
        return response.json()

class CAPClient:
    """
    Client for Coarse-to-Fine inference
    """
    def __init__(self, host="127.0.0.1", port="5000"):
        """
        Initialization
        """
        self.client = RequestClient(host, port)
        print(f"[INFO] Connecting to {host}:{port}")

    def predict_traj(self, obs, max_retries=3, retry_delay=1):
        """
        Send inference request with goal and obs data
        """
        attempts = 0
        # Serialize obs data to JSON-compatible format
        obs = self.obs_format_amend(obs)
        payload = {
            "obs": {
                "robot0_eef_pos": obs["robot0_eef_pos"].tolist(),
                "robot0_eef_quat": obs["robot0_eef_quat"].tolist(),
                "robot0_gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
                "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"].tolist(),
                "agentview_image": obs["agentview_image"].tolist()
            }
        }

        # Inference loop with retries
        while attempts < max_retries:
            response = None
            try:
                response = self.client.predict(payload)
                action_pred = np.array(response["actions"])  # Convert response to numpy array
                return action_pred
            except Exception as e:
                print(f"[ERROR] response: {response} :  {e}")
                traceback.print_exc()
                time.sleep(retry_delay)  # wait before retrying
            finally:
                attempts += 1

        print("[ERROR] Maximum retries reached, operation failed")
        return False

    def obs_format_amend(self, obs):
        rgb_keys = ["agentview_image", "robot0_eye_in_hand_image"]
        lowdim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        for key in rgb_keys:
            # move channel last to channel first
            # B,T,H,W,C -> B,T,C,H,W
            # convert uint8 image to float32
            obs[key] = np.moveaxis(obs[key],-1,2).astype(np.float32) / 255.
        for key in lowdim_keys:
            obs[key] = obs[key][:].astype(np.float32)
        return obs

    def close(self):
        """
        Close resources if needed
        """
        pass

if __name__ == "__main__":
    # Example usage
    client = CAPClient()
    
    # Generate example obs data
    obs = dict()
    obs["robot0_eef_pos"] = np.random.randn(1, 1, 3)
    obs["robot0_eef_quat"] = np.random.randn(1, 1, 4)
    obs["robot0_gripper_qpos"] = np.random.randn(1, 1, 1)
    obs["robot0_eye_in_hand_image"] = np.random.randn(1, 1, 3, 120, 160)
    obs["agentview_image"] = np.random.randn(1, 1, 3, 120, 160)
    
    # Send goal and obs data to server
    goal = "Reach target position"
    result = client.predict_traj(goal, obs)
    
    if isinstance(result, np.ndarray):
        print("Received result matrix:", result)
    else:
        print("Failed to receive result")
