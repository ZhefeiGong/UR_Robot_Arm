#!/usr/bin/env python

import numpy as np
from flask import Flask, request, jsonify
from dp_inference import build, infer

## load model 
policy = build(ckpt_pth = "/home/robot/UR_Robot_Arm/coarse2fine/ckpt/dp-cup-epoch=0450-val_loss=0.060.ckpt")
n_obs_steps = 1
n_action_steps = 8

## server
app = Flask(__name__)

## initialize the server port
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get data
        data = request.get_json()
        obs_data = data.get("obs")
        obs = {
            "agentview_image": np.array(obs_data["agentview_image"]),
            "robot0_eye_in_hand_image": np.array(obs_data["robot0_eye_in_hand_image"]),
            "robot0_eef_pos": np.array(obs_data["robot0_eef_pos"]),
            "robot0_eef_quat": np.array(obs_data["robot0_eef_quat"]),
            "robot0_gripper_qpos": np.array(obs_data["robot0_gripper_qpos"]),
        }
        # infer
        action_quat = infer(policy, obs, n_obs_steps, n_action_steps)
        # send back
        response = {
            "actions": action_quat.tolist()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

## setup
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

