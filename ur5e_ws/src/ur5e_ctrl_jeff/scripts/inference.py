#!/usr/bin/env python

import logging
import sys
import numpy as np
import motion_commander
from motion_commander import MotionCommander
from vla_client import VLAClient
from vla_client import load_image, get_completeTraj, get_trajNdArray
from scipy.spatial.transform import Rotation as R
import geometry_msgs.msg as geometry_msgs


def euler_to_quaternion(action_arrary):
    """
    change the rotation from rx,ry,rz to x,y,z,w

    """
    cartesian = action_arrary[:, 0:3]
    euler = action_arrary[:, 3:6]
    gripper = action_arrary[:, 6:]

    rotation = R.from_euler("xyz", euler, degrees=False)
    quaternions = rotation.as_quat()

    return np.concatenate((cartesian, quaternions, gripper), axis=1)


def action_to_command(action_arrary_quaternion, first_duration=10):
    """
    get the inputs for trajectory moving

    """

    duration = 1

    pose_list = []
    grip_list = []
    duration_list = []

    for idx, action in enumerate(action_arrary_quaternion):
        
        # (x,y,z) + (x,y,z,w)
        pose_list.append(
            geometry_msgs.Pose(
                geometry_msgs.Vector3(action[0], action[1], action[2]),
                geometry_msgs.Quaternion(action[3], action[4], action[5], action[6]),
            )
        )

        # gripper
        grip_list.append(action[7])

        # duration
        if idx == 0:
            duration_list.append(first_duration)
        else:
            duration_list.append(duration)

    return pose_list, grip_list, duration_list

def ask_confirmation(prompt=""):
    """
    ask the user to confirm the movement of the next step
    """

    confirmed = False
    valid = False
    while not valid:
        
        input_str = input(
            prompt + 
            "\n Please type 'y' to proceed or 'n' to abort: "
        )

        valid = input_str in ["y", "n"]

        if not valid:
            logging.info(
                "[INPUT] Please confirm by entering 'y' or abort by entering 'n'"
            )
        else:
            confirmed = input_str == "y"
        
        if not confirmed:
            logging.info("[INFO] Exiting as requested by user.")
            sys.exit(0)

def run():
    """
    run the whole process

    """

    # initialization
    motion_client = MotionCommander()
    vla_client = VLAClient(host="172.16.78.10", port=36095)
    
    # get the images
    ask_confirmation(prompt="we'll capture the image of the scene and wrist...")

    # 
    robot_state = motion_client.get_arm_cartesian_state()

    # get the actions
    initialImg = load_image("/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/init.png")
    infer_param = {
        "initialImg" : initialImg,
        # "finalImg" : finalImg,
        "instruction" : "move the door to the left side",
        "template" : "12:37:15", # "0:0:*" for random dialogue
        "reward" : 0,
        "prompt_img" : False,
        "last_actions" : "",
        "maximumLength" : 220,
        "robot_state" : "[0.0259, -0.2313, 0.5713, 3.0905, -0.0291, 1.5001, 1]",
    }
    response = vla_client.infer_traj(infer_param)
    action_arrary = get_trajNdArray(get_completeTraj(response))

    """
    action_arrary = np.array(
        [[0.01891, -0.22252, 0.57389, -0.01629, -0.07064, 1.48513, 1],
        [0.01547, -0.21443, 0.57787, -0.01629, -0.0649, 1.48513, 1],
        [0.00859, -0.20635, 0.57986, -0.01629, -0.0649, 1.48513, 1],
        [0.00172, -0.19826, 0.58186, -0.0234, -0.0649, 1.48513, 1],
        [-0.00516, -0.19018, 0.58385, -0.03051, -0.0649, 1.48513, 1],
        [-0.01203, -0.18479, 0.58385, -0.04473, -0.05916, 1.48513, 1],
        [-0.01891, -0.174, 0.58385, -0.05184, -0.04768, 1.50968, 1],
        [-0.02922, -0.16592, 0.58186, -0.05184, -0.04193, 1.50968, 1],
        [-0.03953, -0.15783, 0.58186, -0.03762, -0.03619, 1.50968, 1],
        [-0.04984, -0.14975, 0.57986, -0.03762, -0.03045, 1.53423, 1],
        [-0.06016, -0.14166, 0.57986, -0.03762, -0.03045, 1.53423, 1],
        [-0.07047, -0.13357, 0.58186, -0.04473, -0.03045, 1.55878, 1],
        [-0.08078, -0.12549, 0.58186, -0.03051, -0.02471, 1.55878, 1],
        [-0.09453, -0.1201, 0.57986, -0.01629, -0.01896, 1.58332, 1],
        [-0.10484, -0.11471, 0.57986, -0.00207, -0.01322, 1.60787, 1],]
    )
    """

    action_arrary_quaternion = euler_to_quaternion(action_arrary)
    print(action_arrary_quaternion)
    pose_list, grip_list, duration_list = action_to_command(action_arrary_quaternion)

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
