#!/usr/bin/env python

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import geometry_msgs.msg as geometry_msgs

def capture_image(camera, path):
    """
    capture the image from the camera
    """
    
    frame = camera.get_frame()
    if frame is not None:
        cv2.imwrite(path, frame)
        print(f"照片已保存为 {path}")
    else:
        print("无法获取照片")


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
            print("[INPUT] Please confirm by entering 'y' or abort by entering 'n'")
        else:
            confirmed = input_str == "y"
        if not confirmed:
            print("[INFO] Exiting as requested by user.")
            sys.exit(0)


def euler_to_quaternion(action_array):
    """
    @change the rotation from rx,ry,rz to x,y,z,w
    @action_array is a 2-d array

    """

    cartesian = action_array[:, 0:3]
    euler = action_array[:, 3:6]
    gripper = action_array[:, 6:]

    rotation = R.from_euler("xyz", euler, degrees=False)
    quaternions = rotation.as_quat() # [x,y,z,w]
    
    return np.concatenate((cartesian, quaternions, gripper), axis=1)


def quaternion_to_euler(action_array):
    """
    @change the rotation from x,y,z,w to rx,ry,rz
    @action_array is a 2-d array

    """

    cartesian = action_array[:, 0:3]
    quaternions = action_array[:, 3:7]
    gripper = action_array[:, 7:]

    rotation = R.from_quat(quaternions) # [x,y,z,w]
    euler = rotation.as_euler("xyz", degrees=False)

    return np.concatenate((cartesian, euler, gripper), axis=1)


def format_state_array(state_array):
    """
    format the state array to the specified string format.

    """

    state = state_array[0]
    state_str = "[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:d}]".format(
        state[0], state[1], state[2], state[3], state[4], state[5], int(state[6])
      )
    
    return state_str


def action_to_command(action_arrary_quaternion, first_duration=10, duration=2, fix_num = 4):
    """
    get the inputs for trajectory moving

    """
    
    action_arrary_quaternion = np.round(action_arrary_quaternion,fix_num)
    
    gripper_crateria = 0.5

    pose_list = []
    grip_list = []
    duration_list = []

    time_count = 0

    for idx, action in enumerate(action_arrary_quaternion):
        
        # (x,y,z) + (x,y,z,w)
        pose_list.append(
            geometry_msgs.Pose(
                # [x,y,z]
                geometry_msgs.Vector3(x=action[0], y=action[1], z=action[2]),
                # [x,y,z,w]
                geometry_msgs.Quaternion(x=action[3], y=action[4], z=action[5], w=action[6]),
            )
        )

        # gripper
        if action[7] > gripper_crateria : 
            grip_list.append(1)
        else:
            grip_list.append(0)
        
        # duration
        if idx == 0:
            time_count += first_duration
        else:
            time_count += duration
        duration_list.append(time_count)
    
    return pose_list, grip_list, duration_list


def cartesian_linear_mapping(robot_state, cart, cart_m):
    """
        [ cart ] --> [ cart_m ]
    * : [mim, max] --> [min, max]
    x : [0.18, 0.68]  --> [-0.80, 0.00]
    y : [-0.27, 0.38] --> [-0.80, 0.00]
    z : [-0.20, 0.20] --> [0.35, 0.95]
    rx : [-1.0,1.0] --> no need (pi)
    ry : [-0.51,0.40] --> no need
    rz : [0.77,2.60] --> no need

    @robot_state : [[x,y,z,rx,ry,rz]] | 2-dimensional
    @formula : y=ax+b
    
    """
    
    # init
    state = robot_state.copy()
    dim = len(cart)
    min_sg = 0
    max_sg = 1

    # calculate
    a = (cart_m[:,max_sg]-cart_m[:,min_sg]) / ((cart[:,max_sg]-cart[:,min_sg]))     # [6,]
    b = cart_m[:,min_sg] - a * cart[:,min_sg]                                       # [6,]
    a = a.reshape((1,dim))                                                          # [1,6]
    b = b.reshape((1,dim))                                                          # [1,6]

    # transform
    state[:,:6] = a*state[:,:6] + b                                     # [n,6] = [1,6] * [n,6] + [1,6]

    return state

def curtail_duplicate_action(action_arrary):
    """
    cut off the same action compared with the former one

    @action_arrary has shape : [n,7]
    
    """

    cur_action_arrary = [action_arrary[0]]

    for idx in range(1, action_arrary.shape[0]):
        if not np.array_equal(action_arrary[idx], action_arrary[idx-1]):
            cur_action_arrary.append(action_arrary[idx])
    
    return np.array(cur_action_arrary)



if __name__ == "__main__":

    # img_path_scene = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/scene/test.jpg"
    # img_path_wrist = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/wrist/test.jpg"
    # img_path_initial = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/init.jpg"

    # # img_path_scene = "/Users/zhefeigong/Downloads/workspace/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/scene/test.jpg"
    # # img_path_wrist = "/Users/zhefeigong/Downloads/workspace/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/wrist/test.jpg"
    # # img_path_initial = "/Users/zhefeigong/Downloads/workspace/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/init_.jpg"
    
    # concateImage(img_path_scene, img_path_wrist, img_path_initial)
    
    quaternions = np.array([0.11625579, 0.94370034, -0.30423459, 0.05792736])
    rotation = R.from_quat(quaternions) # [x,y,z,w]
    euler = rotation.as_euler("xyz", degrees=False)
    print(euler)