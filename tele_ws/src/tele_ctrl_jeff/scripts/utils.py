#!/usr/bin/env python

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

def axis_to_euler(axis_angle):
    """
    @input : rx,ry,rz - [axis angle] | array
    @output : rx,ty,tz - [euler] | array
    """
    theta = np.linalg.norm(axis_angle)
    axis = axis_angle / theta
    half_theta = theta/2
    q_w = np.cos(half_theta)
    q_xyz = axis*np.sin(half_theta)
    quat = np.array([q_xyz[0],q_xyz[1],q_xyz[2],q_w]) # [x,y,z,w]
    rotation = R.from_quat(quat) # [x,y,z,w]
    euler = rotation.as_euler("xyz", degrees=False)
    return euler

def axis_to_quat(axis_angle):
    """
    @input : rx,ry,rz - [axis angle] | array
    @output : qx,qy,qz,qw - [quat] | array
    """
    theta = np.linalg.norm(axis_angle)
    axis = axis_angle / theta
    half_theta = theta/2
    q_w = np.cos(half_theta)
    q_xyz = axis*np.sin(half_theta)
    quat = np.array([q_xyz[0],q_xyz[1],q_xyz[2],q_w]) # [x,y,z,w]
    return quat

def euler_to_axis(euler_angles):
    """
    Converts Euler angles to axis-angle representation.
    @input : rx, ry, rz - [euler angles] | array
    @output : rx, ry, rz - [axis angle] | array
    """
    rotation = R.from_euler("xyz", euler_angles, degrees=False)
    axis_angle = rotation.as_rotvec()
    return axis_angle

def quat_to_axis(quaternion):
    """
    Converts quaternion to axis-angle representation.
    @input : qx, qy, qz, qw - [quaternion] | array
    @output : rx, ry, rz - [axis angle] | array
    """
    quaternion = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    rotation = R.from_quat(quaternion)
    axis_angle = rotation.as_rotvec()
    return axis_angle

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

def print_2d_arr(info,actions):
    """
    @fun :
    """
    if info is not None:
        print(info)
    for row in actions: 
        print('[', end="")
        for v in row:
            print(f"{v:.7f}", end="")
            print(' ', end="")
        print(']')
    return

def preprocess_image(scene_image, wrist_image, resize_width=160, resize_height=120):
    """
    Preprocess scene and wrist images by resizing, rotating, and formatting as numpy arrays.
    Args:
        scene_image (PIL.Image): Image from the scene camera.
        wrist_image (PIL.Image): Image from the wrist camera.
        resize_width (int): Width to resize the image.
        resize_height (int): Height to resize the image.
    Returns:
        tuple: Preprocessed scene and wrist images as numpy arrays with shape (H, W, C).
    """
    # Preprocess scene image
    scene_image = scene_image.convert("RGB")
    scene_image = scene_image.transpose(Image.ROTATE_180)               # Rotate by 180 degrees if necessary
    scene_image = scene_image.resize((resize_width, resize_height))     # Resize to desired dimensions
    scene_image = np.array(scene_image)                                 # Convert to numpy array with shape (H, W, C)
    scene_image = cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB)          # to RGb
    # Preprocess wrist image
    wrist_image = wrist_image.convert("RGB")
    wrist_image = wrist_image.resize((resize_width, resize_height))     # Resize to desired dimensions
    wrist_image = np.array(wrist_image)                                 # Convert to numpy array with shape (H, W, C)
    wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)          # to RGB
    return scene_image, wrist_image



