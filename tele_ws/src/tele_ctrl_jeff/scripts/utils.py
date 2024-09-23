#!/usr/bin/env python

import numpy as np
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