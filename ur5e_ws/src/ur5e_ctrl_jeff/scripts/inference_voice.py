#!/usr/bin/env python

import sys
import cv2
import rospy
from copy import deepcopy
import numpy as np
import geometry_msgs.msg as geometry_msgs
import ur5e_ctrl_jeff.msg
import motion_commander
from motion_commander import MotionCommander
from joint_listener import JointStateListener
from PIL import Image

from vla_client_voice import VLAClient

from utils import ask_confirmation
from utils import euler_to_quaternion, quaternion_to_euler, format_state_array, action_to_command
from utils import cartesian_linear_mapping, curtail_duplicate_action

from wrist_camera import WristSubscriber
from scene_camera import SceneSubscriber

"""
joint0 joint1 joint2 joint3 joint4 joint5 x y z qx qy qz qw gripper_is_closed action_blocked

x <-> [-0.70,0.00]
y <-> [-0.70,0.00]
z <-> [0.20,0.68]
"""

if sys.version_info[0] < 3:
    """ 
    @func : compatibility for python2 and python3 
    """
    input = raw_input


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
    wrist_Image = Image.merge("RGB", (b,g,r)) # berkeleyur5-bgr
    wrist_Image = wrist_Image.transpose(Image.ROTATE_180) # berkeleyur5-rotate

    return scene_Image, wrist_Image

def preprocess_state(joints, poses, action_blocked=0.0):
    """
    @func : 
    """
    robot_obs=[]
    for joint in joints:
        robot_obs.append(joint) # joint0 joint1 joint2 joint3 joint4 joint5
    for pose in poses:
        robot_obs.append(pose)
    robot_obs.append(action_blocked) # x y z qx qy qz qw gripper_is_closed
    return robot_obs

def postprocess_action(robot_state, actions):
    """
    @func : 
    """
    action_dim = 7
    action_num = len(actions)//action_dim
    robot_state = robot_state[0]
    robot_actions = []
    for idx_num in range(action_num):
        for idx_dim in range(action_dim):
            if idx_dim + 1 != action_dim:
                robot_state[idx_dim] += actions[idx_num*action_dim + idx_dim]
            else:
                robot_state[idx_dim] = actions[idx_num*action_dim + idx_dim]
        print(robot_state)
        robot_actions.append(deepcopy(robot_state))
    robot_actions = np.stack(robot_actions, axis=0)
    return robot_actions

def run():
    """
    @func : run the whole process
    """
    
    ### initialization
    rospy.init_node("inference")
    motion_client = MotionCommander()
    joint_subscriber = JointStateListener()
    scene_image_subscriber = SceneSubscriber()
    wrist_image_subscriber = WristSubscriber()

    ### get the initial image
    ask_confirmation(prompt="we'll start the client to recieve the msg from VLA")
    vla_client = VLAClient(host="192.168.2.7", port=5050)
    
    ### 
    goal = "Take the tiger out of the red bowl and put it in the grey bowl"
    scene_pth = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/voice/scene.jpg"
    wrist_pth = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/voice/wrist.jpg"

    if_ask_confirmation= True
    
    ### run
    while True:
        
        ### get the initial image
        if if_ask_confirmation: ask_confirmation(prompt="we'll capture the image of the scene and wrist...")
        scene_cur_image = Image.fromarray(scene_image_subscriber.get_current_image())
        wrist_cur_image = Image.fromarray(wrist_image_subscriber.get_current_image())
        scene_cur_image,wrist_cur_image = preprocess_image(scene_cur_image,wrist_cur_image)
        scene_cur_image.save(scene_pth)
        wrist_cur_image.save(wrist_pth)

        ### get the robot state
        if if_ask_confirmation: ask_confirmation(prompt="we'll build the current state of the robot...")
        robot_state = motion_client.get_state() # [1,7]
        robot_state_euler = quaternion_to_euler(robot_state) # [1,6]
        print('[INFO] robot state | quat : \n', robot_state)
        print('[INFO] robot state | euler : \n', robot_state_euler)
        robot_joint = np.array([joint_subscriber.get_joint_states('position')]) # [1,6]
        print('[INFO] robot joint : \n', robot_joint)
        robot_state = list(robot_state[0])
        robot_joint = list(robot_joint[0])
        robot_obs = preprocess_state(joints=robot_joint,poses=robot_state)
        print('[INFO] robot obs : \n', robot_obs)

        ### get the actions
        if if_ask_confirmation: ask_confirmation(prompt="we'll recieve the state from ur5e...")
        actions = vla_client.predict_traj(goal=goal,robot_obs=robot_obs,scene_pth=scene_pth,wrist_pth=wrist_pth)
        print('[INFO] robot action | raw : \n', actions)
        print('[INFO] robot state | euler | before : \n', robot_state_euler)
        robot_actions = postprocess_action(robot_state_euler, actions)
        print('[INFO] robot action | len : \n', len(robot_actions))
        print('[INFO] robot action | euler : \n', robot_actions)
        robot_actions = curtail_duplicate_action(robot_actions)
        print('[INFO] robot action | euler | curtailed : \n', robot_actions)

        ### get the command
        if if_ask_confirmation: ask_confirmation(prompt="we'll converse the instruction...")
        action_arrary_quaternion = euler_to_quaternion(robot_actions)
        pose_list, grip_list, duration_list = action_to_command(action_arrary_quaternion, first_duration=1, duration=1)
        print('[INFO] robot action | pose : \n',pose_list)
        print('[INFO] robot action | gripper : \n',grip_list)
        print('[INFO] robot action | duration : \n',duration_list)
        
        if if_ask_confirmation: ask_confirmation(prompt="we'll execute the trajectory...")
        motion_client.execute_arm_gripper_trajectory(pose_list, grip_list, duration_list, is_ask_conf=False)

if __name__ == "__main__":
    run()

"""
🌟 category :
1. Take the tiger out of the red bowl and put it in the grey bowl.
2. Sweep the green cloth to the left side of the table.
3. Pick up the blue cup and put it into the brown cup.
4. Put the ranch bottle into the pot.    
"""

"""
🌟 range :
x : -0.75 ~ 0.00
y : -0.75 ~ 0.00
z : 0.20 ~ 0.75
"""

"""
🌟🌟🌟 Tiger-4 🌟🌟🌟

🌟 original
[[-0.28779358 -0.50560178  0.39258813 -0.39346165 -0.91257039  0.10261079 0.04329246  0.        ]
 [-0.34448933 -0.5799649   0.24045356 -0.05108385 -0.97837303  0.19527888 0.0451975   1.        ]
 [-0.35433142 -0.52749795  0.37972002 -0.07073421 -0.96183267  0.25523876 0.0687587   1.        ]
 [-0.75433192 -0.44718822  0.30989092 -0.29757578 -0.8897512   0.29923767 0.17392031  0.        ]
 [-0.69341253 -0.45089179  0.40752026 -0.28544193 -0.93144124  0.20839132 0.08667859  0.        ]]

🌟 mapped
[[-0.02341261 -0.34111358  0.41555742 -0.39346165 -0.91257039  0.10261079 0.04329246  0.        ]
 [-0.09801227 -0.4443957   0.29067083 -0.05108385 -0.97837303  0.19527888 0.0451975   1.        ]
 [-0.1109624  -0.37152493  0.40499405 -0.07073421 -0.96183267  0.25523876 0.0687587   1.        ]
 [-0.63727884 -0.25998364  0.34767165 -0.29757578 -0.8897512   0.29923767 0.17392031  0.        ]
 [-0.55712175 -0.26512748  0.42781514 -0.28544193 -0.93144124  0.20839132 0.08667859  0.        ]]
"""


"""
🌟🌟🌟 Bottle-6 🌟🌟🌟

🌟 original
[[-0.55935583 -0.57534852  0.3647796   0.39036752  0.9064039  -0.1595691 0.02414259  0.        ]
 [-0.6042515  -0.34438318  0.25835688 -0.47975619 -0.8670687   0.13370965 0.012149    0.        ]
 [-0.60700114 -0.35516387  0.15417322 -0.53291503 -0.84438056  0.05414724 0.00954518  1.        ]
 [-0.58911591 -0.341283    0.33939717 -0.52256426 -0.83134177  0.18321391 0.0472241   1.        ]
 [-0.49689463 -0.63534945  0.29619845  0.38136535  0.90666088 -0.17278228 0.05169911  1.        ]
 [-0.50287908 -0.64575297  0.21685555  0.38568214  0.91043886 -0.13875036 0.05566601  0.        ]
 [-0.53426955 -0.56432181  0.32394524 -0.39318641 -0.9166355   0.05521894 0.04620256  0.        ]]

🌟 mapped
[[-0.38073135 -0.43798405  0.39272952  0.39036752  0.9064039  -0.1595691 0.02414259  0.        ]
 [-0.4398046  -0.11719886  0.30536759 -0.47975619 -0.8670687   0.13370965 0.012149    0.        ]
 [-0.44342255 -0.13217204  0.21984369 -0.53291503 -0.84438056  0.05414724 0.00954518  1.        ]
 [-0.41988936 -0.11289305  0.3718932  -0.52256426 -0.83134177  0.18321391 0.0472241   1.        ]
 [-0.29854557 -0.52131868  0.33643156  0.38136535  0.90666088 -0.17278228 0.05169911  1.        ]
 [-0.30641984 -0.53576802  0.27129933  0.38568214  0.91043886 -0.13875036 0.05566601  0.        ]
 [-0.34772309 -0.42266918  0.35920878 -0.39318641 -0.9166355   0.05521894 0.04620256  0.        ]]

"""
