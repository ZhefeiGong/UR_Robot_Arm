#!/usr/bin/env python

import sys
import cv2
import rospy
import numpy as np
import geometry_msgs.msg as geometry_msgs

import ur5e_ctrl_jeff.msg

import motion_commander
from motion_commander import MotionCommander
from PIL import Image

from vla_client import VLAClient
from vla_client import load_image, get_completeTraj, get_trajNdArray

from utils import capture_image, ask_confirmation, generate_initial_img
from utils import euler_to_quaternion, quaternion_to_euler, format_state_array, action_to_command
from utils import cartesian_linear_mapping

from wrist_camera import WristSubscriber
from scene_camera import SceneSubscriber


if sys.version_info[0] < 3:
    """ 
    compatibility for python2 and python3 
    
    """

    input = raw_input


cart_real_all = np.array([[-0.84, -0.27],    # x
                            [-0.80, -0.26],  # y
                            [0.13, 0.80],    # z
                            [0.00, 1.00],    # rx
                            [0.00, 1.00],    # ry
                            [0.00, 1.00]])   # rz

cart_real = np.array([[-0.75, 0.00],    # x
                        [-0.75, 0.00],  # y
                        [0.20, 0.75],   # z
                        [0.00, 1.00],   # rx
                        [0.00, 1.00],   # ry
                        [0.00, 1.00]])  # rz


def deal():
    robot_state_euler = np.array([[-0.02341261,-0.34111358,0.41555742,-0.39346165,-0.91257039,0.10261079,0.04329246,0.        ],
                                    [-0.09801227,-0.4443957,0.29067083,-0.05108385,-0.97837303,0.19527888,0.0451975,1.        ],
                                    [-0.1109624,-0.37152493,0.40499405,-0.07073421,-0.96183267,0.25523876,0.0687587,1.        ],
                                    [-0.63727884,-0.25998364,0.34767165,-0.29757578,-0.8897512,0.29923767,0.17392031,0.        ],
                                    [-0.55712175,-0.26512748,0.42781514,-0.28544193,-0.93144124,0.20839132,0.08667859,0.        ]] ,float)
    robot_state_euler = cartesian_linear_mapping(robot_state=robot_state_euler, cart=cart_real, cart_m=cart_real_all)
    print(robot_state_euler)

def run():
    """
    run the whole process

    """
    
    ### initialization
    rospy.init_node("inference")
    motion_client = MotionCommander()
    scene_image_subscriber = SceneSubscriber()
    wrist_image_subscriber = WristSubscriber()

    ### get the initial image
    ask_confirmation(prompt="we'll start the client to recieve the msg from VLA")
    vla_client = VLAClient(host="192.168.2.3", port=5050)
    
    ### run
    while True:
        # get the initial image
        # ask_confirmation(prompt="we'll capture the image of the scene and wrist...")
        img_path_scene = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/scene/test.jpg"
        img_path_wrist = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/wrist/test.jpg"
        scene_current_image = Image.fromarray(scene_image_subscriber.get_current_image())
        wrist_current_image = Image.fromarray(wrist_image_subscriber.get_current_image())
        img_path_initial = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/init.jpg"
        generate_initial_img( image_scene_path=img_path_scene, 
                                image_wrist_path=img_path_wrist, 
                                img_path_initial=img_path_initial, 
                                scene_current_image=scene_current_image, 
                                wrist_current_image=wrist_current_image,
                                is_path=False)

        # get the initial state
        # ask_confirmation(prompt="we'll recieve the state from ur5e...")
        robot_state = motion_client.get_state()
        print('[INFO] robot state | quat : \n', robot_state)
        robot_state_euler = quaternion_to_euler(robot_state)
        print('[INFO] robot state | euler : \n', robot_state_euler)
        robot_state_euler = cartesian_linear_mapping(robot_state=robot_state_euler, cart=cart_real, cart_m=cart_real_all)
        print('[INFO] robot state | euler | mapped: \n', robot_state_euler)
        robot_state_euler_str= format_state_array(robot_state_euler)
        print('[INFO] robot state | euler | mapped | str: \n', robot_state_euler_str)

        # get the actions
        # ask_confirmation(prompt="we'll recieve the action prediction from VLA...")
        initialImg = load_image(img_path_initial)
        infer_param = {
            "initialImg" : initialImg,
            # "finalImg" : finalImg,
            "instruction" : "Take the tiger out of the red bowl and put it in the grey bowl", # 
            "template" : "12:36:12", # "class_id : index : num_action"
            "reward" : 0,
            "prompt_img" : False,
            "last_actions" : "",
            "maximumLength" : 1024,
            "robot_state" : robot_state_euler_str,
        }

        response = vla_client.infer_traj(infer_param)
        action_arrary = get_trajNdArray(get_completeTraj(response['traj']))
        print('[INFO] robot state | euler : \n', action_arrary)
        action_arrary = cartesian_linear_mapping(robot_state=action_arrary, cart=cart_real_all, cart_m=cart_real)
        print('[INFO] robot state | euler | mapped: \n', action_arrary)

        ask_confirmation(prompt="we'll converse the instruction...")
        action_arrary_quaternion = euler_to_quaternion(action_arrary)
        pose_list, grip_list, duration_list = action_to_command(action_arrary_quaternion, first_duration=5, duration=3)
        print(pose_list)
        print(grip_list)
        print(duration_list)

        ask_confirmation(prompt="we'll execute the trajectory...")
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

