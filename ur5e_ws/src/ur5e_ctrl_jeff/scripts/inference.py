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
from utils import cartesian_linear_mapping, curtail_duplicate_action

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
    robot_state_euler = np.array([] ,float)
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
        robot_state_euler_mapped = cartesian_linear_mapping(robot_state=robot_state_euler, cart=cart_real, cart_m=cart_real_all)
        print('[INFO] robot state | euler | mapped: \n', robot_state_euler_mapped)
        robot_state_euler_mapped_str= format_state_array(robot_state_euler_mapped)
        print('[INFO] robot state | euler | mapped | str: \n', robot_state_euler_mapped_str)
        
        # get the actions
        # ask_confirmation(prompt="we'll recieve the action prediction from VLA...")
        initialImg = load_image(img_path_initial)
        infer_param = {
            "initialImg" : initialImg,
            # "finalImg" : finalImg,
            "instruction" : "Put the ranch bottle into the pot", # 
            "template" : "12:36:12", # "class_id : index : num_action"
            "reward" : 0,
            "prompt_img" : False,
            "last_actions" : "",
            "maximumLength" : 1024,
            "robot_state" : robot_state_euler_mapped_str,
        }

        print('[INFO] BEGIN TO INFER ... ')
        response = vla_client.infer_traj(infer_param)
        action_arrary = get_trajNdArray(get_completeTraj(response['traj']))
        print('[INFO] robot state | euler : \n', action_arrary)
        action_arrary = cartesian_linear_mapping(robot_state=action_arrary, cart=cart_real_all, cart_m=cart_real)
        print('[INFO] robot state | euler | mapped : \n', action_arrary)
        action_arrary = curtail_duplicate_action(action_arrary)
        print('[INFO] robot state | euler | mapped | before : \n', robot_state_euler)
        print('[INFO] robot state | euler | mapped | curtailed : \n', action_arrary)
        
        # ask_confirmation(prompt="we'll converse the instruction...")
        action_arrary_quaternion = euler_to_quaternion(action_arrary)
        pose_list, grip_list, duration_list = action_to_command(action_arrary_quaternion, first_duration=5, duration=3)

        # print(pose_list)
        # print(grip_list)
        # print(duration_list)

        # ask_confirmation(prompt="we'll execute the trajectory...")
        motion_client.execute_arm_gripper_trajectory(pose_list, grip_list, duration_list, is_ask_conf=False)
        

if __name__ == "__main__":

    run()

    # deal()


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

