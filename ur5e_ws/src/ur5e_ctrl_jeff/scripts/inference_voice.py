#!/usr/bin/env python

import sys
import cv2
import rospy
import math
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
    wrist_Image = Image.merge("RGB", (r,g,b)) # berkeleyur5-bgr
    # wrist_Image = wrist_Image.transpose(Image.ROTATE_180) # berkeleyur5-rotate

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
    @input : robot_state := 2d array, actions := 2d array
    """
    action_dim = 7
    action_num = len(actions)
    robot_state = deepcopy(robot_state[0])
    robot_actions = []
    for idx_num in range(action_num):

        # for idx_dim in range(action_dim):
        #     if idx_dim + 1 != action_dim:
        #         robot_state[idx_dim] += actions[idx_num*action_dim + idx_dim]
        #     else:
        #         robot_state[idx_dim] = actions[idx_num*action_dim + idx_dim]            
        #     # # rx build gap
        #     # if robot_state[3] < -math.pi:
        #     #     robot_state[3] = math.pi + (robot_state[3] + math.pi)
        #     # elif robot_state[3] >= math.pi:
        #     #     robot_state[3] = -math.pi + (robot_state[3] - math.pi)
        #     # # ry build gap
        #     # if robot_state[4] < -math.pi:
        #     #     robot_state[4] = math.pi + (robot_state[4] + math.pi)
        #     # elif robot_state[4] >= math.pi:
        #     #     robot_state[4] = -math.pi + (robot_state[4] - math.pi)
        #     # # rz build gap
        #     # if robot_state[5] < -math.pi:
        #     #     robot_state[5] = math.pi + (robot_state[5] + math.pi)
        #     # elif robot_state[5] >= math.pi:
        #     #     robot_state[5] = -math.pi + (robot_state[5] - math.pi)
        
        ### update
        robot_state[:6] += actions[idx_num,:6]
        robot_state[6] = actions[idx_num,6]

        ### build the [-pi ~ pi] gap | rx,ry,rz
        for i in range(3,6):
            if robot_state[i] < -math.pi:
                robot_state[i] = math.pi + (robot_state[i] + math.pi)
            elif robot_state[i] >= math.pi:
                robot_state[i] = -math.pi + (robot_state[i] - math.pi)
        
        ### update
        robot_actions.append(deepcopy(robot_state))

    robot_actions = np.stack(robot_actions, axis=0)
    return robot_actions

def process_npz(root_path):
    """
    """
    from PIL import Image
    max_num = 24
    actions = list()
    print(np.load(f"{root_path}/auto_lang_ann.npy", allow_pickle=True))
    for i in range(max_num):
        formatted_num = str(i+1).zfill(7)
        data_path = f"{root_path}/{formatted_num}.npz"
        data = np.load(data_path)
        rel_actions = data['rel_actions']
        rgb_static = data['rgb_static']
        rgb_gripper = data['rgb_gripper']
        robot_obs = data['robot_obs']
        print(rel_actions)
        actions.append(rel_actions)
        # print("arrays :", data.files)
        # print(rel_actions.shape)
        # print(rgb_static.shape)
        # print(rgb_gripper.shape)
        # print(robot_obs.shape)
        # import cv2
        # cv2.imwrite('/home/robot/rgb_static.jpg', rgb_static) # save in bgr order
        # cv2.imwrite('/home/robot/rgb_gripper.jpg', rgb_gripper) # save in bgr order

        # rgb_static_img = Image.fromarray(rgb_static)
        # rgb_gripper_img = Image.fromarray(rgb_gripper)
        # rgb_static_img.save(f'{root_path}/image/rgb_static_{formatted_num}.png') # save in rgb order
        # rgb_gripper_img.save(f'{root_path}/image/rgb_gripper_{formatted_num}.png') # save in rgb order
    # actions_arr = np.array(actions)
    # np.savez(f"{root_path}/actions.npz", actions = actions_arr)

def get_process_actions(path='/home/robot/data_tmp/training_test/actions.npz'):
    data = np.load(path)
    return data['actions']

def print_2d_arr(info,actions):
    """"""
    print(info)
    for row in actions: 
        print('[', end="")
        for v in row:
            print(f"{v:.7f}", end="")
            print(' ', end="")
        print(']')
    return

def normalize_action(actions, means, stds):
    """
    @input : actions:=2d array, mean:=1d array, stds:=1d array
    @func : 1.2450980
    """
    cratera = 1.24
    coef = 2.5
    actions_norm = (actions-means)/stds
    actions_coef = np.ones(actions_norm.shape,actions_norm.dtype)
    actions_coef[abs(actions_norm)>cratera]=coef
    return actions_norm, actions_coef


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
    vla_client = VLAClient(host="192.168.2.4", port=5050)
    
    ### 
    goal = "put the smaller green bowl into the grey bowl"
    scene_pth = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/voice/scene.jpg"
    wrist_pth = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/voice/wrist.jpg"

    """
    put the smaller green bowl into the grey bowl
    take the tiger out of the red bowl and put it in the green bowl
    pick up the green cup for me
    """

    if_ask_confirmation= True

    ###@test@
    # img_idx = 1 #@test@
    # root_path = '/home/robot/data_tmp/training_test' #@test@


    means = np.array([0.00245544, -0.00174869, -0.00113085, 0.0006589, 0.00253147, 0.00102762, 0.50635778]) # x.y,z + rx,ry,rz
    stds = np.array([0.02039579, 0.01705823, 0.02590469, 0.01555712, 0.01804757, 0.02337257, 0.49995958]) # x.y,z + rx,ry,rz

    ### run
    while True:
        
        # #@test@
        # if img_idx > 24:
        #     break
        
        ### get the initial image
        if if_ask_confirmation: ask_confirmation(prompt="we'll capture the image of the scene and wrist...")
        scene_cur_image = Image.fromarray(scene_image_subscriber.get_current_image())
        wrist_cur_image = Image.fromarray(wrist_image_subscriber.get_current_image())
        scene_cur_image,wrist_cur_image = preprocess_image(scene_cur_image,wrist_cur_image)
        scene_cur_image.save(scene_pth)
        wrist_cur_image.save(wrist_pth)
        
        # #@test@
        # formatted_num = str(img_idx).zfill(7)
        # data_path = f"{root_path}/{formatted_num}.npz"
        # print(data_path)
        # data = np.load(data_path)
        # rgb_static = data['rgb_static']
        # rgb_gripper = data['rgb_gripper']
        # rgb_static_img = Image.fromarray(rgb_static)
        # rgb_gripper_img = Image.fromarray(rgb_gripper)
        # rgb_static_img.save(scene_pth) # save in rgb order
        # rgb_gripper_img.save(wrist_pth) # save in rgb order
        
        ### get the robot state
        if if_ask_confirmation: ask_confirmation(prompt="we'll build the current state of the robot...")
        robot_state = motion_client.get_state() # [1,7]
        robot_state_euler = quaternion_to_euler(robot_state) # [1,6]
        print_2d_arr('[INFO] robot state | quat : ', robot_state)
        print_2d_arr('[INFO] robot state | euler : ', robot_state_euler)
        robot_joint = np.array([joint_subscriber.get_joint_states('position')]) # [1,6]
        print_2d_arr('[INFO] robot joint : ', robot_joint)
        robot_state = list(robot_state[0])
        robot_joint = list(robot_joint[0])
        robot_obs = preprocess_state(joints=robot_joint,poses=robot_state) # [15,]
        print_2d_arr('[INFO] robot obs : ', [robot_obs])
        
        ### get the actions
        if if_ask_confirmation: ask_confirmation(prompt="we'll recieve the state from ur5e...")
        actions = vla_client.predict_traj(goal=goal,robot_obs=robot_obs,scene_pth=scene_pth,wrist_pth=wrist_pth)
        actions_raw = np.array(actions).reshape(-1,7)
        # actions = np.array(get_process_actions()).reshape(-1,7) #@test@ 
        print_2d_arr('[INFO] robot action | raw : ', actions_raw)

        ### normal
        actions_norm, actions_coef = normalize_action(actions_raw, means, stds)
        print_2d_arr('[INFO] robot action | raw | norm : ', actions_norm)
        print_2d_arr('[INFO] robot action | raw | coef : ', actions_coef)
        # actions_shift = actions_raw * actions_coef
        # print_2d_arr('[INFO] robot action | raw | shift : ', actions_shift)
        
        ### get obs actions
        robot_actions = postprocess_action(robot_state_euler, actions_raw) # raw or shift
        print_2d_arr('[INFO] robot state | euler | before : ', robot_state_euler)
        print_2d_arr('[INFO] robot action | euler : ', robot_actions)
        robot_actions = curtail_duplicate_action(robot_actions)
        print_2d_arr('[INFO] robot action | euler | curtailed :', robot_actions)

        ### get the command
        if if_ask_confirmation: ask_confirmation(prompt="we'll converse the instruction...")
        action_arrary_quaternion = euler_to_quaternion(robot_actions)
        pose_list, grip_list, duration_list = action_to_command(action_arrary_quaternion, first_duration=3, duration=3)
        print('[INFO] robot action | pose : \n',pose_list)
        print('[INFO] robot action | gripper : \n',grip_list)
        print('[INFO] robot action | duration : \n',duration_list)
        
        ### run
        if if_ask_confirmation: ask_confirmation(prompt="we'll execute the trajectory...")
        motion_client.execute_arm_gripper_trajectory(pose_list, grip_list, duration_list, is_ask_conf=False)
        
        # img_idx += 5 #@test@
        
if __name__ == "__main__":

    run()
    # root_path = '/home/robot/data_tmp/training_test'
    # process_npz(root_path)


"""
🌟 category :
1. Take the tiger out of the red bowl and put it in the grey bowl.
2. Sweep the green cloth to the left side of the table.
3. Pick up the blue cup and put it into the brown cup.
4. Put the ranch bottle into the pot.    
"""