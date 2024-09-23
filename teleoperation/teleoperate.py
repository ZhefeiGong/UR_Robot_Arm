#!/usr/bin/env python
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface 
from rtde_io import RTDEIOInterface as RTDEIO
import robotiq_gripper
import numpy as np
import time
from space_mouse import SpaceMouse
from utils import axis_to_euler

# Define robot parameters
ROBOT_HOST = "192.168.2.4"  # IP address of the robot controller

def run():
    sm = SpaceMouse()
    sm.start()
    # Initialize RTDEControlInterface
    rtde_c = RTDEControlInterface(ROBOT_HOST)
    rtde_r = RTDEReceiveInterface(ROBOT_HOST)
    rtde_io = RTDEIO(ROBOT_HOST)

    print("Creating gripper...")
    gripper = robotiq_gripper.RobotiqGripper()
    print("Connecting to gripper...")
    gripper.connect(ROBOT_HOST, 63352)
    print("Activating gripper...")
    gripper.activate()
    
    try:
        while True:
            if rtde_r.getRobotMode() == 7:
                # Read motion state from SpaceMouse
                motion_state = sm.get_motion_state_transformed()
                # print("Current motion state" , motion_state)
                
                # send command to robot 
                rtde_c.speedL(motion_state, acceleration = 1.5, time = 0.1) # adjust the acceleration if required 

                # get TCP velocity of robot
                actual_velocity = rtde_r.getActualTCPSpeed()
                actual_velocity = [0 if abs(x) < 0.01 else x for x in actual_velocity] #filter out extremely small numbers
                # print("Current velocity vector" , actual_velocity)

                # get TCP pose of robot
                actual_pose = rtde_r.getActualTCPPose()
                # print("Current pose vector" , actual_pose)

                axis_cart = np.array(actual_pose[:3])
                axis_angle = np.array(actual_pose[3:])
                euler = axis_to_euler(axis_angle)
                print("Current pose" , np.array([axis_cart,euler]))

                if sm.is_button_pressed(0):
                    gripper.move(gripper.get_open_position(), 255, 255)
                
                if sm.is_button_pressed(1):
                    gripper.move(gripper.get_closed_position(), 255, 255)

                # if sm.is_button_pressed(0):
                #     gripper_position += 3
                #     gripper.move(gripper_position, 155, 255)
                # if sm.is_button_pressed(1):
                #     gripper_position -= 3
                #     gripper.move(gripper_position, 155, 255)
                # if gripper_position < 0:
                #     gripper_position = 0
                # if gripper_position > 255:
                #     gripper_position = 255
                
                print("Gripper Position (0 to 255): ", gripper.get_current_position())

                # wait awhile before proceeding 
                time.sleep(1/100)

            else:
                print("Robot is not ready.")
                time.sleep(1)  # Wait longer if robot is not ready

    except KeyboardInterrupt:
        # Handle graceful shutdown here
        print("Stopping robot")
        rtde_c.stopScript()
        sm.stop()

if __name__ == "__main__":
    run()
