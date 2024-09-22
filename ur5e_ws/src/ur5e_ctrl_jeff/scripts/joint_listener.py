#!/usr/bin/env python

import sys
import rospy
import numpy as np
import geometry_msgs.msg as geometry_msgs
from sensor_msgs.msg import JointState
import ur5e_ctrl_jeff.msg
from utils import quaternion_to_euler

# Compatibility for python2 and python3
if sys.version_info[0] < 3:
    input = raw_input

class JointStateListener():
    """Subscribe to the joint states of the UR5e robot"""
    
    def __init__(self, verbose: bool = False):
        self.is_verbose = verbose
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_efforts = []
        self.joint_names = []
        
        # Subscribe to the /joint_states topic
        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
    
    def joint_state_callback(self, msg):
        """Callback function for joint state messages"""
        
        # Store joint information
        self.joint_positions = msg.position
        self.joint_velocities = msg.velocity
        self.joint_efforts = msg.effort
        self.joint_names = msg.name

        # # If verbose, print joint information
        # if self.is_verbose:
        #     print("[MSG] Joint Names: ", self.joint_names)
        #     print("[MSG] Joint Positions: ", self.joint_positions)
        #     print("[MSG] Joint Velocities: ", self.joint_velocities)
        #     print("[MSG] Joint Efforts: ", self.joint_efforts)
    
    def get_joint_states(self, joint_info_type='position'):
        """Get the current joint states (positions, velocities, efforts)"""
        info_types = ['name', 'position', 'velocity', 'effort' ]
        if joint_info_type not in info_types:
            return {
                'name': self.joint_names,
                'position': self.joint_positions,
                'velocity': self.joint_velocities,
                'effort': self.joint_efforts
            }
        else:
            if joint_info_type=='name':
                return self.joint_names
            elif joint_info_type=='position':
                return self.joint_positions
            elif joint_info_type=='velocity':
                return self.joint_velocities
            elif joint_info_type=='effort':
                return self.joint_efforts

def main():
    """Main Function"""

    print("===== JointStateListener =====")
    
    # Initialize the ROS node
    rospy.init_node('joint_state_listener')

    # Create an instance of the JointStateListener class
    listener = JointStateListener(verbose=True)
    
    # Keep the node running to continuously receive joint state updates
    rospy.spin()


if __name__ == "__main__":
    main()

