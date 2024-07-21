#!/usr/bin/env python

import sys
import rospy
import geometry_msgs.msg as geometry_msgs
from tf2_msgs.msg import TFMessage

# Compatibility for python2 and python3
if sys.version_info[0] < 3:
    input = raw_input

class CartesianStateListener():
    """Subscribe the cartesian position of the end-effector of the UR5e robot"""
    
    def __init__(self,
                 verbose: bool = False):
        
        self.is_verbose = verbose
        self.cartesian_pose = geometry_msgs.Pose()

        rospy.init_node('tf_listener')
        rospy.Subscriber("tf", TFMessage, self.tf_callback)
        
    def tf_callback(self, msg):
        """Initialize the callback function of the subscriber"""

        assert len(msg.transforms) == 1, "[ERROR] the size of each tf msg is not equal to one"
        for transform in msg.transforms : 
            # print the info
            if self.is_verbose:
                print("[MSG] Frame ID : ", transform.header.frame_id)
                print("[MSG] Child Frame ID : ", transform.header.frame_id)
                print("[MSG] Translation : ", transform.transform.translation)
                print("[MSG] Rotation : ", transform.transform.rotation)
            # save the cartesian info
            self.cartesian_pose.position = transform.transform.translation          # x,y,z | geometry_msgs.Vector3
            self.cartesian_pose.orientation = transform.transform.rotation          # x,y,z,w | geometry_msgs.Quaternion
    
    def get_actual_cartesian(self):
        """Get the cartesian position infomation"""

        if self.cartesian_pose is not None:
            return self.cartesian_pose
        else:
            raise ValueError("[ERROR] haven't assign any value to cartesian_pose")

if __name__ == "__main__":
    print("===== CartesianStateListener =====")
