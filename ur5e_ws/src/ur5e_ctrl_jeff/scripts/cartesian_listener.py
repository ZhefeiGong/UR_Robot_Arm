#!/usr/bin/env python

import sys
import rospy
import numpy as np
import geometry_msgs.msg as geometry_msgs
from tf2_msgs.msg import TFMessage
import ur5e_ctrl_jeff.msg
from utils import quaternion_to_euler

# Compatibility for python2 and python3
if sys.version_info[0] < 3:
    input = raw_input

class CartesianStateListener():
    """Subscribe the cartesian position of the end-effector of the UR5e robot"""
    
    def __init__(self,
                 verbose: bool = False):
        
        self.is_verbose = verbose
        self.cartesian_pose = geometry_msgs.Pose()

        # rospy.init_node('tf_listener')
        rospy.Subscriber("tf", TFMessage, self.tf_callback)
        
    def tf_callback(self, msg):
        """Initialize the callback function of the subscriber"""
    
        num_tf_end_effector = 1
        
        if len(msg.transforms) == num_tf_end_effector:

            # get the info
            transform = msg.transforms[0]
            
            # print the info
            if self.is_verbose:
                
                print("[MSG] Frame ID : ", transform.header.frame_id) # the name of the coordinate we use
                print("[MSG] Child Frame ID : ", transform.child_frame_id) # the name of the child coordinate (end-effector | gripper)
                print("[MSG] Translation : \n", transform.transform.translation)
                print("[MSG] Rotation : \n", transform.transform.rotation)
                
                # state = np.array([[transform.transform.translation.x, 
                #                     transform.transform.translation.y, 
                #                     transform.transform.translation.z,
                #                     transform.transform.rotation.x,
                #                     transform.transform.rotation.y,
                #                     transform.transform.rotation.z,
                #                     transform.transform.rotation.w,]])
                # state_euler = quaternion_to_euler(state)
                # print(state_euler[:,3:])
            
            # save the cartesian info
            self.cartesian_pose.position = transform.transform.translation          # x,y,z | geometry_msgs.Vector3
            self.cartesian_pose.orientation = transform.transform.rotation          # x,y,z,w | geometry_msgs.Quaternion
    
    def get_actual_cartesian(self):
        """Get the cartesian position infomation"""

        if self.cartesian_pose is not None:
            return self.cartesian_pose
        else:
            raise ValueError("[ERROR] haven't assign any value to cartesian_pose")



def main():
    """Main Function"""

    print("===== CartesianStateListener =====")

    rospy.init_node('tf_listener')
    listener = CartesianStateListener(verbose=True)
    rospy.spin()


if __name__ == "__main__":
    main()