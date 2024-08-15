#!/usr/bin/env python

# import the neccessary repo
import sys
import rospy
import actionlib
import numpy as np

# Joint-Based Controller
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

# Controller Manager
from controller_manager_msgs.srv import SwitchControllerRequest, SwitchController
from controller_manager_msgs.srv import LoadControllerRequest, LoadController
from controller_manager_msgs.srv import ListControllers, ListControllersRequest

# Cartesian-Based Controller
import geometry_msgs.msg as geometry_msgs
from cartesian_control_msgs.msg import (
    FollowCartesianTrajectoryAction,
    FollowCartesianTrajectoryGoal,
    CartesianTrajectoryPoint,
)

# 
from ur5e_ctrl_jeff.msg import Robotiq2FGripper_robot_output
from cartesian_listener import CartesianStateListener
from gripper_listener import GripperStateListener  

# Compatibility for python2 and python3
if sys.version_info[0] < 3:
    input = raw_input

# If your robot description is created with a tf_prefix, those would have to be adapted
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# All of those controllers can be used to execute joint-based trajectories.
# The scaled versions should be preferred over the non-scaled versions.
JOINT_TRAJECTORY_CONTROLLERS = [
    "scaled_pos_joint_traj_controller",
    "scaled_vel_joint_traj_controller",
    "pos_joint_traj_controller",
    "vel_joint_traj_controller",
    "forward_joint_traj_controller",
]
JOINT_CONTROLLER_SELLECT = 0
"""
🔥scaled_pos_joint_traj_controller🔥 : 
    It controls the movement of each joint to follow a desired position trajectory, with an added scaling factor to modulate the speed and smoothness of the motion.
scaled_vel_joint_traj_controller : 
    It controls the velocity of each joint to follow a desired velocity trajectory, with a scaling factor to adjust the speed of the movements.
pos_joint_traj_controller : 
    It controls each joint to follow a specified position trajectory directly, without any additional scaling factors.
vel_joint_traj_controller : 
    It directly controls the velocity of each joint to follow a specified velocity trajectory, without any scaling factors.
forward_joint_traj_controller : 
    It uses a predefined joint trajectory to generate control inputs directly, often without relying heavily on feedback.
"""

# All of those controllers can be used to execute Cartesian trajectories.
# The scaled versions should be preferred over the non-scaled versions.
CARTESIAN_TRAJECTORY_CONTROLLERS = [
    "pose_based_cartesian_traj_controller",
    "joint_based_cartesian_traj_controller",
    "forward_cartesian_traj_controller",
]
CARTESIAN_CONTROLLER_SELLECT = 0
"""
🔥pose_based_cartesian_traj_controller🔥 : 
    It calculates the difference between the current pose and the target pose, 
    generating the necessary control inputs to gradually move the robot's end-effector (typically the Tool Center Point, TCP) to the desired pose.
joint_based_cartesian_traj_controller : 
    Given a Cartesian trajectory, it computes the desired positions for each joint to move the end-effector along the specified Cartesian path. 
    It combines inverse kinematics solving and trajectory tracking.
forward_cartesian_traj_controller : 
    It generates control inputs based on a predetermined Cartesian trajectory, driving the end-effector along the path directly,
    usually without using sensor feedback to adjust the trajectory.
"""

# We'll have to make sure that none of these controllers are running, as they will
# be conflicting with the joint trajectory controllers
CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "twist_controller"]
"""
joint_group_vel_controller : 
    It sets the velocity for each joint in the specified group, allowing for coordinated movement based on the desired velocity inputs.
twist_controller : 
    It commands the end-effector to move with specified linear and angular velocities, typically used for direct teleoperation or Cartesian space control.
"""

# We set gripper into two situations : open or close.
GRIPPER_SLEEP_INTERVAL = 0.1

GRIPPER_OPEN = 0
GRIPPER_CLSOE = 1

"""
rACT: First action to be made prior to any other actions, rACT bit will activate the Gripper. Clear rACT to reset the Gripper and clear
fault status.
l 0x0 - Deactivate Gripper.
l 0x1 - Activate Gripper (must stay on after activation routine is completed).

rGTO: The "Go To" action moves the Gripper fingers to the requested position using the configuration defined by the other registers,
rGTO will engage motion while byte 3, 4 and 5 will determine aimed position, force and speed. The only motions performed without
the rGTO bit are activation and automatic release routines.
l 0x0 - Stop.
l 0x1 - Go to requested position.

rATR: Automatic Release routine action slowly opens the Gripper fingers until all motion axes reach their mechanical limits. After all
motion is completed, the Gripper sends a fault signal and needs to be reactivated before any other motion is performed. The rATR bit
overrides all other commands excluding the activation bit (rACT).
l 0x0 - Normal.
l 0x1 - Emergency auto-release.

rARD: Auto-release direction. When auto-releasing, rARD commands the direction of the movement. The rARD bit should be set prior
to or at the same time as the rATR bit, as the motion direction is set when the auto-release is initiated.
l 0x0 - Closing auto-release
l 0x1 - Opening auto-release

rPR: POSITION REQUEST, This register is used to set the target position for the Gripper's fingers. The positions 0x00 and 0xFF correspond respectively to the fully
opened and fully closed mechanical stops. For detailed finger trajectory, please refer to the Specifications section.
l 0x00 - Open position, with 85 mm or 140 mm opening respectively
l 0xFF - Closed
l Opening / count: 0.4 mm (for 85 mm stroke) and 0.65 mm (for 140 mm stroke)

rSP: SPEED, This register is used to set the Gripper closing or opening speed in real time, however, setting a speed will not initiate a motion.
l 0x00 - Minimum speed
l 0xFF - Maximum speed

rFR: FORCE, The force setting defines the final gripping force for the Gripper. The force will fix the maximum current sent to the motor while in
motion. If the current limit is exceeded, the fingers stop and trigger an object detection notification. Please refer to the Robot Input
Registers & Status section for details on force control.
l 0x00 - Minimum force
l 0xFF - Maximum force
"""

class GripperCommander:
    """A commander to control the 2f-85 Gripper"""

    def __init__(self):
        
        # Initialization
        # rospy.init_node('Robotiq2FGripperSimpleController')
        self.gripper_pub = rospy.Publisher('Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output, queue_size=10)
        self.command = Robotiq2FGripper_robot_output();
        
        # Activate the gripper first
        self.gripper_activate()

    def gripper_activate(self):
        """Activate the gripper before utilizing"""

        rospy.sleep(GRIPPER_SLEEP_INTERVAL)

        self.command.rACT = 1    # Activate the gripper
        self.command.rGTO = 1    # Go to the position
        self.command.rSP  = 255  # Set the speed
        self.command.rFR  = 150  # Set the force

        activate_interval = GRIPPER_SLEEP_INTERVAL*10
        self.gripper_pub.publish(self.command)
        rospy.sleep(activate_interval)

    def gripper_close(self):
        """Close the gripper"""
        
        self.command.rPR = 255   # Fully close the gripper

        self.gripper_pub.publish(self.command)
        rospy.sleep(GRIPPER_SLEEP_INTERVAL)
    
    def gripper_open(self):
        """Open the gripper"""

        self.command.rPR = 0   # Fully open the gripper

        self.gripper_pub.publish(self.command)
        rospy.sleep(GRIPPER_SLEEP_INTERVAL)


class MotionCommander:
    """A trajectory client to run a joint trajectory"""

    def __init__(self, 
                 verbose: bool = True, 
                 timeout_wait_duration: int=5,
                 joint_controller_index: int = JOINT_CONTROLLER_SELLECT, 
                 cartesian_controller_index: int = CARTESIAN_CONTROLLER_SELLECT):
        
        # params initialization
        self.is_verbose = verbose
        self.wait_duration = timeout_wait_duration
        
        # initialize
        # rospy.init_node("motion_commander")
        timeout = rospy.Duration(self.wait_duration)

        # initialize the services we need
        self.switch_srv = rospy.ServiceProxy("controller_manager/switch_controller", SwitchController)
        self.load_srv = rospy.ServiceProxy("controller_manager/load_controller", LoadController)
        self.list_srv = rospy.ServiceProxy("controller_manager/list_controllers", ListControllers)
        
        # waiting for the switch service
        try:
            self.switch_srv.wait_for_service(timeout.to_sec()) # wait for the service 
        except rospy.exceptions.ROSException as err:
            rospy.logerr("Could not reach controller switch service. Msg: {}".format(err))
            sys.exit(-1)

        # initialize the robot controller | joint_trajectory_controller or cartesian_trajectory_controller
        self.joint_trajectory_controller = JOINT_TRAJECTORY_CONTROLLERS[joint_controller_index]
        self.cartesian_trajectory_controller = CARTESIAN_TRAJECTORY_CONTROLLERS[cartesian_controller_index]

        # 
        self.cartesian_state_listener = CartesianStateListener()
        self.gripper_state_listener = GripperStateListener(interval=GRIPPER_SLEEP_INTERVAL, timeout_wait_duration=self.wait_duration)
        self.gripper_commander = GripperCommander() 

    def send_joint_trajectory(self, position_list=[], velocity_list=[], duration_list=[]):
        """Send a trajectory using the selected action server"""

        # switch the controller to what we chose
        self.switch_controller(self.joint_trajectory_controller)

        # initialize a joint traj client
        trajectory_client = actionlib.SimpleActionClient(
            "{}/follow_joint_trajectory".format(self.joint_trajectory_controller), # the name of the server in ROS network
            FollowJointTrajectoryAction, # the type of the message
        )

        # wait for action server to be ready
        timeout = rospy.Duration(self.wait_duration)
        if not trajectory_client.wait_for_server(timeout):
            rospy.logerr("[ERROR] Could not reach controller action server.")
            sys.exit(-1)

        # create trajectory goal
        goal = FollowJointTrajectoryGoal()
        
        # name the each joint
        goal.trajectory.joint_names = JOINT_NAMES
        
        # check the size of each list in the trajectory
        assert len(position_list)==len(velocity_list)==len(duration_list), "[ERROR] receive different sizes of each list in joint trajectory"
        
        # append the joint-based trajectory 
        for i, position in enumerate(position_list):
            point = JointTrajectoryPoint()
            point.positions = position
            point.velocities = velocity_list[i]
            point.time_from_start = rospy.Duration(duration_list[i])
            goal.trajectory.points.append(point)
    
        # ask the user to confitm the following actions
        if self.is_verbose:
            self.ask_confirmation(position_list)
            rospy.loginfo("[INFO] Executing trajectory using the {}".format(self.joint_trajectory_controller))

        # send the goals and wait for answer
        trajectory_client.send_goal(goal)
        trajectory_client.wait_for_result()

        # get the results from server
        result = trajectory_client.get_result()
        rospy.loginfo("[INFO] Trajectory execution finished in state {}".format(result.error_code))

    def send_cartesian_trajectory(self, pose_list=[], duration_list=[]):
        """Send a Cartesian trajectory it using the selected action server"""

        # switch to the controller that we chose
        self.switch_controller(self.cartesian_trajectory_controller)

        # create the cartesian goal
        goal = FollowCartesianTrajectoryGoal()

        # initialize the trajectory client
        trajectory_client = actionlib.SimpleActionClient(
            "{}/follow_cartesian_trajectory".format(self.cartesian_trajectory_controller), # the name of the server in ROS network
            FollowCartesianTrajectoryAction, # the type of the message
        )

        # wait for action server to be ready
        timeout = rospy.Duration(self.wait_duration)
        if not trajectory_client.wait_for_server(timeout):
            rospy.logerr("[ERROR] Could not reach controller action server.")
            sys.exit(-1)
        
        # check the size of each list in the trajectory
        assert len(pose_list)==len(duration_list), "[ERROR] receive different sizes of each list in pose trajectory"

        # append the cartesian-based trajectory 
        for i, pose in enumerate(pose_list):
            point = CartesianTrajectoryPoint()
            point.pose = pose
            point.time_from_start = rospy.Duration(duration_list[i])
            goal.trajectory.points.append(point)
        
        # ask the user to confitm the following actions
        if self.is_verbose:
            self.ask_confirmation(pose_list)
            rospy.loginfo("[INFO] Executing trajectory using the {}".format(self.cartesian_trajectory_controller))

        print("=========== HERE ===========")

        # send the goals and wait for answer
        trajectory_client.send_goal(goal)
        trajectory_client.wait_for_result()
        
        # get the results from server
        result = trajectory_client.get_result()
        rospy.loginfo("[INFO] Trajectory execution finished in state {}".format(result.error_code))

    def search_gripper_mutation(self, gripper_list=[]):
        """
        find the mutational state of gripper
        
        param@gripper_list : list[GRIPPER_OPEN or GRIPPER_CLOSE]
        
        """

        assert all(x in [GRIPPER_OPEN, GRIPPER_CLSOE] for x in gripper_list), "[ERROR] the gripper list contains elements other than open and close. "

        cur_gripper_state = self.get_gripper_state()
        girpper_mutation_indexes = []
        girpper_mutation_actions = []

        for index, state in enumerate(gripper_list):
            if state != cur_gripper_state:
                girpper_mutation_actions.append(state)
                girpper_mutation_indexes.append(index)
                cur_gripper_state = state

        return girpper_mutation_indexes, girpper_mutation_actions
    
    def split_list(self, lst, indices):
        """
        split the pose and duration list according to the indices

        param@lst : 
        param@indices
        """

        result = [] # Initialize the result list
        start = 0 # Start index of the first slice

        # Loop through each index in indices
        for index in indices:
            result.append(lst[start:index+1])   # Create a slice from start to index (inclusive)
            start = index + 1 # Update the start index for the next slice

        # Add the remaining elements after the last index
        if start < len(lst):
            result.append(lst[start:])

        return result

    def execute_arm_gripper_trajectory(self, pose_list=[], grip_list=[], duration_list=[], is_ask_conf=True):
        """
        Execute the whole trajectory combining robot arm and gripper
        
        param@pose_list : cartesian position
        param@grip_list : 0/1 only
        param@duration_list : the interval between each movement

        """

        # check the size of each list in the trajectory
        assert len(pose_list)==len(grip_list)==len(duration_list), "[ERROR] receive different sizes of each list in pose trajectory"

        # switch to the controller that we chose
        self.switch_controller(self.cartesian_trajectory_controller)

        # initialize the trajectory client
        trajectory_client = actionlib.SimpleActionClient(
            "{}/follow_cartesian_trajectory".format(self.cartesian_trajectory_controller), # the name of the server in ROS network
            FollowCartesianTrajectoryAction, # the type of the message
        )

        # wait for action server to be ready
        timeout = rospy.Duration(self.wait_duration)
        if not trajectory_client.wait_for_server(timeout):
            rospy.logerr("[ERROR] Could not reach controller action server.")
            sys.exit(-1)

        # ask the user to confitm the following actions
        if is_ask_conf:
            self.ask_confirmation(pose_list)
        if self.is_verbose:
            rospy.loginfo("[INFO] Executing trajectory using the {}".format(self.cartesian_trajectory_controller))
        
        # split the poses according to the mutation of gripper action
        mutation_indexes, mutation_actions = self.search_gripper_mutation(grip_list)
        pose_list_split = self.split_list(pose_list, mutation_indexes)
        duration_list_split = self.split_list(duration_list, mutation_indexes)

        # print(mutation_indexes)
        # print(mutation_actions)
        # print(pose_list_split)
        # print(duration_list_split)

        # run the trajectory for each time
        for mut_idx,(poses, durations) in enumerate(zip(pose_list_split, duration_list_split)):
            
            # initial goals
            goal = FollowCartesianTrajectoryGoal()
            for mv_idx, (pose, duration) in enumerate(zip(poses, durations)):
                point = CartesianTrajectoryPoint()
                point.pose = pose
                point.time_from_start = rospy.Duration(duration)
                goal.trajectory.points.append(point)
            
            # # visulize the movement of the arm
            # if is_ask_conf:
            #     self.ask_confirmation(poses)
            # if self.is_verbose : 
            #     rospy.logwarn("[INFO] The robot will move to the following waypoints: {}".format(poses))

            # send the goals and wait for answer
            trajectory_client.send_goal(goal)
            trajectory_client.wait_for_result()

            # get the results from server
            if self.is_verbose:
                result = trajectory_client.get_result()
                rospy.loginfo("[INFO] Trajectory execution finished in state {}".format(result.error_code))
            
            ####### ####### ####### ####### ####### ####### ####### ####### ####### 

            # change the state of gripper
            if mut_idx <= len(mutation_actions) - 1:
                
                # show the movement of gripper
                if self.is_verbose:
                    rospy.loginfo("[INFO] The Gripper changed to {}".format(mutation_actions[mut_idx]))

                # command the gripper to move
                if mutation_actions[mut_idx]==GRIPPER_OPEN :
                    print("OPEN")
                    self.gripper_commander.gripper_open()
                elif mutation_actions[mut_idx]==GRIPPER_CLSOE :
                    print("CLOSE")
                    self.gripper_commander.gripper_close()
                else:
                    raise ValueError("[ERROR] the value of grip_list is neither 1 nor 0")
                
                # finish only until the gripper is done
                self.gripper_state_listener.wait_for_gripper()
        
        rospy.loginfo("[INFO] Trajectory execution finished successfully")
    
    def ask_confirmation(self, waypoint_list):
        """Ask the user for confirmation. This function is obviously not necessary, but makes sense
        in a testing script when you know nothing about the user's setup."""
        
        rospy.logwarn("[INFO] The robot will move to the following waypoints: \n{}".format(waypoint_list))
        confirmed = False
        valid = False
        while not valid:
            input_str = input(
                "Please confirm that the robot path is clear of obstacles.\n"
                "Keep the EM-Stop available at all times.\n"
                "Please type 'y' to proceed or 'n' to abort: "
            )
            
            valid = input_str in ["y", "n"]

            if not valid:
                rospy.loginfo("[INPUT] Please confirm by entering 'y' or abort by entering 'n'")
            else:
                confirmed = input_str == "y"
        
        if not confirmed:
            rospy.loginfo("[INFO] Exiting as requested by user.")
            sys.exit(0)

    def switch_controller(self, target_controller):
        """Activates the desired controller and stops all others from the predefined list above"""
    
        # all of the controllers
        other_controllers = (
            JOINT_TRAJECTORY_CONTROLLERS
            + CARTESIAN_TRAJECTORY_CONTROLLERS
            + CONFLICTING_CONTROLLERS
        )
        other_controllers.remove(target_controller)

        # check whether the target controller is running
        srv = ListControllersRequest()
        response = self.list_srv(srv)
        for controller in response.controller:
            if controller.name == target_controller and controller.state == "running":
                return

        # load the target controller if it's not already loaded
        srv = LoadControllerRequest()
        srv.name = target_controller
        self.load_srv(srv)

        # start the target_controller and stop the other controllers
        srv = SwitchControllerRequest()
        srv.stop_controllers = other_controllers
        srv.start_controllers = [target_controller]
        srv.strictness = SwitchControllerRequest.BEST_EFFORT
        self.switch_srv(srv)

    def get_arm_cartesian_state(self):
        """Get the position and orientation of the arm's end-effector"""

        return self.cartesian_state_listener.get_actual_cartesian()

    def get_gripper_state(self):
        """Get whether the gripper is closed or not"""

        if self.gripper_state_listener.get_is_closed():
            return GRIPPER_CLSOE
        else:
            return GRIPPER_OPEN
    
    def get_state(self):
        """Get the state including cartesian and gripper, return a 7-dimension array"""

        pose = self.get_arm_cartesian_state()
        gripper = self.get_gripper_state()
        state = np.array([[pose.position.x,
                        pose.position.y,
                        pose.position.z,
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                        gripper
                        ]], float)
        return state


if __name__ == "__main__":
    
    rospy.init_node("motion_commander")
    client = MotionCommander()

    print("===== MotionCommander =====")
    
    # # JOINT TRAJECTORY CONTROLLER
    # # the following list are arbitrary positions | Change to your own needs if desired
    # position_list = [[0, -1.57, -1.57, 0, 0, 0]]
    # position_list.append([0.2, -1.57, -1.57, 0, 0, 0])
    # position_list.append([-0.5, -1.57, -1.2, 0, 0, 0])
    # velocity_list = [[0.2, 0, 0, 0, 0, 0]]
    # velocity_list.append([-0.2, 0, 0, 0, 0, 0])
    # velocity_list.append([0, 0, 0, 0, 0, 0])
    # duration_list = [5.0, 10.0, 15.0]
    # client.send_joint_trajectory(position_list, velocity_list, duration_list)
    
    # # POSE TRAJECTORY CONTROLLER
    # # the following list are arbitrary positions | Change to your own needs if desired | Position([3]) + Quaternion([4])
    # pose_list = [
    #     geometry_msgs.Pose(
    #         geometry_msgs.Vector3(-1.519791009070174947e-01,-3.131022253507395048e-01,1.013676147114260129e+00), 
    #         geometry_msgs.Quaternion(-3.817350884507677566e-01,-7.129251057315758588e-01,5.668133858421183779e-01,1.572854141149138407e-01)
    #     ),
    #     geometry_msgs.Pose(
    #         geometry_msgs.Vector3(-5.712233991576016745e-01,-4.502260737368067867e-01,7.264138187685256209e-01), 
    #         geometry_msgs.Quaternion(5.036091149290695679e-01,7.681067146008192514e-01,-3.869030021392148577e-01,8.182909801016656492e-02)
    #     ),
    #     geometry_msgs.Pose(
    #         geometry_msgs.Vector3(-7.632043884707433445e-01,-3.879991053728846229e-01,3.876896953747783203e-01), 
    #         geometry_msgs.Quaternion(5.447764712748347504e-01,8.383633324432937517e-01,-1.910714013280295775e-02,6.605723731065349293e-04)
    #     ),
    #     geometry_msgs.Pose(
    #         geometry_msgs.Vector3(-2.433018494073919402e-01,-4.929757010216171409e-01,8.479811315436367458e-01), 
    #         geometry_msgs.Quaternion(-2.764537739283712825e-01,-9.060502273573407539e-01,2.989711706867989038e-01,1.151630821254677334e-01)
    #     ),
    # ]
    # duration_list = [10.0, 10.0, 10.0, 10.0]
    # client.send_cartesian_trajectory(pose_list, duration_list)
    # print(client.get_arm_cartesian_state())
    
    # # POSITION + GRIPPER
    # # the following list are arbitrary positions | Change to your own needs if desired | Position([3]) + Quaternion([4])
    # pose_list = [
    #     geometry_msgs.Pose(
    #         geometry_msgs.Vector3(0.2, 0.2, 0.45), geometry_msgs.Quaternion(0, 0, 0, 1)
    #     ),
    #     geometry_msgs.Pose(
    #         geometry_msgs.Vector3(0.3, 0.6, 0.65), geometry_msgs.Quaternion(0, 0, 0, 1)
    #     ),
    #     geometry_msgs.Pose(
    #         geometry_msgs.Vector3(0.5, 0.7, 0.85), geometry_msgs.Quaternion(0, 0, 0, 1)
    #     ),
    # ]
    # duration_list = [8.0, 16.0, 24.0]
    # grip_list = [0, 1, 0]
    # client.execute_arm_gripper_trajectory(pose_list, grip_list, duration_list)
    # print(client.get_arm_cartesian_state())
    
    # POSITION + GRIPPER
    # the following list are arbitrary positions | Change to your own needs if desired | Position([3]) + Quaternion([4])

    """
    position: 
        x: -0.5140640000000001
        y: -0.6890584615384615
        z: 0.5902399999999999
        orientation: 
        x: -0.41373427121606693
        y: -0.8780025890263642
        z: 0.11449318779703765
        w: 0.21172320711812312
    """
    
    # pose_list = [
    #     geometry_msgs.Pose(
    #         geometry_msgs.Vector3(x=-0.2862943306897481, y=-0.6872491842681708, z=0.5960513700027534), 
    #         geometry_msgs.Quaternion(x=-0.4125877485290233, y=-0.8766820482510818, z=0.12091663142395062, w=0.2158219272528257)
    #     ),
    # ]
    # duration_list = [12.0]
    # grip_list = [0]
    # client.execute_arm_gripper_trajectory(pose_list, grip_list, duration_list)
    
    print(client.get_arm_cartesian_state())
    print(client.get_state())
    
    # raise ValueError(
    #     "I only understand types 'joint_based' and 'cartesian', but got '{}'".format(
    #         trajectory_type
    #     )
    # )

