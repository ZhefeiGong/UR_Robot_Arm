## set env
cd /home/robot/UR_Robot_Arm/ur5e_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
## run
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.2.2 kinematics_config:=/home/robot/UR_Robot_Arm/ur5e_ws/my_robot_calibration.yaml
