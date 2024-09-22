## set env
cd /home/robot/UR_Robot_Arm/ur5e_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
## run
sudo usermod -a -G dialout $USER
dmesg | grep tty
rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0
