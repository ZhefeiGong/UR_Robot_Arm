# 🦾 UR5e 📏
> UR5e | Robotiq 2f-85 gripper


## ⚙️ Setting | UR5e

>[manual-en](https://www.universal-robots.com/manuals/EN/PDF/SW5_19/user-manual-UR5e-PDF_online/710-965-00_UR5e_User_Manual_en_Global.pdf) | [manual-zh](https://s3-eu-west-1.amazonaws.com/ur-support-site/165903/99419_UR5e_User_Manual_zh_Global.pdf)

### 🔩 Drive with Source Ros

##### 1. Environment
* UR5e
* Ubuntu 20.04.6
* Noetic
* Python3.8

##### 2. Intall ROS | Noetic | Ubuntu 20.04 
[Neotic Installation](https://wiki.ros.org/noetic/Installation/Ubuntu#Ubuntu_install_of_ROS_Noetic)

```bash
#
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# (or maybe we need the following due to the firewall)
# sudo sh -c '. /etc/lsb-release && echo "deb http://mirrors.ustc.edu.cn/ros/ubuntu/ `lsb_release -cs` main" > /etc/apt/sources.list.d/ros-latest.list'

#
sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

#
sudo apt update

#
sudo apt install ros-noetic-desktop-full

#
sudo apt search ros-noetic

#
source /opt/ros/noetic/setup.bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

#
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
```

##### 3. Install moveit 

```bash
sudo apt install ros-noetic-moveit
```

##### 4. Install universal_robots_ros_driver | universal_robot

[universal_robots_ros_driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver) | [dirver_intro](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/tree/master/ur_robot_driver)| [universal_robot](https://github.com/ros-industrial/universal_robot) | [real-time / linux](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/real_time.md)

Binary-Version

```bash
# we set ${ROS_DISTRO} as noetic
sudo apt install ros-noetic-ur-robot-driver
sudo apt install ros-noetic-ur-calibration
sudo apt-get install ros-noetic-universal-robots
# set the source
source /opt/ros/${ROS_DISTRO}/setup.bash
```

🔥Development-Version🔥

```bash
# source global ros
source /opt/ros/noetic/setup.bash

# create a catkin workspace
mkdir -p catkin_ws/src && cd catkin_ws

# clone the driver
git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git src/Universal_Robots_ROS_Driver

# clone the description. 
git clone -b noetic-devel https://github.com/ros-industrial/universal_robot.git src/universal_robot

# install dependencies
sudo apt update -qq
rosdep update
rosdep install --from-paths src --ignore-src -y

# build the workspace
catkin_make

# activate the workspace (ie: source it)
source devel/setup.bash

```

```bash
#@NOTICE : we need use the python/boost/... from ubuntu20.04 rather than anaconda
$ sudo gedit ~/.bashrc
#@NOTICE : remove the initialization of anaconda or any other environments
```

##### 5. Simulation Test

```bash
# simulate the robot and environment in Gazebo
roslaunch ur_gazebo ur5e_bringup.launch
# use MoveIt for motion planning and control
roslaunch ur5e_moveit_config moveit_planning_execution.launch sim:=true
# finally visualize the whole process through RViz
roslaunch ur5e_moveit_config moveit_rviz.launch
```

##### 6. Install External-Control on robot | link PC and robot through TCP-IP

[External-Control](https://github.com/UniversalRobots/Universal_Robots_ExternalControl_URCap/releases) | [InstallGuide](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/install_urcap_e_series.md)

```python
###### CONNECT ROBOT and PC ######
# the PC IPv4 
address : 192.168.1.10
netmask : 255.255.255.0
gateway : 192.168.1.1
# the robot IPv4
address : 192.168.1.60
netmask : 255.255.255.0
gateway : 192.168.1.1
```

##### 7. Communicate with the Robot | rs485

[rs485](https://github.com/UniversalRobots/Universal_Robots_ToolComm_Forwarder_URCap/releases) | [Communication](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/setup_tool_communication.md) 

```bash
# launch the ros score
roslaunch ur_robot_driver ur5e_bringup.launch  use_tool_communication:=true tool_voltage:=24 tool_parity:=0 tool_baud_rate:=115200 tool_stop_bits:=1 tool_rx_idle_chars:=1.5 tool_tx_idle_chars:=3.5 tool_device_name:=/tmp/ttyUR

# test the ttyUR connection
rosrun ur_robot_driver tool_communication

# utilize just like the true serial
rosrun imaginary_drivers rs485_node device:=/tmp/ttyUR
```

##### 8. Prepare the ROS PC

Extract calibration information

```bash
roslaunch ur_calibration calibration_correction.launch robot_ip:=192.168.1.60 target_filename:="home/robot/my_robot_calibration.yaml"
```

##### 9. Get Start

[usage examples](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/usage_example.md)	

>Notice : the version of URSoftware for e-series robots should no less than 5.5.1, if not we can update the software version of the UR pad through [updates](https://www.universal-robots.cn/articles/ur/documentation/legacy-download-center/)

* Starting the driver and visualizing the robot in RViz
* Control the robot
* Control the robot using MoveIt

```bash
###### Visualizing ######
# launch the driver
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.1.60 kinematics_config:=/home/robot/my_robot_calibration.yaml
# in another terminal run rviz for visualization
roslaunch ur_robot_driver example_rviz.launch

###### Controling ######
rosrun ur_robot_driver test_move

###### Control the robot using Moveit ######
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.1.60 kinematics_config:=/home/robot/my_robot_calibration.yaml

roslaunch ur5e_moveit_config moveit_planning_execution.launch

roslaunch ur5e_moveit_config moveit_rviz.launch rviz_config:=/home/robot/ur5e_ws/src/universal_robot/ur5e_moveit_config/launch/moveit.rviz

```

### 🔩 Drive with Python-urx

>[pythonb-urx](https://github.com/SintefManufacturing/python-urx) | [guideline-zh](https://blog.csdn.net/rocachilles/article/details/102667474)


### 🔩 Drive with ur_rtde

>[python-api](https://pypi.org/project/ur-rtde/) | [guideline](https://sdurobotics.gitlab.io/ur_rtde/index.html)



## ⚙️ Setting | Gripper | Robotiq-2f-85

>[manual](https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_CB-Series_PDF_20190122.pdf)


### 1. Robotiq Environment

>[github](https://github.com/jr-robotics/robotiq.git)

```bash
# # official - only support until Melodic
# git clone https://github.com/ros-industrial/robotiq.git src/robotiq

# support for Noetic
git clone https://github.com/jr-robotics/robotiq.git src/robotiq

# install dependencies
sudo apt update -qq
rosdep update
rosdep install --from-paths src --ignore-src -y

# build the workspace
catkin_make

# activate the workspace (ie: source it)
source devel/setup.bash
```


### 2. Connection

**🔥Method-1🔥** : 2f-85 $\rightarrow$ rs485 $\rightarrow$ usb $\leftarrow$ computer
[csdn](https://blog.csdn.net/gyxx1998/article/details/118710774?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%A6%82%E4%BD%95%E6%8E%A7%E5%88%B6UR%E6%9C%BA%E6%A2%B0%E8%87%82%E4%B8%8A%E7%9A%842F-85&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-118710774.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187) | [ros_tutorial](https://wiki.ros.org/robotiq/Tutorials/Control%20of%20a%202-Finger%20Gripper%20using%20the%20Modbus%20RTU%20protocol%20%28ros%20kinetic%20and%20newer%20releases%29)

**Method-2** : 2f-85 $\rightarrow$ rs485 $\rightarrow$ tool communication $\leftarrow$ computer
[issue-1](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/issues/506#issuecomment-1256338704) | [issue-2](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/issues/691) | [tool_communication](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/setup_tool_communication.md)

🔥**Method-3**🔥 : 2f-85 $\rightarrow$ rs485 $\rightarrow$ usb $\rightarrow$ pendant $\leftarrow$ 65332 $\leftarrow$ computer
[issue-1](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/issues/506#issuecomment-1292417947) | [code](https://gitlab.com/sdurobotics/ur_rtde/-/blob/master/doc/_static/robotiq_gripper.py)


### 3. Run

```bash
# test the conection by Method-1
sudo usermod -a -G dialout $USER
dmesg | grep tty

# 
roscore

# run the node
rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0

# run the controller
rosrun robotiq_2f_gripper_control Robotiq2FGripperSimpleController.py

# run the listener 
rosrun robotiq_2f_gripper_control Robotiq2FGripperStatusListener.py
```


## ⚙️ Setting | Teleoperation | 3D Connexion

>refer to [UR-Teleop](https://github.com/keitheorem/3DConnexion-Spacemouse-UR-Teleop) | [DiffsuionPolicy](https://github.com/real-stanford/diffusion_policy/tree/main/diffusion_policy/real_world)

### 1. Env Setting
* `ur_rtde`
* `spnav`
* `robotiq gripper` | `3D Connexion spacemouse` | `ur5e`

### 2. Build

* download dependencies of spacemouse
```bash
sudo apt install libspnav-dev spacenavd; sudo systemctl start spacenavd
pip install spnav
```
* check if spacemouse is connected to workstation
```bash
lsusb
```
* download `RTDE` library
```bash
pip install ur_rtde
```
In the `spnav` library, `PyCObject_AsVoidPtr` is deprecated. `find . -name spnav` on terminal to find `spnav` folder. Replace all instances of `PyCObject_AsVoidPtr` with `PyCapsule_GetPointer` in `init.py`


## ⚙️ Coding

>ROS | Robotics Operating System | [tutorial](https://wiki.ros.org/rospy_tutorials/Tutorials)

### 🔧 Rospy Intro From GPT

1. Initialize a ros `Node`
	`rospy.init_node('node_name', anonymous=True)`
2. Publish the `msg`
	`rospy.Publisher('chatter', String, queue_size=10)`
3. Subscribe the `msg`
	`rospy.Subscriber('chatter', String, callback)`
4. Utilize the `service`
	* `Server` side
		`rospy.Service('add_two_ints', AddTwoInts, handle_add_two_ints)`
	* `Client` side
		`rospy.wait_for_service('add_two_ints')`
		`rospy.ServiceProxy('add_two_ints', AddTwoInts)`


### 🔧 Debug

```bash
# show all of the topic we can use
rostopic list

# show a specific topic
rostopic info /topic_name
rostopic echo /topic_name
rostopic type /topic_name

# show the communication graph
rqtgraph

```


### 🔧 Code Space

* 🔥 [Repository](https://github.com/ZhefeiGong/UR_Robot_Arm) 🔥
	1. ubuntu 20.04
	2. ur5e + robotiq-2f-85


## 📖 Reference

#### 🔩 Drive with `ur-ros`
* [universal_robots_ros_driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver)

#### 🔩 Drive with `python-urx`
* [pythonb-urx](https://github.com/SintefManufacturing/python-urx)
* [guideline-zh](https://blog.csdn.net/rocachilles/article/details/102667474)

#### 🔩 Drive with `ur_rtde`
* [python-api](https://pypi.org/project/ur-rtde/)
* [guideline](https://sdurobotics.gitlab.io/ur_rtde/index.html)


