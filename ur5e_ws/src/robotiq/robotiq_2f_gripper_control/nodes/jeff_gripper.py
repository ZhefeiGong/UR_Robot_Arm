#!/usr/bin/env python3

from __future__ import print_function

import rospy

import roslib; roslib.load_manifest('robotiq_2f_gripper_control')
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output  as outputMsg
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input  as inputMsg

from time import sleep
from std_msgs.msg import String

# Python2 or Python3
try:
    input = raw_input
except NameError:
    pass


"""
##################### Publisher #####################
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

# 
def genCommand(char, command):
    """Update the command according to the character entered by the user."""

    # activcate
    if char == 'a':
        command = outputMsg.Robotiq2FGripper_robot_output();
        command.rACT = 1
        command.rGTO = 1
        command.rSP  = 255
        command.rFR  = 150
    
    # reset
    if char == 'r':
        command = outputMsg.Robotiq2FGripper_robot_output();
        command.rACT = 0

    # close
    if char == 'c':
        command.rPR = 255

    # open
    if char == 'o':
        command.rPR = 0

    # position
    # If the command entered is a int, assign this value to rPRA
    try:
        command.rPR = int(char)
        if command.rPR > 255:
            command.rPR = 255
        if command.rPR < 0:
            command.rPR = 0
    except ValueError:
        pass

    # faster
    if char == 'f':
        command.rSP += 25
        if command.rSP > 255:
            command.rSP = 255

    # slower
    if char == 'l':
        command.rSP -= 25
        if command.rSP < 0:
            command.rSP = 0

    # increase the force
    if char == 'i':
        command.rFR += 25
        if command.rFR > 255:
            command.rFR = 255

    # decrease the force
    if char == 'd':
        command.rFR -= 25
        if command.rFR < 0:
            command.rFR = 0

    return command


def askForCommand(command):
    """Ask the user for a command to send to the gripper."""

    currentCommand  = 'Simple 2F Gripper Controller\n-----\nCurrent command:'
    currentCommand += '  rACT = '  + str(command.rACT)
    currentCommand += ', rGTO = '  + str(command.rGTO)
    currentCommand += ', rATR = '  + str(command.rATR)
    currentCommand += ', rPR = '   + str(command.rPR )
    currentCommand += ', rSP = '   + str(command.rSP )
    currentCommand += ', rFR = '   + str(command.rFR )


    print(currentCommand)

    strAskForCommand  = '-----\nAvailable commands\n\n'
    strAskForCommand += 'r: Reset\n'
    strAskForCommand += 'a: Activate\n'
    strAskForCommand += 'c: Close\n'
    strAskForCommand += 'o: Open\n'
    strAskForCommand += '(0-255): Go to that position\n'
    strAskForCommand += 'f: Faster\n'
    strAskForCommand += 'l: Slower\n'
    strAskForCommand += 'i: Increase force\n'
    strAskForCommand += 'd: Decrease force\n'
    
    strAskForCommand += '-->'

    return input(strAskForCommand)


def publisher():
    """Main loop which requests new commands and publish them on the Robotiq2FGripperRobotOutput topic."""

    rospy.init_node('Robotiq2FGripperSimpleController')

    pub = rospy.Publisher('Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output)
    
    command = outputMsg.Robotiq2FGripper_robot_output();
    
    while not rospy.is_shutdown():

        command = genCommand(askForCommand(command), command)

        print(command)

        pub.publish(command)

        rospy.sleep(0.1)


"""
##################### Listener #####################
gACT: Activation status, echo of the rACT bit (activation bit).
l 0x0 - Gripper reset.
l 0x1 - Gripper activation.

gGTO: Action status, echo of the rGTO bit (go to bit).
l 0x0 - Stopped (or performing activation / automatic release).
l 0x1 - Go to Position Request.

gSTA: Gripper status, returns the current status & motion of the Gripper fingers.
l 0x00 - Gripper is in reset ( or automatic release ) state. See Fault Status if Gripper is activated.
l 0x01 - Activation in progress.
l 0x02 - Not used.
l 0x03 - Activation is completed.

gOBJ: Object detection status, is a built-in feature that provides information on possible object pick-up. Ignore if gGTO == 0.
l 0x00 - Fingers are in motion towards requested position. No object detected.
l 0x01 - Fingers have stopped due to a contact while opening before requested position. Object detected opening.
l 0x02 - Fingers have stopped due to a contact while closing before requested position. Object detected closing.
l 0x03 - Fingers are at requested position. No object detected or object has been loss / dropped.

gFLT: Fault status returns general error messages that are useful for troubleshooting. Fault LED (red) is present on the Gripper chassis,
LED can be blue, red or both and be solid or blinking.
l 0x00 - No fault (LED is blue)
l Priority faults (LED is blue)
l 0x05 - Action delayed, activation (reactivation) must be completed prior to perfmoring the action.
l 0x07 - The activation bit must be set prior to action.

gPR: Echo of the requested position for the Gripper, value between 0x00 and 0xFF.
l 0x00 - Full opening.
l 0xFF - Full closing

gPO: Actual position of the Gripper obtained via the encoders, value between 0x00 and 0xFF.
l 0x00 - Fully opened.
l 0xFF - Fully closed.

gCU: The current is read instantaneously from the motor drive, value between 0x00 and 0xFF, approximate current equivalent is 10 *
value read in mA.

"""

def printStatus(status):
    """Print the status string generated by the statusInterpreter function."""

    print(statusInterpreter(status))

def Robotiq2FGripperStatusListener():
    """Initialize the node and subscribe to the Robotiq2FGripperRobotInput topic."""

    rospy.init_node('Robotiq2FGripperStatusListener')
    rospy.Subscriber("Robotiq2FGripperRobotInput", inputMsg.Robotiq2FGripper_robot_input, printStatus)
    rospy.spin()

def statusInterpreter(status):
    """Generate a string according to the current value of the status variables."""

    output = '\n-----\n2F gripper status interpreter\n-----\n'

    #gACT
    output += 'gACT = ' + str(status.gACT) + ': '
    if(status.gACT == 0):
       output += 'Gripper reset\n'
    if(status.gACT == 1):
       output += 'Gripper activation\n'

    #gGTO
    output += 'gGTO = ' + str(status.gGTO) + ': '
    if(status.gGTO == 0):
        output += 'Standby (or performing activation/automatic release)\n'
    if(status.gGTO == 1):
        output += 'Go to Position Request\n'

    #gSTA
    output += 'gSTA = ' + str(status.gSTA) + ': '
    if(status.gSTA == 0):
        output += 'Gripper is in reset ( or automatic release ) state. see Fault Status if Gripper is activated\n'
    if(status.gSTA == 1):
        output += 'Activation in progress\n'
    if(status.gSTA == 2):
        output += 'Not used\n'
    if(status.gSTA == 3):
        output += 'Activation is completed\n'

    #gOBJ
    output += 'gOBJ = ' + str(status.gOBJ) + ': '
    if(status.gOBJ == 0):
        output += 'Fingers are in motion (only meaningful if gGTO = 1)\n'
    if(status.gOBJ == 1):
        output += 'Fingers have stopped due to a contact while opening\n'
    if(status.gOBJ == 2):
        output += 'Fingers have stopped due to a contact while closing \n'
    if(status.gOBJ == 3):
        output += 'Fingers are at requested position\n'

    #gFLT
    output += 'gFLT = ' + str(status.gFLT) + ': '
    if(status.gFLT == 0x00):
        output += 'No Fault\n'
    if(status.gFLT == 0x05):
        output += 'Priority Fault: Action delayed, initialization must be completed prior to action\n'
    if(status.gFLT == 0x07):
        output += 'Priority Fault: The activation bit must be set prior to action\n'
    if(status.gFLT == 0x09):
        output += 'Minor Fault: The communication chip is not ready (may be booting)\n'
    if(status.gFLT == 0x0B):
        output += 'Minor Fault: Automatic release in progress\n'
    if(status.gFLT == 0x0E):
        output += 'Major Fault: Overcurrent protection triggered\n'
    if(status.gFLT == 0x0F):
        output += 'Major Fault: Automatic release completed\n'

    #gPR
    output += 'gPR = ' + str(status.gPR) + ': '
    output += 'Echo of the requested position for the Gripper: ' + str(status.gPR) + '/255\n'

    #gPO
    output += 'gPO = ' + str(status.gPO) + ': '
    output += 'Position of Fingers: ' + str(status.gPO) + '/255\n'
    
    #gCU
    output += 'gCU = ' + str(status.gCU) + ': '
    output += 'Current of Fingers: ' + str(status.gCU * 10) + ' mA\n'

    return output


if __name__ == '__main__':

    publisher()
    
    # Robotiq2FGripperStatusListener()
