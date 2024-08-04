#!/usr/bin/env python

import sys
import rospy
from ur5e_ctrl_jeff.msg import Robotiq2FGripper_robot_input  


# Compatibility for python2 and python3
if sys.version_info[0] < 3:
    input = raw_input

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

class GripperStateListener():
    """Subscribe the status of the gripper"""

    def __init__(self,
                 verbose: bool = False,
                 interval: float = 0.1,
                 timeout_wait_duration: int = 5):

        self.GIRPPER_STOP_SIGN_CLP = 0x02
        self.GIRPPER_STOP_SIGN_NOR = 0x03
        self.is_verbose = verbose
        self.gripper_state = None
        self.is_gripper_stopped = True
        self.mim_interval = interval
        self.wait_duration = timeout_wait_duration
        
        # rospy.init_node('Robotiq2FGripperStatusListener')
        rospy.Subscriber("Robotiq2FGripperRobotInput", Robotiq2FGripper_robot_input, self.gripper_callback)
    
    def gripper_callback(self, msg):
        """Define the callback function of gripper subscriber"""

        if self.is_verbose:
            print(self.status_interpreter(msg))
        
        self.gripper_state = msg
        self.is_gripper_stopped = (msg.gOBJ == self.GIRPPER_STOP_SIGN_NOR or msg.gOBJ == self.GIRPPER_STOP_SIGN_CLP)

    def get_gripper_state(self):
        """Get gripper state info"""

        if self.gripper_state is not None:
            return self.gripper_state
        else:
            raise ValueError("[ERROR] haven't assign any value to gripper_state")
    
    def get_is_gripper_stopped(self):
        """Get the info to judge whether the gripper is stopped"""

        return self.is_gripper_stopped

    def get_is_closed(self):
        """Get the info to judge whether the gripper is closed"""

        if self.gripper_state.gPR == 0xFF :
            return True
        else:
            return False
    
    def wait_for_gripper(self):
        """Wait for the gripper's execution"""

        # initialize the params
        start_time = rospy.Time.now()
        timeout = rospy.Duration(self.wait_duration)

        print("BEGIN 2 WAIT")

        # begin to wait
        while not rospy.is_shutdown():
            
            # check whether the gripper is done
            if self.is_gripper_stopped:
                print("DONE")
                return

            # check whether executiing time is out
            if (rospy.Time.now()-start_time).to_sec() > timeout.to_sec():
                raise ValueError("[ERROR] gripper executing time is too long")

            # sleep a little bit time
            rospy.sleep(self.mim_interval)
    
    def status_interpreter(self, status):
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

def main():
    """Main Function"""

    print("===== GripperStateListener =====")

    rospy.init_node('Robotiq2FGripperStatusListener')
    listener = GripperStateListener(verbose=True)
    rospy.spin()

if __name__ == "__main__":
    main()
    