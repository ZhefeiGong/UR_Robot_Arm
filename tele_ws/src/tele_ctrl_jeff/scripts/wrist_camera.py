#!/usr/bin/env python

import rospy
import cv2
import threading
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def image_publisher(camera_id=0):
    """
    @func : the publisher of camera node
    """

    print("[INFO] wrist camera is runnning...")

    rospy.init_node('wrist_image_publisher_node', anonymous=True)
    image_pub = rospy.Publisher('camera/wrist_image', Image, queue_size=10)
    bridge = CvBridge()
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    ### EXPOSURE
    # auto
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0) # auto
    # # manual
    # exposure = -5.5
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0) 
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0) # manual
    # cap.set(cv2.CAP_PROP_EXPOSURE, 10000*2**exposure)


    if not cap.isOpened():
        rospy.logerr("Unable to open camera")
        return

    rate = rospy.Rate(10)  # 10Hz

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            image_message = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_pub.publish(image_message)

        rate.sleep()

    cap.release()

class WristSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.current_frame = None
        rospy.Subscriber('camera/wrist_image', Image, self.image_callback)

    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def get_current_image(self):
        return self.current_frame

def test():

    # Open the default camera (usually the first camera found, index 0)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Verify if the resolution was set correctly
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Actual resolution: {int(actual_width)}x{int(actual_height)}")

    # Set desired resolution (e.g., 1280x720)
    desired_width = 640 # 1280
    desired_height = 480 # 720
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print(f"Desired resolution: {int(desired_width)}x{int(desired_height)}")
    print(f"Current resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    ### EXPOSURE
    # auto
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0) # auto
    # # manual
    # exposure = -5.5
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0) 
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0) # manual
    # cap.set(cv2.CAP_PROP_EXPOSURE, 10000*2**exposure)
    # print(f"Desired exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    
    time.sleep(2)

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR")

        cv2.imshow("Camera",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    # # test the camera
    # test()
    
    # launch the publisher
    try:
        image_publisher()
    except rospy.ROSInterruptException:
        pass

