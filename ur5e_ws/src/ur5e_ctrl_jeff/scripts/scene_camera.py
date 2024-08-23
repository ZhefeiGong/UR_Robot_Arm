#!/usr/bin/env python

import rospy
import cv2
import threading
import time

import ur5e_ctrl_jeff.msg
from utils import capture_image

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def image_publisher(camera_id=2):
    """

    """
    
    rospy.init_node('scene_image_publisher_node', anonymous=True)
    image_pub = rospy.Publisher('camera/scene_image', Image, queue_size=10)
    bridge = CvBridge()
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0) # auto
    
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

class SceneSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.current_frame = None
        rospy.Subscriber('camera/scene_image', Image, self.image_callback)

    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def get_current_image(self):
        return self.current_frame

class SceneCamera:
    """
    the class for wrist camera | open another thread for picture capturing

    """
    
    def __init__(self, camera_id=2):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

        self.cap.set(cv2.CAP_PROP_FPS, 30.0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)

        exposure = -5.5
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 10000*2**exposure)

        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.update_frame, daemon=True).start()
    
    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

    def wait_for_ready(self, timeout=10, wake_up_pause=3):
        
        import time
        start_time = time.time()
        check_interval = 0.1
        is_ready=False

        while time.time() - start_time < timeout:
            with self.lock:
                if self.frame is not None:
                    is_ready=True
                    break
            time.sleep(check_interval)
        
        if is_ready:
            time.sleep(wake_up_pause)
            return True
        else:
            return False


def test():

    # Open the default camera (usually the first camera found, index 0)
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

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

    ### Set the exposure

    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
    
    exposure = -5.5
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
    cap.set(cv2.CAP_PROP_EXPOSURE, 10000*2**exposure)
    print(f"Desired exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")

    time.sleep(2)

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR")

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        cv2.imshow("Camera",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # # capture one image
    # camera = SceneCamera()
    # camera.start()
    # if camera.wait_for_ready():
    #     capture_image(camera,"/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/scene/test.jpg")
    # else:
    #     print("等待相机准备超时")
    # camera.stop()
    # cv2.destroyAllWindows()

    # test the camera
    test()

    # try:
    #     image_publisher()
    # except rospy.ROSInterruptException:
    #     pass

