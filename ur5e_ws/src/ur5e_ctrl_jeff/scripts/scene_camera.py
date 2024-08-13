#!/usr/bin/env python

import rospy
import cv2
import threading
import time

import ur5e_ctrl_jeff.msg
from utils import capture_image

class SceneCamera:
    """
    the class for wrist camera | open another thread for picture capturing

    """
    
    def __init__(self, camera_id=2):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

        self.cap.set(cv2.CAP_PROP_FPS, 30.0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

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
    desired_width = 1280
    desired_height = 720
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print(f"Desired resolution: {int(desired_width)}x{int(desired_height)}")

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

    # 
    test()
