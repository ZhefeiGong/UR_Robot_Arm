#!/usr/bin/env python

import rospy
import cv2
import threading

import ur5e_ctrl_jeff.msg
from utils import capture_image

class WristCamera:
    """
    the class for wrist camera | open another thread for picture capturing

    """

    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
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


if __name__ == "__main__":

    camera = WristCamera()
    camera.start()

    if camera.wait_for_ready():
        capture_image(camera,"/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/wrist/test.jpg")
    else:
        print("等待相机准备超时")

    camera.stop()
    cv2.destroyAllWindows()

