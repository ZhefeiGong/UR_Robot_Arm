#!/usr/bin/env python

import cv2
import threading

class WristCamera:
    """
    the class for wrist camera

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

    def wait_for_ready(self):
        while True:
            if self.frame is not None:
                break
        
        return

def capture_image(camera, path="/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/wrist/wristcaptured_image.jpg"):
    """

    """

    frame = camera.get_frame()
    if frame is not None:
        cv2.imwrite(path, frame)
        print(f"照片已保存为 {path}")
    else:
        print("无法获取照片")


if __name__ == "__main__":

    camera = WristCamera()
    camera.start()
    camera.wait_for_ready()

    capture_image(camera)

    camera.stop()

    cv2.destroyAllWindows()

