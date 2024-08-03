#!/usr/bin/env python

import cv2               
import threading                 
import numpy as np                                 
import pyrealsense2 as rs                
import PIL.Image
import os

from utils import capture_image

class RealsenseCamera:
    """
    The class for realsense camera | open another thread for picture capturing
    """

    def __init__(self):

        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        # color | resolution-width | resolution-height | image format | frame rate
        self.cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

        self.profile = None
        self.frameset = None
        self.running = False

        self.lock = threading.Lock()
    
    def start(self):
        if not self.running:
            self.profile = self.pipe.start(self.cfg)
            self.running = True
            threading.Thread(target=self.update_frame, daemon=True).start()
    
    def update_frame(self):
        while self.running:
            frameset = self.pipe.wait_for_frames()
            with self.lock:
                self.frameset = frameset
    
    def get_frame(self):
        with self.lock:
            if self.frameset is None:
                return None
            color_frame = self.frameset.get_color_frame()
            if not color_frame:
                return None
            color_img = np.asanyarray(color_frame.get_data())
            return color_img

    def stop(self):
        self.running = False
        self.pipe.stop()

    def wait_for_ready(self, timeout=10, wake_up_pause=3):
        
        import time
        start_time = time.time()
        check_interval = 0.1
        is_ready=False

        while time.time() - start_time < timeout:
            with self.lock:
                if self.frameset is not None:
                    is_ready=True
                    break
            time.sleep(check_interval)
        
        if is_ready:
            time.sleep(wake_up_pause)
            return True
        else:
            return False


if __name__ == "__main__":

    camera = RealsenseCamera()
    camera.start()

    if camera.wait_for_ready():
        capture_image(camera,"/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/scene/test.jpg")
    else:
        print("等待相机准备超时")
    
    camera.stop()
    cv2.destroyAllWindows()
