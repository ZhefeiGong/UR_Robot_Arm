#!/usr/bin/env python

import rospy
import cv2               
import threading                 
import numpy as np                                 
import pyrealsense2 as rs 
import PIL.Image
import os

import ur5e_ctrl_jeff.msg
from utils import capture_image

class RealsenseCamera:
    """
    The class for realsense camera | open another thread for picture capturing
    """

    def __init__(self):

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        
        # color | resolution-width | resolution-height | image format | frame rate
        self.cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 10)

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
            
            b,g,r = cv2.split(color_img)
            img_rgb = cv2.merge([r,g,b])
            return img_rgb

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

def test():
    """
    test

    """

    import time
    import numpy as np
    import pyrealsense2 as rs
    import cv2
    framerate = 15
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, framerate)
    pipe_profile = pipeline.start(config)
    save_path = "./images_jpg/"
    shot_flag = False
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        img_color = np.asanyarray(color_frame.get_data())
        cv2.imshow("q", img_color)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            shot_flag = ~shot_flag
            print("shot flag "+str(shot_flag))
        if shot_flag:
            cv2.imwrite(save_path + str(time.time()) + ".jpg", img_color)

if __name__ == "__main__":

    # capture one image
    camera = RealsenseCamera()
    camera.start()
    if camera.wait_for_ready():
        capture_image(camera,"/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/scene/test.jpg")
    else:
        print("等待相机准备超时")
    camera.stop()
    cv2.destroyAllWindows()
    
    # # 
    # test()
