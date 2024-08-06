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


def test():
    # Open the default camera (usually the first camera found, index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # # Set desired resolution (e.g., 1280x720)
    # desired_width = 1280
    # desired_height = 720
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # Verify if the resolution was set correctly
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(f"Set resolution: {desired_width}x{desired_height}")
    print(f"Actual resolution: {int(actual_width)}x{int(actual_height)}")

    # # Check for the maximum resolution supported by the camera
    # # This method may not directly give the max resolution, hence iterating to find the max is one approach
    # max_width = 0
    # max_height = 0

    # for width in [1920, 1280, 640, 320]:
    #     for height in [1080, 720, 480, 240]:
    #         cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    #         actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #         actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #         if actual_width == width and actual_height == height:
    #             max_width = max(max_width, width)
    #             max_height = max(max_height, height)

    # print(f"Maximum resolution: {max_width}x{max_height}")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # capture one image
    camera = WristCamera()
    camera.start()
    if camera.wait_for_ready():
        capture_image(camera,"/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/wrist/test.jpg")
    else:
        print("等待相机准备超时")
    camera.stop()
    cv2.destroyAllWindows()

    # # 
    # test()