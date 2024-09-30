#!/usr/bin/env python

import rospy
import cv2
import threading
import time

import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def run():

    # Open the default camera (usually the first camera found, index 0)
    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Verify if the resolution was set correctly
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Actual resolution: {int(actual_width)}x{int(actual_height)}")

    # Set desired resolution (e.g., 1280x720)
    desired_width = 1920 # 1280 | 640
    desired_height = 1080 # 720 | 480
    fps = 30.0
    cap.set(cv2.CAP_PROP_FPS, fps)
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
    print(f"Desired exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    
    time.sleep(2)
    
    is_recording = False
    out = None
    record_count = 0
    blank_image = np.zeros((500,500,3),np.uint8)
    
    # run
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera",blank_image)
            key = cv2.waitKey(1) & 0xFF 
            if key == ord('b'):
                if not is_recording:
                    print("Start recording...")
                    is_recording = True
                    filename = f"output_{record_count}.mp4"
                    record_count += 1
                    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'),fps,(desired_width, desired_height))
            if is_recording:
                out.write(frame)
            if key == ord('e'):
                if is_recording:
                    print("Stop recording...")
                    is_recording = False
                    out.release()
            if key == ord('q'):
                break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()