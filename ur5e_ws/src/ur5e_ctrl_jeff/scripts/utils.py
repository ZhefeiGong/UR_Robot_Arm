#!/usr/bin/env python

import cv2

def capture_image(camera, path):
    """
    capture the image from the camera
    """
    
    frame = camera.get_frame()
    if frame is not None:
        cv2.imwrite(path, frame)
        print(f"照片已保存为 {path}")
    else:
        print("无法获取照片")


def ask_confirmation(prompt=""):
    """
    ask the user to confirm the movement of the next step
    """

    confirmed = False
    valid = False
    while not valid:
        input_str = input(
            prompt + 
            "\n Please type 'y' to proceed or 'n' to abort: "
        )
        valid = input_str in ["y", "n"]
        if not valid:
            print("[INPUT] Please confirm by entering 'y' or abort by entering 'n'")
        else:
            confirmed = input_str == "y"
        if not confirmed:
            print("[INFO] Exiting as requested by user.")
            sys.exit(0)