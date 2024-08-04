#!/usr/bin/env python

import rospy
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


from PIL import Image, ImageDraw, ImageFont
def generate_initial_img(image1_path, image2_path, output_path, direction='vertical'):
    """
    Concatenate two images either vertically or horizontally.

    :param image1_path: Path to the first image
    :param image2_path: Path to the second image
    :param output_path: Path to save the concatenated image
    :param direction: 'vertical' or 'horizontal' direction to concatenate images
    """
    # Open images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Determine size of new image
    if direction == 'vertical':
        new_width = max(image1.width, image2.width)
        new_height = image1.height + image2.height
    elif direction == 'horizontal':
        new_width = image1.width + image2.width
        new_height = max(image1.height, image2.height)
    else:
        raise ValueError("Direction must be 'vertical' or 'horizontal'")

    # Create a new blank image
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    # Paste the images into the new image
    if direction == 'vertical':
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, image1.height))
    elif direction == 'horizontal':
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1.width, 0))

    # Save the result
    new_image.save(output_path)
    print(f"Concatenated image saved to {output_path}")