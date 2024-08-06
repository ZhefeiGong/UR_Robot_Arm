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


def concateImage(image1_path, image2_path, width=1300, height=700, horizonMargin=2, font_size=16):
    """

    """

    # Open images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    image1_width, image1_height = image1.size  # ( 480, 640, 3 )
    image2_width, image2_height = image2.size  # hand_image": ( 480, 640, 3 ).

    # Create a new image with a white background
    new_image = Image.new("RGB", (width, height), "white")

    # Draw a black border
    draw = ImageDraw.Draw(new_image)
    border_color = "black"
    border_width = 2  # Width of the border; you can adjust this
    # draw.rectangle(
    #     [(0, 0), (width - 1, height - 1)], outline=border_color, width=border_width
    # )

    # Calculate positions for the images
    image1_x = border_width + horizonMargin  # Adjust for border width
    image1_y = (height - image1_height) // 2  # Vertically centered
    image2_x = (
        width - image2_width - border_width - horizonMargin
    )  # Adjust for border width
    image2_y = (height - image2_height) // 2  # Vertically centered

    # Paste the images onto the new image
    new_image.paste(image1, (image1_x, image1_y))
    new_image.paste(image2, (image2_x, image2_y))

    # Specify the size of your font
    font_size = font_size  # Change this to your desired font size

    # Load the font (use a .ttf file if you have a specific font in mind)
    # For a specific font: font = ImageFont.truetype('path/to/font.ttf', font_size)
    font = ImageFont.load_default().font_variant(size=font_size)
    # font = ImageFont.truetype(size=font_size)

    # Center the text horizontally relative to its corresponding image
    text1 = "global view"
    text2 = "gripper"
    text2_2 = "view"
    
    text1_width = draw.textlength(text1, font=font)
    text2_width = draw.textlength(text2, font=font)
    text2_2_width = draw.textlength(text2_2, font=font)

    text1_x = image1_x + (image1_width - text1_width) // 2
    text2_x = image2_x + (image2_width - text2_width) // 2
    text2_2_x = image2_x + (image2_width - text2_2_width) // 2

    text1_y = image1_y + image1_height
    text2_y = image2_y + image2_height
    text2_2_y = image2_y + image2_height + 20

    draw.text((text1_x, text1_y), text1, fill="black", font=font)
    draw.text((text2_x, text2_y), text2, fill="black", font=font)
    draw.text((text2_2_x, text2_2_y), text2_2, fill="black", font=font)
    
    return new_image

if __name__ == "__main__":
    img_path_scene = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/scene/test.jpg"
    img_path_wrist = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/wrist/test.jpg"
    img_path_initial = "/home/robot/UR_Robot_Arm/ur5e_ws/src/ur5e_ctrl_jeff/img/init_.jpg"

    concateImage(img_path_scene, img_path_wrist).save(output_path)