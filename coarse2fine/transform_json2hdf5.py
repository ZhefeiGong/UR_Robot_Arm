# :date: 2nd.Nov
# :author: Jeff

import os
import cv2
import json
import h5py
import numpy as np

def get_task_all_json(dataset_root="/liujinxin/zhefei/ARGen4IL/workspace/data/DATASET", 
                      task_name = "bowl"):
    """
    :func:
    """

    json_name = "data.json"
    traj_info = []
    for date_folder in os.listdir(dataset_root):
        date_path = os.path.join(dataset_root, date_folder)
        if os.path.isdir(date_path):
            bowl_path = os.path.join(date_path, task_name)
            if os.path.isdir(bowl_path):
                for traj_folder in os.listdir(bowl_path):
                    traj_path = os.path.join(bowl_path, traj_folder)
                    if os.path.isdir(traj_path):
                        data_json_path = os.path.join(traj_path, json_name)
                        if os.path.exists(data_json_path):
                            traj_info.append({
                                "date": date_folder,
                                "traj": traj_folder,
                                "data": data_json_path
                            })
    print(f"[INFO] the scale of {task_name} task: {len(traj_info)} trajectories")
    # for info in traj_info: print(f"Date: {info['date']}, Trajectory: {info['traj']}, Data: {info['data']}")
    return traj_info

def store_as_hdf5(traj_info, 
                  task_name="bowl", 
                  save_root="/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/"):
    """
    :func:
    """
    RESIZE_HEIGHT = 120 # 480
    RESIZE_WIDTH = 160  # 640
    hdf5_path = os.path.join(save_root, f"{task_name}.hdf5")
    with h5py.File(hdf5_path, "w") as hdf5_file:
        data_group = hdf5_file.create_group("data")
        demo_idx = 0
        for info in traj_info:
            date_info = info['date']
            run_info = info['traj']
            json_path = info['data']
            root_path = json_path.split("/data.json", 1)[0]

            ### each trajectory
            if json_path.endswith(".json"):

                print(f'[INFO] current trajectory: {date_info}/{run_info}')

                demo_name = f"demo_{demo_idx}"
                demo_group = data_group.create_group(demo_name)
                
                with open(json_path, "r") as f:
                    data = json.load(f)
                    data_lenghth = len(data)

                    ### instruction
                    data_instr = data[0]['task']
                    demo_group.create_dataset("instruction",
                                              data=data_instr)

                    ### actions | [1:]
                    data_actions = []
                    for i in range(data_lenghth)[1:]: # Length-1
                        data_actions.append(data[i]['pose'])
                    data_actions = np.array(data_actions)
                    demo_group.create_dataset("actions",
                                              data=data_actions.astype(np.float32))

                    ### observation | [:-1]
                    obs_group = demo_group.create_group("obs")
                    # robot0_eef_pos
                    data_eef_pos = []
                    for i in range(data_lenghth)[:-1]: # Length-1
                        data_eef_pos.append(data[i]['pose'][:3])
                    data_eef_pos = np.array(data_eef_pos)
                    obs_group.create_dataset("robot0_eef_pos", 
                                             data = data_eef_pos.astype(np.float32))
                    # robot0_eef_quat
                    data_eef_quat = []
                    for i in range(data_lenghth)[:-1]: # Length-1
                        data_eef_quat.append(data[i]['pose'][3:-1])
                    data_eef_quat = np.array(data_eef_quat)
                    obs_group.create_dataset("robot0_eef_quat", 
                                             data = data_eef_quat.astype(np.float32))
                    # robot0_gripper_qpos
                    data_gripper_qpos = []
                    for i in range(data_lenghth)[:-1]: # Length-1
                        data_gripper_qpos.append(data[i]['pose'][-1:])
                    data_gripper_qpos = np.array(data_gripper_qpos)
                    obs_group.create_dataset("robot0_gripper_qpos", 
                                             data = data_gripper_qpos.astype(np.float32))
                    # agentview_image
                    data_agentview_image = []
                    for i in range(data_lenghth)[:-1]: # Length-1
                        img_pth = data[i]['imgs'].split("/image/", 1)[-1]
                        img_pth = os.path.join(f"{root_path}/image", img_pth)
                        image = cv2.imread(img_pth)                                 # BGR
                        image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))    # BGR | (480，640，3) -> (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # RGB
                        # cv2.imwrite("output_image_bgr.jpg", image)                
                        # cv2.imwrite("output_image_rgb.jpg", image_rgb)            
                        # print("Image shape:", image_rgb.shape)
                        # image_rgb = np.transpose(image_rgb, (2, 0, 1))              # (3, RESIZE_HEIGHT, RESIZE_WIDTH)
                        data_agentview_image.append(image_rgb)                      # 
                    data_agentview_image = np.array(data_agentview_image)           # BHWC | [0,255]
                    # min_value = np.min(data_agentview_image)    # vis | 
                    # max_value = np.max(data_agentview_image)    # vis |
                    obs_group.create_dataset("agentview_image", 
                                             data = data_agentview_image.astype(np.uint8))
                    # robot0_eye_in_hand_image
                    data_eye_in_hand_image = []
                    for i in range(data_lenghth)[:-1]: # Length-1
                        img_pth = data[i]['imgw'].split("/image/", 1)[-1]
                        img_pth = os.path.join(f"{root_path}/image", img_pth)
                        image = cv2.imread(img_pth)                                 # BGR
                        image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))    # BGR | (480，640，3) -> (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # RGB
                        # cv2.imwrite("output_image_bgr.jpg", image) 
                        # cv2.imwrite("output_image_rgb.jpg", image_rgb) 
                        # print("Image shape:", image_rgb.shape)
                        # image_rgb = np.transpose(image_rgb, (2, 0, 1))              # (3, RESIZE_HEIGHT, RESIZE_WIDTH)
                        data_eye_in_hand_image.append(image_rgb)                    # 
                    data_eye_in_hand_image = np.array(data_eye_in_hand_image)       # BHWC | [0,255]
                    # min_value = np.min(data_eye_in_hand_image)    # vis | 
                    # max_value = np.max(data_eye_in_hand_image)    # vis |
                    obs_group.create_dataset("robot0_eye_in_hand_image", 
                                             data = data_eye_in_hand_image.astype(np.uint8)) 
            # renew
            demo_idx += 1
            
    print('[INFO] done !!!')


if __name__ == "__main__":
    
    ### param
    task_name = "bowl"
    dataset_root="/liujinxin/zhefei/ARGen4IL/workspace/data/DATASET"
    save_root="/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data"

    ### get all traj
    traj_info = get_task_all_json(dataset_root, task_name)

    ### save as hdf5
    store_as_hdf5(traj_info, task_name, save_root)
