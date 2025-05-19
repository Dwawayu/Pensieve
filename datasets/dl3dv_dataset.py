import json
import os
import random
from PIL import Image
import glob
from datasets.base_dataset import BaseDataset
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from utils.general_utils import listdir_nohidden
from utils.matrix_utils import matrix_to_quaternion

import numpy as np

class DL3DV10KDataset(BaseDataset):

    def get_video_folders(self):
        sub_root_list = listdir_nohidden(self.config["data_path"])
        sub_root_list.sort()
        sub_root_list = sub_root_list
        dataset = []
        
        with open("./evaluation_jsons/evaluation_index_dl3dv_5view.json", "r") as f:
            eval_idx = json.load(f)

        for sub_dataset in sub_root_list:
            videos_sub_root_path = os.path.join(self.config["data_path"], sub_dataset)
            if not os.path.isdir(videos_sub_root_path):
                continue
            videos_file_path = listdir_nohidden(videos_sub_root_path)
            videos_file_path.sort()
            for video_path in videos_file_path:
                if video_path in eval_idx:
                    continue
                ref_video_path = os.path.join(videos_sub_root_path, video_path, "images_4")
                if not os.path.isdir(ref_video_path):
                    continue
                dataset.append(ref_video_path)

        return dataset
    
    def get_image_names(self, video_folder):
        image_names = listdir_nohidden(video_folder)
        image_names.sort()
        return image_names
    
    def read_image(self, video_folder, selected_image_names):
        selected_image_paths = [os.path.join(video_folder, name) for name in selected_image_names]
        images = []
        for image_path in selected_image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                return None
            
            images.append(np.array(image))
        images_array = np.array(images)
        
        video_tensor = torch.tensor(images_array).permute(0, 3, 1, 2)
        video_tensor = video_tensor / 255.
        return video_tensor
    
    def get_camera_folders(self):
        camera_folders_list = [ref_video_path.replace("images_4", "transforms.json") for ref_video_path in self.video_folders_list]
        return camera_folders_list
            
    
    def read_camera(self, camera_folder, selected_image_names):
        if not os.path.exists(camera_folder):
            return None
        with open(camera_folder) as camera_file:
            contents = json.load(camera_file)
            frames = contents["frames"]
            c2ws = []
            name_to_c2w = {frame["file_path"]: frame["transform_matrix"] for frame in frames}
            for image_name in selected_image_names:
                image_name = os.path.join("images", image_name)
                if image_name in name_to_c2w:
                    c2ws.append(name_to_c2w[image_name])
                else:
                    return None

            c2ws = torch.tensor(c2ws, dtype=torch.float32)
            c2ws[..., :3, 1:3] *= -1
            outputs = {}
            N = c2ws.shape[0]
            outputs["quaternion"] = matrix_to_quaternion(c2ws[:, :3, :3])
            outputs["t"] = c2ws[:, :3, 3]
            outputs["_cx"] = torch.tensor(contents["cx"], dtype=torch.float32).repeat(N)
            outputs["_cy"] = torch.tensor(contents["cy"], dtype=torch.float32).repeat(N)
            outputs["fx"] = torch.tensor(contents["fl_x"], dtype=torch.float32).repeat(N)
            outputs["fy"] = torch.tensor(contents["fl_y"], dtype=torch.float32).repeat(N)
            outputs["width"] = contents["w"]
            outputs["height"] = contents["h"]
            return outputs

    def get_flow_folders(self):
        flow_folders = [video_folder.replace("/datasets/", "/datasets_processed/").replace("/images_4", "/optical_flow") for video_folder in self.video_folders_list]
        return flow_folders

    def read_flow(self, flow_folder, selected_image_names):
        def readFlow(fn):
            """ Read .flo file in Middlebury format"""
            # Code adapted from:
            # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

            # WARNING: this will work on little-endian architectures (eg Intel x86) only!
            # print 'fn = %s'%(fn)
            with open(fn, 'rb') as f:
                magic = np.fromfile(f, np.float32, count=1)
                if 202021.25 != magic:
                    print('Magic number incorrect. Invalid .flo file')
                    return None
                else:
                    w = np.fromfile(f, np.int32, count=1)
                    h = np.fromfile(f, np.int32, count=1)
                    # print 'Reading %d x %d flo file\n' % (w, h)
                    data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                    # Reshape testdata into 3D array (columns, rows, bands)
                    # The reshape here is for visualization, the original code is (w,h,2)
                    return np.resize(data, (int(h), int(w), 2))

        flow_list = []
        for image_name in selected_image_names[:-1]:
            flow_name = image_name[:-4] + "_pred.flo"
            flow_path = os.path.join(flow_folder, flow_name)
            if not os.path.exists(flow_path):
                return None
            flow_image = readFlow(flow_path)
            flow_image = torch.tensor(flow_image).permute(2, 0, 1) # 2, H, W
            flow_image[0] /= flow_image.shape[2]
            flow_image[1] /= flow_image.shape[1] # [0, 1]
            flow_list.append(flow_image)

        flow_list = torch.stack(flow_list)
        return flow_list # L-1, 2, H, W