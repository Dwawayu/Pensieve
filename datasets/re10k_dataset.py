import json
import os
from io import BytesIO
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from datasets.base_dataset import BaseDataset
from utils.general_utils import listdir_nohidden
from utils.matrix_utils import matrix_to_quaternion


class RE10KDataset(BaseDataset):

    def get_video_folders(self):
        
        index_file_path = os.path.join(self.config["data_path"], "index.json")
        with open(index_file_path, 'r') as index_file:
            index_data = json.load(index_file)

        file_scene_list = []
             
        for key, value in index_data.items():
            file_scene_list.append((os.path.join(self.config["data_path"], value), key))
            
        file_scene_list.sort(key=lambda x: x[0])

        self.file_scene_list = file_scene_list

        return file_scene_list
    
    def get_image_names(self, video_folder):
        if self.data_cache is not None:
            if video_folder[1] in self.data_cache:
                self.now_scene = self.data_cache[video_folder[1]]
            else:
                chunk = torch.load(video_folder[0], weights_only=False)
                for scene in chunk:
                    self.data_cache[scene["key"]] = scene
                self.now_scene = self.data_cache[video_folder[1]]
        else:
            chunk = torch.load(video_folder[0], weights_only=False)
            for scene in chunk:
                if scene["key"] == video_folder[1]:
                    self.now_scene = scene
                    break
        image_names = list(range(len(self.now_scene['images'])))
        return image_names#[:1] * 100 # debug
    
    def read_image(self, video_folder, selected_image_names):
        video_tensor = []
        for idx in selected_image_names:
            video_tensor.append(torch.tensor(np.array(Image.open(BytesIO(self.now_scene['images'][idx].numpy().tobytes())))))
        video_tensor = torch.stack(video_tensor)
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = video_tensor / 255.
        return video_tensor
    
    def get_camera_folders(self):
        return self.file_scene_list
    
    def read_camera(self, camera_folder, selected_image_names):
        scene = self.now_scene

        cameras = scene['cameras'][selected_image_names]
        w2cs = cameras[:, 6:].reshape(-1, 3, 4)
        c2w_R = w2cs[:, :, :3].transpose(1, 2)
        c2w_t = -c2w_R @ w2cs[:, :, 3:]
        
        camera_dict = {}
        camera_dict["quaternion"] = matrix_to_quaternion(c2w_R)
        camera_dict["t"] = c2w_t.squeeze(-1)
        camera_dict["_cx"] = cameras[:, 2]
        camera_dict["_cy"] = cameras[:, 3]
        camera_dict["fx"] = cameras[:, 0]
        camera_dict["fy"] = cameras[:, 1]
        camera_dict["width"] = 1
        camera_dict["height"] = 1
        
        return camera_dict