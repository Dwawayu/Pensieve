import json
import os
from io import BytesIO
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

class RE10KEvaluationJson(BaseDataset):

    def get_video_folders(self):
        with open(os.path.join(self.config["data_path"], "index.json"), "r") as f:
            self.index_data = json.load(f)
            
        with open("./evaluation_jsons/evaluation_index_re10k.json", "r") as f:
            self.eval_idx = json.load(f)

        return list(self.eval_idx.keys())
    
    def get_image_names(self, video_folder):
        file_name = self.index_data[video_folder]
        chunk = torch.load(os.path.join(self.config["data_path"], file_name), weights_only=False)
        for scene in chunk:
            if scene["key"] == video_folder:
                self.now_scene = scene
                break
        if self.eval_idx[video_folder] is None:
            return None
        self.context_idx = self.eval_idx[video_folder]['context']
        self.target_idx = self.eval_idx[video_folder]['target']
        return self.context_idx


    def read_image(self, video_folder, selected_image_names):
        images = []
        for idx in selected_image_names:
            images.append(torch.tensor(np.array(Image.open(BytesIO(self.now_scene['images'][idx].numpy().tobytes())))))
        images = torch.stack(images)
        video_tensor = images.permute(0, 3, 1, 2)
        video_tensor = video_tensor / 255.
        return video_tensor
    
    def get_camera_folders(self):
        return list(self.eval_idx.keys())
    
    def read_camera(self, camera_folder, selected_image_names):
        scene = self.now_scene
        cameras = []
        
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
    
    def read_misc(self, outputs):
        image_names = []
        outputs["target_cameras"] = self.read_camera(None, self.target_idx)
        outputs["target_images"] = self.read_image(None, self.target_idx)
        
        return outputs