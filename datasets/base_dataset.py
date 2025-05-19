from math import floor
import torch
import torchvision
import os
from PIL import Image

from utils.config_utils import get_instance_from_config, GlobalState
import random
import numpy as np

from utils.general_utils import sample_sublist

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        **config
    ):
        self.config = config
        self.video_folders_list = self.get_video_folders()
        self.transforms = self.get_transforms()
        
        if self.config.get("read_camera", False):
            self.camera_folders_list = self.get_camera_folders()

        if self.config.get("read_depth", False):
            self.depth_folders_list = self.get_depth_folders()

        if self.config.get("read_flow", False):
            self.flow_folders_list = self.get_flow_folders()
            
        if self.config.get("data_cache", False):
            self.data_cache = {}
        else:
            self.data_cache = None

    def get_transforms(self):
        if "transforms" not in self.config:
            return []
        transforms_list = []
        for transform_name, transform_config in self.config["transforms"].items():
            transforms_list.append(get_instance_from_config(transform_config))
        return torchvision.transforms.v2.Compose(transforms_list)
    
    # def get_camera_transforms(self):
    #     if "transforms" not in self.config:
    #         return []
    #     transforms_list = []
    #     for transform_name, transform_config in self.config["transforms"].items():
    #         if "size" not in transform_config["params"]:
    #             print(f"Skip creating camera transform for {transform_config['class']} since it does not have size parameter.")
    #             continue

    #         transform_class = transform_config["class"].split(".")[-1]
    #         if transform_class == "Resize":
    #             transform_config["class"] = "utils.camera.Resize"
    #         else:
    #             raise NotImplementedError(f"Camera transform class {transform_class} is not implemented.")
            
        #     transforms_list.append(get_instance_from_config(transform_config))

        # return torchvision.transforms.Compose(transforms_list)

    def get_video_folders(self):
        raise NotImplementedError("Get video folders to traverse images in it")
    
    def get_image_names(self, video_folder):
        raise NotImplementedError("Get image names in a video folder in order to read images in in the same __getitem__")
    
    def get_camera_folders(self):
        raise NotImplementedError("Get camera folders.")

    def read_image(self, video_folder, selected_image_names):
        selected_image_paths = [os.path.join(video_folder, name) for name in selected_image_names]
        images_array = np.array([np.array(Image.open(path).convert("RGB")) for path in selected_image_paths])
        video_tensor = torch.tensor(images_array).permute(0, 3, 1, 2)
        video_tensor = video_tensor / 255.
        return video_tensor
    
    def read_camera(self, camera_folder):
        raise NotImplementedError(" Since the format of the camera folders is dataset specific,"
                                  " this reading function should be implemented in the dataset class.")
    
    def __len__(self):
        return len(self.video_folders_list)
    
    def __getitem__(self, idx):
        
        outputs = {}
        video_folder = self.video_folders_list[idx]
        image_names_list = self.get_image_names(video_folder)
        if image_names_list is None:
            return None
        if self.config.get("multi_views", 1) > 1:
            multi_views = self.config["multi_views"]
            assert len(image_names_list) == multi_views, f"The number of view list should be equal to the multi_views parameter."
            all_image_names_list = image_names_list
            image_names_list = all_image_names_list[0]
            
        min_video_length = self.config["min_video_length"]
        max_video_length = self.config["max_video_length"]
        if len(image_names_list) < min_video_length:
            return None

        if "init_max_step" in self.config:
            warmup_max_step = self.config["init_max_step"] + (self.config["max_step"] - self.config["init_max_step"]) * GlobalState["global_step"] / self.config["warmup_steps"]
            warmup_max_step = int(warmup_max_step)
            max_step = min(warmup_max_step, self.config["max_step"])
        else:
            max_step = self.config["max_step"]
        
        if self.config.get("multi_views", 1) > 1:
            _, sub_slice = sample_sublist(image_names_list, min_video_length, max_video_length, self.config["min_step"], max_step, self.config.get("step_mode", "constant"))
            selected_image_names = [sublist[sub_slice] for sublist in all_image_names_list]
            selected_image_names = [item for sublist in zip(*selected_image_names) for item in sublist]
        else:
            selected_image_names, _ = sample_sublist(image_names_list, min_video_length, max_video_length, self.config["min_step"], max_step, self.config.get("step_mode", "constant"))
        
        video_tensor = self.read_image(video_folder, selected_image_names)
        if video_tensor is None:
            return None

        outputs["video_tensor"] = video_tensor

        if self.config.get("read_camera", False):
            camera_folder = self.camera_folders_list[idx]
            camera_dict = self.read_camera(camera_folder, selected_image_names)
            if not camera_dict:
                return None
            outputs["camera_dict"] = camera_dict

        if self.config.get("read_depth", False):
            depth_folder = self.depth_folders_list[idx]
            depth_tensor = self.read_depth(depth_folder, selected_image_names)
            if depth_tensor is None:
                return None
            outputs["depth_tensor"] = depth_tensor

        if self.config.get("read_flow", False):
            # Shape should be L-1, 2, H, W
            # Value should in [0, 1]
            flow_folder = self.flow_folders_list[idx]
            flow_tensor = self.read_flow(flow_folder, selected_image_names)
            if flow_tensor is None:
                return None
            outputs["flow_tensor"] = flow_tensor
            
        if self.config.get("read_misc", False):
            outputs = self.read_misc(outputs)
            if outputs is None:
                return None
        
        outputs = self.transforms(outputs)
        
        return outputs