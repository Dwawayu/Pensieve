import copy
import random

import torch

from utils.GS_utils import gs_cat, gs_trans, render
from utils.camera import average_intrinsics, norm_extrinsics
from utils.config_utils import get_instance_from_config
from utils.matrix_utils import align_camera_dict, quaternion_multiply, quaternion_to_matrix, quaternion_translation_inverse, quaternion_translation_multiply
    
    
class RefineInference:
    def __init__(self, trainer, camera_min, camera_max, gs_min, gs_max, random_order=True, lvsm_render=True, gs_render=True,
                 lower_weight=0.1):
        self.trainer = trainer
        self.camera_min = camera_min
        self.camera_max = camera_max
        self.gs_min = gs_min
        self.gs_max = gs_max
        self.random_order = random_order
        self.lvsm_render = lvsm_render
        self.gs_render = gs_render
        self.lower_weight = lower_weight
        
    def __call__(self, inputs):
        backbone = self.trainer.backbones["shared_backbone"]
        camera_decoder = self.trainer.decoders["camera_decoder"]
        if self.lvsm_render:
            lvsm_decoder = self.trainer.decoders["lvsm_decoder"]
        if self.gs_render:
            gs_decoder = self.trainer.decoders["gs_decoder"]
        
        camera_num = random.randint(self.camera_min, self.camera_max)
        gs_min = min(self.gs_min, camera_num-1)
        gs_max = min(self.gs_max, camera_num-1)
        gs_num = random.randint(gs_min, gs_max)
        camera_idx = list(range(camera_num))
        if self.random_order:
            random.shuffle(camera_idx)
            
        inputs["video_tensor"] = inputs["video_tensor"][:, camera_idx]
        gs_idx = random.sample(range(inputs["video_tensor"].shape[1]), gs_num)
        gs_idx.sort()
        
        inputs["gs_idx"] = gs_idx
        # features_camera, features_gs_0, features_gs_1 = backbone(images, gs_idx)
        inputs = torch.utils.checkpoint.checkpoint(backbone, inputs, gs_idx, use_reentrant=False)
        
        for l in range(inputs["camera_features"].shape[1]):
            inputs["now_idx"] = l
            # inputs = camera_decoder(inputs)
            inputs = torch.utils.checkpoint.checkpoint(camera_decoder, inputs, use_reentrant=False)
        if self.trainer.config["single_intrinsic"]:
            inputs["cameras_list"] = average_intrinsics(inputs["cameras_list"])
        if self.trainer.config["norm_extrinsic"]:
            inputs["cameras_list"] = norm_extrinsics(inputs["cameras_list"], idx=gs_idx[0])
            
        if self.lvsm_render:
            inputs["tgt_idx"] = list(range(len(inputs["cameras_list"])))
            inputs = torch.utils.checkpoint.checkpoint(lvsm_decoder, inputs, use_reentrant=False)
            loss_weight = [self.lower_weight if i in gs_idx else 1. for i in inputs["tgt_idx"]]
            inputs["rets_dict"][("lvsm",)] = {"render": inputs["lvsm_prediction"],
                                             "loss_weight": loss_weight}
            if "lvsm_weight" in inputs:
                inputs["rets_dict"][("lvsm",)]["weight"] = inputs["lvsm_weight"]
        
        if self.gs_render:
            gs_inputs = {}
            gs_inputs["cameras_list"] = [inputs["cameras_list"][i] for i in gs_idx]
            gs_inputs["video_tensor"] = inputs["video_tensor"][:, gs_idx]
            gs_inputs["gs_list"] = []
            gs_inputs["gs_features"] = inputs["gs_features"]
            
            for l in range(inputs["gs_features"].shape[1]):
                gs_inputs["now_idx"] = l
                # gs_inputs = gs_decoder(gs_inputs)
                gs_inputs = torch.utils.checkpoint.checkpoint(gs_decoder, gs_inputs, use_reentrant=False)
            inputs["gs_list"] = gs_inputs["gs_list"]
                
            if "predict_depth" in gs_inputs:
                inputs["predict_depth"] = torch.zeros_like(inputs["video_tensor"][:, :, None, :1])
                predict_depth = gs_inputs["predict_depth"]
                if predict_depth.shape[-2:] != inputs["video_tensor"].shape[-2:]:
                    predict_depth_shape = predict_depth.shape[:-2]
                    predict_depth = predict_depth.reshape(-1, 1, *predict_depth.shape[-2:])
                    predict_depth = torch.nn.functional.interpolate(predict_depth, size=inputs["video_tensor"].shape[-2:], mode="bilinear", align_corners=False)
                    predict_depth = predict_depth.reshape(*predict_depth_shape, *predict_depth.shape[-2:])
                inputs["predict_depth"][:, gs_idx] = predict_depth
            
            gs_to_render = gs_cat(gs_inputs["gs_list"])
            for i in range(len(inputs["cameras_list"])):
                if i in gs_idx:
                    loss_weight = self.lower_weight
                else:
                    loss_weight = 1.
                rets = render(inputs["cameras_list"][i], gs_to_render)
                rets["loss_weight"] = loss_weight
                
                if self.trainer.config["alpha_bg"] == "GT_detach":
                    rets["render"] = rets["rend_alpha"].detach() * rets["render"] + (1. - rets["rend_alpha"].detach()) * inputs["video_tensor"][:, i]
                elif self.trainer.config["alpha_bg"] == "noise":
                    rets["render"] = rets["rend_alpha"] * rets["render"] + (1. - rets["rend_alpha"]) * torch.rand_like(rets["render"])
                elif self.trainer.config["alpha_bg"] == "GT_grad":
                    rets["render"] = rets["rend_alpha"] * rets["render"] + (1. - rets["rend_alpha"]) * inputs["video_tensor"][:, i]
                else:
                    raise UserWarning("Using Black Rendering Background")
                    
                inputs["rets_dict"][("gs", i)] = rets
                
                        
        return inputs