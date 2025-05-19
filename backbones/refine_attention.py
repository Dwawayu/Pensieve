from backbones.transformers import PerFrameTransformer
from backbones.base_model import BaseModel
from utils.config_utils import GlobalState, get_instance_from_config

import torch

class RefineAttention(BaseModel):
    def _init_model(self):
        self.model_self = get_instance_from_config(self.config["self_transformer"])
        
        self.self2gs = torch.nn.Conv2d(self.model_self.ch_feature+6, self.config["gs_transformer"]["params"]["in_channels"], 1)
        self.self2camera = torch.nn.Conv2d(self.model_self.ch_feature+6, self.config["camera_transformer"]["params"]["in_channels"], 1)
        
        self.gs_cross = get_instance_from_config(self.config["gs_transformer"])
        self.camera_cross = get_instance_from_config(self.config["camera_transformer"])
        assert self.gs_cross.ch_feature == self.camera_cross.ch_feature
        self.ch_feature = self.gs_cross.ch_feature + 5
        
    def _print_info(self):
        print("Using RefineAttention model.")
        
    def normalize_image(self, images):
        return (images - 0.45) / 0.226
        
    def embedding_uv(self, images):
        B, L, _, H, W = images.shape
        width_list = torch.arange(W, device=images.device) / (W - 1.)
        height_list = torch.arange(H, device=images.device) / (H - 1.)
        
        width_list = width_list[None, None, None, None, :].expand(B, L, -1, H, -1)
        height_list = height_list[None, None, None, :, None].expand(B, L, -1, -1, W)
        images = torch.cat([images, height_list, width_list], dim=2)
        
        return images
    
    def embedding_idx(self, images, norm=False):
        B, L, _, H, W = images.shape
        idx_list = torch.arange(L, device=images.device) / (L - 1. + 1e-4)
        if norm:
            idx_list = (idx_list - 0.45) / 0.226

        idx_list = idx_list[None, :, None, None, None].expand(B, -1, -1, H, W)
        images = torch.cat([images, idx_list], dim=2)
        
        return images
        
    def self_encode(self, images):
        
        B, L, C, H, W = images.shape
        # single frame
        images = images.reshape(B*L, 1, C, H, W)
        
        features_self = self.model_self(images) # (B*L, 1, F, H, W)
        features_self = features_self.reshape(B, L, -1, H, W)
        
        return features_self
    
    def cross_encode_gs(self, images):
        B, G, _, H, W = images.shape
        images = self.self2gs(images.reshape(B*G, -1, H, W)).reshape(B, G, -1, H, W)
        images = self.gs_cross(images)
        return images
    
    def cross_encode_camera(self, images):
        B, C, _, H, W = images.shape
        images = self.self2camera(images.reshape(B*C, -1, H, W)).reshape(B, C, -1, H, W)
        images = self.camera_cross(images)
        return images
    
    def forward(self, inputs, gs_idx):
        images = inputs["video_tensor"]
        images_uv = self.embedding_uv(images)
        images_uv = self.normalize_image(images_uv)
        features_self = self.self_encode(images_uv)
        
        features_self = torch.cat([features_self, images_uv], dim=2)
        
        features_camera = self.cross_encode_camera(self.embedding_idx(features_self, True))
        features_camera = torch.cat([features_camera, images_uv], dim=2)
        
        features_gs_0 = features_self[:, gs_idx]
        features_gs_1 = self.cross_encode_gs(self.embedding_idx(features_gs_0, True))
        features_gs_1 = torch.cat([features_gs_1, images_uv[:, gs_idx]], dim=2)
        
        inputs["gs_features"] = features_gs_1
        inputs["camera_features"] = features_camera
        
        return inputs
    
    
class RefineAttentionTriple(RefineAttention):
    def forward(self, inputs, gs_idx):
        gs_idx = [0, 2]
        inputs["gs_idx"] = gs_idx
        images = inputs["video_tensor"]
        images_uv = self.embedding_uv(images)
        images_uv = self.normalize_image(images_uv)
        features_self = self.self_encode(images_uv)
        
        features_self = torch.cat([features_self, images_uv], dim=2)
        
        features_camera_0 = self.cross_encode_camera(self.embedding_idx(features_self[:, 0:2], True))
        features_camera_0 = torch.cat([features_camera_0, images_uv[:, 0:2]], dim=2)
        
        features_camera_1 = self.cross_encode_camera(self.embedding_idx(features_self[:, 1:3], True))
        features_camera_1 = torch.cat([features_camera_1, images_uv[:, 1:3]], dim=2)
        features_camera = torch.cat([features_camera_0, features_camera_1], dim=1)
        
        features_gs_0 = features_self[:, gs_idx]
        features_gs_1 = self.cross_encode_gs(self.embedding_idx(features_gs_0, True))
        features_gs_1 = torch.cat([features_gs_1, images_uv[:, gs_idx]], dim=2)
        
        inputs["gs_features"] = features_gs_1
        inputs["camera_features"] = features_camera
        
        return inputs