import torch
from utils.camera import Camera, BatchCameras
from omegaconf import OmegaConf
import os
from utils.config_utils import get_instance_from_config
from utils.matrix_utils import quaternion_translation_multiply, quaternion_translation_inverse, quaternion_inverse

class BaseCameraDecoder(torch.nn.Module):
    def __init__(self, ch_feature, **config):
        super(BaseCameraDecoder, self).__init__()
        
        self.config = config
        self._init_decoder(ch_feature)
        self._init_converters()
        
        self.mode = self.config.get("mode", "relative")
        
    def _init_decoder(self, ch_feature):
        pass
    
    def _infer_model(self, inputs):
        inputs['rel_quaternion_raw'] = inputs['rel_quaternion_raw_stacked'][:, inputs["now_idx"]]
        inputs['rel_translation_raw'] = inputs['rel_translation_raw_stacked'][:, inputs["now_idx"]]
        return {}, inputs
    
    def _init_converters(self):
        # in some cases, the converter needs to get attributes from its parent, so we pass self to it.
        self.convert_to_quaternion = get_instance_from_config(self.config["convert_to_quaternion"], self)
        self.convert_to_translation = get_instance_from_config(self.config["convert_to_translation"], self)
        self.convert_to_focal = get_instance_from_config(self.config["convert_to_focal"], self)
        self.convert_to_principal = get_instance_from_config(self.config["convert_to_principal"], self)
    
    def _raw_to_camera(self, inputs):        
        relative_quaternion = self.convert_to_quaternion(inputs)
        rel_translation = self.convert_to_translation(inputs)
        rfx, rfy = self.convert_to_focal(inputs)
        cx, cy = self.convert_to_principal(inputs)

        if relative_quaternion.shape[0] == 2 * inputs['video_tensor'].shape[0]:
            B = inputs['video_tensor'].shape[0]
            relative_quaternion, inverse_quaternion = relative_quaternion[:B], relative_quaternion[-B:]
            rel_translation, inverse_translation = rel_translation[:B], rel_translation[-B:]
            rfx = rfx[:B]
            rfy = rfy[:B]
            cx = None # cx[:B]
            cy = None # cy[:B]
            inverse_quaternion_loss = - (relative_quaternion * quaternion_inverse(inverse_quaternion)).sum(-1).abs()
            inverse_translation_loss = (rel_translation - quaternion_translation_inverse(inverse_quaternion, inverse_translation)[1]).square().sum(-1).clamp(1e-6).sqrt()
            if "inverse_quaternion_loss" not in inputs:
                inputs["inverse_quaternion_loss"] = []
            inputs["inverse_quaternion_loss"].append(inverse_quaternion_loss)
            if "inverse_translation_loss" not in inputs:
                inputs["inverse_translation_loss"] = []
            inputs["inverse_translation_loss"].append(inverse_translation_loss)

        batch_camera = BatchCameras()
        if self.mode == "relative":
            last_frame_cam = inputs['cameras_list'][-1]
            last_quaternion = last_frame_cam.quaternion
            last_translation = last_frame_cam.t
            next_quaternion, next_translation = quaternion_translation_multiply(last_quaternion, last_translation, relative_quaternion, rel_translation)
            batch_camera.width = last_frame_cam.width
            batch_camera.height = last_frame_cam.height
            batch_camera.device = last_frame_cam.device
        elif self.mode == "direct":
            next_quaternion, next_translation = relative_quaternion, rel_translation
            batch_camera.width = inputs["video_tensor"].shape[-1]
            batch_camera.height = inputs["video_tensor"].shape[-2]
            batch_camera.device = inputs["video_tensor"].device
        batch_camera.quaternion = next_quaternion.float()
        batch_camera.t = next_translation.float() # + torch.tensor([[0, 0, -1]], device=next_translation.device)
        batch_camera.fx = rfx.float() * batch_camera.width
        batch_camera.fy = rfy.float() * batch_camera.height
        batch_camera.cx = cx
        batch_camera.cy = cy

        
        return batch_camera, inputs
    
    def forward(self, inputs):
        if isinstance(inputs['camera_features'], torch.Tensor) and inputs['camera_features'].ndim == 3 and inputs["now_idx"] == 0:
            inputs['camera_features'] = inputs['camera_features'][..., None, None]
        
        if self.mode == "relative":
        
            if inputs["now_idx"] == inputs["video_tensor"].shape[1] - 1:
                return inputs
            
            if inputs["now_idx"] == 0:
                assert len(inputs["cameras_list"]) == 0
                first_camera = BatchCameras()
                B, L, _, H, W = inputs["video_tensor"].shape
                first_camera.t = torch.zeros([B, 3], device=inputs['video_tensor'].device)
                first_camera._quaternion = torch.zeros([B, 4], device=inputs['video_tensor'].device)
                first_camera._quaternion[..., 0] = 1.
                first_camera.width = W
                first_camera.height = H
                first_camera.device = inputs['video_tensor'].device
                inputs["cameras_list"].append(first_camera)
        
        outputs, inputs = self._infer_model(inputs)
        inputs.update(outputs)
        
        cameras, inputs = self._raw_to_camera(inputs)
        inputs["cameras_list"].append(cameras)
        
        if self.mode == "relative":
            if len(inputs["cameras_list"]) == 2:
                first_camera = inputs['cameras_list'][0]
                first_camera.fx = inputs['cameras_list'][1].fx
                first_camera.fy = inputs['cameras_list'][1].fy
                first_camera._cx = inputs['cameras_list'][1]._cx
                first_camera._cy = inputs['cameras_list'][1]._cy
                inputs['cameras_list'][0] = first_camera
                
        return inputs