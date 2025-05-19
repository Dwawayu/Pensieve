import torch
from torch import nn
import torch.nn.functional as F

class CenterCrop(nn.Module):
    """Support: video_tensor, camera_dict
        In each support, we should check the shape with video_tensor
    """
    def __init__(self, size):
        super().__init__()
        self.h_out, self.w_out = size

    def forward(self, inputs):
        H, W = inputs["video_tensor"].shape[-2:]
        h_s = (H - self.h_out) // 2
        w_s = (W - self.w_out) // 2
        
        # do center crop for all keys with video or image
        for key, value in inputs.items():
            if "video" in key or "image" in key:
                assert value.shape[-2] == H and value.shape[-1] == W and value.shape[-3] == 3
                inputs[key] = value[..., h_s:h_s+self.h_out, w_s:w_s+self.w_out]
        
            elif "camera" in key:
                assert inputs[key]["width"] == W and inputs[key]["height"] == H
                inputs[key]["_cx"] = inputs[key]["_cx"] - w_s
                inputs[key]["_cy"] = inputs[key]["_cy"] - h_s
                inputs[key]["width"] = self.w_out
                inputs[key]["height"] = self.h_out
            
        if "flow_tensor" in inputs:
            assert inputs["flow_tensor"].shape[-1] == W and inputs["flow_tensor"].shape[-2] == H
            inputs["flow_tensor"][..., 0, :, :] *= W / self.w_out
            inputs["flow_tensor"][..., 1, :, :] *= H / self.h_out
            inputs["flow_tensor"] = inputs["flow_tensor"][..., h_s:h_s+self.h_out, w_s:w_s+self.w_out]
        
        return inputs
    
    
class Resize(nn.Module):
    def __init__(self, size):
        """Support: video_tensor, camera_dict, flow_tensor
        each support will be resized to the same size
        """
        super().__init__()
        self.h_out, self.w_out = size

    def forward(self, inputs):
        
        for key, value in inputs.items():
            if "video" in key or "image" in key:
                o_shape = value.shape
                assert o_shape[-3] == 3
                value = value.reshape(-1, *o_shape[-3:])
                value = F.interpolate(value, (self.h_out, self.w_out), mode="bilinear", align_corners=True)
                value = value.reshape(*o_shape[:-2], self.h_out, self.w_out)
                inputs[key] = value
        
            elif "camera" in key:
                o_h, o_w = inputs[key]["height"], inputs[key]["width"]
                ratio_x = self.w_out / o_w
                ratio_y = self.h_out / o_h
                inputs[key]["width"] = self.w_out
                inputs[key]["height"] = self.h_out
                if inputs[key]["_cx"] is not None:
                    inputs[key]["_cx"] *= ratio_x
                if inputs[key]["_cy"] is not None:
                    inputs[key]["_cy"] *= ratio_y
                if inputs[key]["fx"] is not None:
                    inputs[key]["fx"] *= ratio_x
                if inputs[key]["fy"] is not None:
                    inputs[key]["fy"] *= ratio_y
                
        if "flow_tensor" in inputs:
            o_shape = inputs["flow_tensor"].shape
            assert o_shape[-3] == 2
            flow_tensor = inputs["flow_tensor"].reshape(-1, *o_shape[-3:])
            flow_tensor = F.interpolate(flow_tensor, (self.h_out, self.w_out), mode="bilinear", align_corners=True)
            flow_tensor = flow_tensor.reshape(*o_shape[:-2], self.h_out, self.w_out)
            inputs["flow_tensor"] = flow_tensor
                
        return inputs