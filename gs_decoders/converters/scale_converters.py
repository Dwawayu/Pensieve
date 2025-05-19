from utils.GS_utils import map_to_GS
import torch
import torch.nn.functional as F
from utils.config_utils import GlobalState

class ShiftMinMax(object):
    def __init__(self, parent_model, shift=0, min_scale=None, max_scale=None, activation="torch.exp", clamp_or_rescale="clamp"):
        super(ShiftMinMax, self).__init__()
        self.shift = shift
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.activation = eval(activation)
        self.clamp_or_rescale = clamp_or_rescale
        

    def __call__(self, outputs):
        scale = self.activation(outputs["scale_raw"] + self.shift)
        if scale.ndim == 5:
            scale = map_to_GS(scale)
        if self.clamp_or_rescale == "clamp":
            if self.min_scale is not None:
                scale = torch.clamp_min(scale, self.min_scale)
            if self.max_scale is not None:
                scale = torch.clamp_max(scale, self.max_scale)
        elif self.clamp_or_rescale == "rescale":
            scale = self.min_scale + (self.max_scale - self.min_scale) * scale
        return scale

class ExpWithConstantShift(object):
    def __init__(self, parent_model, shift=0, min_scale=None, max_scale=None):
        super(ExpWithConstantShift, self).__init__()
        self.shift = shift
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, outputs):
        scale = torch.exp(outputs["scale_raw"] + self.shift)
        if scale.ndim == 5:
            scale = map_to_GS(scale)
        if self.min_scale is not None:
            scale = torch.clamp_min(scale, self.min_scale)
        if self.max_scale is not None:
            scale = torch.clamp_max(scale, self.max_scale)
        return scale


class SoftplusWithConstantShift:
    def __init__(self, parent_model, shift=-2.5):
        super().__init__()
        self.shift = shift

    def __call__(self, outputs):
        scale = F.softplus(outputs["scale_raw"] + self.shift)
        if scale.ndim == 5:
            scale = map_to_GS(scale)
        return scale


class InitAccordingBins(object):
    def __init__(self, parent_model):
        super(InitAccordingBins, self).__init__()
        N_bins = parent_model.config["N_bins"]
        min_bin = parent_model.convert_to_xyz.min_bin
        max_bin = parent_model.convert_to_xyz.max_bin
        depth_levels = torch.arange(N_bins, device="cuda", dtype=torch.float32)
        disps = (1. / min_bin) * (min_bin / max_bin)**(depth_levels / (N_bins-1)) # N
        bins = 1. / disps # N
        self.bins = bins[None, :, None, None, None] # 1, N, 1, 1, 1

    def __call__(self, outputs):
        H, W = outputs["gs_camera"].height, outputs["gs_camera"].width
        init_scale = torch.cat([self.bins / W, self.bins / H], dim=2) # 1, N, 2, 1, 1
        init_shift = torch.log(init_scale)
        return map_to_GS(torch.exp(outputs["scale_raw"] + init_shift))


class ScaleAccordingDepth:
    def __init__(self, parent_model, activation="torch.abs", **kwargs):
        super().__init__()
        def min_sigmoid_max(x):
            x = torch.sigmoid(x)
            x = kwargs["min_scale"] + (kwargs["max_scale"] - kwargs["min_scale"]) * x
            return x
        self.activation = eval(activation)
        
    def get_reference(self, outputs):
        depth = outputs["predict_depth"][:, outputs["now_idx"]]
        depth = map_to_GS(depth) # B, N, 1
        return depth

    def __call__(self, outputs):
        H, W = outputs["gs_camera"].height, outputs["gs_camera"].width
        depth = self.get_reference(outputs) # B, N, 1
        init_scale = torch.cat([depth / W, depth / H], dim=-1) # B, N, 2
        if GlobalState["dim_mode"].lower() == '3d':
            init_scale = torch.cat([init_scale, init_scale.mean(-1, True)], dim=-1)
        return map_to_GS(self.activation(outputs["scale_raw"])) * init_scale
    

class ScaleAccordingDistance(ScaleAccordingDepth):   
    def get_reference(self, outputs):
        xyz = outputs["gs_list"][-1]["xyz"] # B, N, 3
        norm = xyz.norm(dim=-1, keepdim=True) # B, N, 1
        return norm