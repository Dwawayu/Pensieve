from utils.GS_utils import map_to_GS
import torch
from utils.matrix_utils import create_camera_plane

class ExpBinsPixelResidual(object):
    def __init__(self, parent_model, min_bin, max_bin):
        super(ExpBinsPixelResidual, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin

    def __call__(self, outputs):
        B, N, _, H, W = outputs["depth_residual_raw"].shape
        residual_levels = torch.sigmoid(outputs["depth_residual_raw"]) * 2. - 1. # B, N, 1, H, W
        residual_levels = residual_levels * (N-1)
        depth_levels = torch.arange(N, device=outputs["depth_residual_raw"].device).float()
        depth_levels = depth_levels[None, :, None, None, None] + residual_levels
        disps = (1. / self.min_bin) * (self.min_bin / self.max_bin)**(depth_levels / (N-1)) # B, N, 1, H, W
        depths = 1. / disps # B, N, 1, H, W
        pixel_residual = torch.sigmoid(outputs["pixel_residual_raw"]) * 2. - 1. # B, N, 2, H, W
        pixel_residual = pixel_residual * 4.
        camera_planes = create_camera_plane(outputs["gs_camera"], pix_residual=pixel_residual)
        
        xyz = camera_planes * depths # B, N, 3, H, W

        xyz = outputs["gs_camera"].R.unsqueeze(1) @ xyz.reshape(B, N, 3, H*W) + outputs["gs_camera"].t[:, None, :, None]
        xyz = xyz.reshape(B, N, 3, H, W)
        return map_to_GS(xyz)

class ScaledExpBinsPixelResidual(ExpBinsPixelResidual):
    def __call__(self, outputs):
        B, N, _, H, W = outputs["depth_residual_raw"].shape
        residual_levels = torch.sigmoid(outputs["depth_residual_raw"]) * 2. - 1. # B, N, 1, H, W
        residual_levels = residual_levels * (N-1)
        depth_levels = torch.arange(N, device=outputs["depth_residual_raw"].device).float()
        depth_levels = depth_levels[None, :, None, None, None] + residual_levels
        disps = (1. / self.min_bin) * (self.min_bin / self.max_bin)**(depth_levels / (N-1)) # B, N, 1, H, W
        depths = 1. / disps # B, N, 1, H, W
        pixel_residual = torch.sigmoid(outputs["pixel_residual_raw"]) * 2. - 1. # B, N, 2, H, W
        pixel_residual = pixel_residual * 4.
        camera_planes = create_camera_plane(outputs["gs_camera"], pix_residual=pixel_residual)
        
        xyz = camera_planes * depths # B, N, 3, H, W

        xyz = xyz * outputs["scale"] # scaled here

        xyz = outputs["gs_camera"].R.unsqueeze(1) @ xyz.reshape(B, N, 3, H*W) + outputs["gs_camera"].t[:, None, :, None]
        xyz = xyz.reshape(B, N, 3, H, W)
        return map_to_GS(xyz)

class BackProjection:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, outputs):
        camera_planes = create_camera_plane(outputs["gs_camera"]).unsqueeze(1) # B, 1, 3, H, W
        depths = outputs["predict_depth"][:, outputs["now_idx"]] # B, N, 1, H, W
        
        xyz = camera_planes * depths # B, N, 3, H, W
        B, N, _, H, W = xyz.shape

        xyz = outputs["gs_camera"].R.unsqueeze(1) @ xyz.reshape(B, N, 3, H*W) + outputs["gs_camera"].t[:, None, :, None]
        xyz = xyz.reshape(B, N, 3, H, W)
        return map_to_GS(xyz)
    

class SigmoidDepth:
    def __init__(self, parent_model, min_depth=0., max_depth=500., inv=False):
        super().__init__()
        if inv:
            self.min_depth = 1. / max_depth
            self.max_depth = 1. / min_depth
        else:
            self.min_depth = min_depth
            self.max_depth = max_depth
        self.inv = inv

    def __call__(self, outputs):
        B, N, _, H, W = outputs["depth_residual_raw"].shape
        depths = self.min_depth + torch.sigmoid(outputs["depth_residual_raw"]) * (self.max_depth - self.min_depth) # B, N, 1, H, W
        if self.inv:
            depths = 1. / depths
        # if depths.shape[-2:] != outputs["video_tensor"].shape[-2:]:
        #     predict_depth = torch.nn.functional.interpolate(depths.squeeze(2), size=outputs["video_tensor"].shape[-2:], mode='bilinear', align_corners=True) # B, N, H, W
        if outputs["now_idx"] == 0:
            outputs["predict_depth"] = depths.unsqueeze(1)
        else:
            outputs["predict_depth"] = torch.cat([outputs["predict_depth"], depths.unsqueeze(1)], dim=1)
        if "pixel_residual_raw" in outputs:
            pixel_residual = torch.sigmoid(outputs["pixel_residual_raw"]) * 2. - 1. # B, N, 2, H, W
            pixel_residual = pixel_residual * 8.
            camera_planes = create_camera_plane(outputs["gs_camera"], pix_residual=pixel_residual)
        else:
            camera_planes = create_camera_plane(outputs["gs_camera"]).unsqueeze(1)
        
        xyz = camera_planes * depths # B, N, 3, H, W

        xyz = outputs["gs_camera"].R.unsqueeze(1) @ xyz.reshape(B, N, 3, H*W) + outputs["gs_camera"].t[:, None, :, None]
        xyz = xyz.reshape(B, N, 3, H, W)
        return map_to_GS(xyz)


class ExpDepth(object):
    def __init__(self, parent_model, min_depth=0.):
        super().__init__()
        self.min_depth = min_depth

    def __call__(self, outputs):
        B, N, _, H, W = outputs["depth_residual_raw"].shape
        depths = torch.expm1(outputs["depth_residual_raw"].abs()) + self.min_depth # B, N, 1, H, W
        if outputs["now_idx"] == 0:
            outputs["predict_depth"] = depths.unsqueeze(1)
        else:
            outputs["predict_depth"] = torch.cat([outputs["predict_depth"], depths.unsqueeze(1)], dim=1)
        if "pixel_residual_raw" in outputs:
            pixel_residual = torch.sigmoid(outputs["pixel_residual_raw"]) * 2. - 1. # B, N, 2, H, W
            pixel_residual = pixel_residual * 4.
            camera_planes = create_camera_plane(outputs["gs_camera"], pix_residual=pixel_residual)
        else:
            camera_planes = create_camera_plane(outputs["gs_camera"]).unsqueeze(1)
        
        xyz = camera_planes * depths # B, N, 3, H, W

        xyz = outputs["gs_camera"].R.unsqueeze(1) @ xyz.reshape(B, N, 3, H*W) + outputs["gs_camera"].t[:, None, :, None]
        xyz = xyz.reshape(B, N, 3, H, W)
        return map_to_GS(xyz)
    

class Expm1W:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, outputs):
        xyz_raw = outputs["xyz_raw"]
        norm = torch.norm(xyz_raw, dim=2, keepdim=True)
        xyz_raw = xyz_raw / (norm + 1e-4)
        xyz = xyz_raw * torch.expm1(norm)
        
        if xyz.ndim == 5:
            xyz = map_to_GS(xyz)
        
        return xyz
    
class IdentityW:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, outputs):
        xyz = outputs["xyz_raw"]
        
        if xyz.ndim == 5:
            xyz = map_to_GS(xyz)
        
        return xyz