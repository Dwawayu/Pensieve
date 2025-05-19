from utils.GS_utils import RGB2SH, map_to_GS
import torch

class SigmoidCat:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, ouputs):
        B, N, _, H, W = ouputs["rgb_raw"].shape
        features_dc = RGB2SH(torch.sigmoid(ouputs["rgb_raw"])) # B, N, 3, H, W
        features_dc = map_to_GS(features_dc).unsqueeze(-2) # B, N, 1, 3
        if ouputs["sh_raw"].numel() <= 0.:
            return features_dc

        features_rest = ouputs["sh_raw"] # B, N, 3*F, H, W
        features_rest = map_to_GS(features_rest) # B, N, 3 * F
        return torch.cat([features_dc, features_rest.reshape(B, N*H*W, -1, 3)], dim=-2)

class Cat:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, ouputs):
        features_dc = ouputs["rgb_raw"] # B, N, 3, H, W
        if features_dc.ndim == 5:
            features_dc = map_to_GS(features_dc) # B, N, 3
        features_dc = features_dc.unsqueeze(-2) # B, N, 1, 3
        if "sh_raw" not in ouputs or ouputs["sh_raw"].numel() <= 0.:
            return features_dc

        features_rest = ouputs["sh_raw"]
        if features_rest.ndim == 5:
            features_rest = map_to_GS(features_rest) # B, N, F*3

        return torch.cat([features_dc, features_rest.reshape(*features_rest.shape[:2], -1, 3)], dim=-2)
    

class ResidualCat:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, outputs):
        if outputs["rgb_raw"].ndim == 5:
            rgb = outputs["video_tensor"][:, outputs["now_idx"]] # B, 3, H, W
            if rgb.shape[-2:] != outputs["rgb_raw"].shape[-2:]:
                rgb = torch.nn.functional.interpolate(rgb, size=outputs["rgb_raw"].shape[-2:], mode='bilinear', align_corners=True)
            features_dc = RGB2SH(rgb)
            features_dc = features_dc.unsqueeze(1) # B, 1, 3, H, W
            features_dc = features_dc + outputs["rgb_raw"] # B, N, 3, H, W
            features_dc = map_to_GS(features_dc) # B, N, 3
        else:
            rgb = outputs["video_tensor"][:, outputs["now_idx"]].unsqueeze(1) # B, 1, 3, H, W
            # TODO interpolate
            features_dc = RGB2SH(rgb)
            features_dc = map_to_GS(features_dc)
            features_dc = features_dc + outputs["rgb_raw"] # B, N, 3
        features_dc = features_dc.unsqueeze(-2) # B, N, 1, 3
        if "sh_raw" not in outputs or outputs["sh_raw"].numel() <= 0.:
            return features_dc

        features_rest = outputs["sh_raw"]
        if features_rest.ndim == 5:
            features_rest = map_to_GS(features_rest) # B, N, F*3

        return torch.cat([features_dc, features_rest.reshape(*features_rest.shape[:2], -1, 3)], dim=-2)