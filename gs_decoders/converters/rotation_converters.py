from utils.GS_utils import map_to_GS
from utils.matrix_utils import quaternion_multiply
import torch

class ResidualOnInit:
    def __init__(self, parent_model, init_rotation):
        super(ResidualOnInit, self).__init__()
        init_rotation_list = []
        for (N, init_rotation) in init_rotation:
            init_rotation_list = init_rotation_list + [init_rotation] * N
        init_rotation_list = torch.tensor(init_rotation_list, device="cuda")
        self.init_rotation = init_rotation_list[None, :, :, None, None] # 1, N, 4, 1, 1

    def __call__(self, inputs):
        rotation = self.init_rotation + inputs["rotation_raw"]
        # rotation = torch.nn.functional.normalize(rotation, dim=2) # B, N, 4, H, W
        rotation = rotation / (rotation.norm(dim=2, keepdim=True) + 1e-5)
        B, N, _, H, W = rotation.shape
        rotation = rotation.permute(0, 1, 3, 4, 2).reshape(B, -1, 4) # B, N*H*W, 4
        rotation = quaternion_multiply(inputs["gs_camera"].quaternion.unsqueeze(1).expand(-1, N*H*W, -1), rotation)
        return rotation


class Normalization:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, inputs):
        rotation = inputs["rotation_raw"] / (inputs["rotation_raw"].norm(dim=2, keepdim=True) + 1e-5)
        B, N, _, H, W = rotation.shape
        rotation = rotation.permute(0, 1, 3, 4, 2).reshape(B, -1, 4) # B, N*H*W, 4
        rotation = quaternion_multiply(inputs["gs_camera"].quaternion.unsqueeze(1).expand(-1, N*H*W, -1), rotation)
        return rotation
    

class NormalizationW:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, inputs):
        rotation = inputs["rotation_raw"] / (inputs["rotation_raw"].norm(dim=2, keepdim=True) + 1e-5)
        if rotation.ndim == 5:
            rotation = map_to_GS(rotation)
        return rotation