from utils.GS_utils import map_to_GS
import torch

class Sigmoid(object):
    def __init__(self, parent_model, shift=0.):
        super(Sigmoid, self).__init__()
        self.shift = shift

    def __call__(self, ouputs):
        opacity = torch.sigmoid(ouputs["opacity_raw"] + self.shift)
        if opacity.ndim == 5:
            opacity = map_to_GS(opacity)
        return opacity

class ConvertFromData:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, ouputs):
        opacity = ouputs.pop("opacity")
        return map_to_GS(opacity) # B, N, 1, H, W -> B, NHW, 1

class SigmoidWithConfShift:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, ouputs):
        return map_to_GS(torch.sigmoid(ouputs["opacity_raw"] + ouputs["conf"][:, ouputs["now_idx"]]))