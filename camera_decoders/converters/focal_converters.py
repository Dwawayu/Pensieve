import torch

class Sigmoid:
    def __init__(self, parent_model, min_value=0.1, max_value=3.):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, inputs):
        rfx = self.min_value + torch.sigmoid(inputs["fx_raw"].squeeze(-1)) * (self.max_value - self.min_value)
        rfy = self.min_value + torch.sigmoid(inputs["fy_raw"].squeeze(-1)) * (self.max_value - self.min_value)
        return rfx, rfy
    
class ConvertFromData:
    def __init__(self, parent_model):
        super().__init__()
    
    def __call__(self, inputs):
        now_idx = inputs["now_idx"]
        return inputs["camera_dict"]["fx"][..., now_idx] / inputs["camera_dict"]["width"], inputs["camera_dict"]["fy"][..., now_idx] / inputs["camera_dict"]["height"]