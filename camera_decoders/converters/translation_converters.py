import torch

class Identity:
    def __init__(self, parent_model, scale=None):
        super().__init__()
        self.scale = scale

    def __call__(self, inputs):
        if self.scale is not None:
            return inputs["rel_translation_raw"] * self.scale
        return inputs["rel_translation_raw"]
    
class Shift:
    def __init__(self, parent_model, shift):
        super().__init__()
        self.shift = torch.tensor(shift, device="cuda").unsqueeze(0)
        

    def __call__(self, inputs):
        return inputs["rel_translation_raw"] + self.shift