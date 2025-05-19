import torch

class Normalization:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, inputs):
        relative_quaternion = inputs["rel_quaternion_raw"]
        relative_quaternion = relative_quaternion / (relative_quaternion.norm(dim=-1, keepdim=True) + 1e-5)
        return relative_quaternion