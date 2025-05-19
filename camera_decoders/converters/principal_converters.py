import torch

class ReturnNone:
    def __init__(self, parent_model):
        super().__init__()

    def __call__(self, inputs):
        return None, None