import torch
from utils.config_utils import get_instance_from_config

class Empty(torch.nn.Module):

    def __init__(self, ch_feature, **config):
        super().__init__()
        self.config = config
    
    def forward(self, inputs):
        return inputs