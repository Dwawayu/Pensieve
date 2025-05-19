import torch
from omegaconf import OmegaConf
import os

import torch.distributed as dist

class BaseModel(torch.nn.Module):
    def __init__(self, **config):
        super(BaseModel, self).__init__()
        self.config = config

        self.ch_feature = self.config.get("ch_feature", None)
        self._init_model()
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            self._print_info()
        
    def _init_model(self):
        raise NotImplementedError("init_model method is not implemented")
    
    def _print_info(self):
        raise NotImplementedError("Model should print some important information.")
    
    def _preprocess_inputs(self, inputs):
        raise NotImplementedError()
    
    def _encode_features(self, inputs):
        raise NotImplementedError()
    
    def forward(self, inputs):
        '''
        inputs["images"]: [0, 1]
        '''
        self._preprocess_inputs(inputs)
        features, inputs = self._encode_features(inputs)
        
        return features, inputs