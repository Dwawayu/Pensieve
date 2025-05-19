from camera_decoders.base_camera_decoder import BaseCameraDecoder
import torch

import torch.nn as nn
class LinearDecoder(BaseCameraDecoder):
    def _init_decoder(self, ch_feature):
        self.out_names = []
        self.out_dims = []
        self.out_weights = []
        for name, (dim, weight) in self._get_raw_params_dict().items():
            self.out_names.append(name)
            self.out_dims.append(dim)
            self.out_weights.append(weight)
            
        self.map_layer = nn.Conv2d(ch_feature, self.config["feature_dim"], 1, 1, 0, bias=self.config["bias"])
        
        self.layers = []
        for i in range(self.config["num_layers"]):
            self.layers.append(torch.nn.Linear(self.config["feature_dim"], self.config["feature_dim"], bias=self.config["bias"]))
            self.layers.append(torch.nn.LayerNorm(self.config["feature_dim"], bias=self.config["bias"]))
            if i < self.config["num_layers"] - 1:
                self.layers.append(torch.nn.GELU())
        self.layers.append(torch.nn.Linear(self.config["feature_dim"], sum(self.out_dims), bias=self.config["bias"]))
        self.layers = torch.nn.Sequential(*self.layers)
        
        
        torch.nn.init.constant_(self.layers[-1].weight, 0)
        if self.config["bias"]:
            torch.nn.init.constant_(self.layers[-1].bias, 0)
            self.layers[-1].bias.data[0] = 1.
        else:
            self.layers[-1].weight.data[0] = 1.

    def _infer_model(self, inputs):
        
        camera_features = self.map_layer(inputs['camera_features'][:, inputs["now_idx"]])
        camera_features = camera_features.mean([-1, -2])
        camera_raw = self.layers(camera_features)

        # outputs = dict(zip(self.out_names, torch.split(camera_raw, self.out_dims, dim=-1)))
        outputs = torch.split(camera_raw, self.out_dims, dim=-1)
        outputs_dict = {}
        for i in range(len(self.out_names)):
            outputs_dict[self.out_names[i]] = outputs[i] * self.out_weights[i] # + (outputs[i] * (1. - self.out_weights[i])).detach()
        outputs_dict["camera_raw"] = camera_raw
        return outputs_dict, inputs
    
    def _get_raw_params_dict(self):
        raw_params_dict = {
            'rel_quaternion_raw': (4, 1.),
            'rel_translation_raw': (3, 1.),
            'fx_raw': (1, 0.1),
            'fy_raw': (1, 0.1),
            'cx_raw': (1, 0.1),
            'cy_raw': (1, 0.1),
        }
        return raw_params_dict