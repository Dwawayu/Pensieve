from gs_decoders.base_gs_decoder import BaseGSDecoder
import torch

from utils.config_utils import GlobalState

class RegAttributes(BaseGSDecoder):
    '''
    Keys required in inputs:
        gs_features
        
    '''
    def _init_decoder(self, ch_feature):
        disabled_attributes = self.config["disabled_attributes"]
        self.out_names = []
        self.out_dims = []
        self.out_weights = []
        for name, (dim, weight) in self._get_raw_params_dict().items():
            if name not in disabled_attributes:
                self.out_names.append(name)
                if name == 'sh_raw':
                    dim = ((self.config["sh_degree"] + 1)**2 - 1) * 3
                self.out_dims.append(dim)
                self.out_weights.append(weight)
        print("Predict attributes: ", self.out_names)
        print("dims: ", self.out_dims)
        print("weights: ", self.out_weights)
        print("ch_feature: ", ch_feature)
        self.convs = []
        for i in range(self.config["num_layers"]):
            if i == 0:
                conv = torch.nn.Conv2d(ch_feature, self.config["feature_dim"], 3, 1, 1, bias=self.config["bias"])
            else:
                conv = torch.nn.Conv2d(self.config["feature_dim"], self.config["feature_dim"], 3, 1, 1, bias=self.config["bias"])
            self.convs.append(conv)
            # self.convs.append(torch.nn.BatchNorm2d(self.config["feature_dim"]))
            self.convs.append(torch.nn.InstanceNorm2d(self.config["feature_dim"]))
            if i < self.config["num_layers"] - 1:
                self.convs.append(torch.nn.GELU())
            if i < self.config["downsample_2"]:
                self.convs.append(torch.nn.AvgPool2d(2, 2, 0))
                # self.convs.append(torch.nn.Identity())
        
        if self.config["num_layers"] > 0:
            self.convs.append(torch.nn.Conv2d(self.config["feature_dim"], self.config["N_bins"] * sum(self.out_dims), 1, 1, 0, bias=self.config["bias"]))
        else:
            self.convs.append(torch.nn.Conv2d(ch_feature, self.config["N_bins"] * sum(self.out_dims), 1, 1, 0, bias=self.config["bias"]))

        self.convs = torch.nn.Sequential(*self.convs)
        
    def _infer_model(self, inputs):
        outputs = self.convs(inputs["gs_features"][:, inputs["now_idx"]])
        outputs = outputs.reshape(outputs.shape[0], self.config["N_bins"], -1, outputs.shape[2], outputs.shape[3])
        outputs = torch.split(outputs, self.out_dims, dim=2)
        outputs_dict = {}
        for i in range(len(self.out_names)):
            outputs_dict[self.out_names[i]] = outputs[i] * self.out_weights[i] # + (outputs[i] * (1. - self.out_weights[i])).detach()
        return outputs_dict, inputs
    
    def _get_raw_params_dict(self):
        raw_attributes_dict = {
            'depth_residual_raw': (1, 1.),
            'pixel_residual_raw': (2, 1.),
            'rotation_raw': (4, 1.),
            'opacity_raw': (1, 1.),
            'rgb_raw': (3, 1.),
            'sh_raw': (None, 1. / 20.),
            'xyz_raw': (3, 1.),
        }
        if GlobalState["dim_mode"].lower() == '2d':
            raw_attributes_dict["scale_raw"] = (2, 1.)
        elif GlobalState["dim_mode"].lower() == '3d':
            raw_attributes_dict["scale_raw"] = (3, 1.)
        else:
            raise ValueError("Invalid dim_mode")
        return raw_attributes_dict