import torch
from torch import nn
from utils.config_utils import get_instance_from_config

class BaseGSDecoder(torch.nn.Module):

    def __init__(self, ch_feature, **config):
        super(BaseGSDecoder, self).__init__()
        
        self.config = config
        
        if self.config.get("cat_image", False):
            ch_feature += 3

        self._init_decoder(ch_feature)
        self._init_converters()
        
        if self.config.get("use_bilgrid", False):
            self.bilgrid_net = []
            for i in range(self.config['bilgrid_downsample_2']):
                self.bilgrid_net.append(nn.Conv2d(ch_feature, ch_feature, kernel_size=3, stride=1, padding=1))
                # self.bilgrid_net.append(nn.BatchNorm2d(ch_feature))
                self.bilgrid_net.append(nn.InstanceNorm2d(ch_feature))
                self.bilgrid_net.append(nn.ELU())
                self.bilgrid_net.append(nn.AvgPool2d(2, 2, 0))
            self.bilgrid_net.append(nn.Conv2d(ch_feature, 12 * self.config['bilgrid_depth'], kernel_size=1, stride=1, padding=0))
            self.bilgrid_net[-1].weight.data = torch.zeros_like(self.bilgrid_net[-1].weight.data)
            
            bias = torch.tensor([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.])
            bias = bias.unsqueeze(1).repeat(1, self.config['bilgrid_depth']).flatten()
            self.bilgrid_net[-1].bias.data = bias
            self.bilgrid_net = nn.Sequential(*self.bilgrid_net)
        
    def _init_decoder(self, ch_feature):
        pass
    
    def _infer_model(self, inputs):
        return {}, inputs
    
    def _init_converters(self):
        # in some cases, the converter needs to get attributes from its parent, so we pass self to it.
        self.convert_to_opacity = get_instance_from_config(self.config["convert_to_opacity"], self)
        self.convert_to_features = get_instance_from_config(self.config["convert_to_features"], self)
        self.convert_to_xyz = get_instance_from_config(self.config["convert_to_xyz"], self)
        self.convert_to_rotation = get_instance_from_config(self.config["convert_to_rotation"], self)
        self.convert_to_scale = get_instance_from_config(self.config["convert_to_scale"], self)

    def _preprocess_inputs(self, inputs):
        if isinstance(inputs["gs_features"], torch.Tensor):
            if inputs["gs_features"].ndim == 4:
                # transformer style: B, L, PP, C
                inputs["gs_features"] = inputs["gs_features"].permute(0, 1, 3, 2) # B, L, C, PP
                input_H, input_W = inputs["video_tensor"].shape[-2], inputs["video_tensor"].shape[-1]
                n_pixel = input_H * input_W
                scale = (inputs["gs_features"].shape[-1] / n_pixel)**0.5
                inputs["gs_features"] = inputs["gs_features"].reshape(*inputs["gs_features"].shape[:-1], int(input_H*scale), int(input_W*scale))
            assert inputs["gs_features"].ndim == 5
            if self.config.get("cat_image", False):
                image = inputs["video_tensor"]
                image = image * 2. - 1.
                inputs["gs_features"] = torch.cat([inputs["gs_features"], image], dim=-3)
        return inputs
    
    def forward(self, inputs):
        if inputs["now_idx"] == 0:
            inputs = self._preprocess_inputs(inputs)
        outputs, inputs = self._infer_model(inputs)
        
        inputs["gs_camera"] = inputs["cameras_list"][inputs["now_idx"]]
        if outputs == {}:
            first_value = inputs["opacity_raw"]
        else:
            first_value = next(iter(outputs.values()))
        if first_value.shape[-2] != inputs["gs_camera"].height or first_value.shape[-1] != inputs["gs_camera"].width:
            inputs["gs_camera"] = inputs["gs_camera"].resize(first_value.shape[-1], first_value.shape[-2])
        
        if self.config.get("use_bilgrid", False):
            bilgrid = self.bilgrid_net(inputs["gs_features"][:, inputs["now_idx"]])
            inputs["cameras_list"][inputs["now_idx"]].bilgrid = bilgrid.reshape(bilgrid.shape[0], 12, self.config['bilgrid_depth'], bilgrid.shape[2], bilgrid.shape[3])
        
        inputs.update(outputs)
        GS_params = {}
        inputs["gs_list"].append(GS_params)
        GS_params["sh_degree"] = self.config["sh_degree"]
        GS_params["opacity"] = self.convert_to_opacity(inputs)
        GS_params["features"] = self.convert_to_features(inputs)
        GS_params["xyz"] = self.convert_to_xyz(inputs)
        GS_params["rotation"] = self.convert_to_rotation(inputs)
        GS_params["scale"] = self.convert_to_scale(inputs)

        return inputs