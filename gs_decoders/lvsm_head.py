import torch

from utils.config_utils import get_instance_from_config

class LVSMHead(torch.nn.Module):
    def __init__(self, ch_feature, **config):
        super(LVSMHead, self).__init__()
        self.config = config
        assert self.config["lvsm_transformer"]["params"]["in_channels"] == ch_feature + 6
        assert self.config["lvsm_transformer"]["params"]["out_channels"] == 3 + int(self.config.get("predict_weight", False))

        self.lvsm = get_instance_from_config(self.config["lvsm_transformer"])
            
        
    def infer_lvsm(self, features_gs, plucker):
        features = torch.cat([features_gs, plucker.unsqueeze(1)], dim=1)
        features = self.lvsm(features)
        frame = features[:, -1, :3]
        frame = torch.sigmoid(frame) # B, 3, H, W
        return frame, features[:, -1, 3:]
        
    def forward(self, inputs):
        
        src_plucker = []
        for idx in inputs["gs_idx"]:
            camera = inputs["cameras_list"][idx]
            plucker_embedding = camera.plucker_ray
            src_plucker.append(plucker_embedding)
        src_plucker = torch.stack(src_plucker, dim=1) # B, G, 6, H, W
        gs_features_plucker = torch.cat([inputs["gs_features"], src_plucker], dim=2) # B, G, F+6, H, W
        
        zero_gs = torch.zeros_like(inputs["gs_features"][:, 0]) # B, F, H, W
        
        lvsm_prediction = []
        if self.config.get("predict_weight", False):
            lvsm_weight = []
        for l in inputs["tgt_idx"]: # range(len(inputs["cameras_list"])):
            camera = inputs["cameras_list"][l]
            plucker_embedding = camera.plucker_ray # B, 6, H, W
            plucker_embedding = torch.cat([zero_gs, plucker_embedding], dim=1) # B, F+6, H, W
            frame, others = self.infer_lvsm(gs_features_plucker, plucker_embedding)
            lvsm_prediction.append(frame)
            if self.config.get("predict_weight", False):
                lvsm_weight.append(others[:, :1])
        lvsm_prediction = torch.stack(lvsm_prediction, dim=1) # B, L, 3, H, W
        inputs["lvsm_prediction"] = lvsm_prediction
        if self.config.get("predict_weight", False):
            lvsm_weight = torch.stack(lvsm_weight, dim=1)
            inputs["lvsm_weight"] = lvsm_weight # B, L, 1, H, W
        
        return inputs