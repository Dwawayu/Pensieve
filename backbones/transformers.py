import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.base_model import BaseModel
from backbones.dpt import DPT
from utils.config_utils import get_instance_from_config

class ImageTokenizer(nn.Module):
    def __init__(self, patch_size=8, in_channels=5, embed_dim=1024, bias=False):
        super(ImageTokenizer, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear_proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias=bias)
        self.ln = nn.LayerNorm(embed_dim, bias=bias)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        B, C, H, W = x.shape

        # Ensure image dimensions are divisible by patch size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "Image dimensions must be divisible by the patch size."

        # Extract patches and flatten
        x = self.unfold(x)  # Shape: (batch_size, C*patch_size*patch_size, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, C*patch_size*patch_size)

        # Linear projection and layer normalization
        x = self.linear_proj(x)
        x = self.ln(x)

        return x  # Shape: (batch_size, num_patches, embed_dim)


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0., RMSNorm=False, bias=False):
        super(CustomMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout_p

        # Define the projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        if RMSNorm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, q, k, v):
        B, N, C = q.shape

        # Compute Q, K, V
        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)

        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.q_norm is not None:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        # Compute scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.dropout_p if self.training else 0.0, is_causal=False)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)

        # Final linear projection
        output = self.out_proj(attn_output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16, mlp_hidden_dim=4096, dropout_p=0., RMSNorm=False, bias=False):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim, bias=bias)
        # self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn = CustomMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout_p=dropout_p, RMSNorm=RMSNorm, bias=bias)
        self.ln2 = nn.LayerNorm(embed_dim, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim, bias=bias)
        )

    def forward(self, x):
        # Pre-layer normalization and self-attention
        x_norm = self.ln1(x)
        attn_output = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output  # Residual connection

        # Pre-layer normalization and MLP
        x_norm = self.ln2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output  # Residual connection

        return x  # Shape: (batch_size, sequence_length, embed_dim)


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=24, embed_dim=1024, num_heads=16, mlp_hidden_dim=4096, 
                 dropout_p=0., RMSNorm=False, bias=False, hooks=[]):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, dropout_p=dropout_p if i > 0 else 0., RMSNorm=RMSNorm, bias=bias)
            for i in range(num_layers)
        ])
        
        self.hooks = hooks
        if self.hooks:
            print("Using hooks at layers:", hooks)

    def forward(self, x):
        # Pass through all transformer layers
        if self.hooks:
            features = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i in self.hooks:
                features.append(x)
        if self.hooks:
            features.append(x)
            x = features
        return x  # Shape: (batch_size, sequence_length, embed_dim)


class Unpatchify(nn.Module):
    def __init__(self, patch_size=8, out_channels=32, embed_dim=1024, bias=False):
        super(Unpatchify, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.ln = nn.LayerNorm(embed_dim, bias=bias)
        self.linear_proj = nn.Linear(embed_dim, out_channels * patch_size * patch_size, bias=bias)

    def forward(self, x, output_size):
        # Layer normalization and linear projection
        x = self.ln(x)
        x = self.linear_proj(x)  # Shape: (batch_size, num_patches, out_channels * patch_size * patch_size)

        # Prepare for folding back into image
        x = x.transpose(1, 2)  # Shape: (batch_size, out_channels * patch_size * patch_size, num_patches)

        # Reconstruct the image
        x = F.fold(x, output_size=output_size, kernel_size=self.patch_size, stride=self.patch_size)

        return x  # Shape: (batch_size, out_channels, img_size, img_size)


class VisionTransformer(nn.Module):
    def __init__(self, patch_size=8, in_channels=6, out_channels=32,
                 num_layers=24, embed_dim=1024, num_heads=16, mlp_hidden_dim=4096, dropout_p=0., RMSNorm=False, bias=False, hooks=[], hook_fusion="DPT"):
        super(VisionTransformer, self).__init__()
        self.ch_feature = out_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.hook_fusion = hook_fusion
        self.tokenizer = ImageTokenizer(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim, bias=bias)
        self.transformer = TransformerEncoder(num_layers=num_layers, embed_dim=embed_dim,
                                              num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
                                              dropout_p=dropout_p, RMSNorm=RMSNorm, bias=bias, hooks=hooks)
        if not hooks:
            self.unpatchify = Unpatchify(patch_size=patch_size, out_channels=out_channels, embed_dim=embed_dim, bias=bias)
        else:
            if self.hook_fusion == "MLP":
                self.hook_linear = nn.Linear(embed_dim*(len(hooks)+1), embed_dim)
                self.unpatchify = Unpatchify(patch_size=patch_size, out_channels=out_channels, embed_dim=embed_dim, bias=bias)
            elif self.hook_fusion == "DPT":
                self.unpatchify = DPT(patch_size=patch_size, out_channels=out_channels, embed_dim=embed_dim, bias=bias)
            else:
                raise NotImplementedError

    def forward(self, x, RNN_features=None):
        
        B, L, _, H, W = x.shape
        x = x.reshape(B*L, _, H, W)
        
        # Tokenize image into patches
        x = self.tokenizer(x)  # Shape: (batch_size, num_patches, embed_dim)
        
        BL, num_patches, embed_dim = x.shape
        x = x.reshape(B, L*num_patches, embed_dim)
        
        if RNN_features is not None:
            x = x + RNN_features

        # Pass through the transformer encoder
        x = self.transformer(x)  # Shape: (batch_size, num_patches, embed_dim)
        
        if isinstance(x, list):
            if self.hook_fusion == "DPT":
                x = [feat.reshape(B*L, num_patches, embed_dim) for feat in x]
            if self.hook_fusion == "MLP":
                x = torch.cat(x, dim=-1)
                x = self.hook_linear(x)
                x = x.reshape(B*L, num_patches, embed_dim)
        else:
            x = x.reshape(B*L, num_patches, embed_dim)
        # Reconstruct the image from patches
        x = self.unpatchify(x, output_size=(H, W))  # Shape: (batch_size, out_channels, img_size, img_size)
        x = x.reshape(B, L, -1, H, W)
        
        return x


class TransformerCrossBlock(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16, mlp_hidden_dim=4096, dropout_p=0.):
        super().__init__()
        self.ln1_q = nn.LayerNorm(embed_dim)
        self.ln1_kv = nn.LayerNorm(embed_dim)
        # self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn = CustomMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout_p=dropout_p)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, q, kv):
        # Pre-layer normalization and cross-attention
        q_norm = self.ln1_q(q)
        kv_norm = self.ln1_kv(kv)
        
        attn_output = self.attn(q_norm, kv_norm, kv_norm)
        q = q + attn_output  # Residual connection

        # Pre-layer normalization and MLP
        q_norm = self.ln2(q)
        mlp_output = self.mlp(q_norm)
        q = q + mlp_output  # Residual connection

        return q  # Shape: (batch_size, sequence_length, embed_dim)


class TransformerCrossEncoder(nn.Module):
    def __init__(self, num_layers=2, embed_dim=1024, num_heads=16, mlp_hidden_dim=4096, dropout_p=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerCrossBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, dropout_p=dropout_p if i > 0 else 0.)
            for i in range(num_layers)
        ])

    def forward(self, q, kv):
        # Pass through all transformer layers
        for layer in self.layers:
            q = layer(q, kv)
        return q  # Shape: (batch_size, sequence_length, embed_dim)


class AllFrameTransformer(BaseModel):
    def _init_model(self):
        self.model = get_instance_from_config(self.config["vision_transformer"])
        self.ch_feature = self.model.ch_feature
        self.patch_size = self.model.patch_size
        
        self.apply(self._init_weights)
        
        if self.config.get("use_RNN", False):
            self.RNN_begin = nn.Linear(self.ch_feature*self.patch_size*self.patch_size, self.model.embed_dim)
            torch.nn.init.constant_(self.RNN_begin.weight, 0)
            torch.nn.init.constant_(self.RNN_begin.bias, 0)
            self.RNN_end = nn.Conv2d(self.ch_feature, self.ch_feature, 3, 1, 1)
            torch.nn.init.constant_(self.RNN_end.weight, 0)
            torch.nn.init.constant_(self.RNN_end.bias, 0)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def _print_info(self):
        print(self.config["vision_transformer"])
        
    def _preprocess_inputs(self, inputs):
        inputs["images"] = inputs["video_tensor"]
        B, L, _, H, W = inputs["images"].shape
        width_list = torch.arange(W, device=inputs["images"].device) / (W - 1.)
        height_list = torch.arange(H, device=inputs["images"].device) / (H - 1.)
        idx_list = torch.arange(L, device=inputs["images"].device) / (L - 1.)
        meshgrid = torch.meshgrid(width_list, height_list, indexing='xy')
        pix_coords = torch.stack(meshgrid, axis=0) # 2, H, W
        pix_coords = pix_coords[None, None].expand(B, L, -1, -1, -1)
        idx_list = idx_list[None, :, None, None, None].expand(B, -1, -1, H, W)
        inputs["images"] = torch.cat([inputs["images"], pix_coords, idx_list], dim=2)
        inputs["images"] = inputs["images"] * 2. - 1.
        return inputs
    
    def _encode_features(self, inputs):
        B, L, _, H, W = inputs["images"].shape
        RNN_features = None
        if self.config.get("use_RNN", False):
            if "backbone_RNN_features" not in inputs:
                inputs["backbone_RNN_features"] = torch.zeros(B, L, self.ch_feature, H, W, device=inputs["images"].device)
            RNN_features = inputs["backbone_RNN_features"]
            RNN_features = F.unfold(RNN_features.reshape(B*L, self.ch_feature, H, W), self.patch_size, stride=self.patch_size).transpose(-1, -2)
            RNN_features = self.RNN_begin(RNN_features).reshape(B, -1, self.model.embed_dim)
        inputs["transformer_features"] = self.model(inputs["images"], RNN_features)
        if self.config.get("use_RNN", False):
            RNN_features = self.RNN_end(inputs["backbone_RNN_features"].reshape(B*L, self.ch_feature, H, W)).reshape(B, L, self.ch_feature, H, W)
            inputs["transformer_features"] = inputs["transformer_features"] + RNN_features
            inputs["backbone_RNN_features"] = inputs["transformer_features"]
        return inputs["transformer_features"], inputs


class PerFrameTransformer(BaseModel):
    def _init_model(self):
        self.model = get_instance_from_config(self.config["vision_transformer"])
        self.ch_feature = self.model.ch_feature
        self.patch_size = self.model.patch_size
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)    
    
    def _print_info(self):
        print(self.config["vision_transformer"])
        
    def _preprocess_inputs(self, inputs):
        inputs["images"] = inputs["video_tensor"]
        B, L, _, H, W = inputs["images"].shape
        width_list = torch.arange(W, device=inputs["images"].device) / (W - 1.)
        height_list = torch.arange(H, device=inputs["images"].device) / (H - 1.)
        meshgrid = torch.meshgrid(width_list, height_list, indexing='xy')
        pix_coords = torch.stack(meshgrid, axis=0) # 2, H, W
        pix_coords = pix_coords[None, None].expand(B, L, -1, -1, -1)
        inputs["images"] = torch.cat([inputs["images"], pix_coords], dim=2)
        inputs["images"] = inputs["images"] * 2. - 1.

        return inputs
    
    def _encode_features(self, inputs):
        B, L, _, H, W = inputs["images"].shape
        inputs["transformer_features"] = self.model(inputs["images"].reshape(B*L, 1, _, H, W)).reshape(B, L, -1, H, W)
        return inputs["transformer_features"], inputs


class TwoFrameTransformer(AllFrameTransformer):
    def _preprocess_inputs(self, inputs):
        if inputs["now_idx"] == 0:
            inputs["images"] = torch.stack([inputs["video_tensor"][:, 0], inputs["video_tensor"][:, -1]], dim=1)
            B, L, _, H, W = inputs["images"].shape
            width_list = torch.arange(W, device=inputs["images"].device) / (W - 1.)
            height_list = torch.arange(H, device=inputs["images"].device) / (H - 1.)
            idx_list = torch.arange(L, device=inputs["images"].device) / (L - 1.)
            meshgrid = torch.meshgrid(width_list, height_list, indexing='xy')
            pix_coords = torch.stack(meshgrid, axis=0) # 2, H, W
            pix_coords = pix_coords[None, None].expand(B, L, -1, -1, -1)
            idx_list = idx_list[None, :, None, None, None].expand(B, -1, -1, H, W)
            inputs["images"] = torch.cat([inputs["images"], pix_coords, idx_list], dim=2)
            inputs["images"] = inputs["images"] * 2. - 1.
            RNN_features = None
            if self.config.get("use_RNN", False):
                if "backbone_RNN_features" not in inputs:
                    inputs["backbone_RNN_features"] = torch.zeros(B, L, self.ch_feature, H, W, device=inputs["images"].device)
                RNN_features = inputs["backbone_RNN_features"]
                RNN_features = F.unfold(RNN_features.reshape(B*L, self.ch_feature, H, W), self.patch_size, stride=self.patch_size).transpose(-1, -2)
                RNN_features = self.RNN_begin(RNN_features).reshape(B, -1, self.model.embed_dim)
            inputs["transformer_features"] = self.model(inputs["images"], RNN_features)
            if self.config.get("use_RNN", False):
                RNN_features = self.RNN_end(inputs["backbone_RNN_features"].reshape(B*L, self.ch_feature, H, W)).reshape(B, L, self.ch_feature, H, W)
                inputs["transformer_features"] = inputs["transformer_features"] + RNN_features
                inputs["backbone_RNN_features"] = inputs["transformer_features"]
        return inputs
    
    def _encode_features(self, inputs):
        if inputs["now_idx"] == 0:
            return inputs["transformer_features"][:, inputs["now_idx"]], inputs
        elif inputs["now_idx"] == inputs["video_tensor"].shape[1] - 1:
            return inputs["transformer_features"][:, -1], inputs
        return torch.zeros_like(inputs["transformer_features"][:, 0]), inputs

class PointQueryTransformer(BaseModel):
    def _init_model(self):
        downsample_2 = self.config["downsample_2"]
        self.sh_degree = self.config["sh_degree"]
        
        self.tokenizer = get_instance_from_config(self.config["tokenizer"])
        self.transformer = get_instance_from_config(self.config["transformer"])

        embed_dim = self.tokenizer.embed_dim
        patch_size = self.tokenizer.patch_size
        
        self.point_linear = nn.Linear(embed_dim, 3 * patch_size * patch_size // (2**downsample_2))
        self.point_query = nn.Linear(3, embed_dim)
        
        self.cross_transformer = get_instance_from_config(self.config["cross_transformer"])
        self.attribute_linear = nn.Linear(embed_dim, 10 + ((self.sh_degree + 1)**2 - 1) * 3)
        
        self.camera_proj = nn.Linear(embed_dim, embed_dim)
        self.camera_linear = nn.Linear(embed_dim, 11)
        torch.nn.init.constant_(self.camera_linear.weight, 0)
        torch.nn.init.constant_(self.camera_linear.bias, 0)
        self.camera_linear.bias.data[0] = 1.
        
        self.unpatchify = get_instance_from_config(self.config["unpatchify"])
        self.ch_feature = self.unpatchify.out_channels
        
        self.gs_decoder = get_instance_from_config(self.config["gs_decoder"], self.ch_feature)
        self.camera_decoder = get_instance_from_config(self.config["camera_decoder"], self.ch_feature)

    def _print_info(self):
        print(self.config)

    def _preprocess_inputs(self, inputs):
        if inputs["now_idx"] == 0:
            inputs["images"] = inputs["video_tensor"] * 2. - 1.
            width_list = torch.arange(inputs["images"].shape[-1], device=inputs["images"].device) / (inputs["images"].shape[-1] - 1.)
            height_list = torch.arange(inputs["images"].shape[-2], device=inputs["images"].device) / (inputs["images"].shape[-2] - 1.)
            meshgrid = torch.meshgrid(width_list, height_list, indexing='xy')
            pix_coords = torch.stack(meshgrid, axis=0) # 2, H, W
            pix_coords = pix_coords[None, None].expand(*inputs["images"].shape[:2], -1, -1, -1)
            inputs["images"] = torch.cat([inputs["images"], pix_coords], dim=2)
            
            x = inputs["images"]
            B, L, _, H, W = x.shape
            x = x.reshape(B*L, _, H, W)
            
            # Tokenize image into patches
            x = self.tokenizer(x)  # Shape: (batch_size, num_patches, embed_dim)
            
            BL, num_patches, embed_dim = x.shape
            x = x.reshape(B, L*num_patches, embed_dim)

            # Pass through the transformer encoder
            x = self.transformer(x)  # Shape: (batch_size, num_patches, embed_dim)
            
            xyz = self.point_linear(x)
            xyz = xyz.reshape(B, -1, 3)
            
            xyz_query = self.point_query(xyz)
            
            xyz_attributes = self.cross_transformer(xyz_query, x)
            xyz_attributes = self.attribute_linear(xyz_attributes) # B, N, C
            
            inputs["xyz_raw"] = xyz
            inputs["rotation_raw"] = xyz_attributes[:, :, :4]
            inputs["scale_raw"] = xyz_attributes[:, :, 4:6]
            inputs["opacity_raw"] = xyz_attributes[:, :, 6:7]
            inputs["rgb_raw"] = xyz_attributes[:, :, 7:10]
            if self.sh_degree > 0:
                inputs["sh_raw"] = xyz_attributes[:, :, 10:]
                
            camera_attributes = self.camera_proj(x) # B, N, E
            camera_attributes = camera_attributes.reshape(B, L, num_patches, embed_dim)
            camera_attributes = camera_attributes.mean(2)
            camera_attributes = self.camera_linear(camera_attributes) # B, L, C
            camera_attributes = camera_attributes.reshape(B*L, 11)
            inputs["rel_quaternion_raw"] = camera_attributes[:, :4]
            inputs["rel_translation_raw"] = camera_attributes[:, 4:7]
            inputs["fx_raw"] = camera_attributes[:, 7:8]
            inputs["fy_raw"] = camera_attributes[:, 8:9]
            inputs["cx_raw"] = camera_attributes[:, 9:10]
            inputs["cy_raw"] = camera_attributes[:, 10:11]

            x = x.reshape(B*L, num_patches, embed_dim)
            # Reconstruct the image from patches
            x = self.unpatchify(x, output_size=(H, W))  # Shape: (batch_size, out_channels, img_size, img_size)
            
            inputs["camera_features"] = x
            inputs = self.camera_decoder(inputs)
            
            inputs["gs_features"] = x
            inputs = self.gs_decoder(inputs)
            
            assert len(inputs["cameras_list"]) == 1 and len(inputs["gs_list"]) == 1
            inputs["gs_list"] = inputs["gs_list"] * L
            
            inputs["camera_dict"] = inputs["cameras_list"].pop()
            for attr_name in dir(inputs["camera_dict"]):
                attr_obj = getattr(type(inputs["camera_dict"]), attr_name, None)
                if isinstance(attr_obj, property):
                    continue
                attr = getattr(inputs["camera_dict"], attr_name)
                if hasattr(attr, 'shape') and hasattr(attr, 'reshape'):
                    shape = attr.shape
                    if len(shape) > 0 and shape[0] == B * L:
                        setattr(inputs["camera_dict"], attr_name, attr.reshape(B, L, *shape[1:]))
                        
            inputs["transformer_features"] = x.reshape(B, L, -1, H, W)
        
        return inputs
    
    def _encode_features(self, inputs):
        return inputs["transformer_features"][:, inputs["now_idx"]], inputs

# Example usage:
if __name__ == '__main__':
    model = VisionTransformer(img_size=224, patch_size=8, in_channels=3, out_channels=3,
                              num_layers=24, embed_dim=1024, num_heads=16, mlp_hidden_dim=4096)

    x = torch.randn(1, 3, 224, 224)  # Example input
    out = model(x)
    print(out.shape)  # Output shape should be (1, 3, 224, 224)
