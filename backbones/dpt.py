from gs_decoders.base_gs_decoder import BaseGSDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from utils.config_utils import GlobalState

class DPT(nn.Module):

    def __init__(self, patch_size=8, out_channels=32, embed_dim=1024, bias=False, use_bn=False):
        super(DPT, self).__init__()
        self.patch_size = patch_size
        
        layer_dims = [embed_dim//8, embed_dim//4, embed_dim//2, embed_dim]
        self.act_1_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=layer_dims[0],
                kernel_size=1, stride=1, padding=0,
                bias=bias
            ),
            nn.ConvTranspose2d(
                in_channels=layer_dims[0],
                out_channels=layer_dims[0],
                kernel_size=4, stride=4, padding=0,
                bias=bias
            )
        )
        
        self.act_2_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=layer_dims[1],
                kernel_size=1, stride=1, padding=0,
                bias=bias
            ),
            nn.ConvTranspose2d(
                in_channels=layer_dims[1],
                out_channels=layer_dims[1],
                kernel_size=2, stride=2, padding=0,
                bias=bias
            )
        )
        
        self.act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=layer_dims[2],
                kernel_size=1, stride=1, padding=0,
                bias=bias
            )
        )
        
        self.act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=layer_dims[3],
                kernel_size=1, stride=1, padding=0,
                bias=bias
            ),
            nn.Conv2d(
                in_channels=layer_dims[3],
                out_channels=layer_dims[3],
                kernel_size=3, stride=2, padding=1,
                bias=bias
            )
        )
        
        self.scratch = make_scratch(layer_dims, embed_dim, groups=1, expand=False)
        
        self.scratch.refinenet1 = make_fusion_block(embed_dim, use_bn)
        self.scratch.refinenet2 = make_fusion_block(embed_dim, use_bn)
        self.scratch.refinenet3 = make_fusion_block(embed_dim, use_bn)
        self.scratch.refinenet4 = make_fusion_block(embed_dim, use_bn, single_input=True)
        
        self.output_conv1 = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=1, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(embed_dim // 2, out_channels*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x, output_size):
        # x: list of B, num_patches, embed_dim
        features = x
        N_H = output_size[0] // self.patch_size
        N_W = output_size[1] // self.patch_size
        features = [rearrange(feat, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for feat in features]
        feat_1 = self.scratch.layer1_rn(self.act_1_postprocess(features[0]))
        feat_2 = self.scratch.layer2_rn(self.act_2_postprocess(features[1]))
        feat_3 = self.scratch.layer3_rn(self.act_3_postprocess(features[2]))
        feat_4 = self.scratch.layer4_rn(self.act_4_postprocess(features[3]))
        
        path_4 = self.scratch.refinenet4(feat_4, size=feat_3.shape[-2:])
        path_3 = self.scratch.refinenet3(path_4, feat_3, size=feat_2.shape[-2:])
        path_2 = self.scratch.refinenet2(path_3, feat_2, size=feat_1.shape[-2:])
        path_1 = self.scratch.refinenet1(path_2, feat_1)
        
        outputs = self.output_conv1(path_1)
        outputs = F.interpolate(outputs, size=output_size, mode="bilinear", align_corners=True)
        outputs = self.output_conv2(outputs)
        return outputs
    
    
def make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch
    
def make_fusion_block(features, use_bn, size=None, single_input=False):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        single_input=single_input
    )
    

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None,
        single_input=False
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        if not single_input:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)

        return output
    

class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)