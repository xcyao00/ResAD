import math
from typing import Sequence
import torch
import torch.nn as nn
from timm.models.resnet import BasicBlock, create_aa, Bottleneck
from timm.models.layers import create_attn
try:
    from timm.layers import create_act_layer
except ImportError:
    from timm.models.layers import create_act_layer
from timm.models.layers.helpers import make_divisible
from einops import rearrange


class MultiScaleFusion(nn.Module):
    def __init__(self,
                 channels: Sequence[int] = (64, 128, 256)):
        super().__init__()
        
        self.l2_to_l1 = UpsampleConv(channels[1], channels[0], scale_factor=2)
        self.l3_to_l1 = UpsampleConv(channels[2], channels[0], scale_factor=4)
        
        # when groups == in_channels, means depthwise conv
        self.l1_to_l2 = nn.Conv2d(channels[0],  channels[1], stride=2, kernel_size=3, padding=1, groups=channels[0])
        self.l3_to_l2 = UpsampleConv(channels[2], channels[1], scale_factor=2)
        
        self.l1_to_l3 = nn.Conv2d(channels[0], channels[2], stride=4, kernel_size=5, padding=2, groups=channels[0])
        self.l2_to_l3 = nn.Conv2d(channels[1], channels[2], stride=2, kernel_size=3, padding=1, groups=channels[1])

    def forward(self, layer1_x, layer2_x, layer3_x):
        layer2_x_to_1 = self.l2_to_l1(layer2_x)
        layer3_x_to_1 = self.l3_to_l1(layer3_x)
        out1 = layer1_x + layer2_x_to_1 + layer3_x_to_1
        
        layer1_x_to_2 = self.l1_to_l2(layer1_x)
        layer3_x_to_2 = self.l3_to_l2(layer3_x)
        out2 = layer2_x + layer1_x_to_2 + layer3_x_to_2
        
        layer1_x_to_3 = self.l1_to_l3(layer1_x)
        layer2_x_to_3 = self.l2_to_l3(layer2_x)
        out3 = layer3_x + layer1_x_to_3 + layer2_x_to_3
        
        return out1, out2, out3


class MultiScaleFusionAndBasicblock(nn.Module):
    def __init__(self,
                 channels: Sequence[int] = (64, 128, 256)):
        super().__init__()
        
        self.l2_to_l1 = UpsampleConv(channels[1], channels[0], scale_factor=2)
        self.l3_to_l1 = UpsampleConv(channels[2], channels[0], scale_factor=4)
        
        # when groups == in_channels, means depthwise conv
        self.l1_to_l2 = nn.Conv2d(channels[0],  channels[1], stride=2, kernel_size=3, padding=1, groups=channels[0])
        self.l3_to_l2 = UpsampleConv(channels[2], channels[1], scale_factor=2)
        
        self.l1_to_l3 = nn.Conv2d(channels[0], channels[2], stride=4, kernel_size=5, padding=2, groups=channels[0])
        self.l2_to_l3 = nn.Conv2d(channels[1], channels[2], stride=2, kernel_size=3, padding=1, groups=channels[1])

        self.l1_proj = BasicBlock(channels[0], channels[0])
        self.l2_proj = BasicBlock(channels[1], channels[1])
        self.l3_proj = BasicBlock(channels[2], channels[2])
        
    def forward(self, layer1_x, layer2_x, layer3_x):
        layer2_x_to_1 = self.l2_to_l1(layer2_x)
        layer3_x_to_1 = self.l3_to_l1(layer3_x)
        out1 = layer1_x + layer2_x_to_1 + layer3_x_to_1
        out1 = self.l1_proj(out1)
        
        layer1_x_to_2 = self.l1_to_l2(layer1_x)
        layer3_x_to_2 = self.l3_to_l2(layer3_x)
        out2 = layer2_x + layer1_x_to_2 + layer3_x_to_2
        out2 = self.l2_proj(out2)
        
        layer1_x_to_3 = self.l1_to_l3(layer1_x)
        layer2_x_to_3 = self.l2_to_l3(layer2_x)
        out3 = layer3_x + layer1_x_to_3 + layer2_x_to_3
        out3 = self.l3_proj(out3)
        
        return out1, out2, out3


class MultiScaleFusionAndBottleneck(nn.Module):
    def __init__(self,
                 channels: Sequence[int] = (64, 128, 256)):
        super().__init__()
        
        self.l2_to_l1 = UpsampleConv(channels[1], channels[0], scale_factor=2)
        self.l3_to_l1 = UpsampleConv(channels[2], channels[0], scale_factor=4)
        
        # when groups == in_channels, means depthwise conv
        self.l1_to_l2 = nn.Conv2d(channels[0],  channels[1], stride=2, kernel_size=3, padding=1, groups=channels[0])
        self.l3_to_l2 = UpsampleConv(channels[2], channels[1], scale_factor=2)
        
        self.l1_to_l3 = nn.Conv2d(channels[0], channels[2], stride=4, kernel_size=5, padding=2, groups=channels[0])
        self.l2_to_l3 = nn.Conv2d(channels[1], channels[2], stride=2, kernel_size=3, padding=1, groups=channels[1])

        self.l1_proj = Bottleneck(channels[0], channels[0] // 4)
        self.l2_proj = Bottleneck(channels[1], channels[1] // 4)
        self.l3_proj = Bottleneck(channels[2], channels[2] // 4)
        
    def forward(self, layer1_x, layer2_x, layer3_x):
        layer2_x_to_1 = self.l2_to_l1(layer2_x)
        layer3_x_to_1 = self.l3_to_l1(layer3_x)
        out1 = layer1_x + layer2_x_to_1 + layer3_x_to_1
        out1 = self.l1_proj(out1)
        
        layer1_x_to_2 = self.l1_to_l2(layer1_x)
        layer3_x_to_2 = self.l3_to_l2(layer3_x)
        out2 = layer2_x + layer1_x_to_2 + layer3_x_to_2
        out2 = self.l2_proj(out2)
        
        layer1_x_to_3 = self.l1_to_l3(layer1_x)
        layer2_x_to_3 = self.l2_to_l3(layer2_x)
        out3 = layer3_x + layer1_x_to_3 + layer2_x_to_3
        out3 = self.l3_proj(out3)
        
        return out1, out2, out3


class MultiScaleConv(nn.Module):
    def __init__(self,
                 channels: Sequence[int] = (64, 128, 256)):
        super().__init__()

        self.l1_proj = ConvBnAct(channels[0], channels[0])
        self.l2_proj = ConvBnAct(channels[1], channels[1])
        self.l3_proj = ConvBnAct(channels[2], channels[2])
        
    def forward(self, layer1_x, layer2_x, layer3_x):
        out1 = self.l1_proj(layer1_x)
        out2 = self.l2_proj(layer2_x)
        out3 = self.l3_proj(layer3_x)
        
        return out1, out2, out3
    
    
class MultiScaleBasicBlock(nn.Module):
    def __init__(self,
                 channels: Sequence[int] = (64, 128, 256)):
        super().__init__()

        self.l1_proj = BasicBlock(channels[0], channels[0], attn_layer=None)
        self.l2_proj = BasicBlock(channels[1], channels[1], attn_layer=None)
        self.l3_proj = BasicBlock(channels[2], channels[2], attn_layer=None)
        
    def forward(self, layer1_x, layer2_x, layer3_x):
        out1 = self.l1_proj(layer1_x)
        out2 = self.l2_proj(layer2_x)
        out3 = self.l3_proj(layer3_x)
        
        return out1, out2, out3


class MultiScaleBottleneck(nn.Module):
    def __init__(self,
                 channels: Sequence[int] = (64, 128, 256)):
        super().__init__()

        self.l1_proj = Bottleneck(channels[0], channels[0] // 4, attn_layer=None)
        self.l2_proj = Bottleneck(channels[1], channels[1] // 4, attn_layer=None)
        self.l3_proj = Bottleneck(channels[2], channels[2] // 4, attn_layer=None)
        
    def forward(self, layer1_x, layer2_x, layer3_x):
        out1 = self.l1_proj(layer1_x)
        out2 = self.l2_proj(layer2_x)
        out3 = self.l3_proj(layer3_x)
        
        return out1, out2, out3


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x
    
        
class UpsampleConv(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        
        return x


class SinCosPositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.dim = dim
        if self.dim % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(dim))
    
    def forward(self, height, width):
        pos_embed = torch.zeros(self.dim, height, width)
        # Each dimension use half of D
        half_dim = self.dim // 2
        div_term = torch.exp(torch.arange(0.0, half_dim, 2) * -(math.log(1e4) / half_dim))
        pos_w = torch.arange(0.0, width).unsqueeze(1)
        pos_h = torch.arange(0.0, height).unsqueeze(1)
        pos_embed[0:half_dim:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pos_embed[1:half_dim:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pos_embed[half_dim::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pos_embed[half_dim+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        
        return pos_embed


def get_position_encoding(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    
    return P


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        # self.norm = norm_layer(4 * dim)
        # self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2, left top
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2, left bottom
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2, right top
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2, right bottom
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2

        # x = self.norm(x)
        # x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        # self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=2, p2=2, c=C//4)
        # x = x.view(B, -1, C//4)
        # x = self.norm(x)

        return x


class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=None, gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Linear(channels, rd_channels, bias=True)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = create_act_layer(act_layer, inplace=True)
        self.fc2 = nn.Linear(rd_channels, channels, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean(dim=1, keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax(dim=1, keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
    
    
class OrthogonalProjector(nn.Module):
    """ 
    Orthogonal projection layer, we use the mlp in transformer as the projection layer.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True,
                 with_attn=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        
        self.with_attn = with_attn
        if with_attn:
            self.se = SEModule(hidden_features)

    def forward(self, x):
        b, c, h, w = x.shape
        #residual = x
        out = x.permute(0, 2, 3, 1).reshape(b, -1, c)
        out = self.fc1(out)
        if self.with_attn:
            out = self.se(out)
        out = self.act(out)
        out = self.fc2(out)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        #out = out + residual
        
        return out
    
    
class MultiScaleOrthogonalProjector(nn.Module):
    def __init__(self,
                 channels: Sequence[int] = (64, 128, 256, 512),
                 with_attn: bool = True):
        super().__init__()
        
        self.l1_proj = OrthogonalProjector(channels[0], channels[0] * 4, channels[0], with_attn=with_attn)
        self.l2_proj = OrthogonalProjector(channels[1], channels[1] * 4, channels[1], with_attn=with_attn)
        self.l3_proj = OrthogonalProjector(channels[2], channels[2] * 4, channels[2], with_attn=with_attn)
        self.l4_proj = OrthogonalProjector(channels[3], channels[3] * 4, channels[3], with_attn=with_attn)
    
    
    def forward(self, layer1_x, layer2_x, layer3_x, layer4_x):
        out1 = self.l1_proj(layer1_x)
        out2 = self.l2_proj(layer2_x)
        out3 = self.l3_proj(layer3_x)
        out4 = self.l4_proj(layer4_x)
        
        return out1, out2, out3, out4


        
