# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications made by Cynthia Ifeyinwa Ugwu (cugwu), on 2024.
# - Added TSM, GSM, and GSF modules.
# - Modified the original MONAI `blocks` to integrate new functionalities for SSHUNet
# https://dl.acm.org/doi/abs/10.1145/3653946.3653965

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetUpBlock


############################## added by cugwu #####################################
class Tsm(nn.Module):
    """
    TSM: Temporal Shift Module for Efficient Video Understanding
    https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html
    """
    def __init__(self, channels_division=4):
        super(Tsm, self).__init__()
        self.ch_div = channels_division

    def forward(self, x):
        x = self.shift(x, self.ch_div)
        return x, None

    @staticmethod
    def shift(x, ch_div):
        _, c, _, _, _ = x.size()
        fold = c // ch_div
        out = torch.zeros_like(x)
        out[:, :fold, 1:] = x[:, :fold, :-1]
        out[:, -fold:, :-1] = x[:, -fold:, 1:]
        out[:, fold:-fold] = x[:, fold:-fold]
        return out


class Gsm(nn.Module):
    """
    Gate-Shift Networks for Video Action Recognition
    https://openaccess.thecvf.com/content_CVPR_2020/html/Sudhakaran_Gate-Shift_Networks_for_Video_Action_Recognition_CVPR_2020_paper.html
    """
    def __init__(self, fPlane, decoder):
        super(Gsm, self).__init__()

        self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1, padding=(1, 1, 1), groups=2)
        self.tanh = nn.Tanh()
        self.fPlane = fPlane
        self.dec = decoder
        self.bn = nn.InstanceNorm3d(num_features=fPlane)
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def lshift(self, x):
        out = torch.roll(x, shifts=-1, dims=2)
        out[:, :, -1] = 0
        return out

    def rshift(self, x):
        out = torch.roll(x, shifts=1, dims=2)
        out[:, :, 0] = 0
        return out

    def forward(self, x):
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)
        gate = self.tanh(self.conv3D(x_bn_relu))
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)

        x_group1 = x[:, :self.fPlane // 2]
        x_group2 = x[:, self.fPlane // 2:]
        y_group1 = gate_group1 * x_group1
        y_group2 = gate_group2 * x_group2

        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2
        if self.dec:
            y_group1 = self.rshift(y_group1) + r_group1
            y_group2 = self.lshift(y_group2) + r_group2
        else:
            y_group1 = self.lshift(y_group1) + r_group1
            y_group2 = self.rshift(y_group2) + r_group2

        return torch.cat((y_group1, y_group2), dim=1), gate


class Gsf(nn.Module):
    """
    Gate-Shift-Fuse for Video Action Recognition
    https://ieeexplore.ieee.org/abstract/document/10105518
    """
    def __init__(self, fPlane,  decoder, gsf_ch_ratio=25):
        super(Gsf, self).__init__()

        fPlane_temp = int(fPlane * gsf_ch_ratio / 100)
        if fPlane_temp % 2 != 0:
            fPlane_temp += 1
        self.fPlane = fPlane_temp
        self.conv3D = nn.Conv3d(self.fPlane, 2, (3, 3, 3), stride=1, padding=(1, 1, 1), groups=2)
        self.tanh = nn.Tanh()

        self.bn = nn.InstanceNorm3d(num_features=self.fPlane)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.channel_conv1 = nn.Conv2d(2, 1, (3, 3), padding=(3 // 2, 3 // 2))
        self.channel_conv2 = nn.Conv2d(2, 1, (3, 3), padding=(3 // 2, 3 // 2))
        self.sigmoid = nn.Sigmoid()
        self.dec = decoder

    def lshift_zeroPad(self, x):
        out = torch.roll(x, shifts=-1, dims=2)
        out[:, :, -1] = 0
        return out

    def rshift_zeroPad(self, x):
        out = torch.roll(x, shifts=1, dims=2)
        out[:, :, 0] = 0
        return out

    def forward(self, x_full):
        x = x_full[:, :self.fPlane, :, :, :]

        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)
        gate = self.tanh(self.conv3D(x_bn_relu))
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)

        x_group1 = x[:, :self.fPlane // 2]
        x_group2 = x[:, self.fPlane // 2:]

        y_group1 = gate_group1 * x_group1
        y_group2 = gate_group2 * x_group2

        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2  # BxCxNxWxH

        if self.dec:
            y_group1 = self.rshift_zeroPad(y_group1)
            y_group2 = self.lshift_zeroPad(y_group2)
        else:
            y_group1 = self.lshift_zeroPad(y_group1)
            y_group2 = self.rshift_zeroPad(y_group2)

        r_1 = torch.mean(r_group1, dim=-1, keepdim=False)
        r_1 = torch.mean(r_1, dim=-1, keepdim=False).unsqueeze(3)
        r_2 = torch.mean(r_group2, dim=-1, keepdim=False)
        r_2 = torch.mean(r_2, dim=-1, keepdim=False).unsqueeze(3)

        y_1 = torch.mean(y_group1, dim=-1, keepdim=False)
        y_1 = torch.mean(y_1, dim=-1, keepdim=False).unsqueeze(3)
        y_2 = torch.mean(y_group2, dim=-1, keepdim=False)
        y_2 = torch.mean(y_2, dim=-1, keepdim=False).unsqueeze(3)  # BxCxN

        y_r_1 = torch.cat([y_1, r_1], dim=3).permute(0, 3, 1, 2)
        y_r_2 = torch.cat([y_2, r_2], dim=3).permute(0, 3, 1, 2)  # Bx2xCxN

        y_1_weights = self.sigmoid(self.channel_conv1(y_r_1)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        r_1_weights = 1 - y_1_weights
        y_2_weights = self.sigmoid(self.channel_conv2(y_r_2)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        r_2_weights = 1 - y_2_weights

        y_group1 = y_group1 * y_1_weights + r_group1 * r_1_weights
        y_group2 = y_group2 * y_2_weights + r_group2 * r_2_weights

        y = torch.cat((y_group1, y_group2), dim=1)
        y = torch.cat([y, x_full[:, self.fPlane:, :, :, :]], dim=1)

        return y, gate
##################################################################################

class GateUnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        decoder: bool,
        do_gate: bool,
        gate_type: str,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        ############# added by cugwu #################
        self.do_gate = do_gate
        if self.do_gate:
            if gate_type == 'gsm': self.shift = Gsm(fPlane=out_channels, decoder=decoder)
            elif gate_type == 'gsf': self.shift = Gsf(fPlane=out_channels, decoder=decoder)
            elif gate_type == 'tsm': self.shift = Tsm(channels_division=4)
            else: print("The inserted gate type is not supported. The options are: gsf, gsm or tsm")
        ############# added by cugwu #################
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                act=None,
                norm=None,
                conv_only=False,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        ############ added by cugwu #############
        if self.do_gate:
            out, gate = self.shift(out)
        else:
            gate = None
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.do_gate:
            out, gate = self.shift(out)
        else:
            gate = None
        ############ added by cugwu #############

        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.norm2(out)
        out = self.lrelu(out)
        return out, gate


class GateUnetBasicBlock(nn.Module):
    """
    A CNN module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out, None


class GateUnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        decoder: bool,
        do_gate: bool,
        do_basic: bool,
        gate_type: str,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )
        if do_basic:
            self.conv_block = GateUnetBasicBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dropout=dropout,
                norm_name=norm_name,
                act_name=act_name,
            )
        else:
            self.conv_block = GateUnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                decoder=decoder,
                do_gate=do_gate,
                gate_type=gate_type,
                kernel_size=kernel_size,
                stride=1,
                dropout=dropout,
                norm_name=norm_name,
                act_name=act_name,
                )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out, gate = self.conv_block(out)
        return out, gate


class GateUnetOutBlockVFN(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            dropout=dropout,
            bias=True,
            act=None,
            norm=None,
            conv_only=False,
        )

        ############ added by cugwu #############
        self.conv1 = UnetBasicBlock(spatial_dims, out_channels*3, out_channels*6, 3, 2, "instance", "leakyrelu", dropout)
        self.conv2 = UnetBasicBlock(spatial_dims, out_channels*6, out_channels*9, 3, 2, "instance", "leakyrelu", dropout)
        self.bottleneck = UnetBasicBlock(spatial_dims, out_channels*9, out_channels*12, 3, 2, "instance", "leakyrelu", dropout)

        self.up2 = UnetUpBlock(spatial_dims, out_channels*12, out_channels*9, 3, 2, 2, "instance", "leakyrelu", dropout, False)
        self.up1 = UnetUpBlock(spatial_dims, out_channels*9, out_channels*6, 3, 2, 2, "instance", "leakyrelu", dropout, False)
        self.up = UnetUpBlock(spatial_dims, out_channels*6, out_channels*3, 3, 2, 2, "instance", "leakyrelu", dropout, False)
        #########################################

        self.out = get_conv_layer(
            spatial_dims,
            out_channels*3,
            out_channels,
            kernel_size=1,
            stride=1,
            dropout=dropout,
            bias=True,
            act=None,
            norm=None,
            conv_only=False,
        )

    def forward(self, inp):
        inp = self.conv(inp)

        ############ added by cugwu #############
        # Output preparation
        b, c, t, h, w = inp.size()
        out_views = inp.view(3, -1, c, t, h, w)

        out_xz = out_views[1]
        out_xy = out_views[2]
        out_zy = out_views[0]

        # Volumetric Fusion Net
        vfn_inp = torch.cat((out_zy, out_xz, out_xy), dim=1)
        skip1 = self.conv1(vfn_inp)
        skip2 = self.conv2(skip1)
        out = self.bottleneck(skip2)
        out = self.up2(out, skip2)
        out = self.up1(out, skip1)
        out = self.up(out, vfn_inp)
        out = self.out(out)
        #########################################
        return out


class GateUnetOutBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            dropout=dropout,
            bias=True,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.act = get_act_layer(name="leakyrelu")
        self.norm = get_norm_layer(name="instance", spatial_dims=spatial_dims, channels=in_channels)
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            dropout=dropout,
            bias=True,
            act=None,
            norm=None,
            conv_only=False,
        )

    def forward(self, inp):
        ############ added by cugwu #############
        # Output preparation
        b, c, t, h, w = inp.size()
        out_views = inp.view(3, -1, c, t, h, w)
        out_zy = (out_views[0])
        out_xz = (out_views[1]).transpose(2, 3)
        out_xy = (out_views[2]).transpose(2, 4)

        out = self.conv1(out_zy + out_xz + out_xy)
        out = self.norm(out)
        out = self.act(out)
        #########################################
        return self.conv(out)


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Optional[Union[Tuple, str]] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]
