# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import time
import torch
from torch.nn.modules import Module
from typing import TypeVar
import collections
from collections import OrderedDict
from itertools import repeat
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom


T = TypeVar('T', bound=Module)

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

_triple = _ntuple(3)

def _reverse_repeat_tuple(t, n):
    """Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))

class LobeWeight(nn.Module):
    '''
    21/9/28
    generate a lobe weight map to be multiplied on the feature map
    '''
    def __init__(self,
                 #in_channels,
                 #out_channels,
                 kernel_size,
                 maskScale=(1,1,1),
                 stride = 1,
                 padding = 'same',
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode = 'zeros',
                 device = None,
                 dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LobeWeight, self).__init__()
        kernel_size_ = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = padding if isinstance(padding, str) else _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        #self.out_channels = out_channels
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in self.stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)
            if padding == 'same':
                for d, k, i in zip(self.dilation, kernel_size_,
                                   range(len(kernel_size_) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
            
        for v in range(5):#[0,1,2,3,4 for lung lobe]
            maskDir = 'lunglobeAnnotate/lungLobe5-%d.nii.gz' % (v)
            itk_img = sitk.ReadImage(maskDir)
            maskFull = sitk.GetArrayFromImage(itk_img).astype(np.float32)
            maskFull[maskFull>0.5] = 1
            #maskFull[maskFull<0.1] = -0.5
            if any(s !=1 for s in maskScale):
                #print('maskScale', maskScale)
                maskFull = zoom(maskFull, zoom=maskScale, order=1)
            #maskFull = (maskFull-maskFull.mean())/maskFull.std()
            if v == 0:
                self.lobeMaskList = torch.zeros([5,1,1]+list(maskFull.shape))
            #print('maskFull', maskFull.min(), maskFull.max())
            self.lobeMaskList[v,0,0] = torch.Tensor(maskFull)#[4,1,1,z,y,x]
        
        for lobeIdx in range(self.lobeMaskList.shape[0]):#1-4 for mask 1-4, 5 for AVE LUNG      
            self.register_parameter(
                'maskWeightLobe%d'%(lobeIdx),
                nn.Parameter(
                    torch.empty(
                        (1, 1, *kernel_size_),
                        requires_grad=True,
                        **factory_kwargs
                        )
                    )
                )
            if bias:
                self.register_parameter(
                    'maskBiasLobe%d'%(lobeIdx),
                    nn.Parameter(
                        torch.empty(
                            1,
                            requires_grad=True,
                            **factory_kwargs
                            )
                        )
                    )
            else:
                self.register_parameter('maskBiasLobe%d'%(lobeIdx),None)
        self.reset_parameters()
        
    def reset_parameters(self):
        for lobeIdx in range(self.lobeMaskList.shape[0]):#4
            maskWeight = self.get_parameter('maskWeightLobe%d'%(lobeIdx))
            torch.nn.init.kaiming_uniform_(maskWeight, a=math.sqrt(5))
            #torch.nn.init.constant_(maskWeight, 1.0/(maskWeight.shape[-3]*maskWeight.shape[-2]*maskWeight.shape[-1]))
            maskBias = self.get_parameter('maskBiasLobe%d'%(lobeIdx))
            #print(idx+1, 'bias-before1', bias)
            #print('bias-before2', self.get_parameter('biasMask%d'%(idx+1)))
            if maskBias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(maskWeight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(maskBias, -bound, bound)
                #torch.nn.init.constant_(maskBias, 0)
            
    def forward(self, x):
        self.lobeMaskList = self.lobeMaskList.to(device=x.device)
        deviceUse = x.device
        #print('tuple(shapeM)', tuple(shapeM))
        maskOutputList1 = torch.empty(self.lobeMaskList.shape, device=deviceUse)
        
        for lobeIdx in range(self.lobeMaskList.shape[0]):#4:[0-3]
            mask = self.lobeMaskList[lobeIdx]
            maskWeight = self.get_parameter('maskWeightLobe%d'%(lobeIdx))
            maskBias = self.get_parameter('maskBiasLobe%d'%(lobeIdx))
            #print('layer0-lobe', lobeIdx, maskWeight, maskBias)
            convedMask = F.conv3d(
                    mask, 
                    maskWeight, maskBias, self.stride, self.padding, self.dilation, self.groups
                    )#[1,1,z,y,x]
            #print('convedMask', convedMask.shape)
            maskOutputList1[lobeIdx] = convedMask#[4, 1, 1, z, y, x]
        #maskOutputList1 = F.relu(maskOutputList1, inplace=True)
        maskOutputList1 = maskOutputList1.sum(dim=0)#[1,1,z,y,x]
        #print('maskOutputList1', maskOutputList1.min(), maskOutputList1.max())
        return x*maskOutputList1


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, filterNum, drop_rate, useLobeWeightLayer=False, kernel_size=None,
                 maskScale=None):
        super(_DenseLayer, self).__init__()
        self.add_module('bn1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, filterNum, kernel_size=1, stride=1, bias=False))
        self.add_module('bn2', nn.BatchNorm3d(filterNum))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(filterNum, filterNum, kernel_size=3, stride=1, padding=1, bias=False))
        if useLobeWeightLayer:
            if kernel_size == (3, 5, 5):
                paddings = (1, 2, 2)
            elif kernel_size == (3, 3, 3):
                paddings = (1, 1, 1)
            self.add_module('lobeWeight', LobeWeight(kernel_size=kernel_size, padding=paddings, maskScale=maskScale))
        self.drop_rate = drop_rate
        self.filterOutNum = num_input_features + filterNum

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_input_features, filterNumList, drop_rate, useLobeWeightLayer=False, kernel_size=None,
                 maskScale=None):
        super(_DenseBlock, self).__init__()
        filterIn = num_input_features
        for i, filterNum in enumerate(filterNumList):
            layer = _DenseLayer(filterIn, filterNum, drop_rate, useLobeWeightLayer, kernel_size, maskScale)
            filterIn = layer.filterOutNum
            self.add_module('denselayer%d' % (i + 1), layer)
        self.filterOutNum = filterIn


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('bn', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.add_module('upsample', nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True))
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.add_module('bn', nn.BatchNorm3d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))


class FCPNet(nn.Module):
    def __init__(self, filternum_conv1=16, drop_rate=0, args=None):
        super(FCPNet, self).__init__()
        # Encoder
        self.featuresMain1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, filternum_conv1, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1), bias=False))
        ]))
        block1 = _DenseBlock(num_input_features=filternum_conv1, filterNumList=[24, 24], drop_rate=drop_rate)
        self.featuresMain1.add_module('denseblock1', block1)
        num_filters = block1.filterOutNum
        trans1 = _Transition(num_filters, 24)
        self.featuresMain1.add_module('transition1', trans1)
        self.featuresMain1.add_module('maxpool1', nn.MaxPool3d(kernel_size=2, stride=2))
        block2 = _DenseBlock(num_input_features=24, filterNumList=[32, 32], drop_rate=drop_rate)
        self.featuresMain1.add_module('denseblock2', block2)
        num_filters = block2.filterOutNum
        trans2 = _Transition(num_filters, 32)
        self.featuresMain1.add_module('transition2', trans2)
        self.featuresMain1.add_module('maxpool2', nn.MaxPool3d(kernel_size=2, stride=2))
        block3 = _DenseBlock(num_input_features=32, filterNumList=[64, 64], drop_rate=drop_rate)
        self.featuresMain1.add_module('denseblock3', block3)
        num_filters = block3.filterOutNum
        trans3 = _Transition(num_filters, 64)
        self.featuresMain1.add_module('transition3', trans3)
        self.featuresMain1.add_module('maxpool3', nn.MaxPool3d(kernel_size=2, stride=2))
        block4 = _DenseBlock(num_input_features=64, filterNumList=[96, 96], drop_rate=drop_rate)
        self.featuresMain1.add_module('denseblock4', block4)
        num_filters = block4.filterOutNum
        self.featuresMain1.add_module('maxpool4', nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1)))
        trans4 = _Transition(num_filters, 96)
        self.featuresMain1.add_module('transition4', trans4)
        self.featuresMain1.add_module('bn5', nn.BatchNorm3d(96))
        self.featuresMain1.add_module('relu5', nn.ReLU(inplace=True))

        self.AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d((2, 2, 2))

        # Decoder
        self.upsample5 = UpsampleBlock(96, 96, scale_factor=(1, 1, 1))
        self.upsample4 = UpsampleBlock(96, 64, scale_factor=(2, 2, 2))
        self.upsample3 = UpsampleBlock(64, 32, scale_factor=(2, 2, 2))
        self.upsample2 = UpsampleBlock(32, 24, scale_factor=(2, 2, 2))
        self.upsample1 = UpsampleBlock(24, 1, scale_factor=(1, 2, 2))
        # self.final_conv = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

        self.flatten = nn.Linear(1536, 768)

        self.classifier = nn.Linear(768, 2)
        self.args = args

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x1, x2, x3, x4):
        eps = 1e-6

        # --- concise encoder for one view ---
        def encode_once(x):
            pre = self.featuresMain1(x)
            pooled = self.AdaptiveAvgPool3d(pre)
            flat = pooled.view(pooled.size(0), -1)
            return pre, pooled, flat

        # Encode 4 views
        main1_pre, main1, main1_flat = encode_once(x1)
        main2_pre, main2, main2_flat = encode_once(x2)
        main3_pre, main3, main3_flat = encode_once(x3)
        main4_pre, main4, main4_flat = encode_once(x4)

        # Build base feature F from view1+view2 (same as your concat+linear)
        F_base = torch.cat((main1, main2), dim=1).view(main1.size(0), -1)
        F = self.flatten(F_base)  # [B, 768]

        # Stable "vertical component": v_perp = v - proj_F(v)
        denom = (F.norm(dim=1, keepdim=True).pow(2) + eps)

        def perp_to_F(v):
            alpha = (v * F).sum(dim=1, keepdim=True) / denom
            return v - alpha * F  # [B,768]

        vertical_component_main3 = perp_to_F(main3_flat)
        vertical_component_main4 = perp_to_F(main4_flat)

        features = F + vertical_component_main3 + vertical_component_main4
        encoded_output = self.classifier(features)

        # --- concise decoder for one branch ---
        def decode_once(pre):
            x = self.upsample5(pre)
            x = self.upsample4(x)
            x = self.upsample3(x)
            x = self.upsample2(x)
            x = self.upsample1(x)
            return x

        main1_up1 = decode_once(main1_pre)
        main2_up1 = decode_once(main2_pre)
        main3_up1 = decode_once(main3_pre)
        main4_up1 = decode_once(main4_pre)

        return main1_up1, main2_up1, main3_up1, main4_up1, encoded_output



if __name__ == '__main__':
    
    model = FCPNet()
    print(model)