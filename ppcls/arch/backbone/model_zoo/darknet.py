# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
import math

from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "DarkNet53":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DarkNet53_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 filter_size,
                 stride,
                 padding,
                 name=None,
                 lr_mult=1.0,
                 data_format="NCHW"):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            weight_attr=ParamAttr(name=name + ".conv.weights", learning_rate=lr_mult),
            bias_attr=False,
            data_format=data_format)

        bn_name = name + ".bn"
        self._bn = BatchNorm(
            num_channels=output_channels,
            act="relu",
            param_attr=ParamAttr(name=bn_name + ".scale", learning_rate=lr_mult),
            bias_attr=ParamAttr(name=bn_name + ".offset", learning_rate=lr_mult),
            moving_mean_name=bn_name + ".mean",
            moving_variance_name=bn_name + ".var",
            data_layout=data_format)

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._bn(x)
        return x


class BasicBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, name=None, lr_mult=1.0, data_format="NCHW"):
        super(BasicBlock, self).__init__()

        self._conv1 = ConvBNLayer(
            input_channels, output_channels, 1, 1, 0, name=name + ".0", \
                lr_mult=lr_mult, data_format=data_format)
        self._conv2 = ConvBNLayer(
            output_channels, output_channels * 2, 3, 1, 1, name=name + ".1", \
                lr_mult=lr_mult, data_format=data_format)

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        return paddle.add(x=inputs, y=x)


class DarkNet(nn.Layer):
    def __init__(self, class_num=1000, lr_mult=1.0, data_format="NCHW", input_image_channel=3):
        super(DarkNet, self).__init__()

        self.stages = [1, 2, 8, 8, 4]
        self._conv1 = ConvBNLayer(input_image_channel, 32, 3, 1, 1, name="yolo_input", \
            lr_mult=lr_mult, data_format=data_format)
        self._conv2 = ConvBNLayer(
            32, 64, 3, 2, 1, name="yolo_input.downsample", \
                lr_mult=lr_mult, data_format=data_format)

        self._basic_block_01 = BasicBlock(64, 32, name="stage.0.0", \
            lr_mult=lr_mult, data_format=data_format)
        self._downsample_0 = ConvBNLayer(
            64, 128, 3, 2, 1, name="stage.0.downsample", \
                lr_mult=lr_mult, data_format=data_format)

        self._basic_block_11 = BasicBlock(128, 64, name="stage.1.0", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_12 = BasicBlock(128, 64, name="stage.1.1", \
            lr_mult=lr_mult, data_format=data_format)
        self._downsample_1 = ConvBNLayer(
            128, 256, 3, 2, 1, name="stage.1.downsample", \
                lr_mult=lr_mult, data_format=data_format)

        self._basic_block_21 = BasicBlock(256, 128, name="stage.2.0", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_22 = BasicBlock(256, 128, name="stage.2.1", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_23 = BasicBlock(256, 128, name="stage.2.2", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_24 = BasicBlock(256, 128, name="stage.2.3", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_25 = BasicBlock(256, 128, name="stage.2.4", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_26 = BasicBlock(256, 128, name="stage.2.5", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_27 = BasicBlock(256, 128, name="stage.2.6", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_28 = BasicBlock(256, 128, name="stage.2.7", \
            lr_mult=lr_mult, data_format=data_format)
        self._downsample_2 = ConvBNLayer(
            256, 512, 3, 2, 1, name="stage.2.downsample", \
                lr_mult=lr_mult, data_format=data_format)

        self._basic_block_31 = BasicBlock(512, 256, name="stage.3.0", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_32 = BasicBlock(512, 256, name="stage.3.1", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_33 = BasicBlock(512, 256, name="stage.3.2", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_34 = BasicBlock(512, 256, name="stage.3.3", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_35 = BasicBlock(512, 256, name="stage.3.4", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_36 = BasicBlock(512, 256, name="stage.3.5", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_37 = BasicBlock(512, 256, name="stage.3.6", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_38 = BasicBlock(512, 256, name="stage.3.7", \
            lr_mult=lr_mult, data_format=data_format)
        self._downsample_3 = ConvBNLayer(
            512, 1024, 3, 2, 1, name="stage.3.downsample", \
                lr_mult=lr_mult, data_format=data_format)

        self._basic_block_41 = BasicBlock(1024, 512, name="stage.4.0", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_42 = BasicBlock(1024, 512, name="stage.4.1", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_43 = BasicBlock(1024, 512, name="stage.4.2", \
            lr_mult=lr_mult, data_format=data_format)
        self._basic_block_44 = BasicBlock(1024, 512, name="stage.4.3", \
            lr_mult=lr_mult, data_format=data_format)

        self._pool = AdaptiveAvgPool2D(1, data_format=data_format)

        stdv = 1.0 / math.sqrt(1024.0)
        self._out = Linear(
            1024,
            class_num,
            weight_attr=ParamAttr(
                name="fc_weights", initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_offset"))
        
        self.data_format = data_format

    def forward(self, inputs):
        with paddle.static.amp.fp16_guard():
            if self.data_format == "NHWC":
                inputs = paddle.transpose(inputs, [0, 2, 3, 1])
                inputs.stop_gradient = True
            x = self._conv1(inputs)
            x = self._conv2(x)

            x = self._basic_block_01(x)
            x = self._downsample_0(x)

            x = self._basic_block_11(x)
            x = self._basic_block_12(x)
            x = self._downsample_1(x)

            x = self._basic_block_21(x)
            x = self._basic_block_22(x)
            x = self._basic_block_23(x)
            x = self._basic_block_24(x)
            x = self._basic_block_25(x)
            x = self._basic_block_26(x)
            x = self._basic_block_27(x)
            x = self._basic_block_28(x)
            x = self._downsample_2(x)

            x = self._basic_block_31(x)
            x = self._basic_block_32(x)
            x = self._basic_block_33(x)
            x = self._basic_block_34(x)
            x = self._basic_block_35(x)
            x = self._basic_block_36(x)
            x = self._basic_block_37(x)
            x = self._basic_block_38(x)
            x = self._downsample_3(x)

            x = self._basic_block_41(x)
            x = self._basic_block_42(x)
            x = self._basic_block_43(x)
            x = self._basic_block_44(x)

            x = self._pool(x)
            if self.data_format == "NCHW":
                x = paddle.squeeze(x, axis=[2, 3])
            elif self.data_format == "NHWC":
                x = paddle.squeeze(x, axis=[1, 2])
            x = self._out(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def DarkNet53(pretrained=False, use_ssld=False, **kwargs):
    model = DarkNet(**kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["DarkNet53"], use_ssld=use_ssld)
    return model
