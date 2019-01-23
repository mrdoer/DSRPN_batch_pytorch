import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from .custom_transforms import ToTensor

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn
from torch.nn import init

from IPython import embed
from .config import config

class SiameseAlexNet(nn.Module):
    def __init__(self, ):
        super(SiameseAlexNet, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
        )
        self.anchor_num = config.anchor_num
        self.input_size = config.instance_size
        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)
        self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight.data, std=0.0005)
                nn.init.normal_(m.bias.data, std=0.0005)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, template, detection):
        N = template.size(0)
        template_feature = self.featureExtract(template)
        detection_feature = self.featureExtract(detection)

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        score_filters = kernel_score.reshape(-1, 256, 4, 4)
        pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                            self.score_displacement + 1)

        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        reg_filters = kernel_regression.reshape(-1, 256, 4, 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                              self.score_displacement + 1))
        return pred_score, pred_regression

    def track_init(self, template):
        N = template.size(0)
        template_feature = self.featureExtract(template)

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.score_filters = kernel_score.reshape(-1, 256, 4, 4)
        self.reg_filters = kernel_regression.reshape(-1, 256, 4, 4)

    def track(self, detection):
        N = detection.size(0)
        detection_feature = self.featureExtract(detection)

        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                                 self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                   self.score_displacement + 1))
        return pred_score, pred_regression

class RDN(nn.Module):
    def __init__(self):
        super(RDN,self).__init__()
        self.conv1 = nn.Conv2d(3, 72, kernel_size=7, stride=2, padding=0, bias=False)

        self.dense_block1 = DenseBlock(72,36,2)
        self.transition1 = transitionlayer(self.dense_block1.out_channels,36)
        self.dense_block2 = DenseBlock(36,36,4)
        self.transition2 = transitionlayer(self.dense_block2.out_channels,36)
        self.dense_block3 = DenseBlock(36,36,6)
        self.dense_block4 = DenseBlock_b(in_channels=252,kernel_size=3)
    def forward(self,x):
        x = self.conv1(x)

        x = self.dense_block1(x)
        x = self.transition1(x)

        x = self.dense_block2(x)
        x = self.transition2(x)

        x = self.dense_block3(x)
        x = self.dense_block4(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, concat_input=True):
        super(DenseBlock,self).__init__()
        self.concat_input = concat_input
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.out_channels = num_layers * growth_rate
        if self.concat_input:
            self.out_channels += self.in_channels
        for i in range(num_layers):
            self.add_module(f'layer_{i}',
                            DenseLayer(in_channels=in_channels+i*growth_rate,out_channels = growth_rate))

    def forward(self,block_input):
        layer_input = block_input
        layer_output = block_input.new_empty(0)
        all_outputs = [block_input] if self.concat_input else []
        for layer in self._modules.values():
            layer_input = torch.cat([layer_input, layer_output], dim=1)
            layer_output = layer(layer_input)
            all_outputs.append(layer_output)

        return torch.cat(all_outputs, dim=1)

class DenseBlock_b(nn.Sequential):
    def __init__(self, in_channels, kernel_size):
        super(DenseBlock_b,self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.add_module('bn1',nn.BatchNorm2d(num_features=self.in_channels))
        self.add_module('relu1',nn.ReLU(inplace=True))
        self.add_module('conv1',nn.Conv2d(self.in_channels,out_channels=384,kernel_size=self.kernel_size))
        self.add_module('dropout1',nn.Dropout2d(0.2,inplace=True))

        self.add_module('bn2',nn.BatchNorm2d(num_features=384))
        self.add_module('relu2',nn.ReLU(inplace=True))
        self.add_module('conv2',nn.Conv2d(384,out_channels=384,kernel_size=self.kernel_size))
        self.add_module('dropout2',nn.Dropout2d(0.2,inplace=True))

        self.add_module('bn3',nn.BatchNorm2d(num_features=384))
        self.add_module('relu3',nn.ReLU(inplace=True))
        self.add_module('conv3',nn.Conv2d(384,out_channels=256,kernel_size=self.kernel_size))
        self.add_module('dropout3',nn.Dropout2d(0.2,inplace=True))

class DenseLayer(nn.Sequential):
    r"""
    Consists of:
    - Batch Normalization
    - ReLU
    - 3x3 Convolution
    - (Dropout) dropout rate: 0.2
    """
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DenseLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False))
        if dropout > 0:
            self.add_module('drop', nn.Dropout2d(dropout, inplace=True))

class transitionlayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(transitionlayer,self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes,out_planes,kernel_size=1)
        self.average = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self,x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.average(x)
        return x

class DenseSiamese(nn.Module):
    def __init__(self, ):
        super(DenseSiamese, self).__init__()
        self.featureExtract = RDN()
        self.anchor_num = config.anchor_num
        self.input_size = config.instance_size
        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)
        self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('conv')!=-1 or classname.find('Linear')!=-1):
                init.normal_(m.weight.data, 0.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                 init.normal_(m.weight.data, 1.0, gain)
                 init.constant_(m.bias.data, 0.0)
        self.apply(init_func)
    def init_weights_0(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight.data, std=0.0005)
                nn.init.normal_(m.bias.data, std=0.0005)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, template, detection):
        N = template.size(0)
        template_feature = self.featureExtract(template)
        detection_feature = self.featureExtract(detection)

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 7, 7)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 7, 7)
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 7, self.score_displacement + 7)
        score_filters = kernel_score.reshape(-1, 256, 7, 7)
        pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                            self.score_displacement + 1)

        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 7, self.score_displacement + 7)
        reg_filters = kernel_regression.reshape(-1, 256, 7, 7)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                              self.score_displacement + 1))
        return pred_score, pred_regression

    def track_init(self, template):
        N = template.size(0)
        template_feature = self.featureExtract(template)

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 7, 7)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 7, 7)
        self.score_filters = kernel_score.reshape(-1, 256, 7, 7)
        self.reg_filters = kernel_regression.reshape(-1, 256, 7, 7)

    def track(self, detection):
        N = detection.size(0)
        detection_feature = self.featureExtract(detection)

        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 7, self.score_displacement + 7)
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                                 self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 7, self.score_displacement + 7)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                   self.score_displacement + 1))
        return pred_score, pred_regression
import torch
def dsrpn():
    x = torch.ones(2,3,255,255)
    z = torch.ones(2,3,127,127)
    net = DenseSiamese()
    print(net.eval())
    o = net(z,x)
    print(o[0].shape)
    print(o[1].shape)
if __name__ =='__main__':
    dsrpn()
