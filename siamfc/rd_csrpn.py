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

def crop2(x):
    return x[:, :, 2:-2, 2:-2].contiguous()

def crop4(x):
    return x[:, :, 4:-4, 4:-4].contiguous()

def crop6(x):
    return x[:, :, 6:-6, 6:-6].contiguous()

class RDN(nn.Module):
    def __init__(self):
        super(RDN,self).__init__()
        db_layers = [2,4]
        self.conv1 = nn.Conv2d(3, 72, kernel_size=7, stride=2, padding=0, bias=False)

        self.dense_block1 = DenseBlock(72,36,db_layers[0])
        self.res1 = ResLayer(72,self.dense_block1.out_channels)
        self.transition1 = transitionlayer(self.dense_block1.out_channels,36)
        self.dense_block2 = DenseBlock(36,36,db_layers[1])
        self.res2 = ResLayer(36, self.dense_block2.out_channels)
        self.res_long = ResLayer_long(in_c_x0=72, in_c=36, out_c=self.dense_block2.out_channels)
        self.transition2 = transitionlayer(self.dense_block2.out_channels,36)
        # self.dense_block3 = DenseBlock(36,36,6)
        self.dense_block4 = DenseBlock_c(in_channels=36,kernel_size=3)

        # self.ftb1_z = feature_transformation_block(C=384,C_=256,dk=3,shape=[6,6])
        # self.ftb2_z = feature_transformation_block(C=384,C_=256,dk=3,shape=[8,8])
        # self.ftb1_x = feature_transformation_block(C=384,C_=256,dk=3,shape=[22,22])
        # self.ftb2_x = feature_transformation_block(C=384,C_=256,dk=3,shape=[24,24])

    def forward(self,x):
        x0 = self.conv1(x)
        # print('x0 {}'.format(x0.shape))
        x_db1 = self.dense_block1(x0)
        # print('x_db1 {}'.format(x_db1.shape))
        x_res1 = self.res1(x0,x_db1)
        # print('x_res1 {}'.format(x_res1.shape))
        x_crop1 = crop2(x_res1)
        # print('x_crop1 {}'.format(x_crop1.shape))
        x_t1 = self.transition1(x_crop1)
        # print('x_t1 {}'.format(x_t1.shape))
        x_db2 = self.dense_block2(x_t1)
        # print('x_db2 {}'.format(x_db2.shape))
        x = self.res2(x_t1,x_db2)
        # print('x_res2 {}'.format(x.shape))
        x = self.res_long(x0,x)
        # print('x_res_long {}'.format(x.shape))
        x = crop4(x)
        # print('x_crop2 {}'.format(x.shape))
        x = self.transition2(x)
        # print('x_t2 {}'.format(x.shape))
        # x = self.dense_block3(x)
        x1, x2 = self.dense_block4(x)
        # print(x1.shape,x2.shape)
        return x1,x2

class ResLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResLayer,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Conv2d(in_channels=self.in_c,out_channels=self.out_c,kernel_size=1)
    def forward(self, x_i, x):
        x_i = self.conv(x_i)
        x = x_i + x
        return x

class ResLayer_long(nn.Module):
    def __init__(self, in_c_x0, in_c, out_c):
        super(ResLayer_long,self).__init__()
        self.in_c_x0 = in_c_x0
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Conv2d(in_channels=self.in_c_x0,out_channels=self.out_c,kernel_size=1)
    def forward(self, x_i, x):
        x_i = self.conv(x_i)
        x_i = torch.nn.functional.interpolate(x_i,x.shape[-2:])
        x = x_i + x
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
        # self.add_module(f'layer_{self.num_layers}',nn.Conv2d(in_channels=in_channels+self.num_layers*growth_rate,out_channels=64))

    def forward(self,block_input):
        layer_input = block_input
        layer_output = block_input.new_empty(0)
        all_outputs = [block_input] if self.concat_input else []
        for layer in self._modules.values():
            layer_input = torch.cat([layer_input, layer_output], dim=1)
            layer_output = layer(layer_input)
            all_outputs.append(layer_output)
        return torch.cat(all_outputs, dim=1)

class DenseBlock_c(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DenseBlock_c,self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.bn1 = nn.BatchNorm2d(num_features=self.in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.in_channels,out_channels=384,kernel_size=self.kernel_size)
        self.dropout1 = nn.Dropout2d(0.2,inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=384)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(384,out_channels=256,kernel_size=self.kernel_size)
        self.dropout2 = nn.Dropout2d(0.2,inplace=True)
    def forward(self,x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.dropout1(x)

        x1 = self.bn2(x)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.dropout2(x1)
        return x,x1

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
        self.add_module('conv2',nn.Conv2d(384,out_channels=256,kernel_size=self.kernel_size))
        self.add_module('dropout2',nn.Dropout2d(0.2,inplace=True))

        # self.add_module('bn3',nn.BatchNorm2d(num_features=384))
        # self.add_module('relu3',nn.ReLU(inplace=True))
        # self.add_module('conv3',nn.Conv2d(384,out_channels=256,kernel_size=self.kernel_size))
        # self.add_module('dropout3',nn.Dropout2d(0.2,inplace=True))

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DenseLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('norm1', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, out_channels*4, kernel_size=1, padding=0, bias=False))

        self.add_module('norm3', nn.BatchNorm2d(num_features=out_channels*4))
        self.add_module('relu3', nn.ReLU(inplace=True))
        self.add_module('conv3', nn.Conv2d(out_channels*4, out_channels, kernel_size=3, padding=1, bias=False))
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

class feature_transformation_block(nn.Module):
    def __init__(self, C, C_, dk, shape):
        super(feature_transformation_block,self).__init__()
        self.shape = shape
        self.dk = dk
        if not self.dk == 0:
            self.deconv = nn.ConvTranspose2d(in_channels=C,out_channels=C_,kernel_size=dk)
            self.convX = nn.Conv2d(in_channels=C_,out_channels=C_,kernel_size=3)
        else:
            self.convX = nn.Conv2d(in_channels=C,out_channels=C_,kernel_size=3)


        self.convx1 = nn.Conv2d(in_channels=C_, out_channels=C_,kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.convx2 = nn.Conv2d(in_channels=C_,out_channels=C_,kernel_size=3)

    def forward(self, X, x):
        if not self.dk == 0:
            X = self.deconv(X)
        X = self.convX(X)

        x = self.convx1(x)
        x = self.relu(x)

        x = self.convx2(x)
        X = x + X
        X = self.relu(X)
        X = nn.functional.interpolate(X,self.shape)
        return X


class DenseSiamese(nn.Module):
    def __init__(self, ):
        super(DenseSiamese, self).__init__()
        self.featureExtract = RDN()
        self.anchor_num = config.anchor_num
        self.input_size = config.instance_size

        self.ftb_z = feature_transformation_block(C=256,C_=384,dk=0,shape=[6,6])
        self.ftb_x = feature_transformation_block(C=256,C_=384,dk=0,shape=[22,22])

        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)
        self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)
        # stage2
        self.score_displacement_stage2 = int((self.input_size - config.exemplar_size) / config.total_stride_stage2)
        self.conv_cls1_stage2 = nn.Conv2d(384, 384 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.conv_r1_stage2 = nn.Conv2d(384, 384 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.conv_cls2_stage2 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0)
        self.conv_r2_stage2 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0)
        self.regress_adjust_stage2 = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

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

    def forward(self, template, detection):
        N = template.size(0)
        template_feature_stage2, template_feature = self.featureExtract(template)
        detection_feature_stage2, detection_feature = self.featureExtract(detection)

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
        # stage2
        template_feature_stage2 = self.ftb_z(template_feature, template_feature_stage2)
        detection_feature_stage2= self.ftb_x(detection_feature, detection_feature_stage2)

        kernel_score_stage2 = self.conv_cls1_stage2(template_feature_stage2).view(N, 2 * self.anchor_num, 384, 4, 4)
        kernel_regression_stage2 = self.conv_r1_stage2(template_feature_stage2).view(N, 4 * self.anchor_num, 384, 4, 4)
        conv_score_stage2 = self.conv_cls2_stage2(detection_feature_stage2)
        conv_regression_stage2 = self.conv_r2_stage2(detection_feature_stage2)

        conv_scores_stage2 = conv_score_stage2.reshape(1, -1, self.score_displacement_stage2 + 4, self.score_displacement_stage2 + 4)
        score_filters_stage2 = kernel_score_stage2.reshape(-1, 384, 4, 4)
        pred_score_stage2 = F.conv2d(conv_scores_stage2, score_filters_stage2, groups=N).reshape(N, 10, self.score_displacement_stage2 + 1,
                                                                            self.score_displacement_stage2 + 1)
        conv_reg_stage2 = conv_regression_stage2.reshape(1, -1, self.score_displacement_stage2 + 4, self.score_displacement_stage2 + 4)
        reg_filters_stage2 = kernel_regression_stage2.reshape(-1, 384, 4, 4)
        pred_regression_stage2 = self.regress_adjust_stage2(
            F.conv2d(conv_reg_stage2, reg_filters_stage2, groups=N).reshape(N, 20, self.score_displacement_stage2 + 1,
                                                              self.score_displacement_stage2 + 1))

        return pred_score, pred_regression, pred_score_stage2, pred_regression_stage2

    def track_init(self, template):
        N = template.size(0)
        template_feature_stage2, template_feature = self.featureExtract(template)
        template_feature_stage2 = self.ftb_z(template_feature, template_feature_stage2)

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.score_filters = kernel_score.reshape(-1, 256, 4, 4)
        self.reg_filters = kernel_regression.reshape(-1, 256, 4, 4)

        kernel_score_stage2 = self.conv_cls1(template_feature_stage2).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression_stage2 = self.conv_r1(template_feature_stage2).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.score_filters_stage2 = kernel_score_stage2.reshape(-1, 256, 4, 4)
        self.reg_filters_stage2 = kernel_regression_stage2.reshape(-1, 256, 4, 4)

    def track(self, detection):
        N = detection.size(0)
        detection_feature_stage2,detection_feature = self.featureExtract(detection)
        detection_feature_stage2= self.ftb_x(detection_feature, detection_feature_stage2)

        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                                 self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                   self.score_displacement + 1))
        # stage2
        conv_score_stage2 = self.conv_cls2_stage2(detection_feature_stage2)
        conv_regression_stage2 = self.conv_r2_stage2(detection_feature_stage2)

        conv_scores_stage2 = conv_score_stage2.reshape(1, -1, self.score_displacement_stage2 + 4, self.score_displacement_stage2 + 4)
        pred_score_stage2 = F.conv2d(conv_scores_stage2, self.score_filters_stage2, groups=N).reshape(N, 10, self.score_displacement_stage2 + 1,
                                                                                 self.score_displacement_stage2 + 1)
        conv_reg_stage2 = conv_regression_stage2.reshape(1, -1, self.score_displacement + 4, self.score_displacement_stage2 + 4)
        pred_regression_stage2 = self.regress_adjust_stage2(
            F.conv2d(conv_reg_stage2, self.reg_filters_stage2, groups=N).reshape(N, 20, self.score_displacement_stage2 + 1,
                                                                   self.score_displacement_stage2 + 1))
        return pred_score, pred_regression, pred_score_stage2, pred_regression_stage2

import torch
def dsrpn():
    x = torch.ones(1,3,255,255)
    z = torch.ones(1,3,127,127)
    net = DenseSiamese()
    print(net.eval())
    o = net(z,x)
    o1 =net.featureExtract(z)
    o2 =net.featureExtract(x)
    print(o[0].shape)
    print(o[1].shape)
    print(o[2].shape)
    print(o[3].shape)
    # print(o1[0].shape,o1[1].shape)
    # print(o2[0].shape,o2[1].shape)
if __name__ =='__main__':
    dsrpn()
