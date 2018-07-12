from __future__ import print_function, division

import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

from torchsummary import summary


main_log = logging.getLogger('pytorch_main_log')

# data
data_dir = 'data'
fe_dir = os.path.join(data_dir, 'feature_extraction')
fe_model_weight_path = os.path.join(fe_dir, 'fe_model_weight.h5')


def height_same_padding(padding, layer):
    avg_padding = int(padding / 2)
    is_odd = padding % 2 != 0

    if is_odd:
        padding_size = [0, 0, avg_padding, avg_padding + 1]
    else:
        padding_size = [0, 0, avg_padding, avg_padding]

    # create custom conv2d
    model = nn.Sequential()
    model.add_module('zero_padding', nn.ZeroPad2d(padding_size))
    model.add_module('conv_or_deconv', layer)

    return model


def conv2d_height_same_padding(in_channels, out_channels, height, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    # calculate the padding size
    padding = 0

    if type(stride) == tuple:
        padding += stride[0] * (height - 1) - height
    elif type(stride) == int:
        padding += stride * (height - 1) - height
    else:
        main_log.error('conv2d_height_same_padding error: the type of stride is unknown (%s)' % type(stride))
        exit()

    if type(dilation) == tuple and type(kernel_size) == tuple:
        padding += dilation[0] * (kernel_size[0] - 1) + 1
    elif type(dilation) == int and type(kernel_size) == int:
        padding += dilation * (kernel_size - 1) + 1
    else:
        main_log.error('conv2d_height_same_padding error: the types of dilation and kernel_size are unknown (%s and %s)' % (type(dilation), type(kernel_size)))

    # avg_padding = int(padding / 2)
    # is_odd = padding % 2 != 0

    # if is_odd:
    #     padding_size = [0, 0, avg_padding, avg_padding + 1]
    # else:
    #     padding_size = [0, 0, avg_padding, avg_padding]

    # # create custom conv2d
    # model = nn.Sequential()
    # model.add_module(nn.ZeroPad(padding_size))
    # model.add_module(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=0, dilation=1, groups=groups, bias=bias))

    # return model
    return height_same_padding(padding, nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=1, groups=groups, bias=bias))


def conv2d_transpose_height_same_padding(in_channels, out_channels, height, kernel_size, stride=1, output_padding=0, dilation=1, groups=1, bias=True):
    # calculate the padding size
    padding = 0

    temp_t = type(stride)
    if temp_t == tuple:
        padding += height * stride[0] - height - stride[0]
    elif temp_t == int:
        padding += height * stride - height - stride
    else:
        main_log.error('conv2d_transpose_height_same_padding error: the type of stride is unknown (%s)' % type(stride))
        exit()

    temp_t = type(kernel_size)
    if temp_t == tuple:
        padding += kernel_size[0]
    elif temp_t == int:
        padding += kernel_size
    else:
        main_log.error('conv2d_transpose_height_same_padding error: the type of kernel_size is unknown (%s)' % type(kernel_size))
        exit()

    temp_t = type(output_padding)
    if temp_t == tuple:
        padding += output_padding[0]
    elif temp_t == int:
        padding += output_padding
    else:
        main_log.error('conv2d_transpose_height_same_padding error: the type of output_padding is unknown (%s)' % type(output_padding))
        exit()

    return height_same_padding(padding, nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=(padding, 0), output_padding=output_padding, groups=groups, bias=bias, dilation=dilation))


class FeatureExtraction(torch.nn.Module):
    def __init__(self, model_name='vgg', weight_path=None, is_bn=True, is_conv=True, is_deconv=False, trainable=False, is_summary=True, use_cuda=True, last_layer=''):
        super(FeatureExtraction, self).__init__()

        main_log.debug('create feature extraction model...')

        if model_name == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                        'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                        'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
            if last_layer == '':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[: last_layer_idx + 1])
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])
        elif model_name == 'self-defined':
            self.model = nn.Sequential()

            layer_counter = 0

            if is_bn:
                self.model.add_module('bn_' + str(layer_counter), nn.BatchNorm2d(FeatureExtraction.n_conv_channels[0]))

            layer_counter += 1

            if is_conv:
                for i in range(FeatureExtraction.n_conv_layer):
                    # self.model.add_module(nn.Conv2d(FeatureExtraction.n_conv_layer[0], FeatureExtraction.n_conv_layer[1],
                    # kernel_size=(2, 2), strides=(1, 1), )
                    self.model.add_module('conv_' + str(layer_counter), conv2d_height_same_padding(
                        FeatureExtraction.n_conv_channels[i], FeatureExtraction.n_conv_channels[i + 1],
                        FeatureExtraction.data_height, kernel_size=2
                    ))

                    if is_bn:
                        self.model.add_module('bn_' + str(layer_counter), nn.BatchNorm2d(FeatureExtraction.n_conv_channels[i + 1]))

                    self.model.add_module('act_' + str(layer_counter), nn.ReLU())

                    layer_counter += 1

            if is_deconv:
                for i in range(FeatureExtraction.n_deconv_layer):
                    self.model.add_module('deconv_' + str(layer_counter), conv2d_transpose_height_same_padding(
                        FeatureExtraction.n_deconv_channels[i], FeatureExtraction.n_deconv_channels[i + 1],
                        FeatureExtraction.data_height, kernel_size=(2, 2)
                    ))

                    if is_bn:
                        self.model.add_module('bn_' + str(layer_counter), nn.BatchNorm2d(FeatureExtraction.n_deconv_channels[i + 1]))

                    self.model.add_module('act_' + str(layer_counter), nn.ReLU())

                    layer_counter += 1
                
                self.model.add_module('conv_' + str(layer_counter), conv2d_height_same_padding(
                    FeatureExtraction.n_deconv_channels[FeatureExtraction.n_deconv_layer], 1,
                    FeatureExtraction.data_height, kernel_size=(2, 1)))
                
                if is_bn:
                    self.model.add_module('bn_' + str(layer_counter), nn.BatchNorm2d(1))
                
                self.model.add_module('act_' + str(layer_counter), nn.Sigmoid())
        else:
            main_log.error('Feature_extraction_model build error: unexpected method = %s' % model_name)
            return

        # load the pre-trained weights
        if weight_path and os.path.isfile(weight_path):
            main_log.info('model weight exists')

            checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage)
            checkpoint = OrderedDict([(k.replace('model.', ''), v) for k, v in checkpoint.items()])
            # self.model.load_state_dict(checkpoint, strict=False)
            if is_deconv:
                self.model.load_state_dict(checkpoint, strict=True)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

        # freeze parameters
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        # move to GPU
        if use_cuda:
            self.model.cuda()

        # summary
        if is_summary:
            # input_size = (C, H, W)
            summary(self.model, input_size=(1, 2, 10))

        main_log.debug('creating feature extraction model is completed')

    def forward(self, x):
        return self.model(x)

    data_height = 2

    n_conv_layer = 4
    n_conv_channels = [1, 64, 128, 256, 512]

    n_deconv_layer = 4
    n_deconv_channels = [512, 512, 256, 128, 64]


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
#        print(feature.size())
#        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, f_a, f_b):
        b, c, h, wa = f_a.size()
        wb = f_b.size()[3]
        main_log.debug(f_a.size())
        main_log.debug(f_b.size())

        # reshape features for matrix multiplication
        f_a = f_a.transpose(2, 3).contiguous().view(b, c, h * wa)
        f_b = f_b.view(b, c, h * wb).transpose(1, 2)

        # perform matrix mult.
        feature_mul = torch.bmm(f_b, f_a)
        correlation_tensor = feature_mul.view(b, h, wb, h * wa).transpose(2, 3).transpose(1, 2)

        main_log.debug(f_a.size())
        main_log.debug(f_b.size())
        main_log.debug(correlation_tensor.size())

        return correlation_tensor


class FeatureRegression(nn.Module):
    def __init__(self, weight_path=None, is_bn=True, is_summary=True, use_cuda=True):
        super(FeatureRegression, self).__init__()

        main_log.debug('create feature regression model...')

        self.model = nn.Sequential()

        layer_counter = 1

        for i in range(FeatureRegression.n_conv_layer):
            self.model.add_module('conv_' + str(layer_counter), conv2d_height_same_padding(
                FeatureRegression.n_conv_channels[i], FeatureRegression.n_conv_channels[i + 1],
                FeatureRegression.data_height, kernel_size=FeatureRegression.n_conv_kernel_sizes[i]
            ))

            if is_bn:
                self.model.add_module('bn_' + str(layer_counter), nn.BatchNorm2d(FeatureRegression.n_conv_channels[i + 1]))

            self.model.add_module('act_' + str(layer_counter), nn.ReLU())

            layer_counter += 1

        # self.model.add_module('gap', nn.AvgPool2d())

        # load the pre-trained weights
        if weight_path and os.path.isfile(weight_path):
            main_log.info('model weight exists')
            self.model.load_state_dict(torch.load(weight_path))

        # move to GPU
        if use_cuda:
            self.model.cuda()

        # summary
        if is_summary:
            summary(self.model, input_size=(2992, 2, 1496))

        main_log.debug('creating feature regression model is completed')

    def forward(self, x):
        x = self.model(x)
        gap = nn.AvgPool2d(x.size()[2:])
        return gap(x)
    
    data_height = 2

    n_conv_layer = 7
    # 2 * 1500
    n_conv_channels = [2992, 2048, 1024, 512, 256, 64, 16, 4]
    # n_conv_kernel_sizes = [(2, 11), (2, 11), (2, 7), (2, 7), (2, 3), (2, 3)]
    n_conv_kernel_sizes = [(2, 3), (2, 3), (2, 3), (2, 7), (2, 7), (2, 11), (2, 11)]


class CNNGeometric(nn.Module):
    def __init__(self, fe_model_weight_path=None, fr_model_weight_path=None, is_normalized_feature=True, is_normalized_match=True, use_cuda=True):
        super(CNNGeometric, self).__init__()

        self.use_cuda = use_cuda
        self.is_normalized_feature = is_normalized_feature
        self.is_normalized_match = is_normalized_match

        self.fe_model = FeatureExtraction(model_name='self-defined', weight_path=fe_model_weight_path, use_cuda=self.use_cuda)
        self.L2_norm_model = FeatureL2Norm()
        self.fc_model = FeatureCorrelation()
        self.fr_model = FeatureRegression(weight_path=fr_model_weight_path, use_cuda=self.use_cuda)
        self.ReLU = nn.ReLU(inplace=True)

        self.fr_model_weight_path = fr_model_weight_path

        # lr ???

    def forward(self, a, b):
        # do feature extraction
        f_a = self.fe_model(a)
        f_b = self.fe_model(b)

        # normalize feature
        if self.is_normalized_feature:
            f_a = self.L2_norm_model(f_a)
            f_b = self.L2_norm_model(f_b)
        
        # do feature correlation
        correlation = self.fc_model(f_a, f_b)

        # normalize
        if self.is_normalized_match:
            correlation = self.L2_norm_model(self.ReLU(correlation))
            # correlation = self.L2_norm_model(correlation)
        
        # do regression to predict those arguments
        # args = a1, a2, b1, b2
        # del self.fr_model
        # self.fr_model = FeatureRegression(weight_path=self.fr_model_weight_path, use_cuda=self.use_cuda)
        args = self.fr_model(correlation)

        return args