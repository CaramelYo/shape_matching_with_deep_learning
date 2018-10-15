from __future__ import print_function, division

import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
# from torch.autograd import Variable
import torchvision.models as models

from torchsummary import summary


main_log = logging.getLogger('pytorch_main_log')

# data
data_dir = 'data'
fe_dir = os.path.join(data_dir, 'feature_extraction')
fe_model_weight_path = os.path.join(fe_dir, 'fe_model_weight.h5')

contour_data_size = (2, 1500)


# def height_same_padding(padding, layer):
def same_padding_layer(layer, h_padding=-1, w_padding=-1):
    # left, right, top, bottom
    padding_size = [0, 0, 0, 0]

    if h_padding != -1:
        h_avg_padding = int(h_padding / 2)
        h_is_odd = h_padding % 2 != 0

        padding_size[2] = h_avg_padding
        padding_size[3] = h_avg_padding + 1 if h_is_odd else h_avg_padding

    if w_padding != -1:
        w_avg_padding = int(w_padding / 2)
        w_is_odd = w_padding % 2 != 0

        padding_size[0] = w_avg_padding
        padding_size[1] = w_avg_padding + 1 if w_is_odd else w_avg_padding

    # create custom conv2d
    model = nn.Sequential()
    model.add_module('zero_padding', nn.ZeroPad2d(padding_size))
    model.add_module('conv_or_deconv', layer)

    return model


def conv2d_same_padding(in_channels, out_channels, kernel_size, height=-1, width=-1, stride=1, dilation=1, groups=1, bias=True):
    # calculate the padding size
    def type_check(var, index, var_name):
        var_type = type(var)

        if var_type == tuple:
            v = var[index]
        elif var_type == int:
            v = var
        else:
            main_log.error('conv2d_same_padding error: unexpected type of %s = %s' % (var_name, var_type))

        return v

    def cal_padding(value, index):
        padding = -1

        if value != -1:
            s = type_check(stride, index, 'stride')
            d = type_check(dilation, index, 'dilation')
            k = type_check(kernel_size, index, 'kernel_size')

            padding = s * (value - 1) - value + d * (k - 1) + 1

        return padding

    h_padding = cal_padding(height, 0)
    w_padding = cal_padding(width, 1)

    return same_padding_layer(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=1, groups=groups, bias=bias), h_padding, w_padding)


def conv2d_transpose_same_padding(in_channels, out_channels, kernel_size, height=-1, width=-1, stride=1, output_padding=0, dilation=1, groups=1, bias=True):
    # calculate the padding size

    def type_check(var, index, var_name):
        var_type = type(var)

        if var_type == tuple:
            v = var[index]
        elif var_type == int:
            v = var
        else:
            main_log.error('conv2d_transpose_same_padding error: unexpected type of %s = %s' % (var_name, var_type))

        return v

    def cal_padding(value, index):
        padding = -1

        if value != -1:
            s = type_check(stride, index, 'stride')
            k = type_check(kernel_size, index, 'kernel_size')
            o_p = type_check(output_padding, index, 'output_padding')

            padding = value * s - value - s + k + o_p

        return padding

    h_padding = cal_padding(height, 0)
    w_padding = cal_padding(width, 1)

    return same_padding_layer(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=(h_padding, w_padding), output_padding=output_padding, groups=groups, bias=bias, dilation=dilation), h_padding, w_padding)


class FeatureExtraction(torch.nn.Module):
    def __init__(self, model_name='vgg', weight_path=None, is_bn=True, is_conv=True, is_deconv=False, trainable=False, is_summary=True, use_cuda=True, last_layer=''):
        super(FeatureExtraction, self).__init__()

        main_log.debug('create feature extraction model %s ...' % model_name)

        if model_name == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = [
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
            if last_layer == '':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[: last_layer_idx + 1])
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = [
                'conv1', 'bn1', 'relu', 'maxpool',
                'layer1',
                'layer2',
                'layer3',
                'layer4']
            if last_layer == '':
                last_layer = 'layer4'

            last_layer_idx = resnet_feature_layers.index(last_layer)

            resnet_module_list = [
                self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
                self.model.layer1,
                self.model.layer2,
                self.model.layer3,
                self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])
        elif model_name == 'resnet152':
            # change the input layer ??
            self.model = models.resnet152(pretrained=True)
            resnet_feature_layers = [
                'conv1', 'bn1', 'relu', 'maxpool',
                'layer1',
                'layer2',
                'layer3',
                'layer4']
            if last_layer == '':
                last_layer = 'layer4'

            last_layer_idx = resnet_feature_layers.index(last_layer)

            resnet_module_list = [
                self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
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
                    self.model.add_module('conv_' + str(layer_counter), conv2d_same_padding(
                        FeatureExtraction.n_conv_channels[i], FeatureExtraction.n_conv_channels[i + 1],
                        FeatureExtraction.kernel_size,
                        height=FeatureExtraction.height, width=FeatureExtraction.width
                    ))

                    if is_bn:
                        self.model.add_module('bn_' + str(layer_counter), nn.BatchNorm2d(FeatureExtraction.n_conv_channels[i + 1]))

                    self.model.add_module('act_' + str(layer_counter), nn.ReLU())

                    layer_counter += 1

            if is_deconv:
                for i in range(FeatureExtraction.n_deconv_layer):
                    self.model.add_module('deconv_' + str(layer_counter), conv2d_transpose_same_padding(
                        FeatureExtraction.n_deconv_channels[i], FeatureExtraction.n_deconv_channels[i + 1],
                        FeatureExtraction.kernel_size,
                        height=FeatureExtraction.height, width=FeatureExtraction.width
                    ))

                    if is_bn:
                        self.model.add_module('bn_' + str(layer_counter), nn.BatchNorm2d(FeatureExtraction.n_deconv_channels[i + 1]))

                    self.model.add_module('act_' + str(layer_counter), nn.ReLU())

                    layer_counter += 1

                self.model.add_module('conv_' + str(layer_counter), conv2d_same_padding(
                    FeatureExtraction.n_deconv_channels[FeatureExtraction.n_deconv_layer], 1,
                    (2, 1),
                    height=FeatureExtraction.height, width=FeatureExtraction.width
                ))

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
            # self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
            # cudnn.benchmark = True

        # summary
        if is_summary:
            # input_size = (C, H, W)
            # summary(self.model, input_size=(1, 512, 512))
            summary(self.model, input_size=(1, 2, 1500))

        main_log.debug('creating feature extraction model is completed')

    def forward(self, x):
        return self.model(x)

    kernel_size = 2
    height = 2
    width = 1500

    n_conv_layer = 4
    n_conv_channels = [1, 32, 64, 128, 256]

    n_deconv_layer = 4
    n_deconv_channels = [256, 128, 64, 32, 1]


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, f_a, f_b):
        # b, c, h, wa = f_a.size()
        # wb = f_b.size()[3]

        # # reshape features for matrix multiplication
        # f_a = f_a.transpose(2, 3).contiguous().view(b, c, h * wa)
        # f_b = f_b.view(b, c, h * wb).transpose(1, 2)

        # # perform matrix mult.
        # feature_mul = torch.bmm(f_b, f_a)
        # correlation_tensor = feature_mul.view(b, h, wb, h * wa).transpose(2, 3).transpose(1, 2)

        # return correlation_tensor

        b, c, h, w = f_a.size()

        # reshape features for matrix multiplication
        f_a = f_a.transpose(2, 3).contiguous().view(b, c, h * w)
        f_b = f_b.view(b, c, h * w).transpose(1, 2)

        # perform matrix multiplication
        f_mul = torch.bmm(f_b, f_a)
        correlation = f_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)

        return correlation


class FeatureRegression(nn.Module):
    def __init__(self, weight_path=None, is_bn=True, is_summary=True, use_cuda=True):
        super(FeatureRegression, self).__init__()

        main_log.debug('create feature regression model...')

        self.model = nn.Sequential()

        layer_counter = 1

        for i in range(FeatureRegression.n_conv_layer):
            self.model.add_module('conv_' + str(layer_counter), conv2d_same_padding(
                FeatureRegression.n_conv_channels[i], FeatureRegression.n_conv_channels[i + 1],
                FeatureRegression.data_height, kernel_size=FeatureRegression.n_conv_kernel_sizes[i]
            ))

            if is_bn:
                self.model.add_module('bn_' + str(layer_counter), nn.BatchNorm2d(FeatureRegression.n_conv_channels[i + 1]))

            self.model.add_module('act_' + str(layer_counter), nn.ReLU())

            layer_counter += 1

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


class FeatureMatching(nn.Module):
    def __init__(self, weight_path=None, is_bn=True, is_summary=True, use_cuda=True):
        super(FeatureMatching, self).__init__()

        main_log.debug('create feature matching model...')

        self.model = nn.Sequential()

        layer_counter = 1

        for i in range(FeatureMatching.n_conv_layer):
            self.model.add_module('conv_' + str(layer_counter), conv2d_same_padding(
                FeatureMatching.n_conv_channels[i], FeatureMatching.n_conv_channels[i + 1],
                FeatureMatching.n_conv_kernel_size,
                height=FeatureMatching.height, width=FeatureMatching.width
            ))

            if is_bn:
                self.model.add_module('bn_' + str(layer_counter), nn.BatchNorm2d(FeatureMatching.n_conv_channels[i + 1]))

            self.model.add_module('act_' + str(layer_counter), nn.ReLU())

            layer_counter += 1

        # last one
        self.model.add_module('conv_last', conv2d_same_padding(
            FeatureMatching.n_conv_channels[FeatureMatching.n_conv_layer], FeatureMatching.n_conv_channels[FeatureMatching.n_conv_layer]
            (1, 1), height=FeatureMatching.height, width=FeatureMatching.width
        ))

        if is_bn:
            self.model.add_module('bn_last', nn.BatchNorm2d(FeatureMatching.n_conv_channels[FeatureMatching.n_conv_layer]))

        # range from 0 to 1
        self.model.add_module('act_last', nn.Sigmoid())

        # load the pre-trained weights
        if weight_path and os.path.isfile(weight_path):
            main_log.info('model weight exists')
            self.model.load_state_dict(torch.load(weight_path))

        # move to GPU
        if use_cuda:
            self.model.cuda()

        # summary
        if is_summary:
            summary(self.model, input_size=(512 * 512, 512, 512))

        main_log.debug('creating feature matching model is completed')

    def forward(self, corre_b):
        b, c, h, w = corre_b.size()

        corre_a = corre_b.view(b, c, h * w).transpose(1, 2).view(b, c, h, w)

        selected_b = self.model(corre_b)
        selected_a = self.model(corre_a)

        return selected_b, selected_a

    height = 512
    width = 512

    n_conv_layer = 7
    n_conv_channels = [512 * 512, 256 * 256, 128 * 128, 64 * 64, 32 * 32, 8 * 8, 2 * 2, 1]
    n_conv_kernel_size = (3, 3)


class MatchingPointSelection(nn.Module):
    def __init__(self, weight_path=None, is_bn=True, is_summary=True, use_cuda=True):
        super(MatchingPointSelection, self).__init__()

        main_log.debug('create matching point selection model...')

        self.model = nn.Sequential()

        layer_counter = 1

        for i in range(MatchingPointSelection.n_conv_layer):
            self.model.add_module('conv_' + str(layer_counter), conv2d_same_padding(
                MatchingPointSelection.n_conv_channels[i], MatchingPointSelection.n_conv_channels[i + 1],
                MatchingPointSelection.n_conv_kernel_size,
                # height=contour_data_size[0], width=contour_data_size[1]
                height=contour_data_size[0], width=MatchingPointSelection.n_conv_widths[i]
            ))

            # self.model.add_module('conv_' + str(layer_counter), conv2d_same_padding(
            #     MatchingPointSelection.n_conv_channels[i], MatchingPointSelection.n_conv_channels[i + 1],
            #     MatchingPointSelection.n_conv_kernel_size,
            #     height=contour_data_size[0], width=-1
            # ))

            if is_bn:
                self.model.add_module('bn_' + str(layer_counter), nn.BatchNorm2d(MatchingPointSelection.n_conv_channels[i + 1]))

            self.model.add_module('act_' + str(layer_counter), nn.ReLU())

            # decrease the width dimension
            self.model.add_module('max_pool_' + str(layer_counter), nn.MaxPool2d((1, 2)))

            layer_counter += 1

        # last one
        self.model.add_module('conv_last', conv2d_same_padding(
            MatchingPointSelection.n_conv_channels[MatchingPointSelection.n_conv_layer], MatchingPointSelection.n_conv_channels[MatchingPointSelection.n_conv_layer],
            # (1, 1), height=contour_data_size[0], width=contour_data_size[1]
            (1, 1), height=contour_data_size[0], width=MatchingPointSelection.n_conv_widths[MatchingPointSelection.n_conv_layer]
        ))

        if is_bn:
            self.model.add_module('bn_last', nn.BatchNorm2d(MatchingPointSelection.n_conv_channels[MatchingPointSelection.n_conv_layer]))

        # avg pooling
        # self.model.add_module('avg_pool', nn.AvgPool2d(kernel_size=(contour_data_size[0], 1)))
        # let the last width be 4
        self.model.add_module('avg_pool', nn.AvgPool2d(kernel_size=(contour_data_size[0], 3), padding=(0, 1)))

        # range from 0 to 1
        # self.model.add_module('act_last', nn.Sigmoid())
        # range (-1, 1)
        self.model.add_module('act_last', nn.Tanh())

        # load the pre-trained weights
        if weight_path and os.path.isfile(weight_path):
            main_log.info('model weight exists')
            self.model.load_state_dict(torch.load(weight_path))

        # move to GPU
        if use_cuda:
            self.model.cuda()

        # summary
        if is_summary:
            summary(self.model, input_size=(contour_data_size[0] * contour_data_size[1], contour_data_size[0], contour_data_size[1]))

        main_log.debug('creating matching point selection is completed')

    def forward(self, corre_a, corre_b):
        pr_a = self.model(corre_a)
        pr_b = self.model(corre_b)
        
        return pr_a, pr_b

    n_conv_layer = 7
    # 2 * 1500
    n_conv_channels = [contour_data_size[0] * contour_data_size[1], 1500, 750, 188, 46, 12, 3, 1]
    n_conv_widths = [contour_data_size[1], 750, 375, 187, 93, 46, 23, 11]
    n_conv_kernel_size = (3, 3)


class CNNGeometric(nn.Module):
    # def __init__(self, fe_model_weight_path=None, fr_model_weight_path=None, is_normalized_feature=True, is_normalized_match=True, use_cuda=True):
    def __init__(self, fe_model_weight_path=None, fm_model_weight_path=None, is_normalized_feature=True, is_normalized_match=True, use_cuda=True):
        super(CNNGeometric, self).__init__()

        self.use_cuda = use_cuda
        self.is_normalized_feature = is_normalized_feature
        self.is_normalized_match = is_normalized_match

        self.fe_model = FeatureExtraction(model_name='self-defined', weight_path=fe_model_weight_path, use_cuda=self.use_cuda)
        self.L2_norm_model = FeatureL2Norm()
        self.fc_model = FeatureCorrelation()
        # self.fr_model = FeatureRegression(weight_path=fr_model_weight_path, use_cuda=self.use_cuda)
        self.fm_model = FeatureMatching(weight_path=fm_model_weight_path, use_cuda=self.use_cuda)

        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, a, b, input_a, input_b):
        # do feature extraction
        f_a = self.fe_model(a)
        f_b = self.fe_model(b)

        # normalize feature
        if self.is_normalized_feature:
            f_a = self.L2_norm_model(f_a)
            f_b = self.L2_norm_model(f_b)

        # do feature correlation
        # correlation = self.fc_model(f_a, f_b)
        correlation_b = self.fc_model(f_a, f_b)

        # normalize
        if self.is_normalized_match:
            correlation_b = self.L2_norm_model(self.ReLU(correlation_b))
            # correlation = self.L2_norm_model(correlation)

        # do regression to predict those arguments
        # args = a1, a2, b1, b2
        # del self.fr_model
        # self.fr_model = FeatureRegression(weight_path=self.fr_model_weight_path, use_cuda=self.use_cuda)
        # args = self.fr_model(correlation)

        # return args

        selected_b, selected_a = self.fm_model(correlation_b)

        # return the matching result
        selected_mask_b = selected_b.mul(input_b)
        threshold = 0.5
        selected_points_b = torch.tensor([], dtype=torch.float32, requires_grad=True)

        for hi in range(selected_mask_b.size()[2]):
            for wi in range(selected_mask_b.size()[3]):
                if selected_mask_b[0, 0, hi, wi] >= threshold:
                    point = torch.tensor([[wi], [hi]], dtype=torch.float32, requires_grad=True)
                    selected_points_b = torch.cat((selected_points_b, point), 1)
        
        selected_mask_a = selected_a.mul(input_a)
        selected_points_a = torch.tensor([], dtype=torch.float32, requires_grad=True)

        for hi in range(selected_mask_a.size()[2]):
            for wi in range(selected_mask_a.size()[3]):
                if selected_mask_a[0, 0, hi, wi] >= threshold:
                    point = torch.tensor([[wi], [hi]], dtype=torch.float32, requires_grad=True)
                    selected_points_a = torch.cat((selected_points_a, point), 1)
        
        main_log.debug('b size = ' + str(selected_points_b.size()) + ' a size = ' + str(selected_points_a.size()))

        return selected_points_b, selected_points_a


class PrModel(nn.Module):
    def __init__(self, fe_model_weight_path=None, is_normalized_feature=True, is_normalized_match=True, use_cuda=True):
        super(PrModel, self).__init__()

        self.use_cuda = use_cuda
        self.is_normalized_feature = is_normalized_feature
        self.is_normalized_match = is_normalized_match

        self.fe_model = FeatureExtraction(model_name='self-defined', weight_path=fe_model_weight_path, is_deconv=False, use_cuda=self.use_cuda)
        self.L2_norm_model = FeatureL2Norm()
        self.fc_model = FeatureCorrelation()
        # self.fr_model = FeatureRegression(weight_path=fr_model_weight_path, use_cuda=self.use_cuda)
        # self.fm_model = FeatureMatching(weight_path=fm_model_weight_path, use_cuda=self.use_cuda)
        self.mps_model = MatchingPointSelection(weight_path=None, use_cuda=self.use_cuda)
        self.ReLU = nn.ReLU(inplace=True)
    
    # def forward(self, a, b, input_a, input_b):
    def forward(self, a, b):
        # feature extraction
        f_a = self.fe_model(a)
        f_b = self.fe_model(b)

        # normalize feature
        if self.is_normalized_feature:
            f_a = self.L2_norm_model(f_a)
            f_b = self.L2_norm_model(f_b)

        # do feature correlation
        correlation_b = self.fc_model(f_a, f_b)

        # normalize
        if self.is_normalized_match:
            correlation_b = self.L2_norm_model(self.ReLU(correlation_b))

        b, c, h, w = correlation_b.size()
        correlation_a = correlation_b.view(b, c, h * w).transpose(1, 2).view(b, c, h, w)

        pr_a, pr_b = self.mps_model(correlation_a, correlation_b)

        return pr_a, pr_b
