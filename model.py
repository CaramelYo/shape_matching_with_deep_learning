import logging
import os

from keras.models import Sequential
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation
# from keras.applications.resnet50 import ResNet50
# from keras.initializer import Constant


class Feature_extraction_model:
    def build(
        method='resnet', input_tensor=None, input_shape=None, weight_path=None,
        is_training=False, is_bn=True, is_deconv=False, use_cuda=True
    ):
        # what about loading the model here?

        if method == 'resnet':
            # ResNet50(include_top = False, weights = 'imagenet', input_tensor = inputs)
            logging.info('no implementation')
        else:
            # self-defined feature extraction
            if not input_tensor:
                
                if not input_shape:
                    logging.error('input_shape is None when input_tensor is None')
                    return
                
                # input_shape = (h, w, ch)
                # but in our case, it should be (2, n) or (2, n, 1)
                input_tensor = Input(shape=input_shape)
            
            model = Sequential()
            
            model.add(input_tensor)

            # bn our input
            if is_bn:
                model.add(BatchNormalization())
                # model = BatchNormalization(axis=3)
            
            # build those conv layers
            # model = Conv2D(
            #     filters=512, kernel_size=(2, 2), strides=(1, 1), padding='same',
            #     name='conv1')(model)
            layer_counter = 1

            for i in range(Feature_extraction_model.n_conv_layer):
                n_filter = Feature_extraction_model.n_conv_filters[i]

                model.add(Conv2D(
                    filter=n_filter, kernel_size=(2, 2), strides=(1, 1), padding='same',
                    name='conv' + str(layer_counter)))

                if is_bn:
                    model.add(BatchNormalization(name='bn' + str(layer_counter)))
            
                model.add(Activation(Feature_extraction_model.conv_activation, name='act' + str(layer_counter)))

                layer_counter = layer_counter + 1

            # model.add(
            #     Conv2D(filter=1, kernel_size=(2, 1), strides=(1, 1), padding='same',
            #     name='conv' + str(layer_counter)))
            
            # if is_bn:
            #     model.add(BatchNormalization(name='bn' + str(layer_counter)))
            
            # model.add(Activation(Feature_extraction_model.conv_activation, name='act' + str(layer_counter)))

            # layer_counter = layer_counter + 1

            # build those deconv layers
            if is_deconv:
                for i in range(Feature_extraction_model.n_deconv_layer):
                    n_filter = Feature_extraction_model.n_deconv_filters[i]

                    model.add(Conv2DTranspose(
                        filter=n_filter, kernel_size=(2, 2), strides=(1, 1), padding='same',
                        name='deconv' + str(layer_counter)))

                    if is_bn:
                        model.add(BatchNormalization(name='bn' + str(layer_counter)))
                
                    model.add(Activation(Feature_extraction_model.deconv_activation, name='act' + str(layer_counter)))

                    layer_counter = layer_counter + 1

                model.add(
                    Conv2D(filter=1, kernel_size=(2, 1), strides=(1, 1), padding='same',
                    name='conv' + str(layer_counter)))
                
                if is_bn:
                    model.add(BatchNormalization(name='bn' + str(layer_counter)))
                
                model.add(Activation('sigmod', name='act' + str(layer_counter)))

                layer_counter = layer_counter + 1

            # load the previous parameters
            if weight_path and os.path.isfile(weight_path):
                logging.info('model exists')
                model.load_weights(weight_path, by_name=True)

            model.summary()
            
            return model

    conv_activation = 'relu'
    n_conv_filters = [64, 128, 256, 512]
    n_conv_layer = 4
    deconv_activation = 'relu'
    n_deconv_filters = [512, 256, 128, 64, 16, 4]
    n_deconv_layer = 6


# class Shape_matching_model:
#     def build(self, weight_path = None, is_training = False):