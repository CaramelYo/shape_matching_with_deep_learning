import logging
import os

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Lambda, GlobalAveragePooling2D
# from keras.applications.resnet50 import ResNet50
# from keras.initializer import Constant
import keras.backend as K
import tensorflow as tf

main_log = logging.getLogger('main_log')


class Feature_extraction_model:
    def build(
        method='resnet', input_tensor=None, input_shape=None, weight_path=None,
        is_training=False, is_bn=True, is_conv=True, is_deconv=False, use_cuda=True
    ):
        # is_training is unused here
        # what about loading the model here?

        if method == 'resnet':
            # ResNet50(include_top = False, weights = 'imagenet', input_tensor = inputs)
            main_log.info('no implementation')
        elif method == 'self-defined':
            # self-defined feature extraction
            if not input_tensor:
                
                if not input_shape:
                    main_log.error('input_shape is None when input_tensor is None')
                    return
                
                # input_shape = (h, w, ch)
                # but in our case, it should be (2, n) or (2, n, 1)
                # input_tensor = Input(shape=input_shape)
            
            model = Sequential()
            
            # model.add(input_tensor)

            # bn our input
            if is_bn:
                model.add(BatchNormalization(input_shape=input_shape))
                # model = BatchNormalization(axis=3)
            
            # build those conv layers
            # model = Conv2D(
            #     filters=512, kernel_size=(2, 2), strides=(1, 1), padding='same',
            #     name='conv1')(model)
            layer_counter = 1

            if is_conv:
                for i in range(Feature_extraction_model.n_conv_layer):
                    model.add(Conv2D(
                        filters=Feature_extraction_model.n_conv_filters[i], kernel_size=(2, 2), strides=(1, 1), padding='same',
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
                        filters=n_filter, kernel_size=(2, 2), strides=(1, 1), padding='same',
                        name='deconv' + str(layer_counter)))

                    if is_bn:
                        model.add(BatchNormalization(name='bn' + str(layer_counter)))
                
                    model.add(Activation(Feature_extraction_model.deconv_activation, name='act' + str(layer_counter)))

                    layer_counter = layer_counter + 1

                model.add(Conv2D(filters=1, kernel_size=(2, 1), strides=(1, 1), padding='same', name='conv' + str(layer_counter)))
                
                if is_bn:
                    model.add(BatchNormalization(name='bn' + str(layer_counter)))
                
                model.add(Activation('sigmoid', name='act' + str(layer_counter)))

                layer_counter = layer_counter + 1

            # load the previous parameters
            if weight_path and os.path.isfile(weight_path):
                main_log.info('model weight exists')
                model.load_weights(weight_path, by_name=True)

            model.summary()
            
            return model
        else:
            logging.error('Feature_extraction_model build error: unexpected method = %s' % method)
            return

    conv_activation = 'relu'
    n_conv_filters = [64, 128, 256, 512]
    n_conv_layer = 4
    deconv_activation = 'relu'
    n_deconv_filters = [512, 256, 128, 64, 16, 4]
    n_deconv_layer = 6


class Feature_L2_norm:
    def build(model=None):
        if model:
            model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
        else:
            main_log.error('model = ' % model)
            exit()

        return model


class Feature_correlation:
    # def build(model=None):
    def build(a, b):
        # tensorflow
        sess = tf.InteractiveSession()
        a_shape = tf.shape(a)
        b_shape = tf.shape(b)

        a1 = tf.reshape(tf.transpose(a, perm=[0, 3, 2, 1]), [a_shape[0], a_shape[3], a_shape[2] * a_shape[1]])
        b1 = tf.reshape(b, [b_shape[0], b_shape[1] * b_shape[2], b_shape[3]])

        correlation = tf.matmul(b1, a1)

        re_correlation = tf.reshape(correlation, [a_shape[0], a_shape[1], b_shape[2], a_shape[1] * a_shape[2]])

        re_correlation_value = re_correlation.eval()

        sess.close()

        return re_correlation_value


class Feature_regression:
    def build(weight_path=None, is_bn=True):
        model = Sequential()

        layer_counter = 0

        # conv
        for i in range(Feature_regression.n_conv_layer):
            model.add(Conv2D(
                filters=Feature_regression.n_conv_filters[0], kernel_size=Feature_regression.n_conv_kernel_sizes[0], strides=(1, 1), padding='same',
                name='conv' + str(layer_counter)))
        
            if is_bn:
                model.add(BatchNormalization(name='bn' + str(layer_counter)))
                
            model.add(Activation(Feature_regression.conv_activation, name='act' + str(layer_counter)))

            layer_counter += 1
        
        # fully connected layer or global average pooling(GAP) ?
        model.add(GlobalAveragePooling2D())
        
        # load the previous parameters
        if weight_path and os.path.isfile(weight_path):
            main_log.info('model weight exists')
            model.load_weights(weight_path, by_name=True)

        model.summary()
    
        return model
        
    # the last n_conv_filter is the number of arguments I want to train
    n_conv_filters = [128, 64, 32, 16, 8, 3]
    n_conv_kernel_sizes = [(2, 23), (2, 19), (2, 15), (2, 11), (2, 7), (2, 3)]
    conv_activation = 'relu'
    n_conv_layer = 6

# class Shape_matching_model:
#     def build(self, weight_path = None, is_training = False):
