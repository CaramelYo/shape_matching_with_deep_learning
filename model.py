import logging
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Lambda, GlobalAveragePooling2D, subtract, add, dot, concatenate, multiply
# from keras.applications.resnet50 import ResNet50
# from keras.initializer import Constant
from keras import losses
from keras.optimizers import Adam

import keras.backend as K
import tensorflow as tf

main_log = logging.getLogger('main_log')


# data
data_dir = 'data'
fe_dir = os.path.join(data_dir, 'feature_extraction')
fe_model_weight_path = os.path.join(fe_dir, 'fe_model_weight.h5')


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
    # a1, a2, b1, b2
    n_conv_filters = [128, 64, 32, 16, 8, 4]
    n_conv_kernel_sizes = [(2, 23), (2, 19), (2, 15), (2, 11), (2, 7), (2, 3)]
    conv_activation = 'relu'
    n_conv_layer = 6


def get_loss(a, b):
    def mse_mse(y_true, y_pred):
        y_true_temp = (tf.concat([y_true[:, :, 1:, :], y_true[:, :, 0:1, :]], 2)) - y_true
        y_pred_temp = (tf.concat([y_pred[:, :, 1:, :], y_pred[:, :, 0:1, :]], 2)) - y_pred

        loss = losses.mean_squared_error(y_true, y_pred) + losses.mean_squared_error(y_true_temp, y_pred_temp)

        return loss
    
    def matching_loss(y_true, y_pred):
        # # the indexes of a1, a2, b1, b2
        # args = self.regression_model.predict_on_batch(correlation)
        # a and b are the original inputs

        # inverse a or b ??

        a1 = tf.squeeze(a[:, :, y_pred[0], :])
        a2 = tf.squeeze(a[:, :, y_pred[1], :])
        b1 = tf.squeeze(b[:, :, y_pred[2], :])
        b2 = tf.squeeze(b[:, :, y_pred[3], :])

        d = subtract([b1, a1])
        dx = d[0]
        dy = d[1]

        translated_a2 = add([a2, d])

        cos = tf.squeeze(dot([translated_a2 - b1, b2, b1], axes=0, normalize=True))
        sin = tf.sqrt(subtract([1, cos * cos]))
        
        # rotation_m = np.array([
        #     [cos, -sin],
        #     [sin, cos]
        # ])
        # tf_rotation_m = tf.constant(rotation_m, dtype=tf.float32)

        # translation_m = np.array([
        #     [dx],
        #     [dy]
        # ])
        # tf_translation_m = tf.constant(translation_m, dtype=tf.float32)

        affine = np.array([
            [cos, -sin, dx],
            [sin, cos, dy],
            [0, 0, 1]
        ])
        tf_affine = tf.constant(affine, dtype=tf.float32)

        homo = np.ones((1, tf.shape(a)[1].eval()))
        tf_homo = tf.constant(homo, dtype=tf.float32)

        homo_a = concatenate([a, tf_homo], axis=0)
        homo_b = concatenate([b, tf_homo], axis=0)

        new_a_homo = tf.matmul(tf_affine, homo_a)
        new_a = new_a_homo[:, 0:2, :, :]

        new_a1 = tf.squeeze(new_a[:, :, y_pred[0], :])
        new_a2 = tf.squeeze(new_a[:, :, y_pred[1], :])

        pi = tf.constant(3.14159265, dtype=tf.float32)

        # convexity matching cost
        def convexity_cue(contour, i):
            # boundary??
            v1 = tf.squeeze(contour[:, :, i, :])
            v0 = tf.squeeze(contour[:, :, i - 1, :])
            v2 = tf.squeeze(contour[:, :, i + 1, :])
        
            v01 = subtract([v1, v0])
            v21 = subtract([v2, v1])

            zero = tf.constant([[0]], dtype=tf.float32)
            v01 = tf.reshape(tf.concat([v01, zero], axis=0), [3])
            v21 = tf.reshape(tf.concat([v21, zero], axis=0), [3])

            sign = tf.sign(tf.cross(v01, v21)[2])

            theta = tf.acos(cos)

            convexity = multiply([sign, subtract(pi, theta)])

            return convexity
        
        convexity_loss = tf.Variable(1.0, dtype=tf.float32)
        fc_loss = tf.Variable(0, dtype=tf.float32)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        # how to define the bijective function??
        i = tf.constant(y_pred[0], dtype=tf.int32)
        i_last = tf.constant(add(y_pred[1], 1), dtype=tf.int32)
        j = tf.constant(y_pred[2], dtype=tf.int32)
        j_last = tf.constant(add(y_pred[3], 1), dtype=tf.int32)

        while_condition = lambda i, i_last, j, j_last, fc_loss: K.less(i, i_last)

        def while_body(i, i_last, j, j_last, fc_loss):
            main_log.debug('while_body')
            alpha_a = convexity_cue(new_a, i)
            alpha_b = convexity_cue(b, j)

            fc_loss = add(fc_loss, multiply(tf.sign(multiply(alpha_a, alpha_b)), tf.sqrt(tf.abs(multiply(alpha_a, alpha_b)))))
            
            return [add(i, 1), i_last, add(j, 1), j_last, fc_loss]
        
        results = tf.while_loop(while_condition, while_body, [i, i_last, j, j_last, fc_loss])

        fc_loss = results[4]

        convexity_loss = add(convexity_loss, tf.divide(fc_loss, subtract(i_last, i)))

        sess.close()

        return convexity_loss

    return matching_loss


class Shape_matching_model:
    def __init__(self, is_normalized_feature=True, is_normalized_match=True):
        # feature extraction model
        self.fe_model = Feature_extraction_model.build(
            method='self-defined',
            input_shape=(2, None, 1),
            weight_path=fe_model_weight_path,
            is_training=False,
            is_deconv=False
        )

        # a = fe_model.predict_on_batch()
        # b = fe_model.predict_on_batch()

        # self.l2_norm_model = Feature_L2_norm.build()

        # self.correlation_model = Feature_correlation.build()

        # self.regression_model = Feature_regression.build()

        # self.ReLU = 'relu'

        # model = Sequential()
        
        self.is_normalized_feature = is_normalized_feature
        self.is_normalized_match = is_normalized_match

        self.regression_lr = 0.00001

    def run(self, a, b):
        fe_a = self.fe_model.predict_on_batch()
        fe_b = self.fe_model.predict_on_batch()

        # build other model
        if not self.l2_norm_model:
            self.l2_norm_model = Feature_L2_norm.build()

        if not self.regression_model:
            self.regression_model = Feature_regression.build()

        if not self.ReLU:
            self.ReLU = 'relu'

        if self.is_normalized_feature:
            fe_a = self.l2_norm_model.predict_on_batch(fe_a)
            fe_b = self.l2_norm_model.predict_on_batch(fe_b)
        
        # build the correlation model
        if not self.correlation_model:
            self.correlation_model = Feature_correlation.build(fe_a, fe_b)

        correlation = self.correlation_model.predict_on_batch(fe_a, fe_b)

        if self.is_normalized_match:
            correlation = self.l2_norm_model(correlation)
        
        # the indexes of a1, a2, b1, b2
        # args = self.regression_model.predict_on_batch(correlation)

        self.regression_model.compile(loss=get_loss(), optimizer=Adam(lr=self.regression_lr))
        main_log.info('regression model compilation is completed')

        # history = self.regression_model.fit_generator()
