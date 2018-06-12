import logging

from keras.layers import Input, Conv2D
# from keras.applications.resnet50 import ResNet50
# from keras.initializer import Constant


class Feature_extraction_model:
    def build(
        method='resnet', input_tensor=None, input_shape=None, weight_path=None,
        is_training=False, use_cuda=True
    ):

        if method == 'resnet':
            # ResNet50(include_top = False, weights = 'imagenet', input_tensor = inputs)
            logging.info('no implementation')
        else:
            # self-defined feature extraction
            if not input_tensor:
                
                if not input_shape:
                    logging.error('input_shape is None when input_tensor is None')
                    return
                
                # input_shape = (h, w, ch). but in our case, it should be (2, n)
                input_tensor = Input(shape=input_shape)
            
            model = input_tensor
            
            # build those conv layers
            model = Conv2D(
                filters=512, kernel_size=(2, 2), strides=(1, 1), padding='same',
                activation=Feature_extraction_model.conv_activation, name='conv1')(model)

    conv_activation = 'relu'


# class Shape_matching_model:
#     def build(self, weight_path = None, is_training = False):