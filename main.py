import logging
import sys
import os
import time
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import losses
import keras.backend as K
import tensorflow as tf

from model import Feature_extraction_model

# global variable

# data
data_dir = 'data'
fe_dir = os.path.join(data_dir, 'feature_extraction')
fe_training_dir = os.path.join(fe_dir, 'training')
fe_validation_dir = os.path.join(fe_dir, 'validation')
fe_test_dir = os.path.join(fe_dir, 'test')
fe_pred_dir = os.path.join(fe_dir, 'pred')
fe_tb_model_log_dir = os.path.join(fe_dir, 'tb_model_log')
fe_model_weight_path = os.path.join(fe_dir, 'fe_model_weight.h5')

# nn setting
fe_lr = 0.00001
fe_loss = 'mse'
fe_epo = 3000
# batch_size must be 1 because the length of contours are different
fe_batch_size = 1

# logging setting
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    handlers=[logging.FileHandler('main.log', 'w', 'utf-8'), ])
main_log = logging.getLogger('main_log')

# log message to stdout
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(message)s'))
main_log.addHandler(console)

# # redirect the stdout and stderr
# stdout_log_path = 'stdout.log'
# stdout_log = open(stdout_log_path, 'w')
# sys.stdout = stdout_log

# stderr_log_path = 'stderr.log'
# stderr_log = open(stderr_log_path, 'w')
# sys.stderr = stderr_log


def count_data(d):
    count = 0

    for data_name in os.listdir(d):
        count += 1

    return count


def load_contour(path):
    if os.path.isfile(path):
        with open(path, 'r') as f:
            line = f.readline()
            line_split = line.strip().split(' ')
            x = np.array([[(float)(line_split[0])], [(float)(line_split[1])]], dtype=np.float32)
            n = 1

            for line in f:
                line_split = line.strip().split(' ')
                x = np.concatenate((x, np.array([[(float)(line_split[0])], [(float)(line_split[1])]], dtype=np.float32)), axis=1)
                n += 1

        x = x.reshape(2, n, 1)
        return x
    else:
        main_log.error('load_contour error: path %s is not a file' % path)
        exit()


def generate_contour_from_dir(d, batch_size, is_infinite=True, is_name=False):
    while True:
        batch_counter = 0
        xs = []
        ys = []

        for contour_name in os.listdir(d):
            contour_path = os.path.join(d, contour_name)
            x = load_contour(contour_path)

            # the input data and ground truth are the same beacause it's an auto-encoder
            xs.append(x)
            ys.append(x)

            del contour_path, x

            # output the input data and ground truth
            batch_counter += 1
            if batch_counter >= batch_size:
                xs = np.array(xs, dtype=np.float32)
                ys = np.array(ys, dtype=np.float32)

                if is_name:
                    yield(xs, ys, contour_name)
                else:
                    yield(xs, ys)

                del batch_counter, xs, ys
                batch_counter = 0
                xs = []
                ys = []

        if len(ys) != 0:
            xs = np.array(xs, dtype=np.float32)
            ys = np.array(ys, dtype=np.float32)
            yield(xs, ys)

        del batch_counter, xs, ys

        if not is_infinite:
            break


def get_loss():
    def binary_crossentropy(y_true, y_pred):
        # result = []
        # y_pred_shape = K.shape(y_pred)
        # main_log.debug(y_pred_shape)
        # y_pred_len = y_pred.get_shape().as_list()
        # main_log.debug(y_pred_len)

        # for i in range(y_pred_len):
        #     y_pred[i] = [max(min(x, 1 - K.epsilon()), K.epsilon()) for x in y_pred[i]]
        #     result.append(-np.mean([y_true[i][j] * K.log(y_pred[i][j]) + (1 - y_true[i][j]) * K.log(1 - y_pred[i][j]) for j in range(len(y_pred[i]))]))
        # return np.mean(result)
        i = tf.constant(0, dtype=tf.int32)
        loss = tf.Variable(0, dtype=tf.float32)
        while_condition = lambda i, loss: K.less(i, tf.shape(y_true)[2])

        main_log.debug(i)

        def while_body(i, loss):
            main_log.debug('yo')
            return [tf.add(i, 1), tf.add(loss, 1)]
        
        r = tf.while_loop(while_condition, while_body, [i, loss])

        main_log.debug(r)
        # main_log.debug(r[1].eval())
        return r[1]
    
    def mse_mse(y_true, y_pred):
        y_true_temp = (tf.concat([y_true[:, :, 1:, :], y_true[:, :, 0:1, :]], 2)) - y_true
        y_pred_temp = (tf.concat([y_pred[:, :, 1:, :], y_pred[:, :, 0:1, :]], 2)) - y_pred

        loss = losses.mean_squared_error(y_true, y_pred) + losses.mean_squared_error(y_true_temp, y_pred_temp)

        return loss
    
    return mse_mse


if __name__ == '__main__':
    # use argparser??
    t0 = time.time()

    if len(sys.argv) >= 2:
        if sys.argv[1] == 'fe-train':
            n_training = count_data(fe_training_dir)
            n_validation = count_data(fe_validation_dir)
            main_log.info('n_training = %d \nn_validation = %d' % (n_training, n_validation))

            model = Feature_extraction_model.build(
                method='self-defined',
                input_shape=(2, None, 1),
                weight_path=fe_model_weight_path,
                is_training=True,
                is_deconv=True
            )
            main_log.info('building model is completed')

            # model.compile(loss=fe_loss, optimizer=Adam(lr=fe_lr))
            model.compile(loss=get_loss(), optimizer=Adam(lr=fe_lr))
            main_log.info('model compilation is completed')

            # is it need to generate the ground truth data in validation_data part??
            history = model.fit_generator(
                generate_contour_from_dir(fe_training_dir, fe_batch_size),
                steps_per_epoch=n_training / fe_batch_size,
                epochs=fe_epo,
                validation_data=generate_contour_from_dir(fe_training_dir, fe_batch_size),
                validation_steps=n_validation / fe_batch_size,
                callbacks=[
                    TensorBoard(log_dir=fe_tb_model_log_dir, batch_size=fe_batch_size),
                    ModelCheckpoint(fe_model_weight_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)
                ]
            )

            # for ele in history.history:
            #     main_log.debug('%s: %s' % (ele, history.history[ele]))

            history_str = ''
            for kind in history.history:
                main_log.debug('%s :' % kind)
                for i in range(len(history.history[kind])):
                    # main_log.debug('%d th : %s' % (i, history.history[kind][i]))
                    history_str += '%d th : %s\n' % (i, history.history[kind][i])
                
                main_log.debug(history_str)

            # # reopen those std file
            # sys.stdout.close()
            # stdout_log = open(stdout_log_path, 'a')
            # sys.stdout = stdout_log

            # sys.stderr.close()
            # stderr_log = open(stderr_log_path, 'a')
            # sys.stderr = stderr_log

            del n_training, n_validation, model
        elif sys.argv[1] == 'fe-test':
            n_test = count_data(fe_test_dir)
            main_log.info('n_test = %d' % n_test)

            model = Feature_extraction_model.build(
                method='self-defined',
                input_shape=(2, None, 1),
                weight_path=fe_model_weight_path,
                is_training=False,
                is_deconv=True
            )
            main_log.info('building model is completed')

            model.compile(loss=fe_loss, optimizer=Adam(lr=fe_lr))

            # test the loss
            scores = model.evaluate_generator(
                generate_contour_from_dir(fe_test_dir, fe_batch_size),
                steps=n_test / fe_batch_size,
                verbose=1
            )

            main_log.info('scores = %s' % scores)

            # output the prediction
            if not os.path.exists(fe_pred_dir):
                logging.info('create the dir = %s' % fe_pred_dir)
                os.mkdir(fe_pred_dir)

            for xs, ys, name in generate_contour_from_dir(fe_test_dir, fe_batch_size, is_infinite=False, is_name=True):
                preds = model.predict_on_batch(xs)

                # post processing

                # get rid of those dim = 1
                preds = np.squeeze(preds)

                f = open(os.path.join(fe_pred_dir, name), 'w')
                for i in range(preds.shape[1]):
                    f.write('%f %f\n' % (preds[0][i], preds[1][i]))
                f.close()

                del preds, f

            del n_test, model, scores

    main_log.info('time cost = %s' % (time.time() - t0))
    del t0
