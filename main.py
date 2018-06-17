import logging
import sys
import os
import time
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

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
fe_epo = 2
fe_batch_size = 1

# logging setting
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    handlers=[logging.FileHandler('main.log', 'w', 'utf-8'), ])
main_log = logging.getLogger('main_log')

# log message to stdout
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
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


def generate_contour_from_file(d, batch_size, is_infinite=True):
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


if __name__ == '__main__':
    # use argparser??
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'fe-train':
            t0 = time.time()

            n_training = count_data(fe_training_dir)
            n_validation = count_data(fe_validation_dir)
            main_log.info('n_training = %d \nn_validation = %d' % (n_training, n_validation))

            model = Feature_extraction_model.build(method='self', input_shape=(2, None, 1), weight_path=fe_model_weight_path, is_training=True, is_deconv=True)
            main_log.info('build model completed')

            adam = Adam(lr=fe_lr)

            model.compile(loss=fe_loss, optimizer=adam)

            # is it need to generate the ground truth data in validation_data part??
            history = model.fit_generator(
                generate_contour_from_file(fe_training_dir, fe_batch_size),
                steps_per_epoch=n_training / fe_batch_size,
                epochs=fe_epo,
                validation_data=generate_contour_from_file(fe_training_dir, fe_batch_size),
                validation_steps=n_validation / fe_batch_size,
                callbacks=[
                    TensorBoard(log_dir=fe_tb_model_log_dir, batch_size=fe_batch_size),
                    ModelCheckpoint(fe_model_weight_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)
                ]
            )

            for ele in history.history:
                main_log.debug('%s: %s' % (ele, history.history[ele]))

            # # reopen those std file
            # sys.stdout.close()
            # stdout_log = open(stdout_log_path, 'a')
            # sys.stdout = stdout_log

            # sys.stderr.close()
            # stderr_log = open(stderr_log_path, 'a')
            # sys.stderr = stderr_log

        main_log.info('training time = %s' % (time.time() - t0))

        del t0, n_training, n_validation, model, adam
