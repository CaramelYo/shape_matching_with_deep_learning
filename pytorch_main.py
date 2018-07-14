from __future__ import print_function, division
import logging
import sys
import os
import numpy as np
import time
import matplotlib.image as mat_img
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_model import FeatureExtraction, CNNGeometric
from pytorch_loss import ShapeMatchingLoss


# for compatibility with Python 2
try:
    input = raw_input
except NameError:
    pass

"""

Script to demonstrate evaluation on a trained model as presented in the CNNGeometric CVPR'17 paper
on the ProposalFlow dataset

"""

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
fr_dir = os.path.join(data_dir, 'feature_regression')
fr_model_weight_path = os.path.join(fr_dir, 'fr_model_weight.h5')

# nn setting
fe_lr = 0.00001
fe_loss = 'mse'
fe_epoch = 2
# batch_size must be 1 because the length of contours are different
fe_batch_size = 1

fr_lr = 0.00001
fr_epoch = 2

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


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def count_data(d):
    return len(os.listdir(d))


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
            # x = load_contour(contour_path)
            x = mat_img.imread(contour_path)

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


def fe_train(epoch, model, loss, optimizer, use_cuda=True):
    model.train()
    train_loss_value = 0
    counter = 0
    total = 1001

    for xs, ys in generate_contour_from_dir(fe_training_dir, fe_batch_size, is_infinite=False, is_name=False):
        xs = np.transpose(xs, (0, 3, 1, 2))
        ys = np.transpose(ys, (0, 3, 1, 2))
        tensor_xs = torch.tensor(xs).to(device)
        tensor_ys = torch.tensor(ys).to(device)

        optimizer.zero_grad()
        # args = model(tensor_xs, tensor_ys)
        # loss_value = loss(tensor_xs, tensor_ys, args)
        pred = model(tensor_xs)
        loss_value = loss(pred, tensor_xs)
        loss_value.backward()
        optimizer.step()

        # train_loss_value += loss_value.data.cpu().numpy()[0]
        train_loss_value += loss_value.data.cpu().numpy()

        main_log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
            epoch, counter, total,
            100. * counter / total, loss_value.data[0]))
    
    train_loss_value /= total

    main_log.info('Train set: Average loss: {:.4f}'.format(train_loss_value))
    return train_loss_value


def fe_test(model, loss, use_cuda=True):
    model.eval()
    test_loss_value = 0
    total = 1001

    for xs, ys in generate_contour_from_dir(fe_validation_dir, fe_batch_size, is_infinite=False, is_name=False):
        # args = model(xs, ys)
        # loss_value = loss(xs, ys, args)
        xs = np.transpose(xs, (0, 3, 1, 2))
        ys = np.transpose(ys, (0, 3, 1, 2))
        tensor_xs = torch.tensor(xs).to(device)
        tensor_ys = torch.tensor(ys).to(device)

        pred = model(tensor_xs)
        loss_value = loss(pred, tensor_xs)
        # test_loss_value += loss_value.data.cpu().numpy()[0]
        test_loss_value += loss_value.data.cpu().numpy()

    test_loss_value /= total
    
    main_log.info('Test set: Average loss: {:.4f}'.format(test_loss_value))
    return test_loss_value


def fr_train(epoch, model, loss, optimizer, use_cuda=True):
    model.train()
    train_loss_value = 0
    counter = 0
    total = 1001

    for xs, ys in generate_contour_from_dir(fe_training_dir, fe_batch_size, is_infinite=False, is_name=False):
        xs = np.transpose(xs, (0, 3, 1, 2))
        ys = np.transpose(ys, (0, 3, 1, 2))
        tensor_xs = torch.tensor(xs).to(device)
        tensor_ys = torch.tensor(ys).to(device)

        optimizer.zero_grad()
        args = model(tensor_xs, tensor_ys)
        loss_value = loss(tensor_xs, tensor_ys, args)
        # pred = model(tensor_xs)
        # loss_value = loss(pred, tensor_xs)
        loss_value.backward()
        optimizer.step()

        # train_loss_value += loss_value.data.cpu().numpy()[0]
        train_loss_value += loss_value.data.cpu().numpy()

        # main_log.debug(loss_value.data[0])

        main_log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
            epoch, counter, total,
            100. * counter / total, loss_value.data))
            # 100. * counter / total, loss_value.data[0, 0]))
    
    train_loss_value /= total

    main_log.info('Train set: Average loss: {:.4f}'.format(train_loss_value))
    # main_log.info('Train set: Average loss: {:.4f}'.format(train_loss_value[0, 0]))
    return train_loss_value


def fr_test(model, loss, use_cuda=True):
    model.eval()
    test_loss_value = 0
    total = 1001

    for xs, ys in generate_contour_from_dir(fe_validation_dir, fe_batch_size, is_infinite=False, is_name=False):
        xs = np.transpose(xs, (0, 3, 1, 2))
        ys = np.transpose(ys, (0, 3, 1, 2))
        tensor_xs = torch.tensor(xs).to(device)
        tensor_ys = torch.tensor(ys).to(device)

        args = model(tensor_xs, tensor_ys)
        loss_value = loss(tensor_xs, tensor_ys, args)
        # pred = model(tensor_xs)
        # loss_value = loss(pred, tensor_xs)
        # test_loss_value += loss_value.data.cpu().numpy()[0]
        test_loss_value += loss_value.data.cpu().numpy()

    test_loss_value /= total
    
    main_log.info('Test set: Average loss: {:.4f}'.format(test_loss_value))
    # main_log.info('Test set: Average loss: {:.4f}'.format(test_loss_value[0, 0]))
    return test_loss_value


if __name__ == '__main__':
    # use argparser??
    t0 = time.time()

    if len(sys.argv) >= 2:
        if sys.argv[1] == 'fe-train':
            n_training = count_data(fe_training_dir)
            n_validation = count_data(fe_validation_dir)
            main_log.info('n_training = %d \nn_validation = %d' % (n_training, n_validation))

            # Create model
            main_log.info('create feature extraction model...')
            # model = FeatureExtraction(model_name='self-defined', weight_path=fe_model_weight_path, is_deconv=True, trainable=True, use_cuda=use_cuda)
            model = FeatureExtraction(model_name='resnet152', trainable=True, use_cuda=use_cuda)

            # loss function
            loss = nn.MSELoss()

            # optimizer
            optimizer = optim.Adam(model.parameters(), lr=fe_lr)

            best_test_loss_value = float('inf')

            main_log.info('start training')

            for epoch in range(fe_epoch):
                train_loss_value = fe_train(epoch, model, loss, optimizer, use_cuda=use_cuda)
                test_loss_value = fe_test(model, loss, use_cuda=use_cuda)

                # save the weight with best loss value
                if test_loss_value < best_test_loss_value:
                    torch.save(model.state_dict(), fe_model_weight_path)

                    best_test_loss_value = test_loss_value

            main_log.info('training is completed')
        elif sys.argv[1] == 'fr-train':
            n_training = count_data(fe_training_dir)
            n_validation = count_data(fe_validation_dir)
            main_log.info('n_training = %d \nn_validation = %d' % (n_training, n_validation))

            # Create model
            main_log.info('Creating CNN model...')
            model = CNNGeometric(fe_model_weight_path, fr_model_weight_path, use_cuda=use_cuda)
            
            # loss function
            loss = ShapeMatchingLoss()
            
            # optimizer
            optimizer = optim.Adam(model.fr_model.parameters(), lr=fr_lr)
            
            best_test_loss_value = float('inf')

            main_log.info('start training')

            for epoch in range(fr_epoch):
                train_loss_value = fr_train(epoch, model, loss, optimizer, use_cuda=use_cuda)
                test_loss_value = fr_test(model, loss, use_cuda=use_cuda)

                # save the weight with best loss value
                if test_loss_value < best_test_loss_value:
                    # torch.save(model.state_dict(), fr_model_weight_path)
                    
                    best_test_loss_value = test_loss_value

            main_log.info('training is completed')

    main_log.info('time cost = %s' % (time.time() - t0))
    del t0
