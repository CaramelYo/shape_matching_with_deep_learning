from __future__ import print_function, division
import logging
import sys
import os
from os import path as osp
import numpy as np
import time
import matplotlib.image as mat_img
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_model import FeatureExtraction, CNNGeometric, PrModel
from pytorch_loss import ShapeMatchingLoss
import cv2 as cv


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

fe_dir = osp.join(data_dir, 'feature_extraction')
fe_training_dir = osp.join(fe_dir, 'training')
fe_validation_dir = osp.join(fe_dir, 'validation')
fe_test_dir = osp.join(fe_dir, 'test')
fe_pred_dir = osp.join(fe_dir, 'pred')
fe_tb_model_log_dir = osp.join(fe_dir, 'tb_model_log')
fe_model_weight_path = osp.join(fe_dir, 'fe_model_weight.h5')

fr_dir = osp.join(data_dir, 'feature_regression')
fr_training_dir = osp.join(fr_dir, 'training')
fr_model_weight_path = osp.join(fr_dir, 'fr_model_weight.h5')

fr_result_dir = osp.join(fr_dir, 'results')


# nn setting
fe_lr = 0.00001
fe_loss = 'mse'
fe_epoch = 2
# batch_size must be 1 because the length of contours are different
fe_batch_size = 1

fr_lr = 0.00001
fr_epoch = 10000

img_size = (512, 512, 3)

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

if use_cuda:
    main_log.info('torch.cuda.FloatTensor')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # main_log.info('torch.cuda.DoubleTensor')
    # torch.set_default_tensor_type('torch.cuda.DoubleTensor')


def count_data(d):
    return len(os.listdir(d))


def load_contour(path):
    if osp.isfile(path):
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


def generate_contour_img_from_dir(d, batch_size, is_infinite=True, is_name=False):
    while True:
        batch_counter = 0
        xs = []
        ys = []

        for contour_name in os.listdir(d):
            contour_path = osp.join(d, contour_name)
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


def generate_contour_data_from_dir(d, mode=2, is_infinite=True):
    def get_contour_data(d_path):
        for c_name in os.listdir(d_path):
            c_path = osp.join(d_path, c_name)

            # a = torch.tensor([], dtype=torch.float32, requires_grad=True)
            # a = torch.tensor([], dtype=torch.float32)
            a = torch.tensor([], requires_grad=True)
            with open(c_path, 'r') as f:
                for line in f.readlines():
                    line_split = line.split(' ')
                    # xy = torch.tensor([float(line_split[0].strip()), float(line_split[1].strip())], dtype=torch.float32).view(2, 1)
                    xy = torch.tensor([float(line_split[0].strip()), float(line_split[1].strip())]).view(2, 1)
                    a = torch.cat([a, xy], dim=1)

            yield a, c_name

    while True:
        if mode == 1:
            for a, name in get_contour_data(d):
                yield a, name
        elif mode == 2:
            for pair_dir in os.listdir(d):
                pair_path = osp.join(d, pair_dir)
                # ab = torch.tensor([], dtype=torch.float32, requires_grad=True)
                # ab = torch.tensor([], dtype=torch.float32)
                ab = torch.tensor([], requires_grad=True)
                ab_names = []

                for a, name in get_contour_data(pair_path):
                    ab = torch.cat([ab, a], dim=0)
                    ab_names.append(name)

                yield ab[0:2], ab[2:4], pair_dir, ab_names

                del pair_path, ab, ab_names

        #     for contour_dir in os.listdir(d):
        #         contour_dir_path = osp.join(d, contour_dir)
        #         ab = torch.tensor([], dtype=torch.float32)
        #         ab_names = []

        #         for contour_name in os.listdir(contour_dir_path):
        #             contour_path = osp.join(contour_dir_path, contour_name)

        #             a = torch.tensor([], dtype=torch.float32)
        #             with open(contour_path, 'r') as f:
        #                 for line in f.readlines():
        #                     line_split = line.split(' ')
        #                     xy = torch.tensor([float(line_split[0].strip()), float(line_split[1].strip())], dtype=torch.float32).view(2, 1)
        #                     a = torch.cat([a, xy], dim=1)

        #             ab = torch.cat([ab, a], dim=0)
        #             ab_names.append(contour_name)

        #         if is_name:
        #             yield ab[0:2], ab[2:4], contour_dir, ab_names
        #         else:
        #             yield ab[0:2], ab[2:4]

        #         del contour_dir_path, ab, contour_path, a, line_split, xy

        if not is_infinite:
            break


# def fe_train(epoch, model, loss, optimizer, use_cuda=True):
#     model.train()
#     train_loss_value = 0
#     counter = 0
#     total = 1001

#     for xs, ys in generate_contour_img_from_dir(fe_training_dir, fe_batch_size, is_infinite=False, is_name=False):
#         xs = np.transpose(xs, (0, 3, 1, 2))
#         ys = np.transpose(ys, (0, 3, 1, 2))
#         tensor_xs = torch.tensor(xs).to(device)
#         tensor_ys = torch.tensor(ys).to(device)

#         optimizer.zero_grad()
#         # args = model(tensor_xs, tensor_ys)
#         # loss_value = loss(tensor_xs, tensor_ys, args)
#         pred = model(tensor_xs)
#         loss_value = loss(pred, tensor_xs)
#         loss_value.backward()
#         optimizer.step()

#         # train_loss_value += loss_value.data.cpu().numpy()[0]
#         train_loss_value += loss_value.data.cpu().numpy()

#         main_log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
#             epoch, counter, total,
#             100. * counter / total, loss_value.data[0]))
    
#     train_loss_value /= total

#     main_log.info('Train set: Average loss: {:.4f}'.format(train_loss_value))
#     return train_loss_value


def tensor_to_numpy(t, use_cuda=True):
    return t.data.cpu().numpy() if use_cuda else t.data.numpy()


def fe_train(epoch, model, loss_fn, optimizer, is_output=False, use_cuda=True):
    model.train()
    train_loss = 0

    counter = 0
    total_count = count_data(fe_training_dir)

    for xs, name in generate_contour_data_from_dir(fe_training_dir, mode=1, is_infinite=False):
        h, w = xs.shape
        xs = xs.view(1, 1, h, w)
        
        optimizer.zero_grad()
        pred = model(xs)
        # xs == ys in auto-encoder
        loss = loss_fn(pred, xs)
        loss.backward()
        optimizer.step()
        train_loss += tensor_to_numpy(loss, use_cuda=use_cuda)

        counter += 1
        # print('train epoch : {} [{} / {} ({:.0f}%)]\tloss : {:.6f}'.format(
        #     epoch, counter, total_count, 100. * counter / total_count,
        #     loss.data))

        if is_output:
            output_contour(xs, name, epoch=epoch, use_cuda=use_cuda)

    train_loss /= total_count

    print('train set : average loss : {:.4f}'.format(train_loss))
    return train_loss

# def fe_test(model, loss, use_cuda=True):
#     model.eval()
#     test_loss_value = 0
#     total = 1001

#     for xs, ys in generate_contour_img_from_dir(fe_validation_dir, fe_batch_size, is_infinite=False, is_name=False):
#         # args = model(xs, ys)
#         # loss_value = loss(xs, ys, args)
#         xs = np.transpose(xs, (0, 3, 1, 2))
#         ys = np.transpose(ys, (0, 3, 1, 2))
#         tensor_xs = torch.tensor(xs).to(device)
#         tensor_ys = torch.tensor(ys).to(device)

#         pred = model(tensor_xs)
#         loss_value = loss(pred, tensor_xs)
#         # test_loss_value += loss_value.data.cpu().numpy()[0]
#         test_loss_value += loss_value.data.cpu().numpy()

#     test_loss_value /= total
    
#     main_log.info('Test set: Average loss: {:.4f}'.format(test_loss_value))
#     return test_loss_value


def fe_test(model, d, loss_fn, is_output=False, use_cuda=True):
    model.eval()
    test_loss = 0

    total_count = count_data(d)

    for xs, name in generate_contour_data_from_dir(d, mode=1, is_infinite=False):
        h, w = xs.shape
        xs = xs.view(1, 1, h, w)

        pred = model(xs)
        loss = loss_fn(pred, xs)
        test_loss += tensor_to_numpy(loss, use_cuda=use_cuda)

        if is_output:
            output_contour(xs, name, use_cuda=use_cuda)

    test_loss /= total_count

    print('test set : average loss : {:.4f}'.format(test_loss))
    return test_loss


def output_contour(xs, name, epoch=-1, use_cuda=use_cuda):
    # contour data
    with open(osp.join(fe_pred_dir, str(epoch) + '_' + name), 'w') as f:
        b, c, h, w = xs.shape

        for i in range(w):
            f.write('%f %f\n' % (xs[0, 0, 0, i], xs[0, 0, 1, i]))

    # contour img
    img = np.ones(img_size, dtype=np.uint8)
    for y in range(img_size[0]):
        for x in range(img_size[1]):
            img[y, x, :] = 255

    # np_xs = xs.view(h, w).data.cpu().numpy() if use_cuda else xs.view(h, w).data.numpy()
    np_xs = tensor_to_numpy(xs.view(h, w))

    for i in range(w):
        y = int(np_xs[1, i] * img_size[0])
        x = int(np_xs[0, i] * img_size[1])

        img[y, x, :] = [0, 0, 0]

    n = name.split('.')[0]
    cv.imwrite(osp.join(fe_pred_dir, str(epoch) + '_' + n + '.png'), img)

# def fr_train(epoch, model, loss, optimizer, use_cuda=True):
#     model.train()
#     train_loss_value = 0
#     counter = 0
#     total = 1001

#     for xs, ys in generate_contour_img_from_dir(fe_training_dir, fe_batch_size, is_infinite=False, is_name=False):
#         xs = np.transpose(xs, (0, 3, 1, 2))
#         ys = np.transpose(ys, (0, 3, 1, 2))
#         tensor_xs = torch.tensor(xs).to(device)
#         tensor_ys = torch.tensor(ys).to(device)

#         optimizer.zero_grad()
#         args = model(tensor_xs, tensor_ys)
#         loss_value = loss(tensor_xs, tensor_ys, args)
#         # pred = model(tensor_xs)
#         # loss_value = loss(pred, tensor_xs)
#         loss_value.backward()
#         optimizer.step()

#         # train_loss_value += loss_value.data.cpu().numpy()[0]
#         train_loss_value += loss_value.data.cpu().numpy()

#         # main_log.debug(loss_value.data[0])

#         main_log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
#             epoch, counter, total,
#             100. * counter / total, loss_value.data))
#             # 100. * counter / total, loss_value.data[0, 0]))
    
#     train_loss_value /= total

#     main_log.info('Train set: Average loss: {:.4f}'.format(train_loss_value))
#     # main_log.info('Train set: Average loss: {:.4f}'.format(train_loss_value[0, 0]))
#     return train_loss_value


def fr_train(epoch, model, loss_fn, optimizer, use_cuda=True):
    model.train()
    train_loss = 0

    counter = 0
    total_count = count_data(fr_training_dir)

    for a, b, d_name, ab_names in generate_contour_data_from_dir(fr_training_dir, is_infinite=False):
        h, w = a.shape
        
        a = a.view(1, 1, h, w).contiguous()
        b = b.view(1, 1, h, w).contiguous()

        optimizer.zero_grad()
        pr_a, pr_b = model(a, b)

        # loss = loss_fn(pr_a, pr_b, a, b)
        loss, selected_a, selected_b = loss_fn(pr_a, pr_b, a, b)

        loss.backward()

        # print(model.mps_model.model[0][1].weight)
        # print('grad = ')
        # print(model.mps_model.model[0][1].weight.grad)
        # exit()

        # print('convexity_loss grad =')
        # print(convexity_loss.grad)

        # print('matching_length_cost grad =')
        # print(matching_length_cost.grad)

        optimizer.step()

        # output the info
        train_loss += tensor_to_numpy(loss, use_cuda=use_cuda)

        counter += 1
        main_log.info('train epoch : {} [{} / {} ({:.0f}%)]\tloss : {:.6f}'.format(
            epoch, counter, total_count, 100. * counter / total_count,
            loss.data[0]
        ))

        # output the result
        # selected_a, selected_b, wa, wb = ShapeMatchingLoss.get_contours(pr_a, pr_b, a, b)

        # need?
        # transformed_selected_a, selected_b, wab = ShapeMatchingLoss.bijective(transformed_selected_a, selected_b, wa, wb)

        # a = tensor_to_numpy(a, use_cuda=use_cuda)
        # b = tensor_to_numpy(b, use_cuda=use_cuda)
        # selected_a = tensor_to_numpy(selected_a, use_cuda=use_cuda)
        # selected_b = tensor_to_numpy(selected_b, use_cuda=use_cuda)

        d_path = osp.join(fr_result_dir, d_name)
        if not osp.isdir(d_path):
            os.mkdir(d_path)

        color = np.array([0, 0, 255], dtype=np.uint8)

        def output_img(c, selected_c, wc, name):
            img = np.zeros(img_size, dtype=np.uint8)
            for y in range(img_size[0]):
                for x in range(img_size[1]):
                    img[y, x, :] = 255

            # print('output_img c.shape = ' + str(c.shape))
            # print('output_img selected_c.shape = ' + str(selected_c.shape))

            for i in range(c.shape[3]):
                y = int(c[0, 0, 1, i] * img_size[0])
                x = int(c[0, 0, 0, i] * img_size[1])

                img[y, x, :] = 0
                # img[c[1, i], c[0, i], :] = 0

            n = name.split('.')[0]
            if osp.isfile(osp.join(d_path, 'origin_' + n + '.png')):
                cv.imwrite(osp.join(d_path, 'origin_' + n + '.png'), img)

            for i in range(wc):
                y = int(selected_c[1, i] * img_size[0])
                x = int(selected_c[0, i] * img_size[1])

                img[y, x, :] = color

            cv.imwrite(osp.join(d_path, str(epoch) + '_matching_' + n + '.png'), img)

        output_img(a, selected_a, selected_a.shape[1], ab_names[0])
        output_img(b, selected_b, selected_b.shape[1], ab_names[1])

    train_loss /= total_count
    
    main_log.info('train set : average loss : {:.4f}'.format(train_loss.data[0]))
    return train_loss


def fr_test(model, loss, use_cuda=True):
    model.eval()
    test_loss_value = 0
    total = 1001

    for xs, ys in generate_contour_img_from_dir(fe_validation_dir, fe_batch_size, is_infinite=False, is_name=False):
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
            model = FeatureExtraction(model_name='self-defined', weight_path=fe_model_weight_path, is_deconv=True, trainable=True, use_cuda=use_cuda)
            # model = FeatureExtraction(model_name='resnet152', trainable=True, use_cuda=use_cuda)

            # loss function
            loss_fn = nn.MSELoss()

            # optimizer
            optimizer = optim.Adam(model.parameters(), lr=fe_lr)

            best_test_loss = float('inf')

            main_log.info('start training')

            is_output = True
            if is_output and not osp.isdir(fe_pred_dir):
                os.mkdir(fe_pred_dir)

            for epoch in range(fe_epoch):
                train_loss = fe_train(epoch, model, loss_fn, optimizer, is_output=is_output, use_cuda=use_cuda)
                test_loss = fe_test(model, fe_validation_dir, loss_fn, is_output=is_output, use_cuda=use_cuda)

                # save the weight with best loss value
                if test_loss < best_test_loss:
                    torch.save(model.state_dict(), fe_model_weight_path)
                    best_test_loss = test_loss

            main_log.info('training is completed')
        elif sys.argv[1] == 'fe-test':
            n_test = count_data(fe_test_dir)
            main_log.info('n_test = %d' % n_test)

            if not osp.isdir(fe_pred_dir):
                os.mkdir(fe_pred_dir)

            # create the model
            main_log.info('create feature extraction model...')
            model = FeatureExtraction(model_name='self-defined', weight_path=fe_model_weight_path, is_deconv=True, use_cuda=use_cuda)

            # loss function
            loss_fn = nn.MSELoss()

            main_log.info('start testing')

            is_output = True
            if is_output and not osp.isdir(fe_pred_dir):
                os.mkdir(fe_pred_dir)

            test_loss = fe_test(model, fe_test_dir, loss_fn, is_output=is_output, use_cuda=use_cuda)

            main_log.info('test is completed')
        elif sys.argv[1] == 'fr-train':
            n_training = count_data(fe_training_dir)
            n_validation = count_data(fe_validation_dir)
            main_log.info('n_training = %d \nn_validation = %d' % (n_training, n_validation))

            # Create model
            main_log.info('Creating CNN model...')
            model = CNNGeometric(fe_model_weight_path, fr_model_weight_path, use_cuda=use_cuda)
            
            # loss function
            loss = ShapeMatchingLoss(use_cuda=use_cuda)
            
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
        elif sys.argv[1] == 'fr-train-test':
            n_training = count_data(fr_training_dir)
            # n_validation = count_data(fr_validation_dir)
            # main_log.info('n_training = %d \nn_validation = %d' % (n_training, n_validation))
            main_log.info('n_training = %d' % n_training)

            # # test
            # test_sum = torch.zeros(1, dtype=torch.float32, requires_grad=True)
            # test_sum = test_sum + torch.tensor([5], dtype=torch.float32, requires_grad=True)
            # test_sum = test_sum * torch.tensor([10], dtype=torch.float32, requires_grad=True)
            # print(test_sum)
            # print(test_sum.grad)
            # exit()

            if not osp.isdir(fr_result_dir):
                os.mkdir(fr_result_dir)

            # create model
            main_log.info('create self-defined model')
            model = PrModel(fe_model_weight_path, use_cuda=use_cuda)

            # loss function
            loss_fn = ShapeMatchingLoss(use_cuda=use_cuda)

            # optimizer
            # optimizer = optim.Adam(model.parameters(), lr=fr_lr)
            optimizer = optim.Adam(model.mps_model.parameters(), lr=fr_lr)

            best_loss = float('inf')

            main_log.info('start training')

            for epoch in range(fr_epoch):
                train_loss = fr_train(epoch, model, loss_fn, optimizer, use_cuda=use_cuda)

                # save the weight with best loss
                if train_loss < best_loss:
                    # torch.save(model.state_dict(), fr_model_weight_path)
                    torch.save(model.mps_model.state_dict(), fr_model_weight_path)
                    best_loss = train_loss

            main_log.info('training is completed')

    main_log.info('time cost = %s' % (time.time() - t0))
    del t0
