from __future__ import print_function, division
import logging
import numpy as np
import torch
import torch.nn as nn
# from torch.autograd import Variable
# from geotnf.point_tnf import PointTnf


main_log = logging.getLogger('pytorch_main_log')


class ShapeMatchingLoss(nn.Module):
    def __init__(self):
        super(ShapeMatchingLoss, self).__init__()





        # self.geometric_model = geometric_model
        # # define virtual grid of points to be transformed
        # axis_coords = np.linspace(-1,1,grid_size)
        # self.N = grid_size*grid_size
        # X,Y = np.meshgrid(axis_coords,axis_coords)
        # X = np.reshape(X,(1,1,self.N))
        # Y = np.reshape(Y,(1,1,self.N))
        # P = np.concatenate((X,Y),1)
        # self.P = Variable(torch.FloatTensor(P),requires_grad=False)
        # self.pointTnf = PointTnf(use_cuda)
        # if use_cuda:
        #     self.P = self.P.cuda();

    def forward(self, a, b, args):
        # # work!
        # loss = nn.functional.mse_loss(args[0, :2], args[0, 2:])

        # exit()

        # # work!
        # loss1 = torch.add(args[0, 0], args[0, 1])
        # loss2 = torch.add(args[0, 2], args[0, 3])
        # loss = torch.add(loss1, torch.neg(loss2))

        # loss_fn = nn.Conv2d(10, 10, 2, dtype=torch.long)

        main_log.debug('yo')
        # main_log.debug(args.size())

        # can't work!
        # long_args = args.type(torch.long)

        # # work!
        # args = args.squeeze()
        # args = args.mul(43)        
        # # main_log.debug(args)
        # # main_log.debug(long_args)

        loss = torch.tensor(0., requires_grad=True)

        # for i in range(args.size()[0] - 1):
        #     loss = loss.add(args[i].cpu().mul(i))
        #     # loss = loss.add(torch.add(args[i].cpu(), args[i + 1].cpu().pow(2)))

        # new_tensor = torch.Tensor.new_zeros(a.size(), dtype=torch.long)
        # new_tensor = torch.tensor((), dtype=torch.long)
        # new_tensor = new_tensor.new_zeros(a.size())

        # main_log.debug(args)

        # for i in range(int(args.size()[0] / 2)):
        #     new_tensor[args[2 * i].type(torch.long), args[2 * i + 1].type(torch.long)] = 1
        
        # new_tensor = new_tensor.type(torch.float32)
        
        # loss = loss.add(a.cpu().mul(new_tensor).norm())

        # # test if => work!
        # args = args.squeeze()
        # main_log.debug(args)
        # for i in range(args.size()[0]):
        #     args[i] = args[i] if args[i] > 0.2 else 0.
        
        # for i in range(args.size()[0]):
        #     loss = loss.add(args[i].cpu())

        # test element-wise max
        args = args.squeeze()
        a = args[:2]
        b = args[2:]
        c = torch.max(a, b).cpu()
        for i in range(c.size()[0]):
            loss = loss.add(c[i])

        # loss won't be upgraded
        # np_args = args.data.cpu().numpy()
        # np_int_args = np_args.astype(int)

        # a1 = a[:, :, :, np_int_args[0][0]].squeeze().cpu()
        # a2 = a[:, :, :, np_int_args[0][1]].squeeze().cpu()
        # b1 = b[:, :, :, np_int_args[0][2]].squeeze().cpu()
        # b2 = b[:, :, :, np_int_args[0][3]].squeeze().cpu()

        # loss = torch.tensor(0., requires_grad=True)
        # loss = loss.add((a1 + a2 + b1 + b2).norm())


        # # expand grid according to batch size
        # batch_size = theta.size()[0]
        # P = self.P.expand(batch_size,2,self.N)
        # # compute transformed grid points using estimated and GT tnfs
        # if self.geometric_model=='affine':
        #     P_prime = self.pointTnf.affPointTnf(theta,P)
        #     P_prime_GT = self.pointTnf.affPointTnf(theta_GT,P)
        # elif self.geometric_model=='tps':
        #     P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).unsqueeze(3),P)
        #     P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT,P)
        # # compute MSE loss on transformed grid points
        # loss = torch.sum(torch.pow(P_prime - P_prime_GT,2),1)
        # loss = torch.mean(loss)
        return loss