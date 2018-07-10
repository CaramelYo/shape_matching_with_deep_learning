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
        main_log.debug(args)

        a1 = a[:, :, :, args[0]]
        a2 = a[:, :, :, args[1]]
        b1 = b[:, :, :, args[2]]
        b2 = b[:, :, :, args[3]]

        main_log.debug(a1)
        main_log.debug(a2)
        main_log.debug(b1)
        main_log.debug(b2)


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