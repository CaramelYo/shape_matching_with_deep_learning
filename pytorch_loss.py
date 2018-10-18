from __future__ import print_function, division
import logging
import numpy as np
import torch
import torch.nn as nn
# from torch.autograd import Variable
# from geotnf.point_tnf import PointTnf


main_log = logging.getLogger('pytorch_main_log')


def tensor_to_numpy(t, use_cuda=True):
    return t.data.cpu().numpy() if use_cuda else t.data.numpy()


class ShapeMatchingLoss(nn.Module):
# class ShapeMatchingLoss(torch.autograd.Function):
    def __init__(self, use_cuda=True):
        super(ShapeMatchingLoss, self).__init__()
        self.use_cuda = use_cuda

    def forward(self, pr_a, pr_b, input_a, input_b):
        # # there are grads in loss value
        # print(pr_a.requires_grad)
        # print(pr_b.requires_grad)
        # print(pr_a.add(-pr_b).sum())

        # return pr_a.add(-pr_b).sum()

        # # test input
        # return pr_a.sum() + pr_b.sum() - input_a.sum() - input_b.sum()

        # print(input_a.shape)

        # max_pr_a, max_pr_a_idx = pr_a.topk(2, dim=3)
        # max_pr_b, max_pr_b_idx = pr_b.topk(2, dim=3)
        
        # max_pr_a = max_pr_a.squeeze()
        # max_pr_b = max_pr_b.squeeze()
        # max_pr_a_idx = max_pr_a_idx.squeeze()
        # max_pr_b_idx = max_pr_b_idx.squeeze()

        # print(max_pr_a.shape)
        # print(max_pr_a_idx.shape)

        # a0_idx = max_pr_a[0] * 1500
        # a1_idx = max_pr_a[1] * 1500
        # b0_idx = max_pr_b[0] * 1500
        # b1_idx = max_pr_b[1] * 1500

        # # a0_idx = max_pr_a_idx[0]
        # # a1_idx = max_pr_a_idx[1]
        # # b0_idx = max_pr_b_idx[0]
        # # b1_idx = max_pr_b_idx[1]

        # print('a0_idx = ' + str(a0_idx))
        # print('a1_idx = ' + str(a1_idx))
        # print('b0_idx = ' + str(b0_idx))
        # print('b1_idx = ' + str(b1_idx))
        # print(a0_idx.device)

        # # a0_idx = a0_idx.type(torch.LongTensor).cuda()
        # # a1_idx = a1_idx.type(torch.LongTensor).cuda()
        # # b0_idx = b0_idx.type(torch.LongTensor).cuda()
        # # b1_idx = b1_idx.type(torch.LongTensor).cuda()

        # print('a0_idx = ' + str(a0_idx))
        # print('a1_idx = ' + str(a1_idx))
        # print('b0_idx = ' + str(b0_idx))
        # print('b1_idx = ' + str(b1_idx))

        # # r = torch.pow((input_a[:, :, :, a0_idx] - input_b[:, :, :, b0_idx]).sum(), 2) + torch.pow((input_a[:, :, :, a1_idx] - input_b[:, :, :, b1_idx]).sum(), 2)
        # r = torch.pow(a0_idx - b0_idx, 2) + torch.pow(a1_idx - b1_idx, 2)
        # print(r)

        # return r

        # test 4 arguments
        # batch and channel size will be 1
        # b, c, h, w = pr_a.size()

        def loss_fn(pr):
            # get the indices
            a_0 = pr[0, 0, 0, 0]
            a_1 = pr[0, 0, 0, 1]
            b_0 = pr[0, 0, 0, 2]
            b_1 = pr[0, 0, 0, 3]

            def swap(x, y, msg=None):
                if x > y:
                    t = y
                    y = x
                    x = t

                    if not msg:
                        print(msg)
                
                return x, y

            # is this acceptable to swap the values of pr ???
            a_0, a_1 = swap(a_0, a_1, 'a swaps')
            b_0, b_1 = swap(b_0, b_1, 'b swaps')
            
            # if pr[0, 0, 0, 0] <= pr[0, 0, 0, 1]:
            #     a_0 = pr[0, 0, 0, 0]
            #     a_1 = pr[0, 0, 0, 1]
            # else:
            #     a_0 = pr[0, 0, 0, 1]
            #     a_1 = pr[0, 0, 0, 0]
            
            # if pr[0, 0, 0, 2] <= pr[0, 0, 0, 3]:
            #     b_0 = pr[0, 0, 0, 2]
            #     b_1 = pr[0, 0, 0, 3]
            # else:
            #     b_0 = pr[0, 0, 0, 3]
            #     b_1 = pr[0, 0, 0, 2]
            
            # print('a_0 = ' + str(a_0))
            # print('a_1 = ' + str(a_1))
            # print('b_0 = ' + str(b_0))
            # print('b_1 = ' + str(b_1))

            # # compose the index pairs
            # a_0_pair = torch.cat((a_0.view(1), torch.zeros(1))).view(b, c, -1, 2)
            # a_1_pair = torch.cat((a_1.view(1), torch.zeros(1))).view(b, c, -1, 2)
            # b_0_pair = torch.cat((b_0.view(1), torch.zeros(1))).view(b, c, -1, 2)
            # b_1_pair = torch.cat((b_1.view(1), torch.zeros(1))).view(b, c, -1, 2)

            # # separate the x and y coordinates from input
            # input_a_x = input_a[:, :, 0:1, :]
            # input_a_y = input_a[:, :, 1:2, :]
            # input_b_x = input_b[:, :, 0:1, :]
            # input_b_y = input_b[:, :, 1:2, :]

            # # add the start point to the selected_a
            # selected_a = torch.cat((
            #     nn.functional.grid_sample(input_a_x, a_0_pair),
            #     nn.functional.grid_sample(input_a_y, a_0_pair)
            # ), dim=3)
            
            # # get those points between the start point(a_0) and end point(a_1) with the interval = 0.001 (about 2 / 1500)
            # # and then add those in-between points to selected_a
            # # it must use two non-dimensional tensors in the "<" judgement
            # a_t = a_0
            # while a_t < a_1:
            #     a_t_pair = torch.cat((a_t.view(1), torch.zeros(1))).view(b, c, -1, 2)

            #     selected_a = torch.cat((
            #         selected_a,
            #         torch.cat((
            #             nn.functional.grid_sample(input_a_x, a_t_pair),
            #             nn.functional.grid_sample(input_a_y, a_t_pair)
            #         ), dim=3)
            #     ), dim=2)

            #     a_t = a_t + torch.tensor([0.001])

            # # add the end point
            # selected_a = torch.cat((
            #     selected_a,
            #     torch.cat((
            #         nn.functional.grid_sample(input_a_x, a_1_pair),
            #         nn.functional.grid_sample(input_a_y, a_1_pair)
            #     ), dim=3)
            # ), dim=2)

            # print(a_0)
            # print(a_1)
            # print(selected_a.shape)

            # return selected_a.sum()

            def get_partial_contour(contour, c_0, c_1):
                # compose the index pairs
                c_0_pair = torch.cat([c_0.view(1), torch.zeros(1)]).view(1, 1, -1, 2)
                c_1_pair = torch.cat([c_1.view(1), torch.zeros(1)]).view(1, 1, -1, 2)

                # separate the x and y coordinates from the original contour
                c_x = contour[:, :, 0:1, :]
                c_y = contour[:, :, 1:2, :]

                # add the start point to the selected_c
                selected_c = torch.cat([
                    nn.functional.grid_sample(c_x, c_0_pair),
                    nn.functional.grid_sample(c_y, c_0_pair)
                ], dim=3)
                
                # get those points between the start point(c_0) and end point(c_1) with the interval = 0.001 (about 2 / 1500)
                # and then add those in-between points to selected_c
                # it must use two non-dimensional tensors in the "<" judgement
                c_t = c_0
                while c_t < c_1:
                    c_t_pair = torch.cat([c_t.view(1), torch.zeros(1)]).view(1, 1, -1, 2)

                    selected_c = torch.cat([
                        selected_c,
                        torch.cat([
                            nn.functional.grid_sample(c_x, c_t_pair),
                            nn.functional.grid_sample(c_y, c_t_pair)
                        ], dim=3)
                    ], dim=2)

                    # don't use the inplace operator to tensor!! ex: "+="
                    c_t = c_t + torch.tensor(0.001)

                # add the end point
                selected_c = torch.cat([
                    selected_c,
                    torch.cat([
                        nn.functional.grid_sample(c_x, c_1_pair),
                        nn.functional.grid_sample(c_y, c_1_pair)
                    ], dim=3)
                ], dim=2)

                return selected_c

            # shape = b, c, a_num, 2
            selected_a = get_partial_contour(input_a, a_0, a_1)
            selected_b = get_partial_contour(input_b, b_0, b_1)

            # resize the shape to 2, a_num
            selected_a = selected_a.view(-1, 2).transpose(0, 1)
            selected_b = selected_b.view(-1, 2).transpose(0, 1)

            # print('selected_a.shape = ' + str(selected_a.shape))
            # print('selected_b.shape = ' + str(selected_b.shape))

            # return selected_a.sum() + selected_b.sum()
            
            # transform a to b

            # b0 - a0 = d
            # d shape = 2
            d = selected_b[:, 0] - selected_a[:, 0]
            # dx shape = 1
            dx = d[0].view(1)
            dy = d[1].view(1)
            
            # cos and sin
            translated_a1 = selected_a[:, -1] + d
            # e0 = translated_a1 - b0
            # e0 shape = 2
            e0 = translated_a1 - selected_b[:, 0]
            # e1 = b1 - b0
            e1 = selected_b[:, -1] - selected_b[:, 0]

            cos = torch.sum(e0 * e1, dim=0) / (e0.norm(2, dim=0) * e1.norm(2, dim=0)).view(1)
            sin = torch.sqrt(1. - cos * cos)

            # special case in sin
            if torch.isnan(sin[0]):
                sin[0] = torch.zeros(1)

            # cross in z-dim
            cross = e0[0] * e1[1] - e0[1] * e1[0]
            cross_lt = cross.lt(0).view(1)

            # scale
            scale = (selected_b[:, -1] - selected_b[:, 0]).norm(2, dim=0) / (selected_a[:, -1] - selected_a[:, 0]).norm(2, dim=0).view(1)
            
            scale_cos = scale * cos
            scale_sin = scale * sin

            # affine
            affine = torch.cat([
                torch.cat([scale_cos, -scale_sin, dx], dim=0).view(1, -1),
                torch.cat([scale_sin, scale_cos, dy], dim=0).view(1, -1),
                torch.cat([torch.zeros(2), torch.ones(1)], dim=0).view(1, -1)
            ], dim=0)

            # correct affine with cross_lt
            if torch.eq(cross_lt[0], 1):
                affine[0, 1] = affine[0, 1].neg()
                affine[1, 0] = affine[1, 0].neg()

            # homoheneous coordinates
            # homo_selected_a shape = 3, a_num
            # homo_selected_a = torch.cat([selected_a, torch.ones(1, selected_a.shape[1], dtype=torch.float32)], dim=0)
            homo_selected_a = torch.cat([selected_a, torch.ones(1, selected_a.shape[1])], dim=0)

            transformed_homo_selected_a = torch.matmul(affine, homo_selected_a)
            # transformed_selected_a shape = 2, a_num
            transformed_selected_a = transformed_homo_selected_a.narrow(0, 0, 2).contiguous()

            # bijective function
            # transformed_selected_a, selected_b, wab = ShapeMatchingLoss.bijective(transformed_selected_a, selected_b, wa, wb)
            # shape = 2, a_num and 2, b_num
            transformed_selected_a, selected_b, wab = ShapeMatchingLoss.bijective(transformed_selected_a, selected_b)
            
            # float_wab = torch.tensor(wab, dtype=torch.float32, requires_grad=True)
            # print('wab = ' + str(wab))

            pi = torch.tensor([3.1416], requires_grad=True)

            total_loss = torch.zeros(1, requires_grad=True)

            # convexity matching cost
            convexity_loss = torch.ones(1, requires_grad=True)

            if wab >= 3:
                def convexity_cue(c, i):
                    # if v0 == v1 or v1 == v2 or v2 == v3
                    # then cos will be nan

                    v1 = c[:, i]
                    v0 = c[:, i - 1]
                    v2 = c[:, i + 1]

                    # main_log.debug('v0 = ' + str(v0))
                    # main_log.debug('v1 = ' + str(v1))
                    # main_log.debug('v2 = ' + str(v2))

                    if torch.equal(v0, v1) or torch.equal(v1, v2) or torch.equal(v0, v2):
                        return False

                    # print('v0 = ' + str(v0))
                    # print('v1 = ' + str(v1))
                    # print('v2 = ' + str(v2))

                    v01 = v1 - v0
                    v12 = v2 - v1

                    # has grad
                    # return (v01 - v12).sum()

                    # main_log.debug('v01 = ' + str(v01))
                    # main_log.debug('v12 = ' + str(v12))

                    # print('v01 = ' + str(v01))
                    # print('v12 = ' + str(v12))

                    # cos
                    cos = torch.sum(v01 * v12, dim=0) / (v01.norm(2, dim=0) * v12.norm(2, dim=0))
                    # print('cos')
                    # print(cos)

                    # return cos

                    # # main_log.debug('cos = ' + str(cos))

                    # cross in z-dim
                    cross = v01[0] * v12[1] - v01[1] * v12[0]
                    cross_sign = cross.sign()

                    # main_log.debug('cross = ' + str(cross))

                    cos_eps = 1e-7
                    theta = torch.acos(cos.clamp(min=-1 + cos_eps, max=1 - cos_eps))

                    # main_log.debug('theta = ' + str(theta))

                    convexity = cross_sign * (pi - theta)

                    return convexity

                fab = torch.zeros(1, requires_grad=True)

                # for i in range(1, wab - 1, 1):
                for i in range(1, 10, 1):
                    convexity_a = convexity_cue(transformed_selected_a, i)
                    convexity_b = convexity_cue(selected_b, i)

                    if not (convexity_a and convexity_b):
                        continue

                    # main_log.debug('convexity_a = ' + str(convexity_a))
                    # main_log.debug('convexity_b = ' + str(convexity_b))

                    fab = fab + torch.sign(convexity_a * convexity_b) * torch.sqrt(torch.abs(convexity_a * convexity_b))
                    # fab = fab + torch.abs(convexity_a * convexity_b)
                    # fab = fab + convexity_a + convexity_b
                    # fab = fab + 

                # print('fab = ' + str(fab))
                fab = fab / wab
                convexity_loss = convexity_loss + fab
                # print(convexity_loss)
                # main_log.info('convexity loss = ' + str(convexity_loss))

            # # matching length cost
            # matching_length_cost = torch.ones(1, requires_grad=True)

            # # problems ??
            # # the sizes of a and b contours are initially the same (1500)
            # # the true sizes between a and b contours may be different
            # # matching_length_cost = matching_length_cost - max(
            # #     # selected_a.shape[1] / input_a.shape[3],
            # #     selected_b.shape[1] / input_b.shape[3]
            # # )

            # # matching_length_cost = matching_length_cost - selected_b.shape[1] / input_b.shape[3]

            # total_loss = convexity_loss + matching_length_cost
            # total_loss = total_loss + matching_length_cost
            total_loss = total_loss + convexity_loss

            return total_loss, selected_a, selected_b

        pr_a_loss, selected_a, selected_b = loss_fn(pr_a)
        pr_b_loss, selected_a, selected_b = loss_fn(pr_b)

        return ((pr_a_loss + pr_b_loss) / 2) + torch.pow(pr_a_loss - pr_b_loss, 2), selected_a, selected_b



        # # b, c, h, w = pr_a.size()
        # b, _, input_h, input_w = input_a.size()
        # # print('input_w type = ' + str(type(input_w)))

        # # loss = torch.tensor([0.], requires_grad=True)

        # # select two matching points
        # # max_pr_a, max_pr_a_idx = torch.max(pr_a, dim=1)
        # # max_pr_b, max_pr_b_idx = torch.max(pr_b, dim=1)

        # # max_pr_a, max_pr_a_idx = pr_a.topk(2, dim=1)
        # # if max_pr_a_idx[0] < max_pr_a_idx[1]:
        # #     selected_a = input_a.narrow(3, max_pr_a_idx[0], max_pr_a_idx[1]).view(b, h, -1)
        # # elif max_pr_a_idx[0] > max_pr_a_idx[1]:
        # #     selected_a = input_a.narrow(3, max_pr_a_idx[1], max_pr_a_idx[0]).view(b, h, -1)
        # # else:
        # #     main_log.error('error when extracting a')

        # # max_pr_b, max_pr_b_idx = pr_b.topk(2, dim=1)
        # # if max_pr_b_idx[0] < max_pr_b_idx[1]:
        # #     selected_b = input_b.narrow(3, max_pr_b_idx[0], max_pr_b_idx[1]).view(b, h, -1)
        # # elif max_pr_b_idx[0] > max_pr_b_idx[1]:
        # #     selected_b = input_b.narrow(3, max_pr_b_idx[1], max_pr_b_idx[0]).view(b, h, -1)

        # # wa = selected_a.shape[2]
        # # wb = selected_b.shape[2]

        # # selected_a, selected_b, wa, wb = ShapeMatchingLoss.get_contours(pr_a, pr_b, input_a, input_b)

        # # # print(pr_a.requires_grad)
        # # # print(pr_b.requires_grad)
        # # # print(pr_a.add(-pr_b).sum())
        # # print(selected_a.shape)
        # # selected_a = selected_a.view(-1)
        # # selected_b = selected_b.view(-1)
        # # print(selected_a.shape)
        # # print(selected_b.shape)
        # # print(selected_a.requires_grad)

        # # loss = selected_a.sum() - selected_b.sum()
        # # print(loss)

        # # a_pr, a_pr_idx, b_pr, b_pr_idx = ShapeMatchingLoss.get_contours(pr_a, pr_b, input_a, input_b)

        # s_a, s_b = ShapeMatchingLoss.get_contours(pr_a, pr_b, input_a, input_b)
        # loss = s_a.sum() + s_b.sum()

        # # loss = a_pr.sum() + a_pr_idx.sum() + b_pr.sum() + b_pr_idx.sum()
        # # loss = a_pr.sum() + b_pr.sum()
        # # loss = a_pr_idx.sum() + b_pr_idx.sum()

        # return loss


        # # transform a to b

        # # b0 - a0 = d
        # d = selected_b[:, :, 0] - selected_a[:, :, 0]
        # dx = d[:, 0].view(b, 1, 1)
        # dy = d[:, 1].view(b, 1, 1)

        # # cos and sin
        # translated_a1 = selected_a[:, :, -1] + d
        # # e0 = translated_a1 - b0
        # e0 = translated_a1 - selected_b[:, :, 0]
        # # e1 = b1 - b0
        # e1 = selected_b[:, :, -1] - selected_b[:, :, 0]

        # cos = torch.sum(e0 * e1, dim=1) / (e0.norm(2, dim=1) * e1.norm(2, dim=1))
        # sin = torch.sqrt(1. - cos * cos)

        # # special case in sin
        # for i in range(sin.shape[0]):
        #     if torch.isnan(sin[i]):
        #         sin[i] = torch.zeros(1)

        # cos = cos.view(b, 1, 1)
        # sin = sin.view(b, 1, 1)

        # # cross in z-dim
        # cross = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
        # cross_lt = cross.lt(0)

        # # scale
        # scale = (selected_b[:, :, -1] - selected_b[:, :, 0]).norm(2, dim=1) / (selected_a[:, :, -1] - selected_a[:, :, 0]).norm(2, dim=1)
        # scale = scale.view(b, 1, 1)

        # scale_cos = scale * cos
        # scale_sin = scale * sin

        # # affine
        # affine = torch.cat([
        #     torch.cat([scale_cos, -scale_sin, dx], dim=2),
        #     torch.cat([scale_sin, scale_cos, dy], dim=2),
        #     torch.cat([torch.zeros(b, 1, 2), torch.ones(b, 1, 1)], dim=2)
        # ], dim=1)

        # # correct affine with cross_lt
        # for i in range(cross_lt.shape[0]):
        #     if torch.eq(cross_lt[i], 1):
        #         affine[i, 0, 1] = affine[i, 0, 1].neg()
        #         affine[i, 1, 0] = affine[i, 1, 0].neg()

        # # homoheneous coordinates
        # # homo = torch.tensor(np.ones((b, 1, w)), dtype=np.float32)
        # homo_selected_a = torch.cat([selected_a, torch.ones(b, 1, wa, dtype=torch.float32)], dim=1)
        # # homo_selected_b = torch.cat([selected_b, homo], dim=1)
        # # homo_selected_b = torch.cat([selected_b, torch.ones(b, 1, wb, dtype=torch.float32)], dim=1)

        # transformed_homo_selected_a = torch.matmul(affine, homo_selected_a)
        # transformed_selected_a = transformed_homo_selected_a.narrow(1, 0, 2).contiguous()

        # # bijective function
        # transformed_selected_a, selected_b, wab = ShapeMatchingLoss.bijective(transformed_selected_a, selected_b, wa, wb)
        # float_wab = torch.tensor(wab, dtype=torch.float32, requires_grad=True)
        # print('wab = ' + str(wab))

        # pi = torch.tensor([3.1416], requires_grad=True)

        # total_loss = torch.zeros(1, requires_grad=True)

        # # # convexity matching cost
        # # convexity_loss = torch.ones(1, requires_grad=True)

        # # if wab >= 3:
        # #     def convexity_cue(c, i):
        # #         v1 = c[:, :, i]
        # #         v0 = c[:, :, i - 1]
        # #         v2 = c[:, :, i + 1]

        # #         main_log.debug('v0 = ' + str(v0))
        # #         main_log.debug('v1 = ' + str(v1))
        # #         main_log.debug('v2 = ' + str(v2))

        # #         v01 = v1 - v0
        # #         v12 = v2 - v1

        # #         main_log.debug('v01 = ' + str(v01))
        # #         main_log.debug('v12 = ' + str(v12))

        # #         # cos
        # #         cos = torch.sum(v01 * v12, dim=1) / (v01.norm(2, dim=1) * v12.norm(2, dim=1))

        # #         main_log.debug('cos = ' + str(cos))

        # #         # cross in z-dim
        # #         cross = v01[:, 0] * v12[:, 1] - e0[:, 1] * e1[:, 0]
        # #         cross_sign = cross.sign()

        # #         main_log.debug('cross = ' + str(cross))

        # #         cos_eps = 1e-7
        # #         theta = torch.acos(cos.clamp(min=-1 + cos_eps, max=1 - cos_eps))

        # #         main_log.debug('theta = ' + str(theta))

        # #         convexity = cross_sign * (pi.expand(b) - theta)

        # #         return convexity

        # #     fab = torch.zeros(1, requires_grad=True)

        # #     for i in range(1, wab - 1, 1):
        # #         convexity_a = convexity_cue(transformed_selected_a, i)
        # #         convexity_b = convexity_cue(selected_b, i)

        # #         main_log.debug('convexity_a = ' + str(convexity_a))
        # #         main_log.debug('convexity_b = ' + str(convexity_b))

        # #         fab = fab + torch.sign(convexity_a * convexity_b) * torch.sqrt(torch.abs(convexity_a * convexity_b))

        # #     print('fab = ' + str(fab))
        # #     fab = fab / float_wab
        # #     convexity_loss = convexity_loss + fab
        # #     main_log.info('convexity loss = ' + str(convexity_loss))

        # # matching length cost
        # matching_length_cost = torch.ones(1, requires_grad=True)

        # # matching_length_cost -= wab / input_w
        # matching_length_cost = matching_length_cost - float_wab / input_w

        # # total_loss = convexity_loss + matching_length_cost
        # total_loss = total_loss + matching_length_cost

        # return total_loss

    # def get_contours(pr_a, pr_b, input_a, input_b):
    #     b = input_a.shape[0]
    #     h = input_a.shape[2]

    #     def get_contour(pr, c):
    #         # max_pr, max_pr_idx = pr.topk(2, dim=3)
    #         # max_pr = max_pr.view(2)
    #         # max_pr_idx = max_pr_idx.view(2)

    #         # if max_pr_idx[0] < max_pr_idx[1]:
    #         #     small = max_pr_idx[0]
    #         #     large = max_pr_idx[1]
    #         # elif max_pr_idx[0] > max_pr_idx[1]:
    #         #     small = max_pr_idx[1]
    #         #     large = max_pr_idx[0]
    #         # else:
    #         #     main_log.error('error when extracting contour')
    #         #     exit()

    #         # selected_c = c.narrow(3, small, large - small + 1).contiguous().view(b, h, -1)

    #         # return selected_c

    #         # print('pr = ' + str(pr.requires_grad))
    #         # print('pr grad = \n' + str(pr.grad))

    #         # max_pr, max_pr_idx = pr.max(dim=3)
    #         # print(max_pr_idx)
    #         # # print('max_pr_idx grad = \n' + str(max_pr_idx.grad))
    #         # print('max_pr_idx = ' + str(max_pr_idx.requires_grad))
    #         # max_pr_idx = max_pr_idx.view(1)
    #         # # print('max_pr_idx grad = \n' + str(max_pr_idx.grad))
    #         # print('max_pr_idx = ' + str(max_pr_idx.requires_grad))
    #         # print(max_pr_idx)
    #         # print(c.shape)

    #         # selected_c = c.index_select(dim=3, index=max_pr_idx)
    #         # # print('selected_c grad = \n' + str(selected_c.grad))
    #         # print('selected_c grad = ' + str(selected_c.requires_grad))
    #         # # exit()

    #         # return selected_c

    #         # return max_pr, max_pr_idx

    #         max_pr, max_pr_idx = pr.topk(2, dim=3)
    #         max_pr = max_pr.view(1, 1, -1, 2)
    #         c = c.view(1, 1, 2, -1)

    #         selected_c = torch.nn.functional.grid_sample(c, max_pr)
    #         return selected_c

    #     selected_a = get_contour(pr_a, input_a)
    #     selected_b = get_contour(pr_b, input_b)

    #     # a_pr, a_pr_idx = get_contour(pr_a, input_a)
    #     # b_pr, b_pr_idx = get_contour(pr_b, input_b)

    #     # wa = torch.tensor([selected_a.shape[2]], dtype=torch.int32, requires_grad=True)
    #     # wb = torch.tensor([selected_b.shape[2]], dtype=torch.int32, requires_grad=True)

    #     # return selected_a, selected_b, wa, wb

    #     # return a_pr, a_pr_idx, b_pr, b_pr_idx
        
    #     return selected_a, selected_b

    def bijective(a, b):
        wa = a.shape[1]
        wb = b.shape[1]

        # upsampling
        if wa > wb:
            # wa > wb
            # print('wa > wb')
            # a = ShapeMatchingLoss.fixed_number_contour(a, wb)
            # w = wb
            # use the upsampling in pytorch
            # warning : the bilinear mode doesn't seem to interpolate between two rows ??
            b = b.view(1, 1, 2, -1)
            b = nn.functional.interpolate(b, (2, wa), mode='bilinear', align_corners=True)
            b = b.view(2, -1)
            w = wa
        elif wa < wb:
            # wa < wb
            # print('wa < wb')
            # b = ShapeMatchingLoss.fixed_number_contour(b, wa)
            # w = wa
            # # use the upsampling in pytorch
            a = a.view(1, 1, 2, -1)
            a = nn.functional.interpolate(a, (2, wb), mode='bilinear', align_corners=True)
            a = a.view(2, -1)
            w = wb
        elif wa == wb:
            # wa == wb
            # main_log.debug('a and b are the same')
            w = wa
        else:
            main_log.error('error in align')

        return a, b, w

    def fixed_number_contour(c, w):
        h, wc = c.shape

        point_length = torch.tensor([3.], requires_grad=True)
        contour_lengths = torch.tensor([0.], requires_grad=True)
        total_contour_length = torch.tensor([0.], requires_grad=True)

        total_contour_length = total_contour_length + point_length
        contour_lengths = torch.tensor(total_contour_length)

        for i in range(wc - 1):
            d = (c[:, i + 1] - c[:, i]).norm(2, dim=0)

            total_contour_length = total_contour_length + d + point_length
            contour_lengths = torch.cat([contour_lengths, total_contour_length], dim=0)

        # step = total_contour_length // w
        step = total_contour_length / w

        # contour_lengths_shape = contour_lengths.shape
        contour_lengths_shape = torch.tensor(contour_lengths.shape, dtype=torch.long, requires_grad=True)

        assert(list(contour_lengths_shape)[0] == wc)

        # create a new contour
        new_c = torch.tensor([], requires_grad=True)
        distance = torch.tensor([0.], requires_grad=True)
        contour_lengths_i = torch.tensor([0], dtype=torch.long, requires_grad=True)
        tensor_wc = torch.tensor([wc], dtype=torch.long, requires_grad=True)
        print('step = ' + str(step))

        for i in range(w):
            distance = distance + step

            while(contour_lengths_i < contour_lengths_shape and contour_lengths[contour_lengths_i] < distance):
                contour_lengths_i = contour_lengths_i + 1

            # ?
            # if contour_lengths_i < contour_lengths_shape:
            #     print('contour_lengths_i < contour_lengths_shape')

            contour_lengths_i = contour_lengths_i if contour_lengths_i < contour_lengths_shape else contour_lengths_i - 1

            length = contour_lengths[contour_lengths_i]
            t = length - distance

            if(t < point_length):
                # on the point
                # no need?
                # b is always 1
                p = c[0, :, contour_lengths_i if contour_lengths_i < tensor_wc else tensor_wc - 1]
                main_log.debug('point')
            else:
                # on the edge
                p0 = c[0, :, contour_lengths_i - 1]
                # no need?
                p1 = c[0, :, contour_lengths_i] if contour_lengths_i < tensor_wc else c[0, :, 0]

                # dv = p0 - p1
                dv = p1 - p0
                d = dv.norm(2, dim=0)
                t -= point_length

                p = p0 + t / d * dv

                main_log.debug('edge')

            new_c = torch.cat([new_c, p], dim=1)

        new_c = new_c.view(1, 2, -1)

        return new_c

        # b, h, wc = c.shape
        # float_w = w.type(torch.cuda.FloatTensor)

        # point_length = torch.tensor([3.], requires_grad=True)
        # contour_lengths = torch.tensor([0.], requires_grad=True)
        # total_contour_length = torch.tensor([0.], requires_grad=True)

        # total_contour_length = total_contour_length + point_length
        # contour_lengths = torch.tensor(total_contour_length)

        # for i in range(wc - 1):
        #     d = (c[0, :, i + 1] - c[0, :, i]).norm(2, dim=0)

        #     total_contour_length = total_contour_length + d + point_length
        #     contour_lengths = torch.cat([contour_lengths, total_contour_length], dim=0)

        # # step = total_contour_length // w
        # step = total_contour_length / float_w
        # # contour_lengths_shape = contour_lengths.shape
        # contour_lengths_shape = torch.tensor(contour_lengths.shape, dtype=torch.long, requires_grad=True)

        # assert(list(contour_lengths_shape)[0] == wc)

        # # create a new contour
        # new_c = torch.tensor([], requires_grad=True)
        # distance = torch.tensor([0.], requires_grad=True)
        # contour_lengths_i = torch.tensor([0], dtype=torch.long, requires_grad=True)
        # tensor_wc = torch.tensor([wc], dtype=torch.long, requires_grad=True)
        # print('step = ' + str(step))

        # for i in range(w):
        #     distance = distance + step

        #     while(contour_lengths_i < contour_lengths_shape and contour_lengths[contour_lengths_i] < distance):
        #         contour_lengths_i = contour_lengths_i + 1

        #     # ?
        #     # if contour_lengths_i < contour_lengths_shape:
        #     #     print('contour_lengths_i < contour_lengths_shape')

        #     contour_lengths_i = contour_lengths_i if contour_lengths_i < contour_lengths_shape else contour_lengths_i - 1

        #     length = contour_lengths[contour_lengths_i]
        #     t = length - distance

        #     if(t < point_length):
        #         # on the point
        #         # no need?
        #         # b is always 1
        #         p = c[0, :, contour_lengths_i if contour_lengths_i < tensor_wc else tensor_wc - 1]
        #         main_log.debug('point')
        #     else:
        #         # on the edge
        #         p0 = c[0, :, contour_lengths_i - 1]
        #         # no need?
        #         p1 = c[0, :, contour_lengths_i] if contour_lengths_i < tensor_wc else c[0, :, 0]

        #         # dv = p0 - p1
        #         dv = p1 - p0
        #         d = dv.norm(2, dim=0)
        #         t -= point_length

        #         p = p0 + t / d * dv

        #         main_log.debug('edge')

        #     new_c = torch.cat([new_c, p], dim=1)

        # new_c = new_c.view(1, 2, -1)

        # return new_c


class ShapeMatchingLoss1(nn.Module):
    def __init__(self):
        super(ShapeMatchingLoss1, self).__init__()
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

    def forward(self, selected_points_b, selected_points_a):
        loss = torch.tensor(0., requires_grad=True)

        # convexity_cue

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
        # args = args.squeeze()
        # a = args[:2]
        # b = args[2:]
        # c = torch.max(a, b).cpu()
        # for i in range(c.size()[0]):
        #     loss = loss.add(c[i])

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