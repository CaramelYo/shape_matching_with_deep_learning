import torch
import torch.optim as optim
from torch.autograd import Function
from torch.nn.modules.module import Module
import numpy as np
import itertools

from _ext import my_lib


class MyLibFunction(Function):
    def __init__(self, height, width, times, use_cuda):
        self.height = height
        self.width = width
        self.times = times
        self.use_cuda = use_cuda

        if not self.use_cuda:
            print('no cuda in init')
            exit()

    def forward(self, input1, input2, weight):
        output = torch.zeros((self.height, self.width), dtype=torch.float32).cuda()

        if not self.use_cuda:
            print('no cuda in forward')
            exit()

        result_state = my_lib.my_lib_add_forward(self.height, self.width, self.times,
                                                 input1, input2, weight, output)

        if result_state == 0:
            print('gg in my_lib_add_forward')
            exit()

        # print('output')
        # self.output = output
        # print(self.output)

        return output

    def backward(self, grad_output):
        assert(grad_output == self.use_cuda)

        grad_input = torch.zeros((self.height, self.width), dtype=torch.float32).cuda()

        result_state = my_lib.my_lib_add_backward(self.height, self.width, self.times,
                                                  grad_output, grad_input)

        if result_state == 0:
            print('gg in my_lib_add_backward')
            exit()
        
        print('grad_output')
        print(grad_output)
        print('grad_input')
        print(grad_input)

        return grad_input, None


class MyLib(Module):
    def __init__(self, height, width, times, use_cuda):
        super(MyLib, self).__init__()

        self.height = height
        self.width = width
        self.times = times
        self.use_cuda = use_cuda

    def forward(self, input1, input2, weight):
        return MyLibFunction(self.height, self.width, self.times, self.use_cuda)(input1, input2, weight)


use_cuda = torch.cuda.is_available()

height = 2
width = 3
times = 1


# x = 20.5
# y = 3.4
# weight = 12.3
# truth_x = torch.zeros((), dtype=torch.float32)
# truth_x = truth_x.new_full((height, width), 20.5)
# truth_y = truth_x.new_full((height, width), 3.4)
# truth_w = truth_x.new_full((height, width), 12.3)
truth_x = np.full((height, width), 20.5, dtype=np.float32)
truth_y = np.full((height, width), 3.4, dtype=np.float32)
truth_w = np.full((height, width), 12.3, dtype=np.float32)

# generate data
n_batch = 10
batch_size = 128
training_data_list = []
truth_data_list = []
for i in range(n_batch):
    training_batch_data = []
    truth_batch_data = []
    for j in range(batch_size):
        # batch_data.append(
        #     torch.rand((height, width), dtype=torch.float32) * truth_w + truth_x * truth_y)
        data = np.random.rand(height, width)

        training_batch_data.append(data)
        truth_batch_data.append(data * truth_w + truth_x * truth_y)

    training_data_list.append(training_batch_data)
    truth_data_list.append(truth_batch_data)

training_tensor_data_list = torch.tensor(training_data_list, dtype=torch.float32)
truth_tensor_data_list = torch.tensor(truth_data_list, dtype=torch.float32)

x = torch.ones((height, width), dtype=torch.float32).cuda().detach().requires_grad_()
y = x.clone().cuda().detach().requires_grad_()
w = x.clone().cuda().detach().requires_grad_()

print('x')
# print(x)
# print('y')
# print(y)
# print('w')
# print(w)

n_epoch = 2
lr = 0.001

my_lib_module = MyLib(height, width, times, use_cuda)
# optimizer = optim.Adam((x, y, w), lr=lr)
# optimizer = optim.Adam(itertools.chain((x, y, w)))
optimizer = optim.Adam([x, y, w], lr=lr)
loss_model = torch.nn.MSELoss()

my_lib_module.train()
train_loss = 0

print('training start')
for epoch in range(n_epoch):
    for i in range(n_batch):
        training_data = training_tensor_data_list[i]
        truth_data = truth_tensor_data_list[i]

        optimizer.zero_grad()
        pred = my_lib_module(training_data, x, y, w)

        loss = loss_model(pred, truth_data)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.data.cpu().numpy()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
            epoch, i, n_batch,
            100. * i / n_batch, loss.data))
    
    train_loss /= n_batch * batch_size
    
    print('Train set: Average loss: {:.4f}'.format(train_loss))

    print('pred')
    print(pred)

print('training end')
