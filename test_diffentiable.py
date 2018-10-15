import torch


torch.set_default_tensor_type('torch.cuda.FloatTensor')

# a = torch.tensor([1.], requires_grad=True)
# # b = a.index_select(dim=0, index=torch.tensor([2]))
# b = a.type(dtype=torch.LongTensor)
# b.requires_grad_()

# print(a)
# print(b)
# print(b.requires_grad)

# print(torch.autograd.grad(b, a, allow_unused=True))

src = torch.arange(6, dtype=torch.float, requires_grad=True).reshape(1, 1, 2, -1)
# [x, y]
indices = torch.tensor([[-1., -1.], [0.5, 0.]], dtype=torch.float).reshape(1, 1, -1, 2)
# indices = torch.tensor([[-1., -1.], [-1, 1.]], dtype=torch.float).reshape(1, 1, -1, 2)
output = torch.nn.functional.grid_sample(src, indices)
print(src)
print(output)
