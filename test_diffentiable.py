import torch


torch.set_default_tensor_type('torch.cuda.DoubleTensor')

print(torch.get_default_dtype())
t = torch.tensor([1.])
print(t.type())
t = torch.tensor([1.], dtype=torch.get_default_dtype())
print(t.type())

exit()

# a = torch.tensor([1.], requires_grad=True)
# # b = a.index_select(dim=0, index=torch.tensor([2]))
# b = a.type(dtype=torch.LongTensor)
# b.requires_grad_()

# print(a)
# print(b)
# print(b.requires_grad)

# print(torch.autograd.grad(b, a, allow_unused=True))

src = torch.arange(3, dtype=torch.float, requires_grad=True).reshape(1, 1, 1, -1)
# [x, y]
indices = torch.tensor([[-1., -1.], [0.5, 0.]], dtype=torch.float).reshape(1, 1, -1, 2)
# indices = torch.tensor([[-1., -1.], [-1, 1.]], dtype=torch.float).reshape(1, 1, -1, 2)
output = torch.nn.functional.grid_sample(src, indices)
print(src)
print(output)

# src = torch.arange(6, dtype=torch.float).reshape(1, 1, 2, -1)
# ws = src.shape[3]
# new_ws = 2 * ws
# src[0, 0, 1, 1] = 4.9

output = torch.nn.functional.interpolate(src, (1, 2), mode='bilinear')

print(src)
print(output)
print(src.shape)
print(output.shape)

# src_1 = torch.rand(1, 1, 3)
# # src_1 = src[:, :, 1, :]

# print(src_1.shape)

# output = torch.nn.functional.interpolate(src_1, (5), mode='linear', align_corners=True)
# print(src_1)
# print(output)
# print(src_1.shape)
# print(output.shape)


src = torch.tensor([
    [3., 7., 8.],
    [4., 24., 15.]]).view(2, -1)

print(src)
print(src.shape)
output = src.norm(2, dim=0)
print(output)
