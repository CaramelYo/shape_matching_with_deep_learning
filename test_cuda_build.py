import os
import torch
from torch.utils.ffi import create_extension


sources = ['test_cuda_src/test_cuda.c']
headers = ['test_cuda_src/test_cuda.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    # sources += ['test_cuda_src/test_cuda.c']
    # headers += ['test_cuda_src/test_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

# this_file = os.path.dirname(__file__)
# print(this_file)
this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

extra_objects = ['test_cuda_src/cuda/test_cuda_kernel.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.my_lib', # _ext/my_lib 编译后的动态 链接库 存放路径。
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()