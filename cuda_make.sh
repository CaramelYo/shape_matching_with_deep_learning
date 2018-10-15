#!/usr/bin/env bash

echo "compiling cuda by nvcc..."

cd test_cuda_src/cuda

nvcc -c -o test_cuda_kernel.o test_cuda_kernel.cu \
     -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

echo "python build"

cd ../../

python test_cuda_build.py