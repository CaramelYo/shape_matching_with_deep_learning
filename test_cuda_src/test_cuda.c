#include <THC/THC.h>
#include <math.h>

#include "cuda/test_cuda_kernel.h"

extern THCState *state;

int my_lib_add_forward(const int height, const int width, const int times,
                       THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *weight, THCudaTensor *output)
{
    // THFloatTensor_cadd(output, input1, 1.0, input2);
    // return 1;

    // if (!THCudaTensor_isSameSizeAs(state, input1, input2))
    // {
    //     printf("gg in size problem in froward");
    //     return 0;
    // }

    // THCudaTensor_resizeAs(state, output, input1);

    float *input1_float = THCudaTensor_data(state, input1);
    float *input2_float = THCudaTensor_data(state, input2);
    float *weight_float = THCudaTensor_data(state, weight);
    float *output_float = THCudaTensor_data(state, output);

    cudaStream_t stream = THCState_getCurrentStream(state);

    int result_state = MyLibAddForwardLauncher(height, width, times, input1_float, input2_float, weight_float, output_float, stream);

    if (result_state == 1)
    {
        // success
        printf("good in forward\n");
        return 1;
    }
    else
    {
        printf("gg in forward\n");
        return 0;
    }
}

int my_lib_add_backward(const int height, const int width, const int times,
                        THCudaTensor *grad_output, THCudaTensor *grad_input)
{
    // THFloatTensor_resizeAs(grad_input, grad_output);
    // THFloatTensor_fill(grad_input, 1);
    // return 1;

    // THCudaTensor_resizeAs(state, grad_input, grad_output);

    float *grad_output_float = THCudaTensor_data(state, grad_output);
    float *grad_input_float = THCudaTensor_data(state, grad_input);

    cudaStream_t stream = THCState_getCurrentStream(state);

    int result_state = MyLibAddBackwardLauncher(height, width, times, grad_output_float, grad_input_float, stream);

    if (result_state == 1)
    {
        // success
        printf("grad_output_float\n");
        // printf(grad_output_float);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                printf("%f ", grad_output_float[y * width + x]);
            }
            printf("\n");
        }
        printf("grad_input_float\n");
        // printf(grad_input_float);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                printf("%f ", grad_input_float[y * width + x]);
            }
            printf("\n");
        }
        // printf("\n");

        printf("good in backward\n");
        return 1;
    }
    else
    {
        printf("gg in backward\n");
        return 0;
    }
}