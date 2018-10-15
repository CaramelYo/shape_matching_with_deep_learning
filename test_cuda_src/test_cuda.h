int my_lib_add_forward(const int height, const int width, const int times,
                       THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *weight, THCudaTensor *output);

int my_lib_add_backward(const int height, const int width, const int times,
                        THCudaTensor *grad_output, THCudaTensor *grad_input);