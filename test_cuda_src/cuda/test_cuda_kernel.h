#ifndef _TEST_CUDA_KERNEL
#define _TEST_CUDA_KERNEL

#ifdef __cplusplus
extern "C"
{
#endif

    // int ROIPoolForwardLaucher(
    //     const float *bottom_data, const float spatial_scale, const int num_rois, const int height,
    //     const int width, const int channels, const int pooled_height,
    //     const int pooled_width, const float *bottom_rois,
    //     float *top_data, int *argmax_data, cudaStream_t stream);

    // int ROIPoolBackwardLaucher(const float *top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    //                            const int height, const int width, const int channels, const int pooled_height,
    //                            const int pooled_width, const float *bottom_rois,
    //                            float *bottom_diff, const int *argmax_data, cudaStream_t stream);

    int MyLibAddForwardLauncher(const int height, const int width, const int times, const float *input1, const float *input2, const float *weight,
                                float *output, cudaStream_t stream);

    int MyLibAddBackwardLauncher(const int height, const int width, const int times, const float *grad_output, 
                                 float *grad_input, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif