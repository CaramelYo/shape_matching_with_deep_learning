#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "test_cuda_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


// __global__ void ROIPoolForward(const int nthreads, const float* bottom_data,
//     const float spatial_scale, const int height, const int width,
//     const int channels, const int pooled_height, const int pooled_width, 
//     const float* bottom_rois, float* top_data, int* argmax_data)
__global__ void MyLibAddForward(const int height, const int width, const float *input1, const float *input2, const float *weight,
                                float *output)
{
    // if (!THFloatTensor_isSameSizeAs(input1, input2))
    //     return;
    
    // THFloatTensor_resizeAs(output, input1);
    // THFloatTensor_cadd(output, input1, 2.0, input2);
    // return;

    // // the value of output won't be accumulated
    // for (int y = 0; y < height; ++y){
    //     for(int x = 0; x < width; ++x){
    //         // output[y][x] = input1[y][x] + weight[0] * input2[y][x];
    //         int index = y * width + x;
    //         output[index] += input1[index] + weight[0] * input2[index];
    //     }
    // }

    // printf("1\n");

    // return;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("index = %d\n", index);
    
    output[index] = input1[index] + weight[0] * input2[index];

    return;


    // CUDA_1D_KERNEL_LOOP(index, nthreads)
    // {
    //     // (n, c, ph, pw) is an element in the pooled output
    //     int n = index;
    //     int pw = n % pooled_width;
    //     n /= pooled_width;
    //     int ph = n % pooled_height;
    //     n /= pooled_height;
    //     int c = n % channels;
    //     n /= channels;

    //     bottom_rois += n * 5;
    //     int roi_batch_ind = bottom_rois[0];
    //     int roi_start_w = round(bottom_rois[1] * spatial_scale);
    //     int roi_start_h = round(bottom_rois[2] * spatial_scale);
    //     int roi_end_w = round(bottom_rois[3] * spatial_scale);
    //     int roi_end_h = round(bottom_rois[4] * spatial_scale);

    //     // Force malformed ROIs to be 1x1
    //     int roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1);
    //     int roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);
    //     float bin_size_h = (float)(roi_height) / (float)(pooled_height);
    //     float bin_size_w = (float)(roi_width) / (float)(pooled_width);

    //     int hstart = (int)(floor((float)(ph) * bin_size_h));
    //     int wstart = (int)(floor((float)(pw) * bin_size_w));
    //     int hend = (int)(ceil((float)(ph + 1) * bin_size_h));
    //     int wend = (int)(ceil((float)(pw + 1) * bin_size_w));

    //     // Add roi offsets and clip to input boundaries
    //     hstart = fminf(fmaxf(hstart + roi_start_h, 0), height);
    //     hend = fminf(fmaxf(hend + roi_start_h, 0), height);
    //     wstart = fminf(fmaxf(wstart + roi_start_w, 0), width);
    //     wend = fminf(fmaxf(wend + roi_start_w, 0), width);
    //     bool is_empty = (hend <= hstart) || (wend <= wstart);

    //     // Define an empty pooling region to be zero
    //     float maxval = is_empty ? 0 : -FLT_MAX;
    //     // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    //     int maxidx = -1;
    //     bottom_data += roi_batch_ind * channels * height * width;
    //     for (int h = hstart; h < hend; ++h) {
    //         for (int w = wstart; w < wend; ++w) {
    // //            int bottom_index = (h * width + w) * channels + c;
    //             int bottom_index = (c * height + h) * width + w;
    //             if (bottom_data[bottom_index] > maxval) {
    //                 maxval = bottom_data[bottom_index];
    //                 maxidx = bottom_index;
    //             }
    //         }
    //     }
    //     top_data[index] = maxval;
    //     if (argmax_data != NULL)
    //         argmax_data[index] = maxidx;
    // }
}


// int ROIPoolForwardLaucher(
//     const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
//     const int width, const int channels, const int pooled_height,
//     const int pooled_width, const float* bottom_rois,
//     float* top_data, int* argmax_data, cudaStream_t stream)
int MyLibAddForwardLauncher(const int height, const int width, const int times, const float *input1, const float *input2, const float *weight,
                                float *output, cudaStream_t stream)
{
    // times = 5
    // const int thread_per_block = 1024;
    const int thread_per_block = height * width;
    cudaError_t cuda_err;

    MyLibAddForward<<<(times + thread_per_block - 1) / thread_per_block, thread_per_block, 0, stream>>>(
        height, width, input1, input2, weight,
        output);

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "cudaCheckError() failed : %s in forward launcher\n", cudaGetErrorString(cuda_err));
        exit(-1);
    }

    return 1;

    // const int kThreadsPerBlock = 1024;
    // const int output_size = num_rois * pooled_height * pooled_width * channels;
    // cudaError_t err;


    // ROIPoolForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
    //   output_size, bottom_data, spatial_scale, height, width, channels, pooled_height,
    //   pooled_width, bottom_rois, top_data, argmax_data);

    // err = cudaGetLastError();
    // if(cudaSuccess != err)
    // {
    //     fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    //     exit( -1 );
    // }

    // return 1;
}


// __global__ void ROIPoolBackward(const int nthreads, const float* top_diff,
//     const int* argmax_data, const int num_rois, const float spatial_scale,
//     const int height, const int width, const int channels,
//     const int pooled_height, const int pooled_width, float* bottom_diff,
//     const float* bottom_rois) {
__global__ void MyLibAddBackward(const int height, const int width, const float *grad_output,
                                 float *grad_input)
{
    // THFloatTensor_fill(grad_input, 1);
    // return 1;

    // for(int y = 0; y < height; ++y){
    //     for(int x = 0; x < width; ++x){
    //         // grad_input[y][x] += 1.;
    //         int index = y * width + x;
    //         grad_input[index] += 1.;
    //     }
    // }

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    grad_input[index] = 1;

    return;


//     CUDA_1D_KERNEL_LOOP(index, nthreads)
//     {

//         // (n, c, ph, pw) is an element in the pooled output
//         int n = index;
//         int w = n % width;
//         n /= width;
//         int h = n % height;
//         n /= height;
//         int c = n % channels;
//         n /= channels;

//         float gradient = 0;
//         // Accumulate gradient over all ROIs that pooled this element
//         for (int roi_n = 0; roi_n < num_rois; ++roi_n)
//         {
//             const float* offset_bottom_rois = bottom_rois + roi_n * 5;
//             int roi_batch_ind = offset_bottom_rois[0];
//             // Skip if ROI's batch index doesn't match n
//             if (n != roi_batch_ind) {
//                 continue;
//             }

//             int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
//             int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
//             int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
//             int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

//             // Skip if ROI doesn't include (h, w)
//             const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
//                                h >= roi_start_h && h <= roi_end_h);
//             if (!in_roi) {
//                 continue;
//             }

//             int offset = roi_n * pooled_height * pooled_width * channels;
//             const float* offset_top_diff = top_diff + offset;
//             const int* offset_argmax_data = argmax_data + offset;

//             // Compute feasible set of pooled units that could have pooled
//             // this bottom unit

//             // Force malformed ROIs to be 1x1
//             int roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1);
//             int roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);

//             float bin_size_h = (float)(roi_height) / (float)(pooled_height);
//             float bin_size_w = (float)(roi_width) / (float)(pooled_width);

//             int phstart = floor((float)(h - roi_start_h) / bin_size_h);
//             int phend = ceil((float)(h - roi_start_h + 1) / bin_size_h);
//             int pwstart = floor((float)(w - roi_start_w) / bin_size_w);
//             int pwend = ceil((float)(w - roi_start_w + 1) / bin_size_w);

//             phstart = fminf(fmaxf(phstart, 0), pooled_height);
//             phend = fminf(fmaxf(phend, 0), pooled_height);
//             pwstart = fminf(fmaxf(pwstart, 0), pooled_width);
//             pwend = fminf(fmaxf(pwend, 0), pooled_width);

//             for (int ph = phstart; ph < phend; ++ph) {
//                 for (int pw = pwstart; pw < pwend; ++pw) {
//                     if (offset_argmax_data[(c * pooled_height + ph) * pooled_width + pw] == index)
//                     {
//                         gradient += offset_top_diff[(c * pooled_height + ph) * pooled_width + pw];
//                     }
//                 }
//             }
//         }
//         bottom_diff[index] = gradient;
//   }
}

// int ROIPoolBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
//     const int height, const int width, const int channels, const int pooled_height,
//     const int pooled_width, const float* bottom_rois,
//     float* bottom_diff, const int* argmax_data, cudaStream_t stream)
int MyLibAddBackwardLauncher(const int height, const int width, const int times, const float *grad_output,
                             float *grad_input, cudaStream_t stream)
{
    // times = 5
    // const int thread_per_block = 4;
    const int thread_per_block = height * width;
    cudaError_t cuda_err;

    MyLibAddBackward<<<(times + thread_per_block - 1) / thread_per_block, thread_per_block, 0, stream>>>(
        height, width, grad_output,
        grad_input);

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "cudaCheckError() failed : %s in back launcher\n", cudaGetErrorString(cuda_err));
        exit(-1);
    }

    return 1;


    // const int kThreadsPerBlock = 1024;
    // const int output_size = batch_size * height * width * channels;
    // cudaError_t err;

    // // ROIPoolBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
    //   output_size, top_diff, argmax_data, num_rois, spatial_scale, height, width, channels, pooled_height,
    //   pooled_width, bottom_diff, bottom_rois);

    // err = cudaGetLastError();
    // if(cudaSuccess != err)
    // {
    //     fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    //     exit( -1 );
    // }

    // return 1;
}


#ifdef __cplusplus
}
#endif