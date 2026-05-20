#include "relu.h"
#include <algorithm>

__global__ void relu_forward_kernel(const float *input, float *output, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void relu_backward_kernel(const float *input, const float *grad_output, float *grad_input, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

void relu_forward(const float *input, float *output, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(input, output, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void relu_backward(const float *input, const float *grad_output, float *grad_input, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(input, grad_output, grad_input, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}