#include "mse.h"
#include <algorithm>

__global__ void mse_loss_kernel(const float *pred, const float *target, float *loss, int size) {
    __shared__ float shared_loss[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    shared_loss[tid] = 0.0f;
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        shared_loss[tid] = diff * diff;
    }
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_loss[tid] += shared_loss[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(loss, shared_loss[0]);
    }
}

__global__ void mse_grad_kernel(const float *pred, const float *target, float *grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 2.0f * (pred[idx] - target[idx]) / size;
    }
}

float compute_mse(const float *d_pred, const float *d_target, int size) {
    float *d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mse_loss_kernel<<<blocks, threads>>>(d_pred, d_target, d_loss, size);
    
    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_loss);
    
    return h_loss / size;
}

void mse_backward(const float *d_pred, const float *d_target, float *d_grad, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mse_grad_kernel<<<blocks, threads>>>(d_pred, d_target, d_grad, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}