#include "linear.h"
#include <algorithm>

LinearLayer::LinearLayer(int in_dim, int out_dim, int bs, float lr) 
    : in_features(in_dim), out_features(out_dim), batch_size(bs), learning_rate(lr) 
{
    
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_weights, in_features * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights, in_features * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias, out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input, batch_size * in_features * sizeof(float)));
    
    // Initialize weights (Xavier initialization)
    float *h_weights = new float[in_features * out_features];
    float *h_bias = new float[out_features];
    
    float scale = sqrtf(2.0f / (in_features + out_features));
    for (int i = 0; i < in_features * out_features; i++) {
        h_weights[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
    for (int i = 0; i < out_features; i++) {
        h_bias[i] = 0.0f;
    }
    
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, in_features * out_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, out_features * sizeof(float), cudaMemcpyHostToDevice));
    
    delete[] h_weights;
    delete[] h_bias;
}

LinearLayer::~LinearLayer() {
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_grad_weights);
    cudaFree(d_grad_bias);
    cudaFree(d_output);
    cudaFree(d_grad_input);
}

__global__ void linear_forward_kernel(const float *input, const float *weights, const float *bias,
                                     float *output, int batch_size, int in_dim, int out_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_dim) {
        float sum = bias[col];
        for (int k = 0; k < in_dim; k++) {
            sum += input[row * in_dim + k] * weights[k * out_dim + col];
        }
        output[row * out_dim + col] = sum;
    }
}

__global__ void linear_backward_kernel(const float *input, const float *weights, const float *grad_output,
                                       float *grad_input, float *grad_weights, float *grad_bias,
                                       int batch_size, int in_dim, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute grad_input
    if (idx < batch_size * in_dim) {
        int row = idx / in_dim;
        int col = idx % in_dim;
        float sum = 0.0f;
        for (int k = 0; k < out_dim; k++) {
            sum += grad_output[row * out_dim + k] * weights[col * out_dim + k];
        }
        grad_input[idx] = sum;
    }
    
    // Compute grad_weights 
    if (idx < in_dim * out_dim) {
        int i = idx / out_dim;
        int j = idx % out_dim;
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += input[b * in_dim + i] * grad_output[b * out_dim + j];
        }
        grad_weights[idx] = sum / batch_size;
    }
    
    // Compute grad_bias
    if (idx < out_dim) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += grad_output[b * out_dim + idx];
        }
        grad_bias[idx] = sum / batch_size;
    }
}

__global__ void update_weights_kernel(float *weights, const float *grad_weights, 
                                      float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * grad_weights[idx];
    }
}

void LinearLayer::forward(float *input) {
    d_input = input;
    dim3 threads(16, 16);
    dim3 blocks((out_features + 15) / 16, (batch_size + 15) / 16);
    linear_forward_kernel<<<blocks, threads>>>(d_input, d_weights, d_bias, d_output, 
                                               batch_size, in_features, out_features);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void LinearLayer::backward(float *grad_output) {
    int threads = 256;
    int max_size = std::max({batch_size * in_features, in_features * out_features, out_features});
    int blocks = (max_size + threads - 1) / threads;
    
    linear_backward_kernel<<<blocks, threads>>>(d_input, d_weights, grad_output,
                                                d_grad_input, d_grad_weights, d_grad_bias,
                                                batch_size, in_features, out_features);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void LinearLayer::update() 
{
    int threads = 256;
    int blocks_w = (in_features * out_features + threads - 1) / threads;
    int blocks_b = (out_features + threads - 1) / threads;
    
    update_weights_kernel<<<blocks_w, threads>>>(d_weights, d_grad_weights, learning_rate, 
                                                 in_features * out_features);
    update_weights_kernel<<<blocks_b, threads>>>(d_bias, d_grad_bias, learning_rate, out_features);
    CUDA_CHECK(cudaDeviceSynchronize());
}