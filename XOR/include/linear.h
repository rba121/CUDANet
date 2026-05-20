#ifndef LINEAR_H
#define LINEAR_H

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


class LinearLayer {
public:
    int in_features, out_features, batch_size;
    float *d_weights, *d_bias;
    float *d_input, *d_output;
    float *d_grad_weights, *d_grad_bias, *d_grad_input;
    float learning_rate;

    LinearLayer(int in_dim, int out_dim, int bs, float lr = 0.01f);
    ~LinearLayer();
  
    void forward(float *input);
    
    void backward(float *grad_output);
    
    /**
     * Update weights and bias using gradients
     * W = W - lr * grad_W
     * b = b - lr * grad_b
     */
    void update();
};

__global__ void linear_forward_kernel(const float *input, const float *weights, const float *bias, float *output, int batch_size, int in_dim, int out_dim);

__global__ void linear_backward_kernel(const float *input, const float *weights, const float *grad_output, float *grad_input, 
                                    float *grad_weights, float *grad_bias, int batch_size, int in_dim, int out_dim);

__global__ void update_weights_kernel(float *weights, const float *grad_weights, float learning_rate, int size);

#endif 