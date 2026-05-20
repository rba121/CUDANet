#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void neural_network(const float* X, const float* W, const float* b, float* Y, int batch_size, int input_dim, int output_dim) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < batch_size && col < output_dim) 
    {
        float sum = b[col];

        for (int k = 0; k < input_dim; k++) 
        {
            sum += X[row * input_dim + k] * W[k * output_dim + col];
        }

        // ReLU
        Y[row * output_dim + col] = (sum > 0.0f) ? sum : 0.0f;
    }
}

int main() {
    const int batch_size = 2;
    const int input_dim = 3;
    const int output_dim = 4;

    size_t X_size = batch_size * input_dim * sizeof(float);
    size_t W_size = input_dim * output_dim * sizeof(float);
    size_t b_size = output_dim * sizeof(float);
    size_t Y_size = batch_size * output_dim * sizeof(float);
    //size_t h_size = input_dim * sizeof(float);


    // Host 
    float h_X[6] = 
    {
        1, 2, 3,
        4, 5, 6
    };

    float h_W[12] = 
    {
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2
    };

    float h_b[4] = {1, 1, 1, 1};
    float h_Y[8];

    float *d_X, *d_W, *d_b, *d_Y;
    cudaMalloc(&d_X, X_size);
    cudaMalloc(&d_W, W_size);
    cudaMalloc(&d_b, b_size);
    cudaMalloc(&d_Y, Y_size);

    cudaMemcpy(d_X, h_X, X_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, W_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks(
        (output_dim + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );
    //dim3 blocks(threads.x + threads.y + batch_size*output_dim - 1);


    neural_network<<<blocks, threads>>>(
        d_X, d_W, d_b, d_Y,
        batch_size, input_dim, output_dim
    );

    cudaMemcpy(h_Y, d_Y, Y_size, cudaMemcpyDeviceToHost);

    
    cout << "Output:\n";
    for (int i = 0; i < batch_size; i++) 
    {
        for (int j = 0; j < output_dim; j++) 
        {
            cout << h_Y[i * output_dim + j] << " ";
        }
        cout << "\n";
    }

    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_Y);

    return 0;
}
