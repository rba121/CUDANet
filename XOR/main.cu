#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "include/linear.h"
#include "include/relu.h"
#include "include/mse.h"

using namespace std;

int main() 
{
    srand(time(NULL));
    cout << "Network: 2 -> 4 -> 1 (with ReLU)\n\n";

    // XOR dataset
    const int batch_size = 4;
    const int input_dim = 2;
    const int hidden_dim = 4;
    const int output_dim = 1;
    
    float h_X[8] = {0, 0,  0, 1,  1, 0,  1, 1};
    float h_Y[4] = {0, 1, 1, 0};
    
    // Allocate device memory for data
    float *d_X, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, batch_size * input_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, batch_size * output_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    LinearLayer layer1(input_dim, hidden_dim, batch_size, 0.1f);
    LinearLayer layer2(hidden_dim, output_dim, batch_size, 0.1f);
    
    // Allocate intermediate activations
    float *d_hidden, *d_hidden_relu, *d_grad_output, *d_grad_hidden, *d_grad_hidden_relu;
    CUDA_CHECK(cudaMalloc(&d_hidden, batch_size * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden_relu, batch_size * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * output_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_hidden, batch_size * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_hidden_relu, batch_size * hidden_dim * sizeof(float)));
    
    const int epochs = 5000;
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        // Forward pass
        layer1.forward(d_X);
        relu_forward(layer1.d_output, d_hidden_relu, batch_size * hidden_dim);
        layer2.forward(d_hidden_relu);
        
        //loss
        if (epoch % 500 == 0) 
        {
            float loss = compute_mse(layer2.d_output, d_Y, batch_size * output_dim);
            cout << "Epoch " << epoch << ": Loss = " << loss << "\n";
        }
        
        // Backward pass
        mse_backward(layer2.d_output, d_Y, d_grad_output, batch_size * output_dim);
        layer2.backward(d_grad_output);
        relu_backward(layer1.d_output, layer2.d_grad_input, d_grad_hidden_relu, batch_size * hidden_dim);
        layer1.backward(d_grad_hidden_relu);
        
        // Update weights
        layer2.update();
        layer1.update();
    }
    
    cout << "\n=== Final Results ===\n";
    layer1.forward(d_X);
    relu_forward(layer1.d_output, d_hidden_relu, batch_size * hidden_dim);
    layer2.forward(d_hidden_relu);
    
    float h_predictions[4];
    CUDA_CHECK(cudaMemcpy(h_predictions, layer2.d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cout << "Input -> Predicted -> Target\n";
    for (int i = 0; i < batch_size; i++) 
    {
        cout << "[" << h_X[i*2] << ", " << h_X[i*2+1] << "] -> " << h_predictions[i] << " -> " << h_Y[i] << "\n";
    }
    
    float final_loss = compute_mse(layer2.d_output, d_Y, batch_size);
    cout << "\nFinal Loss: " << final_loss << "\n";
    
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_hidden);
    cudaFree(d_hidden_relu);
    cudaFree(d_grad_output);
    cudaFree(d_grad_hidden);
    cudaFree(d_grad_hidden_relu);
    
    return 0;
}