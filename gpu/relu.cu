#include "relu.h"

void relu_GPU(float *sz_out, vector<Module*> layers)
{
    int sz_in;
    float* curr_in;

    for(int i = layers.size() - 1 ; i>=0; i--)
    {
        sz_in = layers->sz_in;
        
    }

    rinp = curr_in;

    n_blocks = (16,16);

}

void relu_forward_gpu(float *inp, vector<module> layers, float *out)
{
    int sz_in;
    float *curr_in = inp;
    

    cudaMallocManaged(inp, sizeof(float));
    cudaMallocManaged(curr_in, sizeof(float));

    int threadsperblock = 256;
    int blocksperthread = (sz_in + threadsperblock -1)/ threadsperblock;
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventRecord(start);
    relu_GPU<<<blocksperthread, threadsperblock>>>(inp, layers, out);
    cudaEventRecord(stop);

    cudaFree(inp);
    cudaFree(curr_in);

}

void relu_update_layers(vector<Module> layers)
{
    dim3 n_threads(16, 16);
    dim3 blocks((n_threads + threadIdx - 1)/n_threads);
    for(int i = layers.size() -1; i>=0; i--)
    {
        Module* layer = layers[i];
        layer->update();
        layer->backward();
    }
}

void relu_GPU::update()
{
    relu_update_layers(layers);
}

void relu_GPU::forward(float *inp, float *out)
{
    relu_forward_gpu(inp, layers, out);
}