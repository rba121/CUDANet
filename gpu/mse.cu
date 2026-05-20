#include "mse.h"
using namespace std;

void MSE_GPU::forward(float *_inp, float *_out)
{
    int* _sz_out;
    float *curr_in;

    for(int i=0;i<layers.size()-1;i++)
    {
        Module *layer = layers[i];
        sz_out = layer->sz_out;

        cudaMallocManaged(&curr_out, sz_out*sizeof(float));
        layer->forward(inp, curr_out);

        inp = curr_out;  
         
    }

}
