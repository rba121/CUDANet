#ifndef RELU_H
#define RELU_H

#include <cuda_runtime.h>
#include "linear.h"  

void relu_forward(const float *input, float *output, int size);

void relu_backward(const float *input, const float *grad_output, float *grad_input, int size);

#endif 
