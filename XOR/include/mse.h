#ifndef MSE_H
#define MSE_H

#include <cuda_runtime.h>
#include "linear.h"  

/**
 * Mean Squared Error (MSE) Loss
 * L = (1/N) * Σ(prediction - target)²
 * Used for regression problems
 */

__global__ void mse_loss_kernel(const float *pred, const float *target, float *loss, int size);

__global__ void mse_grad_kernel(const float *pred, const float *target, float *grad, int size);

float compute_mse(const float *d_pred, const float *d_target, int size);

void mse_backward(const float *d_pred, const float *d_target, 
                  float *d_grad, int size);

#endif 