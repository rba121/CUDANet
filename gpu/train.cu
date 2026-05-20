#include <iostream>
using namespace std;

#include "train.h"

void train_gpu(Sequential_GPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs)
{
    MSE_GPU mse(bs);

    int sz_inp = bs*n_in;
    float *cp_inp, *out;
    cudaMallocManaged(&cp_inp, sz_inp*sizeof(float));

    for (int i=0; i<n_epochs; i++)
    {
        set_eq(cp_inp, inp, sz_inp);

        seq.forward(cp_inp, out);
        mse.forward(seq.layers.back()->out, targ);

        mse.backward();
        seq.update();
    }

    seq.forward(inp, out);
    mse._forward;
    d(seq.layers.back()->out, targ);
    cout << "The final loss is: " << targ[bs] <<endl;
}
