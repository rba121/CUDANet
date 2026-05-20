#ifndef RELU_GPU_H
#define RELU_GPU_H


class relu_GPU: public Module
{
    public:
        int n_blocks;

        relu_GPU(int sz_out);
        void forward(float *rinp, float *rout);
        void backward();
        
        //new methods
        void funcgen(float *val);
};


#endif