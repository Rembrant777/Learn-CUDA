#ifndef _LOSS_H_
#define _LOSS_H_

#include "blob.h"
namespace cudl { 

class CrossEntropyLoss {
public:
    CrossEntropyLoss(); 
    ~CrossEntropyLoss(); 

    float loss(Blob<float> *predict, Blob<float> *target); 
    float accuracy(Blob<float> *predict, Blob<float> *target);

    // set function for testing  
    void set_h_loss(float h_loss) {
        h_loss_ = h_loss; 
    } 

    // get function for testing 
    float get_h_loss() {
        return h_loss_; 
    }

    // get function for testing 
    float* get_d_loss() {
        return d_loss_; 
    }

    // get function for testing 
    float *get_d_workspace() {
        return d_workspace_; 
    }

    // add for testing 
    void test_invoke_init_workspace(int batch_size) {
        init_workspace(batch_size); 
    }

    // add for testing clip 
    float test_clip(float prediction, float epsilon = 1e-12) {
        return fmin(fmax(prediction, epsilon), 1.f - epsilon); 
    }

    int get_cuda_dev_num_sms(); 

    int get_num_blocks_per_sm(); 

private:
    // reduced loss 
    // todo here we can support more loss functions 
    float h_loss_ = 0.f; 
    float *d_loss_ = nullptr; 
    float *d_workspace_ = nullptr; 
    void init_workspace(int batch_size); 

}; // class CrossEntropyLoss    

} // namespace cudl 

#endif  // _LOSS_H_