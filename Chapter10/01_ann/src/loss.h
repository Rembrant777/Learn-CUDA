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