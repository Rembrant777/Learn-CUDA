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