#ifndef _BLOB_H_
#define _BLOB_H_

#include <array>
#include <string>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include <cudnn.h>

namespace cudl
{
typedef enum {
    host, 
    cuda 
} DeviceType; 

template <typename ftype>
class Blob {
    
}

}

#endif // _BLOB_H_