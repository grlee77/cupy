// This file is a stub header file of cufft for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUFFTXT_H
#define INCLUDE_GUARD_CUPY_CUFFTXT_H

#ifndef CUPY_NO_CUDA
#  include <cufftXt.h>

#else  // CUPY_NO_CUDA
extern "C" {

typedef enum {} cufftXtWorkAreaPolicy_t;

cufftResult_t cufftXtMakePlanMany(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftXtSetWorkArea(...) {
    return CUFFT_SUCCESS;
}

}  // extern "C"

#endif  // CUPY_NO_CUDA

#endif  // INCLUDE_GUARD_CUPY_CUFFTXT_H
