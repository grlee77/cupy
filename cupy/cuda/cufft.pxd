cdef extern from *:
    ctypedef float Float 'cufftReal'
    ctypedef double Double 'cufftDoubleReal'
    ctypedef int Result 'cufftResult_t'
    ctypedef int Handle 'cufftHandle'
    ctypedef int Type 'cufftType_t'
# should have some check similar to:
#if defined(__CUDACC_VER_MAJOR__) && ((__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ >= 2) || (__CUDACC_VER_MAJOR__ > 9))
    ctypedef int WorkAreaPolicy 'cufftXtWorkAreaPolicy_t'


cpdef enum:
    CUFFT_C2C = 0x29
    CUFFT_R2C = 0x2a
    CUFFT_C2R = 0x2c
    CUFFT_Z2Z = 0x69
    CUFFT_D2Z = 0x6a
    CUFFT_Z2D = 0x6c

    CUFFT_FORWARD = -1
    CUFFT_INVERSE = 1

    CUFFT_WORKAREA_MINIMAL = 0
    CUFFT_WORKAREA_USER = 1
    CUFFT_WORKAREA_PERFORMANCE = 2

cpdef get_current_plan()
