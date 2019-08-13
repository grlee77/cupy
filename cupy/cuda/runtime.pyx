"""Thin wrapper of CUDA Runtime API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDARuntimeError exceptions.
3. The 'cuda' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""
cimport cpython  # NOQA
cimport cython  # NOQA

from cupy.cuda cimport driver

cdef class PointerAttributes:

    def __init__(self, int device, intptr_t devicePointer,
                 intptr_t hostPointer, int isManaged, int memoryType):
        self.device = device
        self.devicePointer = devicePointer
        self.hostPointer = hostPointer
        self.isManaged = isManaged
        self.memoryType = memoryType


cdef class PitchedPtr:

    def __init__(self, size_t pitch, intptr_t ptr, size_t xsize, size_t ysize):
        self.pitch = pitch
        self.ptr = ptr
        self.xsize = xsize
        self.ysize = ysize


cdef class Pos:

    def __init__(self, size_t x, size_t y, size_t z):
        self.x = x
        self.y = y
        self.z = z


cdef class Extent:

    def __init__(self, size_t depth, size_t height, size_t width):
        self.depth = depth
        self.height = height
        self.width = width


class FormatDesc:

    def __init__(self, int f, int w, int x, int y, int z):
        self.f = f
        self.w = w
        self.x = x
        self.y = y
        self.z = z


# cdef class Memcpy3DParms:

#     def __init__(self, driver.Array dstArray, _Pos dstPos, PitchedPtr dstPtr,
#                  Extent extent, MemoryKind kind, driver.Array srcArray,
#                  Pos srcPos, PitchedPtr srcPtr):
#         self.dstArray = dstArray
#         self.dstPos = dstPos
#         self.dstPtr = dstPtr
#         self.extent = extent
#         self.kind = kind
#         self.srcArray = srcArray
#         self.srcPos = srcPos
#         self.srcPtr = srcPtr


###############################################################################
# Extern
###############################################################################
cdef extern from *:
    ctypedef int DeviceAttr 'enum cudaDeviceAttr'
    ctypedef int MemoryAdvise 'enum cudaMemoryAdvise'
    ctypedef int MemoryKind 'enum cudaMemcpyKind'
    ctypedef int ChannelFormatKind 'enum cudaChannelFormatKind'

    ctypedef void StreamCallbackDef(
        driver.Stream stream, Error status, void* userData)
    ctypedef StreamCallbackDef* StreamCallback 'cudaStreamCallback_t'


cdef extern from 'cupy_cuda.h' nogil:

    # Types
    struct _PointerAttributes 'cudaPointerAttributes':
        int device
        void* devicePointer
        void* hostPointer
        int isManaged
        int memoryType

    struct _PitchedPtr 'cudaPitchedPtr':
        size_t pitch
        void* ptr
        size_t xsize
        size_t ysize

    struct _Pos 'cudaPos':
        size_t x
        size_t y
        size_t z

    struct _Extent 'cudaExtent':
        size_t depth
        size_t height
        size_t width

    struct _Memcpy3DParms 'cudaMemcpy3DParms':
        driver.Array dstArray
        _Pos dstPos
        _PitchedPtr dstPtr
        _Extent extent
        MemoryKind kind
        driver.Array srcArray
        _Pos srcPos
        _PitchedPtr srcPtr

    struct _ChannelFormatDesc 'cudaChannelFormatDesc':
        ChannelFormatKind f
        int w
        int x
        int y
        int z

    # Error handling
    const char* cudaGetErrorName(Error error)
    const char* cudaGetErrorString(Error error)
    int cudaGetLastError()

    # Initialization
    int cudaDriverGetVersion(int* driverVersion)
    int cudaRuntimeGetVersion(int* runtimeVersion)

    # Device operations
    int cudaGetDevice(int* device)
    int cudaDeviceGetAttribute(int* value, DeviceAttr attr, int device)
    int cudaGetDeviceCount(int* count)
    int cudaSetDevice(int device)
    int cudaDeviceSynchronize()

    int cudaDeviceCanAccessPeer(int* canAccessPeer, int device,
                                int peerDevice)
    int cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)

    # Memory management
    int cudaMalloc(void** devPtr, size_t size)
    int cudaMallocManaged(void** devPtr, size_t size, unsigned int flags)
    int cudaHostAlloc(void** ptr, size_t size, unsigned int flags)
    int cudaHostRegister(void *ptr, size_t size, unsigned int flags)
    int cudaHostUnregister(void *ptr)
    int cudaFree(void* devPtr)
    int cudaFreeHost(void* ptr)
    int cudaMemGetInfo(size_t* free, size_t* total)
    int cudaMallocPitch(void** devPtr, size_t *pitch, size_t width, size_t height) nogil
    int cudaMallocArray(driver.Array* array, _ChannelFormatDesc* desc,
                        size_t width, size_t height, unsigned int flags) nogil
    int cudaMalloc3DArray(driver.Array* array, _ChannelFormatDesc* desc,
                          _Extent extent, unsigned int flags) nogil
    # TODO: RUNTIME API: cudaArrayGetInfo ( cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array )
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g373dacf191566b0bf5e5b807517b6bf9
    # TODO: RUNTIME API:  cudaFreeArray ( cudaArray_t array )
    #                     cudaMemcpy2DArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )
    #                     cudaMemcpy2DFromArray ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind )
    #                     cudaMemcpy2DFromArrayAsync ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )
    #                     cudaMemcpy2DToArray ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )
    #                     cudaMemcpy2DToArrayAsync ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )
    #                     cudaMemcpyArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )
    #                     cudaMemcpyFromArray ( void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind )
    #                     cudaMemcpyFromArrayAsync ( void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 )
    #                     cudaMemcpyToArray ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind )
    #                     cudaMemcpyToArrayAsync ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 )
    #                     cudaMemset2D ( void* devPtr, size_t pitch, int  value, size_t width, size_t height )
    #                     cudaMemset2DAsync ( void* devPtr, size_t pitch, int  value, size_t width, size_t height, cudaStream_t stream = 0 )
    #                     cudaMemset3D ( cudaPitchedPtr pitchedDevPtr, int  value, cudaExtent extent )
    #                     cudaMemset3DAsync ( cudaPitchedPtr pitchedDevPtr, int  value, cudaExtent extent, cudaStream_t stream = 0 )
    #
    # DRIVER API?: CUresult cuArray3DGetDescriptor ( CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray )
    # DRIVER API?: CUresult cuArrayGetDescriptor ( CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray )
    # DRIVER API?: struct CUDA_ARRAY_DESCRIPTOR
    # DRIVER API?: struct CUDA_ARRAY3D_DESCRIPTOR

    int cudaMemcpy(void* dst, const void* src, size_t count,
                   MemoryKind kind)
    int cudaMemcpyAsync(void* dst, const void* src, size_t count,
                        MemoryKind kind, driver.Stream stream)
    int cudaMemcpyPeer(void* dst, int dstDevice, const void* src,
                       int srcDevice, size_t count)
    int cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                            int srcDevice, size_t count,
                            driver.Stream stream)
    int cudaMemcpy2D(void* dst, size_t dpitch, const void* src,
                     size_t spitch, size_t width, size_t height,
                     MemoryKind kind)
    int cudaMemcpy3D(_Memcpy3DParms* p)
    int cudaMemset(void* devPtr, int value, size_t count)
    int cudaMemsetAsync(void* devPtr, int value, size_t count,
                        driver.Stream stream)
    int cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice,
                             driver.Stream stream)
    int cudaMemAdvise(const void *devPtr, size_t count,
                      MemoryAdvise advice, int device)
    int cudaPointerGetAttributes(_PointerAttributes* attributes,
                                 const void* ptr)

    # Stream and Event
    int cudaStreamCreate(driver.Stream* pStream)
    int cudaStreamCreateWithFlags(driver.Stream* pStream,
                                  unsigned int flags)
    int cudaStreamDestroy(driver.Stream stream)
    int cudaStreamSynchronize(driver.Stream stream)
    int cudaStreamAddCallback(driver.Stream stream, StreamCallback callback,
                              void* userData, unsigned int flags)
    int cudaStreamQuery(driver.Stream stream)
    int cudaStreamWaitEvent(driver.Stream stream, driver.Event event,
                            unsigned int flags)
    int cudaEventCreate(driver.Event* event)
    int cudaEventCreateWithFlags(driver.Event* event, unsigned int flags)
    int cudaEventDestroy(driver.Event event)
    int cudaEventElapsedTime(float* ms, driver.Event start,
                             driver.Event end)
    int cudaEventQuery(driver.Event event)
    int cudaEventRecord(driver.Event event, driver.Stream stream)
    int cudaEventSynchronize(driver.Event event)


###############################################################################
# Error codes
###############################################################################

errorInvalidValue = cudaErrorInvalidValue
errorMemoryAllocation = cudaErrorMemoryAllocation


###############################################################################
# Error handling
###############################################################################

class CUDARuntimeError(RuntimeError):

    def __init__(self, status):
        self.status = status
        cdef bytes name = cudaGetErrorName(<Error>status)
        cdef bytes msg = cudaGetErrorString(<Error>status)
        super(CUDARuntimeError, self).__init__(
            '%s: %s' % (name.decode(), msg.decode()))

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        # to reset error status
        cudaGetLastError()
        raise CUDARuntimeError(status)


###############################################################################
# Initialization
###############################################################################

cpdef int driverGetVersion() except? -1:
    cdef int version
    status = cudaDriverGetVersion(&version)
    check_status(status)
    return version


cpdef int runtimeGetVersion() except? -1:
    cdef int version
    status = cudaRuntimeGetVersion(&version)
    check_status(status)
    return version


###############################################################################
# Device and context operations
###############################################################################

cpdef int getDevice() except? -1:
    cdef int device
    status = cudaGetDevice(&device)
    check_status(status)
    return device


cpdef int deviceGetAttribute(int attrib, int device) except? -1:
    cdef int ret
    status = cudaDeviceGetAttribute(&ret, <DeviceAttr>attrib, device)
    check_status(status)
    return ret


cpdef int getDeviceCount() except? -1:
    cdef int count
    status = cudaGetDeviceCount(&count)
    check_status(status)
    return count


cpdef setDevice(int device):
    status = cudaSetDevice(device)
    check_status(status)


cpdef deviceSynchronize():
    with nogil:
        status = cudaDeviceSynchronize()
    check_status(status)


cpdef int deviceCanAccessPeer(int device, int peerDevice) except? -1:
    cpdef int ret
    status = cudaDeviceCanAccessPeer(&ret, device, peerDevice)
    check_status(status)
    return ret


cpdef deviceEnablePeerAccess(int peerDevice):
    status = cudaDeviceEnablePeerAccess(peerDevice, 0)
    check_status(status)


###############################################################################
# Memory management
###############################################################################

cpdef intptr_t malloc(size_t size) except? 0:
    cdef void* ptr
    with nogil:
        status = cudaMalloc(&ptr, size)
    check_status(status)
    return <intptr_t>ptr


cpdef intptr_t mallocManaged(
        size_t size, unsigned int flags=cudaMemAttachGlobal) except? 0:
    cdef void* ptr
    with nogil:
        status = cudaMallocManaged(&ptr, size, flags)
    check_status(status)
    return <intptr_t>ptr


cpdef intptr_t mallocPitch(
        size_t width, size_t height) except? 0:
    cdef void* ptr
    cdef size_t pitch
    with nogil:
        status = cudaMallocPitch(&ptr, &pitch, width, height)
    check_status(status)
    return <intptr_t>ptr, pitch


# cpdef intptr_t mallocArray(
#         _ChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags) except? 0:
cpdef intptr_t mallocArray(
        object format_desc, size_t width, size_t height, unsigned int flags) except? 0:
    cdef void* arr
    cdef _ChannelFormatDesc _desc
    _desc.f = <ChannelFormatKind>format_desc.f
    _desc.w = format_desc.w
    _desc.x = format_desc.x
    _desc.y = format_desc.y
    _desc.z = format_desc.z
    with nogil:
        status = cudaMallocArray(<driver.Array*>arr, &_desc, width, height, flags)
        # status = cudaMallocArray(<driver.Array*>arr, desc, width, height, flags)
    check_status(status)
    return <intptr_t>arr


# cpdef intptr_t malloc3DArray(
#         _ChannelFormatDesc *desc, _Extent extent, unsigned int flags) except? 0:
cpdef intptr_t malloc3DArray(
        object format_desc, _Extent extent, unsigned int flags) except? 0:
    cdef void* arr
    cdef _ChannelFormatDesc _desc
    _desc.f = <ChannelFormatKind>format_desc.f
    _desc.w = format_desc.w
    _desc.x = format_desc.x
    _desc.y = format_desc.y
    _desc.z = format_desc.z
    with nogil:
        status = cudaMalloc3DArray(<driver.Array*>arr, &_desc, extent, flags)
        # status = cudaMalloc3DArray(<driver.Array*>arr, desc, extent, flags)
    check_status(status)
    return <intptr_t>arr


cpdef intptr_t hostAlloc(size_t size, unsigned int flags) except? 0:
    cdef void* ptr
    with nogil:
        status = cudaHostAlloc(&ptr, size, flags)
    check_status(status)
    return <intptr_t>ptr


cpdef hostRegister(intptr_t ptr, size_t size, unsigned int flags):
    with nogil:
        status = cudaHostRegister(<void*>ptr, size, flags)
    check_status(status)


cpdef hostUnregister(intptr_t ptr):
    with nogil:
        status = cudaHostUnregister(<void*>ptr)
    check_status(status)


cpdef free(intptr_t ptr):
    with nogil:
        status = cudaFree(<void*>ptr)
    check_status(status)


cpdef freeHost(intptr_t ptr):
    with nogil:
        status = cudaFreeHost(<void*>ptr)
    check_status(status)


cpdef memGetInfo():
    cdef size_t free, total
    status = cudaMemGetInfo(&free, &total)
    check_status(status)
    return free, total


cpdef memcpy(intptr_t dst, intptr_t src, size_t size, int kind):
    with nogil:
        status = cudaMemcpy(<void*>dst, <void*>src, size, <MemoryKind>kind)
    check_status(status)


cpdef memcpyAsync(intptr_t dst, intptr_t src, size_t size, int kind,
                  size_t stream):
    with nogil:
        status = cudaMemcpyAsync(
            <void*>dst, <void*>src, size, <MemoryKind>kind,
            <driver.Stream>stream)
    check_status(status)


cpdef memcpyPeer(intptr_t dst, int dstDevice, intptr_t src, int srcDevice,
                 size_t size):
    with nogil:
        status = cudaMemcpyPeer(<void*>dst, dstDevice, <void*>src, srcDevice,
                                size)
    check_status(status)


cpdef memcpyPeerAsync(intptr_t dst, int dstDevice, intptr_t src, int srcDevice,
                      size_t size, size_t stream):
    with nogil:
        status = cudaMemcpyPeerAsync(<void*>dst, dstDevice, <void*>src,
                                     srcDevice, size, <driver.Stream> stream)
    check_status(status)


cpdef memcpy2D(intptr_t dst, size_t dpitch, intptr_t src, size_t spitch,
               size_t width, size_t height, int kind):
    with nogil:
        status = cudaMemcpy2D(<void*>dst, dpitch, <void*> src, spitch, width,
                              height, <MemoryKind> kind)
    check_status(status)


cpdef memset(intptr_t ptr, int value, size_t size):
    with nogil:
        status = cudaMemset(<void*>ptr, value, size)
    check_status(status)


cpdef memsetAsync(intptr_t ptr, int value, size_t size, size_t stream):
    with nogil:
        status = cudaMemsetAsync(<void*>ptr, value, size,
                                 <driver.Stream> stream)
    check_status(status)

cpdef memPrefetchAsync(intptr_t devPtr, size_t count, int dstDevice,
                       size_t stream):
    with nogil:
        status = cudaMemPrefetchAsync(<void*>devPtr, count, dstDevice,
                                      <driver.Stream> stream)
    check_status(status)

cpdef memAdvise(intptr_t devPtr, size_t count, int advice, int device):
    with nogil:
        status = cudaMemAdvise(<void*>devPtr, count,
                               <MemoryAdvise>advice, device)
    check_status(status)


cpdef PointerAttributes pointerGetAttributes(intptr_t ptr):
    cdef _PointerAttributes attrs
    status = cudaPointerGetAttributes(&attrs, <void*>ptr)
    check_status(status)
    return PointerAttributes(
        attrs.device,
        <intptr_t>attrs.devicePointer,
        <intptr_t>attrs.hostPointer,
        attrs.isManaged, attrs.memoryType)


###############################################################################
# Stream and Event
###############################################################################

cpdef size_t streamCreate() except? 0:
    cdef driver.Stream stream
    status = cudaStreamCreate(&stream)
    check_status(status)
    return <size_t>stream


cpdef size_t streamCreateWithFlags(unsigned int flags) except? 0:
    cdef driver.Stream stream
    status = cudaStreamCreateWithFlags(&stream, flags)
    check_status(status)
    return <size_t>stream


cpdef streamDestroy(size_t stream):
    status = cudaStreamDestroy(<driver.Stream>stream)
    check_status(status)


cpdef streamSynchronize(size_t stream):
    with nogil:
        status = cudaStreamSynchronize(<driver.Stream>stream)
    check_status(status)


cdef _streamCallbackFunc(driver.Stream hStream, int status,
                         void* func_arg) with gil:
    obj = <object>func_arg
    func, arg = obj
    func(<size_t>hStream, status, arg)
    cpython.Py_DECREF(obj)


cpdef streamAddCallback(size_t stream, callback, intptr_t arg,
                        unsigned int flags=0):
    func_arg = (callback, arg)
    cpython.Py_INCREF(func_arg)
    with nogil:
        status = cudaStreamAddCallback(
            <driver.Stream>stream, <StreamCallback>_streamCallbackFunc,
            <void*>func_arg, flags)
    check_status(status)


cpdef streamQuery(size_t stream):
    return cudaStreamQuery(<driver.Stream>stream)


cpdef streamWaitEvent(size_t stream, size_t event, unsigned int flags=0):
    with nogil:
        status = cudaStreamWaitEvent(<driver.Stream>stream,
                                     <driver.Event>event, flags)
    check_status(status)


cpdef size_t eventCreate() except? 0:
    cdef driver.Event event
    status = cudaEventCreate(&event)
    check_status(status)
    return <size_t>event

cpdef size_t eventCreateWithFlags(unsigned int flags) except? 0:
    cdef driver.Event event
    status = cudaEventCreateWithFlags(&event, flags)
    check_status(status)
    return <size_t>event


cpdef eventDestroy(size_t event):
    status = cudaEventDestroy(<driver.Event>event)
    check_status(status)


cpdef float eventElapsedTime(size_t start, size_t end) except? 0:
    cdef float ms
    status = cudaEventElapsedTime(&ms, <driver.Event>start, <driver.Event>end)
    check_status(status)
    return ms


cpdef eventQuery(size_t event):
    return cudaEventQuery(<driver.Event>event)


cpdef eventRecord(size_t event, size_t stream):
    status = cudaEventRecord(<driver.Event>event, <driver.Stream>stream)
    check_status(status)


cpdef eventSynchronize(size_t event):
    with nogil:
        status = cudaEventSynchronize(<driver.Event>event)
    check_status(status)


##############################################################################
# util
##############################################################################

cdef int _context_initialized = cpython.PyThread_create_key()


cdef _ensure_context():
    """Ensure that CUcontext bound to the calling host thread exists.

    See discussion on https://github.com/cupy/cupy/issues/72 for details.
    """
    cdef size_t status
    status = <size_t>cpython.PyThread_get_key_value(_context_initialized)
    if status == 0:
        # Call Runtime API to establish context on this host thread.
        memGetInfo()
        cpython.PyThread_set_key_value(_context_initialized, <void *>1)
