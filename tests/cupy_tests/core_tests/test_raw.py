import unittest

import cupy


_test_source = r'''
extern "C" __global__
void test_sum(const float* x1, const float* x2, float* y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + x2[tid];
}
'''


class TestRaw(unittest.TestCase):
    def test_basic(self):
        kern = cupy.RawKernel(_test_source, 'test_sum')
        x1 = cupy.arange(100, dtype=cupy.float32).reshape(10, 10)
        x2 = cupy.ones((10, 10), dtype=cupy.float32)
        y = cupy.zeros((10, 10), dtype=cupy.float32)
        kern((10,), (10,), (x1, x2, y))
        assert (y == x1 + x2).all()

        attributes = kern.attributes
        assert hasattr(attributes, 'binaryVersion')
        assert hasattr(attributes, 'cacheModeCA')
        assert hasattr(attributes, 'constSizeBytes')
        assert hasattr(attributes, 'localSizeBytes')
        assert hasattr(attributes, 'maxDynamicSharedSizeBytes')
        assert hasattr(attributes, 'maxThreadsPerBlock')
        assert hasattr(attributes, 'numRegs')
        assert hasattr(attributes, 'preferredShmemCarveout')
        assert hasattr(attributes, 'ptxVersion')
        assert hasattr(attributes, 'sharedSizeBytes')
        assert attributes.numRegs > 0
        assert attributes.maxThreadsPerBlock > 0
        assert attributes.sharedSizeBytes == 0
