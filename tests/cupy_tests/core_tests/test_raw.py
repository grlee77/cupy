import unittest

import pytest

import cupy


_test_source = r'''
extern "C" __global__
void test_sum(const float* x1, const float* x2, float* y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + x2[tid];
}
'''


class TestRaw(unittest.TestCase):

    def setUp(self):
        self.kern = cupy.RawKernel(_test_source, 'test_sum')

    def test_basic(self):
        x1 = cupy.arange(100, dtype=cupy.float32).reshape(10, 10)
        x2 = cupy.ones((10, 10), dtype=cupy.float32)
        y = cupy.zeros((10, 10), dtype=cupy.float32)
        self.kern((10,), (10,), (x1, x2, y))
        assert (y == x1 + x2).all()

    def test_kernel_attributes(self):
        attributes = self.kern.attributes

        for key in ['binaryVersion',
                    'cacheModeCA',
                    'constSizeBytes',
                    'localSizeBytes',
                    'maxDynamicSharedSizeBytes',
                    'maxThreadsPerBlock',
                    'numRegs',
                    'preferredShmemCarveout',
                    'ptxVersion',
                    'sharedSizeBytes']:
            assert key in attributes

        assert attributes['numRegs'] > 0
        assert attributes['maxThreadsPerBlock'] > 0
        assert attributes['sharedSizeBytes'] == 0

    def test_set_max_dynamic_shared_size_bytes(self):
        current_dynamic_mem = self.kern.attributes['maxDynamicSharedSizeBytes']
        # don't try to set on hardware when the attribute is undefined (-1)
        if current_dynamic_mem >= 0:
            new_dynamic_mem = current_dynamic_mem // 2
            self.kern.set_max_dynamic_shared_size_bytes(new_dynamic_mem)
            updated_val = self.kern.attributes['maxDynamicSharedSizeBytes']
            assert updated_val == new_dynamic_mem

    def test_set_preferred_shmem_carveout(self):
        current_percentage = self.kern.attributes['preferredShmemCarveout']
        # don't try to set on hardware when the attribute is undefined (-1)
        if current_percentage >= 0:
            new_percentage = 50
            self.kern.set_preferred_shmem_carveout(new_percentage)
            updated_val = self.kern.attributes['preferredShmemCarveout']
            assert updated_val == new_percentage

        # ValueError for percentages outside of the range [0, 100]
        with pytest.raises(ValueError):
            self.kern.set_preferred_shmem_carveout(-1)
        with pytest.raises(ValueError):
            self.kern.set_preferred_shmem_carveout(101)
