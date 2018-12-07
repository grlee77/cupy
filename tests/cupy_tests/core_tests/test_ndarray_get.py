import unittest

import cupy
from cupy import cuda
from cupy import testing
import numpy
from numpy import testing as np_testing


@testing.gpu
class TestArrayGet(unittest.TestCase):

    def setUp(self):
        self.stream = cuda.Stream.null

    def check_get(self, f, stream, order='C'):
        a_gpu = f(cupy)
        a_cpu = a_gpu.get(stream, order=order)
        if stream:
            stream.synchronize()
        b_cpu = f(numpy)
        np_testing.assert_array_equal(a_cpu, b_cpu)
        if order == 'F' or (order =='A' and a_gpu.flags.f_contiguous):
            assert a_cpu.flags.f_contiguous
        else:
            assert a_cpu.flags.c_contiguous

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_contiguous_array(self, dtype, order):
        def contiguous_array(xp, order):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype, order=order)
        self.check_get(contiguous_array, None, order=order)

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_non_contiguous_array(self, dtype, order):
        def non_contiguous_array(xp):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype,
                                         order=order)[0::2]
        self.check_get(non_contiguous_array, None, order)

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_contiguous_array_stream(self, dtype, order):
        def contiguous_array(xp):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype, order=order)
        self.check_get(contiguous_array, self.stream, order)

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_non_contiguous_array_stream(self, dtype, order):
        def non_contiguous_array(xp):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype,
                                         order=order)[0::2]
        self.check_get(non_contiguous_array, self.stream)

    @testing.multi_gpu(2)
    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_get_multigpu(self, dtype, order):
        with cuda.Device(1):
            src = testing.shaped_arange((2, 3), xp=cupy, dtype=dtype,
                                        order=order)
            src = cupy.asfortranarray(src)
        with cuda.Device(0):
            dst = src.get()
        expected = testing.shaped_arange((2, 3), xp=numpy, dtype=dtype,
                                         order=order)
        np_testing.assert_array_equal(dst, expected)
