import unittest
import warnings

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(
    {'shape': (2, 3, 4), 'index': Ellipsis},
    {'shape': (2, 3, 4), 'index': 0},
    {'shape': (2, 3, 4), 'index': 10},
    {'shape': (2, 3, 4), 'index': slice(None)},
    {'shape': (2, 3, 4), 'index': slice(None, 10)},
    {'shape': (2, 3, 4), 'index': slice(None, None, 2)},
    {'shape': (2, 3, 4), 'index': slice(None, None, -1)},
    {'shape': (2, 3, 4), 'index': slice(10, None, -1)},
    {'shape': (2, 3, 4), 'index': slice(10, None, -2)},
    {'shape': (), 'index': slice(None)},
    {'shape': (10,), 'index': slice(None)},
)
@testing.gpu
class TestFlatiterSubscript(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a.flat[self.index]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_scalar(self, xp, dtype):
        a = xp.zeros(self.shape, dtype=dtype)
        a.flat[self.index] = 1
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_ndarray_1d(self, xp, dtype):
        if numpy.isscalar(self.index):
            return xp.array([])  # skip
        a = xp.zeros(self.shape, dtype=dtype)
        v = testing.shaped_arange((3,), xp, dtype)
        a.flat[self.index] = v
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_ndarray_nd(self, xp, dtype):
        if numpy.isscalar(self.index):
            return xp.array([])  # skip
        a = xp.zeros(self.shape, dtype=dtype)
        v = testing.shaped_arange((2, 3), xp, dtype)
        a.flat[self.index] = v
        return a

    @testing.for_all_dtypes_combination(('a_dtype', 'v_dtype'))
    @testing.numpy_cupy_array_equal()
    def test_setitem_ndarray_different_types(self, xp, a_dtype, v_dtype):
        if numpy.isscalar(self.index):
            return xp.array([])  # skip
        a = xp.zeros(self.shape, dtype=a_dtype)
        v = testing.shaped_arange((3,), xp, v_dtype)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            a.flat[self.index] = v
        return a


@testing.parameterize(
    {'shape': (2, 3, 4), 'index': None},
    {'shape': (2, 3, 4), 'index': (0,)},
    {'shape': (2, 3, 4), 'index': True},
    {'shape': (2, 3, 4), 'index': cupy.array([0])},
    {'shape': (2, 3, 4), 'index': [0]},
)
@testing.gpu
class TestFlatiterSubscriptIndexError(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_getitem(self, dtype):
        a = testing.shaped_arange(self.shape, cupy, dtype)
        with pytest.raises(IndexError):
            a.flat[self.index]

    @testing.for_all_dtypes()
    def test_setitem(self, dtype):
        a = testing.shaped_arange(self.shape, cupy, dtype)
        v = testing.shaped_arange((1,), cupy, dtype)
        with pytest.raises(IndexError):
            a.flat[self.index] = v
