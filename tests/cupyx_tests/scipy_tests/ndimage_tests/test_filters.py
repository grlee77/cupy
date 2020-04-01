import unittest

import numpy

import cupy
from cupy import testing
import cupyx.scipy.ndimage  # NOQA

try:
    import scipy.ndimage  # NOQA
except ImportError:
    pass


@testing.parameterize(*(
    testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'mode': ['reflect'],
        'cval': [0.0],
        'origin': [0, 1, None],
        'adtype': [numpy.int8, numpy.int16, numpy.int32,
                   numpy.float32, numpy.float64],
        'wdtype': [None, numpy.int32, numpy.float64],
        'output': [None, numpy.int32, numpy.float64],
        'filter': ['convolve', 'correlate']
    }) + testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'mode': ['constant'],
        'cval': [-1.0, 0.0, 1.0],
        'origin': [0],
        'adtype': [numpy.int32, numpy.float64],
        'wdtype': [None],
        'output': [None],
        'filter': ['convolve', 'correlate']
    }) + testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'mode': ['nearest', 'mirror', 'wrap'],
        'cval': [0.0],
        'origin': [0],
        'adtype': [numpy.int32, numpy.float64],
        'wdtype': [None],
        'output': [None],
        'filter': ['convolve', 'correlate']
    })
))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolveAndCorrelate(unittest.TestCase):

    def _filter(self, xp, scp, a, w):
        filter = getattr(scp.ndimage, self.filter)
        if self.origin is None:
            origin = (-1, 1, -1, 1)[:a.ndim]
        else:
            origin = self.origin
        return filter(a, w, output=self.output, mode=self.mode,
                      cval=self.cval, origin=origin)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_convolve_and_correlate(self, xp, scp):
        if self.adtype == self.wdtype or self.adtype == self.output:
            return xp.array(True)
        a = testing.shaped_random(self.shape, xp, self.adtype)
        if self.wdtype is None:
            wdtype = self.adtype
        else:
            wdtype = self.wdtype
        w = testing.shaped_random((self.ksize,) * a.ndim, xp, wdtype)
        return self._filter(xp, scp, a, w)


@testing.parameterize(*(
    testing.product({
        'shape': [(3, 4), ],
        'ksize': [3, ],
        'mode': ['reflect', 'constant'],
        'adtype': [numpy.int32, numpy.float32, numpy.float64, numpy.complex64,
                   numpy.complex128],
        'wdtype': [numpy.int32, numpy.float32, numpy.float64, numpy.complex64,
                   numpy.complex128],
        'output': [None],
        'filter': ['convolve', 'correlate'],
        'single_precision': [True, False],
    })
))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolveAndCorrelateComplex(unittest.TestCase):

    def _filter(self, xp, scp, a, w):
        filter = getattr(scp.ndimage, self.filter)
        return filter(a, w, output=self.output, mode=self.mode,
                      single_precision=self.single_precision)

    def test_convolve_and_correlate_complex(self):
        a = testing.shaped_random(self.shape, cupy, self.adtype)
        if self.wdtype is None:
            wdtype = self.adtype
        else:
            wdtype = self.wdtype
        w = testing.shaped_random((self.ksize,) * a.ndim, cupy, wdtype)

        rtol = atol = 1e-5
        # The expected output in complex-valued cases can be computed from a
        # linear combination of real-valued convolution.
        if a.dtype.kind == 'c' and w.dtype.kind != 'c':
            # complex data, real filter
            expected_out = (
                self._filter(cupy, cupyx.scipy, a.real, w) +
                1j * self._filter(cupy, cupyx.scipy, a.imag, w))
        elif a.dtype.kind != 'c' and w.dtype.kind == 'c':
            # real data, complex filter
            a_float = a.astype(cupy.result_type(a.dtype, cupy.float32))
            expected_out = (
                self._filter(cupy, cupyx.scipy, a_float, w.real) +
                1j * self._filter(cupy, cupyx.scipy, a_float, w.imag))
        elif a.dtype.kind == 'c' and w.dtype.kind == 'c':
            # complex data, complex filter
            expected_out = (
                self._filter(cupy, cupyx.scipy, a.real, w.real) +
                1j * self._filter(cupy, cupyx.scipy, a.real, w.imag) +
                1j * self._filter(cupy, cupyx.scipy, a.imag, w.real) -
                self._filter(cupy, cupyx.scipy, a.imag, w.imag)
            )
        else:
            # real data, real filter cases compared to SciPy in other classes
            return

        out = self._filter(cupy, cupyx.scipy, a, w)

        testing.assert_allclose(
            out,
            expected_out,
            rtol=rtol, atol=atol)


@testing.parameterize(*testing.product({
    'ndim': [2, 3],
    'dtype': [numpy.int32, numpy.float64],
    'filter': ['convolve', 'correlate'],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolveAndCorrelateSpecialCases(unittest.TestCase):

    def _filter(self, scp, a, w, mode='reflect', origin=0, output=None):
        filter = getattr(scp.ndimage, self.filter)
        return filter(a, w, mode=mode, origin=origin, output=output)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_weights_with_size_zero_dim(self, xp, scp):
        a = testing.shaped_random((3, ) * self.ndim, xp, self.dtype)
        w = testing.shaped_random((0, ) + (3, ) * self.ndim, xp, self.dtype)
        return self._filter(scp, a, w)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_array_output(self, xp, scp):
        a = testing.shaped_random((3, ) * self.ndim, xp, self.dtype)
        w = testing.shaped_random((3, ) * self.ndim, xp, self.dtype)
        output = xp.empty_like(a)
        return self._filter(scp, a, w, output=output)

    # @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    # def test_inplace_output(self, xp, scp):
    #     a = testing.shaped_random((3, ) * self.ndim, xp, self.dtype)
    #     w = testing.shaped_random((3, ) * self.ndim, xp, self.dtype)
    #     return self._filter(scp, a, w, output=a)

    # TODO: SciPy works without error in this case
    def test_invalid_inplace_output(self):
        a = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        w = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        with self.assertRaises(RuntimeError):
            return self._filter(cupyx.scipy, a, w, output=a)

    @testing.numpy_cupy_raises(scipy_name='scp')
    def test_invalid_output_shape(self, xp, scp):
        a = testing.shaped_random((3, ) * self.ndim, xp, self.dtype)
        w = testing.shaped_random((3, ) * self.ndim, xp, self.dtype)
        output_shape = (2,) * self.ndim  # wrong shape
        output = xp.zeros(output_shape, dtype=a.dtype)
        self._filter(scp, a, w, output=output)

    def test_invalid_shape_weights(self):
        a = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        w = testing.shaped_random((3, ) * (self.ndim - 1), cupy, self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyx.scipy, a, w)
        w = testing.shaped_random((0, ) + (3, ) * (self.ndim - 1), cupy,
                                  self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyx.scipy, a, w)

    def test_invalid_mode(self):
        a = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        w = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyx.scipy, a, w, mode='unknown')

    # SciPy behavior fixed in 1.2.0: https://github.com/scipy/scipy/issues/822
    @testing.with_requires('scipy>=1.2.0')
    def test_invalid_origin(self):
        a = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        for lenw in [3, 4]:
            w = testing.shaped_random((lenw, ) * self.ndim, cupy, self.dtype)
            for origin in range(-3, 4):
                if (lenw // 2 + origin < 0) or (lenw // 2 + origin >= lenw):
                    with self.assertRaises(ValueError):
                        self._filter(cupyx.scipy, a, w, origin=origin)
                else:
                    self._filter(cupyx.scipy, a, w, origin=origin)


@testing.parameterize(*testing.product({
    'size': [3, 4],
    'footprint': [None, 'random'],
    'mode': ['reflect', 'constant', 'nearest', 'mirror', 'wrap'],
    'origin': [0, None],
    'x_dtype': [numpy.int32, numpy.float32],
    'output': [None, numpy.float64],
    'filter': ['minimum_filter', 'maximum_filter']
}))
@testing.gpu
@testing.with_requires('scipy')
class TestMinimumMaximumFilter(unittest.TestCase):

    shape = (4, 5)
    cval = 0.0

    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        if self.origin is None:
            origin = (-1, 1, -1, 1)[:x.ndim]
        else:
            origin = self.origin
        if self.footprint is None:
            footprint = None
        else:
            shape = (self.size, ) * x.ndim
            r = testing.shaped_random(shape, xp, scale=1)
            footprint = xp.where(r < .5, 1, 0)
            if not footprint.any():
                footprint = xp.ones(shape)
        return filter(x, size=self.size, footprint=footprint,
                      output=self.output, mode=self.mode, cval=self.cval,
                      origin=origin)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_minimum_and_maximum_filter(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.x_dtype)
        return self._filter(xp, scp, x)
