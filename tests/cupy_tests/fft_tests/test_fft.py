import functools
import unittest

import numpy as np

import cupy
from cupy.fft import cache
from cupy.fft import config
from cupy.fft.fft import _default_plan_type

from cupy import testing

import six


def nd_planning_states(states=[True, False], name='enable_nd'):
    """Decorator for parameterized tests with and wihout nd planning

    Tests are repeated with config.enable_nd_planning set to True and False

    Args:
         states(list of bool): The boolean cases to test.
         name(str): Argument name to which specified dtypes are passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixture. Then, it runs the fixtures in parallel
    by passing the each element of ``dtypes`` to the named
    argument.
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            # get original global planning state
            planning_state = config.enable_nd_planning
            try:
                for nd_planning in states:
                    try:
                        # enable or disable nd planning
                        config.enable_nd_planning = nd_planning

                        kw[name] = nd_planning
                        impl(self, *args, **kw)
                    except Exception:
                        print(name, 'is', nd_planning)
                        raise
            finally:
                # restore original global planning state
                config.enable_nd_planning = planning_state

        return test_func
    return decorator


class CachedTestCase(unittest.TestCase):
    """Enable caching of CUFFT plans during the test."""

    def setUp(self):
        cache.enable()
        cache.set_keepalive_time(1.0)  # retain items in cache for 5 seconds

    def tearDown(self):
        cache.disable()


@testing.parameterize(*testing.product({
    'n': [None, 0, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'ortho', ''],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFft(CachedTestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft(a, n=self.n, norm=self.norm)

        # np.fft.fft alway returns np.complex128
        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.gpu
class TestFftCacheEnableDisable(CachedTestCase):

    def test_cache_is_enabled(self):
        assert cache.is_enabled()

        cache.disable()
        assert not cache.is_enabled()


@testing.gpu
@testing.slow
class TestFftAllocate(unittest.TestCase):

    def test_fft_allocate(self):
        # Check CuFFTError is not raised when the GPU memory is enough.
        # See https://github.com/cupy/cupy/issues/1063
        # TODO(mizuno): Simplify "a" after memory compaction is implemented.
        a = []
        for i in six.moves.range(10):
            a.append(cupy.empty(100000000))
        del a
        b = cupy.empty(100000007, dtype=cupy.float32)
        cupy.fft.fft(b)
        # Free huge memory for slow test
        del b
        cupy.get_default_memory_pool().free_all_blocks()


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFft2(CachedTestCase): # unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFftn(CachedTestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    # invalid shape
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    # plan_type='nd' can transform along up to 3 contiguous axes at start or end
    {'shape': (2, 3, 4, 5), 's': None, 'axes': (0, 1, 2), 'norm': None},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (0, 1, 2), 'norm': None},
    {'shape': (2, 3, 4, 5, 6), 's': None, 'axes': (-1, -2, -3), 'norm': None},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFftnPlanNd(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp == cupy:
            extra_kwargs = dict(plan_type='nd')
        else:
            extra_kwargs = {}

        out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm,
                           **extra_kwargs)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp == cupy:
            extra_kwargs = dict(plan_type='nd')
        else:
            extra_kwargs = {}
        out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm,
                           **extra_kwargs)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    def _out_shape(self, a_shape):
        """Return the expected output shape."""
        if self.s is None:
            return a_shape

        out_shape = []
        for n, s in enumerate(self.s):
            if s is None:
                out_shape.append(a_shape[n])
            else:
                out_shape.append(s)
        return tuple(out_shape)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_out(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp == cupy:
            out_shape = self._out_shape(a.shape)
            if (a.dtype in [np.complex64, np.complex128] and
                    out_shape == a.shape):
                # in-place transform possible for complex64 and complex128
                out = a
            else:
                if a.dtype in [np.float16, np.float32, np.complex64]:
                    out_dtype = np.complex64
                else:
                    out_dtype = np.complex128
                out = cupy.empty(out_shape,
                                 dtype=out_dtype)
            extra_kwargs = dict(plan_type='nd', out=out)

            xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm,
                         **extra_kwargs)
        else:
            out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_out(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp == cupy:
            out_shape = self._out_shape(a.shape)
            if (a.dtype in [np.complex64, np.complex128] and
                    out_shape == a.shape):
                # in-place transform possible for complex64 and complex128
                out = a
            else:
                if a.dtype in [np.float16, np.float32, np.complex64]:
                    out_dtype = np.complex64
                else:
                    out_dtype = np.complex128
                out = cupy.empty(out_shape,
                                 dtype=out_dtype)
            extra_kwargs = dict(plan_type='nd', out=out)

            xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm,
                        **extra_kwargs)
        else:
            out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_preplan(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp == cupy:
            out_shape = self._out_shape(a.shape)
            if a.dtype in [np.float16, np.float32, np.complex64]:
                fft_type = cupy.cuda.cufft.CUFFT_C2C
            else:
                fft_type = cupy.cuda.cufft.CUFFT_Z2Z

            plan = cupy.fft.get_cufft_plan_nd(
                out_shape, fft_type=fft_type, axes=self.axes,
                order='C')

            extra_kwargs = dict(plan_type='nd', plan=plan)
        else:
            extra_kwargs = {}

        out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm,
                           **extra_kwargs)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_preplan(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp == cupy:
            out_shape = self._out_shape(a.shape)
            if a.dtype in [np.float16, np.float32, np.complex64]:
                fft_type = cupy.cuda.cufft.CUFFT_C2C
            else:
                fft_type = cupy.cuda.cufft.CUFFT_Z2Z

            plan = cupy.fft.get_cufft_plan_nd(
                out_shape, fft_type=fft_type, axes=self.axes,
                order='C')

            extra_kwargs = dict(plan_type='nd', plan=plan)
        else:
            extra_kwargs = {}

        out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm,
                          **extra_kwargs)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': (-3, -2, -1), 'norm': None},
)
@testing.gpu
class TestFftnContiguity(unittest.TestCase):

    @nd_planning_states([True])
    @testing.for_all_dtypes()
    def test_fftn_orders(self, dtype, enable_nd):
        for order in ['C', 'F']:
            a = testing.shaped_random(self.shape, cupy, dtype)
            if order == 'F':
                a = cupy.asfortranarray(a)
            out = cupy.fft.fftn(a, s=self.s, axes=self.axes)

            plan_type = _default_plan_type(a, s=self.s, axes=self.axes)
            if plan_type == 'nd':
                # nd plans have output with contiguity matching the input
                self.assertEqual(out.flags.c_contiguous, a.flags.c_contiguous)
                self.assertEqual(out.flags.f_contiguous, a.flags.f_contiguous)
            else:
                # 1d planning case doesn't guarantee preserved contiguity
                pass

    @nd_planning_states([True])
    @testing.for_all_dtypes()
    def test_ifftn_orders(self, dtype, enable_nd):
        for order in ['C', 'F']:

            a = testing.shaped_random(self.shape, cupy, dtype)
            if order == 'F':
                a = cupy.asfortranarray(a)
            out = cupy.fft.ifftn(a, s=self.s, axes=self.axes)

            plan_type = _default_plan_type(a, s=self.s, axes=self.axes)
            if plan_type == 'nd':
                # nd plans have output with contiguity matching the input
                self.assertEqual(out.flags.c_contiguous, a.flags.c_contiguous)
                self.assertEqual(out.flags.f_contiguous, a.flags.f_contiguous)
            else:
                # 1d planning case doesn't guarantee preserved contiguity
                pass


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'ortho'],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestRfft(CachedTestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft(self, xp, dtype):
        # the scaling of old Numpy is incorrect
        if np.__version__ < np.lib.NumpyVersion('1.13.0'):
            if self.n is not None:
                return xp.empty(0)

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestRfft2(CachedTestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft2(self, xp, dtype):
        # the scaling of old Numpy is incorrect
        if np.__version__ < np.lib.NumpyVersion('1.13.0'):
            if self.s is not None:
                return xp.empty(0)

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)
        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestRfftn(CachedTestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'ortho'],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestHfft(CachedTestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_hfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.hfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ihfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ihfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'n': 1, 'd': 1},
    {'n': 10, 'd': 0.5},
    {'n': 100, 'd': 2},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFftfreq(CachedTestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fftfreq(self, xp, dtype):
        out = xp.fft.fftfreq(self.n, self.d)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfftfreq(self, xp, dtype):
        out = xp.fft.rfftfreq(self.n, self.d)

        return out


@testing.parameterize(
    {'shape': (5,), 'axes': None},
    {'shape': (5,), 'axes': 0},
    {'shape': (10,), 'axes': None},
    {'shape': (10,), 'axes': 0},
    {'shape': (10, 10), 'axes': None},
    {'shape': (10, 10), 'axes': 0},
    {'shape': (10, 10), 'axes': (0, 1)},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFftshift(CachedTestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fftshift(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fftshift(x, self.axes)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifftshift(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifftshift(x, self.axes)

        return out
