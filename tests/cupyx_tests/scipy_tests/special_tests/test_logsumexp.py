import cupy
import cupyx.scipy.special  # NOQA
import pytest
from cupy import testing

rtol = {
    'default': 1e-8,
    cupy.float16: 1e-2,
    cupy.float32: 1e-6,
    cupy.complex64: 1e-6
}


@testing.gpu
@testing.with_requires('scipy')
class TestLogSumExp():

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_logsumexp(self, xp, scp, dtype):
        a = xp.arange(200, dtype=dtype)
        return getattr(scp.special, 'logsumexp')(a)

    @testing.for_all_dtypes(no_bool=True, name='b_dtype')
    @testing.for_all_dtypes(no_bool=True, name='a_dtype')
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=rtol)  # , atol=atol)
    def test_logsumexp_b(self, xp, scp, a_dtype, b_dtype):
        a = xp.arange(200, dtype=a_dtype)
        b = xp.arange(200, 0, -1, dtype=b_dtype)
        return getattr(scp.special, 'logsumexp')(a, b=b)

    @testing.for_dtypes('efdhilHIL')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_logsumexp_large(self, xp, scp, dtype):
        a = xp.asarray([1000, 1000], dtype=dtype)
        return getattr(scp.special, 'logsumexp')(a)

    @testing.for_dtypes('efdhilHIL')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_logsumexp_b_large(self, xp, scp, dtype):
        a = xp.asarray([1000, 1000], dtype=dtype)
        b = xp.asarray([1.2, 1.2], dtype=dtype)
        return getattr(scp.special, 'logsumexp')(a, b=b)

    @testing.for_dtypes('efdhilHIL')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_logsumexp_large2(self, xp, scp, dtype):
        a = xp.full((1000,), 10000, dtype=dtype)
        return getattr(scp.special, 'logsumexp')(a)

    @testing.for_dtypes('efdhilHIL')
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=rtol)
    @pytest.mark.parametrize('with_b', [False, True])
    @pytest.mark.parametrize('keepdims', [False, True])
    @pytest.mark.parametrize('return_sign', [False, True])
    @pytest.mark.parametrize('axis', [None, 0, 1, (0, 1), (-1, -2)])
    def test_logsumexp_axis(self, xp, scp, dtype, axis, keepdims, with_b,
                            return_sign):
        a = testing.shaped_random((1024, 1024), xp=xp, dtype=dtype)
        if with_b:
            b = testing.shaped_random((1024, 1024), xp=xp, dtype=dtype)
        else:
            b = None
        return getattr(scp.special, 'logsumexp')(a, axis, b, keepdims, return_sign)

    @testing.for_dtypes('efdhilHIL')
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=rtol)
    def test_logsumexp_nonfinite(self, xp, scp, dtype):
        a = xp.asarray([xp.inf, -xp.inf, xp.nan])
        return getattr(scp.special, 'logsumexp')(a)

    @testing.for_dtypes('efdhilHIL')
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=rtol)
    def test_logsumexp_highly_different_magnitudes(self, xp, scp, dtype):
        a = xp.asarray([[1e10, 1e-10], [-1e10, -xp.inf]])
        return getattr(scp.special, 'logsumexp')(a, axis=-1)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=rtol)
    def test_logsumexp_sign(self, xp, scp):
        a = xp.asarray([1, 1, 1])
        b = xp.asarray([1, -1, -1])
        return getattr(scp.special, 'logsumexp')(a, b=b, return_sign=True)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=rtol)
    def test_logsumexp_sign_zero(self, xp, scp):
        a = xp.asarray([1, 1])
        b = xp.asarray([1, -1])
        return getattr(scp.special, 'logsumexp')(a, b=b, return_sign=True)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=rtol)
    def test_logsumexp_b_zero(self, xp, scp):
        a = xp.asarray([1, 10000])
        b = xp.asarray([1, 0])
        return getattr(scp.special, 'logsumexp')(a, b=b)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=rtol, atol=1e-15)
    def test_softmax(self, xp, scp, dtype):
        a = xp.arange(200, dtype=dtype)
        return getattr(scp.special, 'softmax')(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=rtol,
                                 atol={'default': 1e-15,
                                       cupy.float32: 1e-7,
                                       cupy.float16: 1e-3})
    @pytest.mark.parametrize('axis', [None, 0, 1, (0, 1), (-1, -2)])
    def test_softmax_axis(self, xp, scp, dtype, axis):
        a = testing.shaped_random((1024, 1024), xp=xp, dtype=dtype)
        return getattr(scp.special, 'softmax')(a, axis)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=rtol, atol=1e-15)
    def test_log_softmax(self, xp, scp, dtype):
        a = xp.arange(200, dtype=dtype)
        return getattr(scp.special, 'log_softmax')(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=rtol)
    @pytest.mark.parametrize('axis', [None, 0, 1, (0, 1), (-1, -2)])
    def test_log_softmax_axis(self, xp, scp, dtype, axis):
        a = testing.shaped_random((1024, 1024), xp=xp, dtype=dtype)
        return getattr(scp.special, 'log_softmax')(a, axis)
