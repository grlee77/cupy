import numpy
import pytest

import cupy
from cupy import testing
import cupyx.scipy.special  # NOQA


@testing.gpu
@testing.with_requires('scipy')
class TestFDistributions:

    @pytest.mark.parametrize('function', ['fdtr', 'fdtrc', 'fdtri'])
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_arange(self, xp, scp, dtype, function):
        import scipy.special  # NOQA

        func = getattr(scp.special, function)
        dfn = testing.shaped_arange((1, 10, 1), xp, dtype)
        dfd = testing.shaped_arange((10, 1, 1), xp, dtype)
        x = xp.asarray([0, 0.25, 0.5, 0.75, 1]).reshape(1, 1, 5)
        # return scp.special.betainc(a, b, x)
        return func(dfn, dfd, x)

    @pytest.mark.parametrize('function', ['fdtr', 'fdtrc', 'fdtri'])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype, function):
        import scipy.special  # NOQA

        func = getattr(scp.special, function)
        # dfn and dfd should be positive
        dfn = xp.linspace(.001, 21, 10, dtype=dtype)[:, xp.newaxis, xp.newaxis]
        dfd = xp.linspace(.001, 21, 10, dtype=dtype)[xp.newaxis, :, xp.newaxis]
        if function == 'fdtri':
            # should be a probability in the range [0, 1] for fdtri
            x = xp.linspace(0, 1, 5, dtype=dtype)
        else:
            x = xp.linspace(0, 10, 5, dtype=dtype)
        x = x[xp.newaxis, xp.newaxis, :]
        return func(dfn, dfd, x)

    def test_fdtr_specific_vals(self):
        # specific values borrowed from SciPy test suite
        special = cupyx.scipy.special

        assert special.betainc(1, 1, 0) == 0.0
        # Computed using Wolfram Alpha: CDF[FRatioDistribution[1e-6, 5], 10]
        testing.assert_allclose(special.fdtr(1e-6, 5, 10),
                                0.9999940790193488, rtol=1e-12)

    def test_fdtrc_specific_vals(self):
        # specific values borrowed from SciPy test suite
        special = cupyx.scipy.special

        assert special.fdtrc(1, 1, 0) == 1.0
        # Computed using Wolfram Alpha:
        #   1 - CDF[FRatioDistribution[2, 1/10], 1e10]
        testing.assert_allclose(special.fdtrc(2, 0.1, 1e10),
                                0.27223784621293512, rtol=1e-12)


    def test_fdtri_specific_vals(self):
        # specific values borrowed from SciPy test suite
        special = cupyx.scipy.special

        p = cupy.array([0.499, 0.501])
        testing.assert_allclose(special.fdtri(1, 1, p),
                                cupy.array([0.9937365, 1.00630298]), rtol=1e-6)

        # From Wolfram Alpha:
        #   CDF[FRatioDistribution[1/10, 1], 3] = 0.8756751669632105666874...
        p = 0.8756751669632105666874
        testing.assert_allclose(special.fdtri(0.1, 1, p), 3, rtol=1e-12)
