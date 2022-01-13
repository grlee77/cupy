import numpy
import pytest

import cupy
from cupy import testing
import cupyx.scipy.special  # NOQA


@testing.gpu
@testing.with_requires('scipy')
class TestChiSquareDistribution:

    @pytest.mark.parametrize('function', ['chdtr', 'chdtrc', 'chdtri'])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype, function):
        import scipy.special  # NOQA

        func = getattr(scp.special, function)
        v = xp.arange(1, 10, dtype=dtype)[:, xp.newaxis]
        if function == 'chdtri':
            # should be a probability in the range [0, 1] for chdtri
            x = xp.linspace(0, 1, 20, dtype=dtype)
        else:
            x = xp.linspace(0, 10, 20, dtype=dtype)
        x = x[xp.newaxis, :]
        return func(v, x)

    def test_chdtr_specific_vals(self):
        # specific values borrowed from SciPy test suite
        special = cupyx.scipy.special

        assert special.chdtr(1, 0) == 0.0
        assert special.chdtr(0.7, cupy.inf) == 1.0
        testing.assert_allclose(special.chdtr(0.6, 3), 0.957890536704110)

    def test_chdtrc_specific_vals(self):
        # specific values borrowed from SciPy test suite
        special = cupyx.scipy.special

        assert special.chdtrc(1, 0) == 1.0
        testing.assert_allclose(special.chdtrc(0.6, 3), 1 - 0.957890536704110)

    def test_chdtri_specific_vals(self):
        # specific values borrowed from SciPy test suite
        special = cupyx.scipy.special

        assert special.chdtri(1, 1) == 0.0
        testing.assert_allclose(special.chdtri(0.6, 1 - 0.957890536704110), 3)
