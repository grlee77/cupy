import unittest

import numpy

from cupy import testing


@testing.parameterize(*testing.product({
    'decimals': [-2, -1, 0, 1, 2],
}))
class TestRound(unittest.TestCase):

    shape = (20,)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_round(self, xp, dtype):
        if dtype == numpy.bool_:
            # avoid cast problem
            a = testing.shaped_random(self.shape, xp, scale=10, dtype=dtype)
            return a.round(0)
        if dtype == numpy.float16:
            # avoid accuracy problem
            a = testing.shaped_random(self.shape, xp, scale=10, dtype=dtype)
            return a.round(0)
        a = testing.shaped_random(self.shape, xp, scale=100, dtype=dtype)
        return a.round(self.decimals)

    @testing.numpy_cupy_array_equal()
    def test_round_out(self, xp):
        a = testing.shaped_random(self.shape, xp, scale=100, dtype='d')
        out = xp.empty_like(a)
        a.round(self.decimals, out)
        return out


@testing.parameterize(*testing.product({
    # limit to:
    # * <=0: values like 0.35 and 0.035 cannot be expressed exactly in IEEE 754
    # * >-4: to avoid float16 overflow
    'decimals': [-3, -2, -1, 0],
}))
class TestRoundHalfway(unittest.TestCase):

    shape = (20,)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_round_halfway_float(self, xp, dtype):
        # generate [..., -1.5, -0.5, 0.5, 1.5, ...] * 10^{-decimals}
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        a *= 2
        a -= a.size + 1
        scale = 10**abs(self.decimals)
        if self.decimals < 0:
            a *= scale
        else:
            a /= scale
        a /= 2

        print(xp, dtype, self.decimals, a, a.round(self.decimals))
        return a.round(self.decimals)

    @testing.for_signed_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_round_halfway_int(self, xp, dtype):
        # generate [..., -1.5, -0.5, 0.5, 1.5, ...] * 10^{-decimals}
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        a *= 2
        a -= a.size + 1
        scale = 10**abs(self.decimals)
        if self.decimals < 0:
            a *= xp.array(scale, dtype=dtype)
        a >>= 1

        return a.round(self.decimals)

    @testing.for_unsigned_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_round_halfway_uint(self, xp, dtype):
        # generate [0.5, 1.5, ...] * 10^{-decimals}
        a = testing.shaped_arange(self.shape, xp, dtype=dtype)
        a *= 2
        a -= 1
        scale = 10**abs(self.decimals)
        if self.decimals < 0:
            a *= xp.array(scale, dtype=dtype)
        a >>= 1

        return a.round(self.decimals)
