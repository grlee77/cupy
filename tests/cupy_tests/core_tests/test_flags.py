import unittest

import cupy
from cupy.core import flags
from cupy import testing


class TestFlags(unittest.TestCase):

    def setUp(self):
        self.flags = flags.Flags(1, 2, 3)

    def test_c_contiguous(self):
        self.assertEqual(1, self.flags['C_CONTIGUOUS'])

    def test_f_contiguous(self):
        self.assertEqual(2, self.flags['F_CONTIGUOUS'])

    def test_owndata(self):
        self.assertEqual(3, self.flags['OWNDATA'])

    def test_key_error(self):
        with self.assertRaises(KeyError):
            self.flags['unknown key']

    def test_repr(self):
        self.assertEqual('''  C_CONTIGUOUS : 1
  F_CONTIGUOUS : 2
  OWNDATA : 3''', repr(self.flags))


@testing.parameterize(
    *testing.product({
        'case': ['1d', '1d_noncontig', '2d_c', '2d_f', '2d_noncontig'],
    })
)
class TestContiguityFlags(unittest.TestCase):

    def setUp(self):
        self.flags = None

    def init_flags(self, xp):
        if self.case == '1d':
            a = testing.shaped_arange((4, ), xp)
        elif self.case == '1d_noncontig':
            a = testing.shaped_arange((8, ), xp)[::2]
        elif self.case == '2d_c':
            a = testing.shaped_arange((4, 4), xp)
        elif self.case == '2d_f':
            a = xp.asfortranarray(testing.shaped_arange((4, 4), xp))
        elif self.case == '2d_noncontig':
            a = testing.shaped_arange((4, 8), xp)[:, ::2]
        self.flags = a.flags

    @testing.numpy_cupy_equal()
    def test_fnc(self, xp):
        self.init_flags(xp)
        return self.flags.fnc

    @testing.numpy_cupy_equal()
    def test_forc(self, xp):
        self.init_flags(xp)
        return self.flags.forc

    @testing.numpy_cupy_equal()
    def test_f_contiguous(self, xp):
        self.init_flags(xp)
        return self.flags.f_contiguous

    @testing.numpy_cupy_equal()
    def test_c_contiguous(self, xp):
        self.init_flags(xp)
        return self.flags.c_contiguous
