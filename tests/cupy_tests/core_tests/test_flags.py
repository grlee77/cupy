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

    def test_fnc(self):
        a = testing.shaped_random((4, ), cupy)
        a_2d_c = testing.shaped_random((4, 4), cupy)
        a_2d_f = cupy.asfortranarray(testing.shaped_random((4, 4), cupy))
        a_noncontig = testing.shaped_random((4, 8), cupy)[:, ::2]
        assert a.flags.fnc == False
        assert a_2d_c.flags.fnc == False
        assert a_2d_f.flags.fnc == True
        assert a_noncontig.flags.fnc == False

    def test_forc(self):
        a = testing.shaped_random((4, ), cupy)
        a_2d_c = testing.shaped_random((4, 4), cupy)
        a_2d_f = cupy.asfortranarray(testing.shaped_random((4, 4), cupy))
        a_noncontig = testing.shaped_random((4, 8), cupy)[:, ::2]
        assert a.flags.forc == True
        assert a_2d_c.flags.forc == True
        assert a_2d_f.flags.forc == True
        assert a_noncontig.flags.forc == False
