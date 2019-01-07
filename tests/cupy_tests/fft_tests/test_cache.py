#
# Copyright 2017 Knowledge Economy Developments Ltd
#
# Henry Gomersall
# heng@kedevelopments.co.uk
#
# Copyright 2018 Gregory Lee, grlee77@gmail.com
# cache tests from PyFFTW adapted for CuPy by Gregory Lee
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

import unittest

import numpy as np

import cupy
from cupy.fft import cache
from cupy import testing


import threading
import time


'''Test the caching functionality of the cupy.fft module.
'''


def _check_n_cache_threads_running():
    '''Return how many threads have the name 'CuPyCUFFTCacheThread.

    Obviously, this isn't production quality, but it should suffice for
    the tests here.
    '''

    cache_threads = 0
    for each_thread in threading.enumerate():
        if each_thread.name == 'CuPyCUFFTCacheThread':
            cache_threads += 1

    return cache_threads


@testing.gpu
class CacheSpecificInterfacesUtils(unittest.TestCase):

    def test_slow_lookup_no_race_condition(self):
        '''Checks that lookups in _utils longer than the keepalive time are ok.
        '''
        # Any old size, it doesn't matter
        data_shape = (128,)

        # Monkey patch the module with a custom _Cache object
        _Cache_class = cache._Cache

        class _SlowLookupCache(_Cache_class):

            def _lookup(self, key):
                return _Cache_class.lookup(self, key)

            def lookup(self, key):
                time.sleep(0.1)
                return self._lookup(key)

        try:
            cache._Cache = _SlowLookupCache

            cache.enable()

            # something shortish
            cache.set_keepalive_time(0.001)

            ar, ai = np.random.randn(*(2,) + data_shape)
            a = cupy.asarray(ar + 1j*ai)

            # Both the following should work without exception
            # (even if it fails to get from the cache)
            cupy.fft.fft(a)
            cupy.fft.fft(a)

            cache.disable()

        finally:
            # Revert the monkey patching
            cache._Cache = _Cache_class


class InterfacesCacheTest(unittest.TestCase):

    def test_missing_threading(self):
        self.assertIs(cache._cufft_cache, None)

        mod_threading = cache._threading
        cache._threading = None

        with self.assertRaises(ImportError):
            cache.enable()

        cache._threading = mod_threading

    def test_is_enabled(self):
        self.assertIs(cache._cufft_cache, None)

        cache.enable()
        self.assertTrue(cache.is_enabled())

        cache.disable()
        self.assertFalse(cache.is_enabled())

    def test_cache_enable_disable(self):

        self.assertIs(cache._cufft_cache, None)

        cache.enable()
        self.assertIsInstance(
            cache._cufft_cache, cache._Cache)

        cache.disable()
        self.assertIs(cache._cufft_cache, None)

    def test_set_keepalive_time(self):
        with self.assertRaises(cache.CacheError):
            cache.set_keepalive_time(10)

        cache.enable()
        cache.set_keepalive_time(10)

        self.assertTrue(
            cache._cufft_cache.keepalive_time == 10.0)

        cache.disable()


class CacheTest(unittest.TestCase):

    def test_cache_parent_thread_ended(self):
        '''Test ending cache parent thread ends cache thread.
        '''
        # Firstly make sure we've exited any lingering threads from other
        # tests.
        time.sleep(2.0)

        self.assertTrue(_check_n_cache_threads_running() == 0)

        def cache_parent_thread():
            cache._Cache()
            time.sleep(0.5)

        # We give the parent thread the same name as a Cache thread so
        # it is picked up by the _check_n_cache_threads_running function
        parent_t = threading.Thread(
            target=cache_parent_thread, name='CuPyCUFFTCacheThread')
        parent_t.start()

        time.sleep(0.1)
        # Check it's running
        self.assertEqual(_check_n_cache_threads_running(), 2)

        parent_t.join()
        time.sleep(0.1)
        # Check both threads have exited properly
        self.assertTrue(_check_n_cache_threads_running() == 0)

    def test_delete_cache_object(self):
        '''Test deleting a cache object ends cache thread.
        '''
        # Firstly make sure we've exited any lingering threads from other
        # tests.
        time.sleep(1.25)
        self.assertTrue(_check_n_cache_threads_running() == 0)

        _cache = cache._Cache(keepalive_time=0.1)
        time.sleep(0.25)
        self.assertTrue(_check_n_cache_threads_running() == 1)

        del _cache
        time.sleep(0.25)
        self.assertTrue(_check_n_cache_threads_running() == 0)

    def test_insert_and_lookup_item(self):
        _cache = cache._Cache()

        key = 'the key'
        obj = None
        _cache.insert(obj, key)

        self.assertIs(_cache.lookup(key), obj)

    def test_invalid_lookup(self):
        _cache = cache._Cache()

        key = 'the key'
        obj = None
        _cache.insert(obj, key)

        self.assertRaises(KeyError, _cache.lookup, 'wrong_key')

    def test_keepalive_time_update(self):
        _cache = cache._Cache()

        # The default
        self.assertEqual(_cache.keepalive_time, 1.0)

        _cache.set_keepalive_time(0.3)
        self.assertEqual(_cache.keepalive_time, 0.3)

        _cache.set_keepalive_time(10.0)
        self.assertEqual(_cache.keepalive_time, 10.0)

        _cache.set_keepalive_time('0.2')
        self.assertEqual(_cache.keepalive_time, 0.2)

        with self.assertRaises(ValueError):
            _cache.set_keepalive_time('foo')

        with self.assertRaises(TypeError):
            _cache.set_keepalive_time([])

    @testing.gpu
    def test_contains(self):
        _cache = cache._Cache()

        key = 'the key'

        _cache.insert(None, key)

        self.assertTrue(key in _cache)
        self.assertFalse('Not a key' in _cache)

    def test_objects_removed_after_keepalive(self):
        _cache = cache._Cache(keepalive_time=0.25)

        key = 'the key'

        obj = None
        _cache.insert(obj, key)

        self.assertIs(_cache.lookup(key), obj)

        keepalive_time = _cache.keepalive_time

        # make sure to wait longer than the keepalive_time
        time.sleep(keepalive_time * 4)

        self.assertRaises(KeyError, _cache.lookup, key)

        _cache.insert(obj, key)

        new_keepalive_time = _cache.keepalive_time * 0.5
        _cache.set_keepalive_time(new_keepalive_time)

        self.assertIs(_cache.lookup(key), obj)

        time.sleep(new_keepalive_time * 0.95)

        # still should be there
        self.assertIs(_cache.lookup(key), obj)

        # should no longer be there
        time.sleep(new_keepalive_time * 4)
        self.assertRaises(KeyError, _cache.lookup, key)
