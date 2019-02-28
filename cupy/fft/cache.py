import numpy

import cupy


__all__ = ['clear', 'disable', 'enable', 'is_enabled', 'set_max_size']

# global dicts (with device id as the key)
_cufft_cache = {}
_max_size_bytes = {}


def enable(max_size_bytes=None, device_id=None):
    """Enable the CUFFT plan cache.

    Args:
        max_size_bytes (int or None): Maximum allowed size in bytes of the
            cache.  If `None`, no size limit is enforced.
        device_id (int or None): The device id. Default is the current device.
    """
    global _cufft_cache
    global _max_size_bytes

    if device_id is None:
        device_id = cupy.cuda.get_device_id()

    _max_size_bytes[device_id] = max_size_bytes
    if device_id not in _cufft_cache:
        _cufft_cache[device_id] = _PlanCache(max_size_bytes)


def disable(device_id=None):
    """Disable the CUFFT plan cache.

    Args:
        device_id (int or None): The device id. Default is the current device.
    """
    global _cufft_cache

    if device_id is None:
        device_id = cupy.cuda.get_device_id()
    _cufft_cache.pop(device_id, None)


def is_enabled(device_id=None):
    """Return whether the cache is currently enabled.

    Args:
        device_id (int or None): The device id. Default is the current device.

    Returns:
        enabled (bool): boolean indicating whether the cache is enabled.
    """
    if device_id is None:
        device_id = cupy.cuda.get_device_id()
    if device_id in _cufft_cache:
        return True
    else:
        return False


def clear(device_id=None):
    """Clear any cached CUFFT plans stored on the device.

    Args:
        device_id (int or None): The device id. Default is the current device.
    """
    global _cufft_cache

    if device_id is None:
        device_id = cupy.cuda.get_device_id()

    """Clear the CUFFT plan cache."""
    if device_id in _cufft_cache:
        _cufft_cache[device_id].clear()


def set_max_size(max_size_bytes, device_id=None):
    """Set the size in bytes available for cached CUFFT plans on a device.

    Args:
        max_size_bytes (int or None): Maximum allowed size in bytes of the
            cache.  If `None`, no size limit is enforced.
        device_id (int or None): The device id. Default is the current device.

    """
    global _max_size_bytes

    if device_id is None:
        device_id = cupy.cuda.get_device_id()

    _max_size_bytes[device_id] = max_size_bytes
    if device_id in _cufft_cache:
        _cufft_cache[device_id].max_size = max_size_bytes
        if _cufft_cache[device_id]._cached_size_bytes > max_size_bytes:
            _cufft_cache[device_id].clear()


class _PlanCache(object):
    """Cache for CUFFT Plans.

    Each device will have it's own, independent _PlanCache object.

    - Can store both cufft.Plan1d and cufft.PlanNd plans.
    - Can specify `max_size_bytes` to limit the total memory used by all stored
      plans
    """

    def __init__(self, max_size_bytes=None):
        self._cache_dict = {}
        self.max_size = max_size_bytes
        self._cached_size_bytes = 0

    def __contains__(self, key):
        return key in self._cache_dict

    def insert(self, plan, key, overwrite=False):
        """Insert a plan into the cache, referenced by a (hashable) key."""
        if not hasattr(plan, 'work_area'):
            raise ValueError(
                "Expected plan to be a CUFFT Plan1d or PlanNd class with a "
                "work_area attribute.")
        new_size = self._cached_size_bytes + plan.work_area.mem.size
        if self.max_size is None or new_size <= self.max_size:
            self._cache_dict[key] = plan
            self._cached_size_bytes = new_size
        else:
            # do not insert if max_size_bytes would be exceeded.
            # TODO: try clearing some/all existing plans to make room?
            pass

    def lookup(self, key):
        """Retrieve the plan corresponding to the given key.

        Returns None if the key is not in the cache.
        """
        return self._cache_dict.get(key, None)

    def remove(self, key):
        """Remove the plan corresponding to the given key."""
        if key in self._cach_dict:
            plan = self._cache_dict.pop(key)
            self._cached_size_bytes -= plan.work_area.mem.size

    def clear(self):
        """Clear all cached plans."""
        self._cache_dict = {}
        self._cached_size_bytes = 0
