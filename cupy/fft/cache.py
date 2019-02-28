import numpy


__all__ = ['enable', 'disable', 'set_max_size_bytes']

_cufft_cache = None
_max_size_bytes = None

# TODO: should have a _cufft_cache and _max_size_bytes per device?

def enable(max_size_bytes=None):
    """Enable the CUFFT plan cache."""
    global _cufft_cache
    global _max_size_bytes

    if _cufft_cache is None:
        _cufft_cache = _PlanCache(_max_size_bytes)


def set_max_size_bytes(max_size_bytes):
    """Set the size in bytes available for cached CUFFT plans.

    If set to None, no limit is enforced.
    """
    global _max_size_bytes

    _max_size_bytes = max_size_bytes
    if _cufft_cache is not None:
        _cufft_cache.max_size_bytes = max_size_bytes


def disable():
    """Disable the CUFFT plan cache."""
    global _cufft_cache
    _cufft_cache = None


def is_enabled():
    """Return whether the cache is currently enabled."""
    if _cufft_cache is None:
        return False
    else:
        return True


class _PlanCache(object):
    """Cache for CUFFT Plans.

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
