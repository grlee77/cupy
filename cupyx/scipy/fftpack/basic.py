"""
Discrete Fourier Transforms - basic.py
"""
from __future__ import division, print_function, absolute_import

__all__ = ['fft', 'ifft', 'fftn', 'ifftn', 'rfft', 'irfft', 'fft2', 'ifft2']

from cupy.fft.fft import _fft, _fftn
from cupy.cuda import cufft


def fftn(x, shape=None, axes=None, overwrite_x=False):
    """
    Return multidimensional discrete Fourier transform.

    The returned array contains::

      y[j_1,..,j_d] = sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
         x[k_1,..,k_d] * prod[i=1..d] exp(-sqrt(-1)*2*pi/n_i * j_i * k_i)

    where d = len(x.shape) and n = x.shape.

    Parameters
    ----------
    x : array_like
        The (n-dimensional) array to transform.
    shape : int or array_like of ints or None, optional
        The shape of the result.  If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
        length ``shape[i]``.
        If any element of `shape` is -1, the size of the corresponding
        dimension of `x` is used.
    axes : int or array_like of ints or None, optional
        The axes of `x` (`y` if `shape` is not None) along which the
        transform is applied.
        The default is over all axes.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed.  Default is False.

    Returns
    -------
    y : complex-valued n-dimensional numpy array
        The (n-dimensional) DFT of the input array.

    See Also
    --------
    ifftn

    Notes
    -----
    If ``x`` is real-valued, then
    ``y[..., j_i, ...] == y[..., n_i-j_i, ...].conjugate()``.

    Both single and double precision routines are implemented.  Half precision
    inputs will be converted to single precision.  Non floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.
    """
    if overwrite_x:
        out = x
    else:
        out = None
    return _fftn(x, shape, axes, None, cufft.CUFFT_FORWARD, out=out)
