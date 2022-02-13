import cupy


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : cupy.ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
    b : cupy.ndarray, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).

    Returns
    -------
    res : cupy.ndarray
        The result, ``cupy.log(cupy.sum(cupy.exp(a)))`` calculated in a
        numerically more stable way. If `b` is given then
        ``cupy.log(cupy.sum(b*cupy.exp(a)))`` is returned.
    sgn : cupy.ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.

    See Also
    --------
    `func`:scipy.special.logsumexp

    Notes
    -----
    NumPy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.

    """
    if b is not None:
        b = cupy.broadcast_to(b, a.shape)
        # set to -inf at any locations where b is 0
        b_zero_mask = b == 0
        if a.real.dtype.kind != 'f' and cupy.any(b_zero_mask):
            # convert to float in same manner as SciPy
            a = a + 0.
        a[b_zero_mask] = -cupy.inf

    a_max = cupy.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~cupy.isfinite(a_max)] = 0
        out = a - a_max
        a_max_is_zero = False
    elif not cupy.isfinite(a_max):
        a_max_is_zero = True
        out = a

    out = cupy.exp(out)
    if b is not None:
        # not *= b to keep the same dtype behavior as SciPy
        out = b * out
    out = cupy.sum(out, axis=axis, keepdims=keepdims)
    if return_sign:
        sgn = cupy.sign(out)
        out *= sgn  # /= makes more sense but we need zero -> zero
    cupy.log(out, out=out)

    if not a_max_is_zero:
        if not keepdims:
            a_max = cupy.squeeze(a_max, axis=axis)
        out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def softmax(x, axis=None):
    """Softmax function.

    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements. That is, if `x` is a one-dimensional
    numpy array::

        softmax(x) = cupy.exp(x)/sum(cupy.exp(x))

    Parameters
    ----------
    x : cupy.ndarray
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.

    Returns
    -------
    s : cupy.ndarray
        An array the same shape as `x`. The result will sum to 1 along the
        specified axis.

    See Also
    --------
    `func`:scipy.special.softmax

    Notes
    -----
    The `softmax` function is the gradient of `logsumexp`.

    """
    # compute in log space for numerical stability
    out = x - logsumexp(x, axis=axis, keepdims=True)
    cupy.exp(out, out=out)
    return out


def log_softmax(x, axis=None):
    """Logarithm of softmax function.

    Parameters
    ----------
    x : cupy.ndarray
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.

    Returns
    -------
    s : cupy.ndarray
        An array with the same shape as `x`. Exponential of the result will
        sum to 1 along the specified axis. If `x` is a scalar, a scalar is
        returned.

    See Also
    --------
    `func`:scipy.special.log_softmax

    Notes
    -----
    `log_softmax` is more accurate than ``cupy.log(softmax(x))`` with inputs
    that make `softmax` saturate.

    """
    x_max = cupy.amax(x, axis=axis, keepdims=True)

    if x_max.ndim > 0:
        x_max[~cupy.isfinite(x_max)] = 0
    elif not cupy.isfinite(x_max):
        x_max = 0

    out = x - x_max
    exp_tmp = cupy.exp(out)
    s = cupy.sum(exp_tmp, axis=axis, keepdims=True)
    cupy.log(s, out=s)
    if out.real.dtype != 'f':
        out = out - s
    else:
        out -= s
    return out
