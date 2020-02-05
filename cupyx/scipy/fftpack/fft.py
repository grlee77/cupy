from bisect import bisect_left

from numpy import prod

import cupy
from cupy.cuda import cufft
from cupy.fft.fft import (_fft, _default_fft_func, _convert_fft_type,
                          _get_cufft_plan_nd)


def get_fft_plan(a, shape=None, axes=None, value_type='C2C'):
    """ Generate a CUDA FFT plan for transforming up to three axes.

    Args:
        a (cupy.ndarray): Array to be transform, assumed to be either C- or
            F- contiguous.
        shape (None or tuple of ints): Shape of the transformed axes of the
            output. If ``shape`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (None or int or tuple of int):  The axes of the array to
            transform. If `None`, it is assumed that all axes are transformed.

            Currently, for performing N-D transform these must be a set of up
            to three adjacent axes, and must include either the first or the
            last axis of the array.
        value_type ('C2C'): The FFT type to perform.
            Currently only complex-to-complex transforms are supported.

    Returns:
        a cuFFT plan for either 1D transform (``cupy.cuda.cufft.Plan1d``) or
        N-D transform (``cupy.cuda.cufft.PlanNd``).

    .. note::
        The returned plan can not only be passed as one of the arguments of
        the functions in ``cupyx.scipy.fftpack``, but also be used as a
        context manager for both ``cupy.fft`` and ``cupyx.scipy.fftpack``
        functions:

        .. code-block:: python

            x = cupy.random.random(16).reshape(4, 4).astype(cupy.complex)
            plan = cupyx.scipy.fftpack.get_fft_plan(x)
            with plan:
                y = cupy.fft.fftn(x)
                # alternatively:
                y = cupyx.scipy.fftpack.fftn(x)  # no explicit plan is given!
            # alternatively:
            y = cupyx.scipy.fftpack.fftn(x, plan=plan)  # pass plan explicitly

        In the first case, no cuFFT plan will be generated automatically,
        even if ``cupy.fft.config.enable_nd_planning = True`` is set.

    .. warning::
        This API is a deviation from SciPy's, is currently experimental, and
        may be changed in the future version.
    """
    # check input array
    if a.flags.c_contiguous:
        order = 'C'
    elif a.flags.f_contiguous:
        order = 'F'
    else:
        raise ValueError('Input array a must be contiguous')

    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(axes, int):
        axes = (axes,)
    if (shape is not None) and (axes is not None) and len(shape) != len(axes):
        raise ValueError('Shape and axes have different lengths.')

    # check axes
    # n=1: 1d (need axis1D); n>1: Nd
    if axes is None:
        n = a.ndim if shape is None else len(shape)
        axes = tuple(i for i in range(-n, 0))
        if n == 1:
            axis1D = 0
    else:  # axes is a tuple
        n = len(axes)
        if n == 1:
            axis1D = axes[0]
            if axis1D >= a.ndim or axis1D < -a.ndim:
                err = 'The chosen axis ({0}) exceeds the number of '\
                      'dimensions of a ({1})'.format(axis1D, a.ndim)
                raise ValueError(err)
        elif n > 3:
            raise ValueError('Only up to three axes is supported')

    # Note that "shape" here refers to the shape along trasformed axes, not
    # the shape of the output array, and we need to convert it to the latter.
    # The result is as if "a=_cook_shape(a); return a.shape" is called.
    transformed_shape = shape
    shape = list(a.shape)
    if transformed_shape is not None:
        for s, axis in zip(transformed_shape, axes):
            shape[axis] = s
    shape = tuple(shape)

    # check value_type
    fft_type = _convert_fft_type(a, value_type)
    if n > 1 and fft_type not in [cufft.CUFFT_C2C, cufft.CUFFT_Z2Z]:
        raise NotImplementedError('Only C2C and Z2Z are supported for N-dim'
                                  ' transform.')

    # generate plan
    if n > 1:  # ND transform
        plan = _get_cufft_plan_nd(shape, fft_type, axes=axes, order=order)
    else:  # 1D transform
        out_size = shape[axis1D]
        batch = prod(shape) // out_size
        plan = cufft.Plan1d(out_size, fft_type, batch)

    return plan


def fft(x, n=None, axis=-1, overwrite_x=False, plan=None):
    """Compute the one-dimensional FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (cupy.cuda.cufft.Plan1d): a cuFFT plan for transforming ``x``
            over ``axis``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axis)

            Note that `plan` is defaulted to None, meaning CuPy will use an
            auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if that of the input is another.

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.

    .. seealso:: :func:`scipy.fftpack.fft`
    """
    return _fft(x, (n,), (axis,), None, cufft.CUFFT_FORWARD,
                overwrite_x=overwrite_x, plan=plan)


def ifft(x, n=None, axis=-1, overwrite_x=False, plan=None):
    """Compute the one-dimensional inverse FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (cupy.cuda.cufft.Plan1d): a cuFFT plan for transforming ``x``
            over ``axis``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axis)

            Note that `plan` is defaulted to None, meaning CuPy will use an
            auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if that of the input is another.

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.

    .. seealso:: :func:`scipy.fftpack.ifft`
    """
    return _fft(x, (n,), (axis,), None, cufft.CUFFT_INVERSE,
                overwrite_x=overwrite_x, plan=plan)


def fft2(x, shape=None, axes=(-2, -1), overwrite_x=False, plan=None):
    """Compute the two-dimensional FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        shape (None or tuple of ints): Shape of the transformed axes of the
            output. If ``shape`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (cupy.cuda.cufft.PlanNd): a cuFFT plan for transforming ``x``
            over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)

            Note that `plan` is defaulted to None, meaning CuPy will either
            use an auto-generated plan behind the scene if cupy.fft.config.
            enable_nd_planning = True, or use no cuFFT plan if it is set to
            False.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``shape`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fftpack.fft2`

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.
    """
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_FORWARD,
                overwrite_x=overwrite_x, plan=plan)


def ifft2(x, shape=None, axes=(-2, -1), overwrite_x=False, plan=None):
    """Compute the two-dimensional inverse FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        shape (None or tuple of ints): Shape of the transformed axes of the
            output. If ``shape`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (cupy.cuda.cufft.PlanNd): a cuFFT plan for transforming ``x``
            over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)

            Note that `plan` is defaulted to None, meaning CuPy will either
            use an auto-generated plan behind the scene if cupy.fft.config.
            enable_nd_planning = True, or use no cuFFT plan if it is set to
            False.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``shape`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fftpack.ifft2`

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.
    """
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_INVERSE,
                overwrite_x=overwrite_x, plan=plan)


def fftn(x, shape=None, axes=None, overwrite_x=False, plan=None):
    """Compute the N-dimensional FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        shape (None or tuple of ints): Shape of the transformed axes of the
            output. If ``shape`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (cupy.cuda.cufft.PlanNd): a cuFFT plan for transforming ``x``
            over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)

            Note that `plan` is defaulted to None, meaning CuPy will either
            use an auto-generated plan behind the scene if cupy.fft.config.
            enable_nd_planning = True, or use no cuFFT plan if it is set to
            False.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``shape`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fftpack.fftn`

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.
    """
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_FORWARD,
                overwrite_x=overwrite_x, plan=plan)


def ifftn(x, shape=None, axes=None, overwrite_x=False, plan=None):
    """Compute the N-dimensional inverse FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        shape (None or tuple of ints): Shape of the transformed axes of the
            output. If ``shape`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (cupy.cuda.cufft.PlanNd): a cuFFT plan for transforming ``x``
            over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)

            Note that `plan` is defaulted to None, meaning CuPy will either
            use an auto-generated plan behind the scene if cupy.fft.config.
            enable_nd_planning = True, or use no cuFFT plan if it is set to
            False.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``shape`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fftpack.ifftn`

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.
    """
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_INVERSE,
                overwrite_x=overwrite_x, plan=plan)


def rfft(x, n=None, axis=-1, overwrite_x=False):
    """Compute the one-dimensional FFT for real input.

    The returned real array contains

    .. code-block:: python

        [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]  # if n is even
        [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2)),Im(y(n/2))]  # if n is odd

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.

    Returns:
        cupy.ndarray:
            The transformed array.

    .. seealso:: :func:`scipy.fftpack.rfft`
    """
    if n is None:
        n = x.shape[axis]

    shape = list(x.shape)
    shape[axis] = n
    f = _fft(x, (n,), (axis,), None, cufft.CUFFT_FORWARD, 'R2C',
             overwrite_x=overwrite_x)
    z = cupy.empty(shape, f.real.dtype)

    slice_z = [slice(None)] * x.ndim
    slice_f = [slice(None)] * x.ndim

    slice_z[axis] = slice(1)
    slice_f[axis] = slice(1)
    z[slice_z] = f[slice_f].real

    slice_z[axis] = slice(1, None, 2)
    slice_f[axis] = slice(1, None)
    z[slice_z] = f[slice_f].real

    slice_z[axis] = slice(2, None, 2)
    slice_f[axis] = slice(1, n - f.shape[axis] + 1)
    z[slice_z] = f[slice_f].imag

    return z


def irfft(x, n=None, axis=-1, overwrite_x=False):
    """Compute the one-dimensional inverse FFT for real input.

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.

    Returns:
        cupy.ndarray:
            The transformed array.

    .. seealso:: :func:`scipy.fftpack.irfft`
    """
    if n is None:
        n = x.shape[axis]
    m = min(n, x.shape[axis])

    shape = list(x.shape)
    shape[axis] = n // 2 + 1
    if x.dtype in (cupy.float16, cupy.float32):
        z = cupy.zeros(shape, dtype=cupy.complex64)
    else:
        z = cupy.zeros(shape, dtype=cupy.complex128)

    slice_x = [slice(None)] * x.ndim
    slice_z = [slice(None)] * x.ndim

    slice_x[axis] = slice(1)
    slice_z[axis] = slice(1)
    z[slice_z].real = x[slice_x]

    slice_x[axis] = slice(1, m, 2)
    slice_z[axis] = slice(1, m // 2 + 1)
    z[slice_z].real = x[slice_x]

    slice_x[axis] = slice(2, m, 2)
    slice_z[axis] = slice(1, (m + 1) // 2)
    z[slice_z].imag = x[slice_x]

    return _fft(z, (n,), (axis,), None, cufft.CUFFT_INVERSE, 'C2R',
                overwrite_x=overwrite_x)


def next_fast_len(target):
    """Find the next fast size of input data to NVIDIA's cuFFT routines.

    cuFFT has efficient functions for transforms of length
    2**a * 3**b * 5**c * 7**d, where a, b, c, d are positive integers.

    Parameters
    ----------
    target : int
        Length to start searching from.  Must be a positive integer.

    Returns
    -------
    out : int
        The first fast length greater than or equal to `target`.

    """
    hams = (18,    20,    21,    24,    25,    27,    28,    30,    32,
            35,    36,    40,    42,    45,    48,    49,    50,    54,
            56,    60,    63,    64,    70,    72,    75,    80,    81,
            84,    90,    96,    98,   100,   105,   108,   112,   120,
            125,   126,   128,   135,   140,   144,   147,   150,   160,
            162,   168,   175,   180,   189,   192,   196,   200,   210,
            216,   224,   225,   240,   243,   245,   250,   252,   256,
            270,   280,   288,   294,   300,   315,   320,   324,   336,
            343,   350,   360,   375,   378,   384,   392,   400,   405,
            420,   432,   441,   448,   450,   480,   486,   490,   500,
            504,   512,   525,   540,   560,   567,   576,   588,   600,
            625,   630,   640,   648,   672,   675,   686,   700,   720,
            729,   735,   750,   756,   768,   784,   800,   810,   840,
            864,   875,   882,   896,   900,   945,   960,   972,   980,
            1000,  1008,  1024,  1029,  1050,  1080,  1120,  1125,  1134,
            1152,  1176,  1200,  1215,  1225,  1250,  1260,  1280,  1296,
            1323,  1344,  1350,  1372,  1400,  1440,  1458,  1470,  1500,
            1512,  1536,  1568,  1575,  1600,  1620,  1680,  1701,  1715,
            1728,  1750,  1764,  1792,  1800,  1875,  1890,  1920,  1944,
            1960,  2000,  2016,  2025,  2048,  2058,  2100,  2160,  2187,
            2205,  2240,  2250,  2268,  2304,  2352,  2400,  2401,  2430,
            2450,  2500,  2520,  2560,  2592,  2625,  2646,  2688,  2700,
            2744,  2800,  2835,  2880,  2916,  2940,  3000,  3024,  3072,
            3087,  3125,  3136,  3150,  3200,  3240,  3360,  3375,  3402,
            3430,  3456,  3500,  3528,  3584,  3600,  3645,  3675,  3750,
            3780,  3840,  3888,  3920,  3969,  4000,  4032,  4050,  4096,
            4116,  4200,  4320,  4374,  4375,  4410,  4480,  4500,  4536,
            4608,  4704,  4725,  4800,  4802,  4860,  4900,  5000,  5040,
            5103,  5120,  5145,  5184,  5250,  5292,  5376,  5400,  5488,
            5600,  5625,  5670,  5760,  5832,  5880,  6000,  6048,  6075,
            6125,  6144,  6174,  6250,  6272,  6300,  6400,  6480,  6561,
            6615,  6720,  6750,  6804,  6860,  6912,  7000,  7056,  7168,
            7200,  7203,  7290,  7350,  7500,  7560,  7680,  7776,  7840,
            7875,  7938,  8000,  8064,  8100,  8192,  8232,  8400,  8505,
            8575,  8640,  8748,  8750,  8820,  8960,  9000,  9072,  9216,
            9261,  9375,  9408,  9450,  9600,  9604,  9720,  9800, 10000)

    if target <= 16:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= hams[-1]:
        return hams[bisect_left(hams, target)]

    match = float('inf')  # Anything found will be smaller

    # allow any integer powers of 2, 3, 5 or 7
    p7 = 1
    while p7 < target:
        p5_7 = p7
        while p5_7 < target:
            p3_5_7 = p5_7
            while p3_5_7 < target:
                # Ceiling integer division, avoiding conversion to
                # float.
                # (quotient = ceil(target / p35))
                quotient = -(-target // p3_5_7)

                # Quickly find next power of 2 >= quotient
                p2 = 2**((quotient - 1).bit_length())

                N = p2 * p3_5_7
                if N == target:
                    return N
                elif N < match:
                    match = N
                p3_5_7 *= 3
                if p3_5_7 == target:
                    return p3_5_7
            if p3_5_7 < match:
                match = p3_5_7
            p5_7 *= 5
            if p5_7 == target:
                return p5_7
        if p5_7 < match:
            match = p5_7
        p7 *= 7
        if p7 == target:
            return p7
    if p7 < match:
        match = p7
    return match
