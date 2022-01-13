"""Chi square distribution

The source code here is an adaptation with minimal changes from the following
files in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/main/scipy/special/cephes/chdtr.c

Cephes Math Library, Release 2.3:  March, 1995
Copyright 1984, 1995 by Stephen L. Moshier
"""

from cupy import _core
from cupyx.scipy.special._gammainc import _igam_preamble, _igami_preamble


chdtrc_definition = """

__device__ double chdtrc(double df, double x)
{

    if (x < 0.0)
        return 1.0;     /* modified by T. Oliphant */
    return igamc(df / 2.0, x / 2.0);
}
"""


chdtr_definition = """

__device__ double chdtr(double df, double x)
{
    if ((x < 0.0)) {        /* || (df < 1.0) ) */
        return CUDART_NAN;
    }
    return igam(df / 2.0, x / 2.0);
}
"""


chdtri_definition = """
__device__ double chdtri(double df, double y)
{
    double x;

    if ((y < 0.0) || (y > 1.0)) {   /* || (df < 1.0) ) */
        return CUDART_NAN;
    }

    x = igamci(0.5 * df, y);
    return 2.0 * x;
}
"""


chdtrc = _core.create_ufunc(
    "cupyx_scipy_chdtrc",
    ("ff->f", "dd->d"),
    "out0 = out0_type(chdtrc(in0, in1));",
    preamble=_igam_preamble + chdtrc_definition,
    doc="""Chi square survival function.

    Returns the complemented chi-squared distribution function (the integral of the
    density from x to infinity).

    Parameters
    ----------
    v : cupy.ndarray
        Degrees of freedom.
    x : cupy.ndarray
        Upper bound of the integral (nonnegative float).

    Returns
    -------
    y : cupy.ndarray
        The complemented chi-squared distribution function with parameter df at
        x.

    See Also
    --------
    :func:`scipy.special.chdtrc`

    """,
)

chdtri = _core.create_ufunc(
    "cupyx_scipy_chdtri",
    ("ff->f", "dd->d"),
    "out0 = out0_type(chdtri(in0, in1));",
    preamble=_igami_preamble + chdtri_definition,
    doc="""Inverse to `chdtrc` with respect to `x`.

    Parameters
    ----------
    v : cupy.ndarray
        Degrees of freedom.
    p : cupy.ndarray
        Probability.
    p : cupy.ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    x : cupy.ndarray
        Value so that the probability a Chi square random variable with `v`
        degrees of freedom is greater than `x` equals `p`.

    See Also
    --------
    :func:`scipy.special.chdtri`

    """,
)


chdtr = _core.create_ufunc(
    "cupyx_scipy_chdtr",
    ("ff->f", "dd->d"),
    "out0 = out0_type(chdtr(in0, in1));",
    preamble=_igam_preamble + chdtr_definition,
    doc="""F cumulative distribution function.

    Parameters
    ----------
    v : cupy.ndarray
        Degrees of freedom.
    x : cupy.ndarray
        Upper bound of the integral (nonnegative float).

    Returns
    -------
    y : cupy.ndarray
        The CDF of the chi-squared distribution with parameter df at x.

    See Also
    --------
    :func:`scipy.special.chdtr`

    """,
)
