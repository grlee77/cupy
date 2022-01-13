"""F distribution

The source code here is an adaptation with minimal changes from the following
files in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/main/scipy/special/cephes/fdtr.c

Cephes Math Library, Release 2.3:  March, 1995
Copyright 1984, 1995 by Stephen L. Moshier
"""

from cupy import _core
from cupyx.scipy.special._beta import (beta_definition,
                                       beta_preamble,
                                       incbet_definition,
                                       incbi_definition,
                                       lbeta_definition,
                                       lbeta_symp_definition,
                                       lgam_sgn_definition)
from cupyx.scipy.special._digamma import polevl_definition
from cupyx.scipy.special._gamma import gamma_definition
from cupyx.scipy.special._gammainc import p1evl_definition
from cupyx.scipy.special._ndtri import ndtri_definition


fdtrc_definition = """

__device__ double fdtrc(double a, double b, double x)
{
    double w;

    if ((a <= 0.0) || (b <= 0.0) || (x < 0.0)) {
        // sf_error("fdtrc", SF_ERROR_DOMAIN, NULL);
        return CUDART_NAN;
    }
    w = b / (b + a * x);
    return incbet(0.5 * b, 0.5 * a, w);
}
"""


fdtr_definition = """

__device__ double fdtr(double a, double b, double x)
{
    double w;

    if ((a <= 0.0) || (b <= 0.0) || (x < 0.0)) {
        // sf_error("fdtr", SF_ERROR_DOMAIN, NULL);
        return CUDART_NAN;
    }
    w = a * x;
    w = w / (b + w);
    return incbet(0.5 * a, 0.5 * b, w);
}
"""


fdtri_definition = """
__device__ double fdtri(double a, double b, double y)
{
    double w, x;

    if ((a <= 0.0) || (b <= 0.0) || (y <= 0.0) || (y > 1.0)) {
        // sf_error("fdtri", SF_ERROR_DOMAIN, NULL);
        return CUDART_NAN;
    }
    y = 1.0 - y;
    /* Compute probability for x = 0.5.  */
    w = incbet(0.5 * b, 0.5 * a, 0.5);
    /* If that is greater than y, then the solution w < .5.
     * Otherwise, solve at 1-y to remove cancellation in (b - b*w).  */
    if (w > y || y < 0.001) {
        w = incbi(0.5 * b, 0.5 * a, y);
        x = (b - b * w) / (a * w);
    }
    else {
        w = incbi(0.5 * a, 0.5 * b, 1.0 - y);
        x = b * w / (a * (1.0 - w));
    }
    return x;
}
"""


fdtrc = _core.create_ufunc(
    "cupyx_scipy_fdtrc",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(fdtrc(in0, in1, in2));",
    preamble=(
        beta_preamble +
        gamma_definition +
        polevl_definition +
        p1evl_definition +
        ndtri_definition +
        lgam_sgn_definition +
        lbeta_symp_definition +
        beta_definition +
        lbeta_definition +
        incbet_definition +
        incbi_definition +
        fdtrc_definition
    ),

    doc="""F survival function.

    Returns the complemented F-distribution function (the integral of the
    density from x to infinity).

    Parameters
    ----------
    dfn : cupy.ndarray
        First parameter (positive float).
    dfd : cupy.ndarray
        Second parameter (positive float).
    x : cupy.ndarray
        Argument (nonnegative float).

    Returns
    -------
    y : cupy.ndarray
        The complemented F-distribution function with parameters dfn and dfd at
        x.

    .. seealso:: :meth:`scipy.special.fdtrc`

    """,
)

fdtri = _core.create_ufunc(
    "cupyx_scipy_fdtri",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(fdtri(in0, in1, in2));",
    preamble=(
        beta_preamble +
        gamma_definition +
        polevl_definition +
        p1evl_definition +
        ndtri_definition +
        lgam_sgn_definition +
        lbeta_symp_definition +
        beta_definition +
        lbeta_definition +
        incbet_definition +
        incbi_definition +
        fdtri_definition
    ),

    doc="""The p-th quantile of the F-distribution.

    This function is the inverse of the F-distribution CDF, `fdtr`, returning
    the `x` such that `fdtr(dfn, dfd, x)` = `p`.

    Parameters
    ----------
    dfn : cupy.ndarray
        First parameter (positive float).
    dfd : cupy.ndarray
        Second parameter (positive float).
    p : cupy.ndarray
        Cumulative probability, in [0, 1].

    Returns
    -------
    y : cupy.ndarray
        The quantile corresponding to p.

    .. seealso:: :meth:`scipy.special.fdtri`

    """,
)


fdtr = _core.create_ufunc(
    "cupyx_scipy_fdtr",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(fdtr(in0, in1, in2));",
    preamble=(
        beta_preamble +
        gamma_definition +
        polevl_definition +
        p1evl_definition +
        ndtri_definition +
        lgam_sgn_definition +
        lbeta_symp_definition +
        beta_definition +
        lbeta_definition +
        incbet_definition +
        incbi_definition +
        fdtr_definition
    ),

    doc="""F cumulative distribution function.


    Parameters
    ----------
    dfn : cupy.ndarray
        First parameter (positive float).
    dfd : cupy.ndarray
        Second parameter (positive float).
    x : cupy.ndarray
        Argument (nonnegative float).

    Returns
    -------
    y : cupy.ndarray
        The CDF of the F-distribution with parameters dfn and dfd at x.

    .. seealso:: :meth:`scipy.special.fdtr`

    """,
)
