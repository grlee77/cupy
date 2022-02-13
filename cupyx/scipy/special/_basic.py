"""basic special functions

cotdg and tandg implementations are adapted from the following SciPy code:

https://github.com/scipy/scipy/blob/master/scipy/special/cephes/tandg.c

radian is from

https://github.com/scipy/scipy/blob/master/scipy/special/cephes/sindg.c


Cephes Math Library Release 2.0:  April, 1987
Copyright 1984, 1987 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
"""

from cupy import _core
from cupyx.scipy.special._beta import (common_beta_preamble,
                                       beta_definition,
                                       lbeta_definition)
from cupyx.scipy.special._gamma import gamma_definition
from cupyx.scipy.special._poch import poch_definition


# Note: cast complex<single> to complex<double> or tests fail tolerance
log1p = _core.create_ufunc(
    'cupyx_scipy_log1p',
    (('f->f', 'out0 = log1pf(in0)'),
     ('F->F', 'out0 = out0_type(log1p((complex<double>)in0))'),
     'd->d', 'D->D'),
    'out0 = log1p(in0);',
    doc="""Elementwise function for scipy.special.log1p

    Calculates log(1 + x) for use when `x` is near zero.

    Notes
    -----
    This implementation currently does not support complex-valued `x`.

    .. seealso:: :meth:`scipy.special.log1p`

    """,
)


cbrt = _core.create_ufunc(
    'cupyx_scipy_special_cbrt',
    (('f->f', 'out0 = cbrtf(in0)'), 'd->d'),
    'out0 = cbrt(in0)',
    doc='''Cube root.

    .. seealso:: :meth:`scipy.special.cbrt`

    ''')


exp2 = _core.create_ufunc(
    'cupyx_scipy_special_exp2',
    (('f->f', 'out0 = exp2f(in0)'), 'd->d'),
    'out0 = exp2(in0)',
    doc='''Computes ``2**x``.

    .. seealso:: :meth:`scipy.special.exp2`

    ''')


exp10 = _core.create_ufunc(
    'cupyx_scipy_special_exp10',
    (('f->f', 'out0 = exp10f(in0)'), 'd->d'),
    'out0 = exp10(in0)',
    doc='''Computes ``10**x``.

    .. seealso:: :meth:`scipy.special.exp10`

    ''')


expm1 = _core.create_ufunc(
    'cupyx_scipy_special_expm1',
    (('f->f', 'out0 = expm1f(in0)'), 'd->d', 'F->F', 'D->D'),
    'out0 = expm1(in0)',
    doc='''Computes ``exp(x) - 1``.

    .. seealso:: :meth:`scipy.special.expm1`

    ''')


pi180_preamble = """
    __constant__ double PI180 = 1.74532925199432957692E-2;  // pi/180
"""

cosdg = _core.create_ufunc(
    'cupyx_scipy_special_cosdg',
    (('f->f', 'out0 = cosf(PI180 * in0)'), 'd->d'),
    'out0 = cos(PI180 * in0)',
    preamble=pi180_preamble,
    doc='''Cosine of x with x in degrees.

    .. seealso:: :meth:`scipy.special.cosdg`

    ''')


sindg = _core.create_ufunc(
    'cupyx_scipy_special_sindg',
    (('f->f', 'out0 = sinf(PI180 * in0)'), 'd->d'),
    'out0 = sin(PI180 * in0)',
    preamble=pi180_preamble,
    doc='''Sine of x with x in degrees.

    .. seealso:: :meth:`scipy.special.sindg`

    ''')


tancot_implementation = pi180_preamble + """


// include for CUDART_INF
#include <cupy/math_constants.h>

__constant__ double  lossth = 1.0e14;

__device__ static double tancot(double, int);

__device__ static double tandg(double x)
{
    return tancot(x, 0);
}


__device__ static double cotdg(double x)
{
    return tancot(x, 1);
}


__device__ static double tancot(double xx, int cotflg)
{
    double x;
    int sign;

    /* make argument positive but save the sign */
    if (xx < 0) {
        x = -xx;
        sign = -1;
    }
    else {
        x = xx;
        sign = 1;
    }

    if (x > lossth) {
        // sf_error("tandg", SF_ERROR_NO_RESULT, NULL);
        return 0.0;
    }

    /* modulo 180 */
    x = x - 180.0 * floor(x / 180.0);
    if (cotflg) {
        if (x <= 90.0) {
            x = 90.0 - x;
        } else {
            x = x - 90.0;
            sign *= -1;
        }
    } else {
        if (x > 90.0) {
            x = 180.0 - x;
            sign *= -1;
        }
    }
    if (x == 0.0) {
        return 0.0;
    }
    else if (x == 45.0) {
        return sign * 1.0;
    }
    else if (x == 90.0) {
        // sf_error((cotflg ? "cotdg" : "tandg"), SF_ERROR_SINGULAR, NULL);
        return CUDART_INF;
    }
    /* x is now transformed into [0, 90) */
    return sign * tan(x * PI180);
}
"""

tandg = _core.create_ufunc(
    'cupyx_scipy_special_tandg', ('f->f', 'd->d'),
    'out0 = tandg(in0)',
    preamble=tancot_implementation,
    doc='''Tangent of x with x in degrees.

    .. seealso:: :meth:`scipy.special.tandg`

    ''')


cotdg = _core.create_ufunc(
    'cupyx_scipy_special_cotdg', ('f->f', 'd->d'),
    'out0 = cotdg(in0)',
    preamble=tancot_implementation,
    doc='''Cotangent of x with x in degrees.

    .. seealso:: :meth:`scipy.special.cotdg`

    ''')


radian_implementation = """
/* 1 arc second, in radians*/
__constant__ double P64800 =
    4.848136811095359935899141023579479759563533023727e-6;

template <typename T>
__device__ T radian(T d, T m, T s)
{
    return (((d * 60.0 + m) * 60.0 + s) * P64800);
}
"""

radian = _core.create_ufunc(
    'cupyx_scipy_special_radian', ('fff->f', 'ddd->d'),
    'out0 = radian(in0, in1, in2)',
    preamble=radian_implementation,
    doc='''Degrees, minutes, seconds to radians:

    .. seealso:: :meth:`scipy.special.radian`

    ''')


factorial_approx_implementation = """
__device__ double factorial_approx(double n)
{
    if (n < 0) {
        return 0;
    }
    return Gamma(n + 1);
}
"""

_factorial_approx = _core.create_ufunc(
    'cupyx_scipy_factorial', ('f->f', 'd->d'),
    'out0 = out0_type(factorial_approx(in0))',
    preamble=gamma_definition + factorial_approx_implementation,
    doc='''Fast approximation to factorial(n) via Gamma(n + 1).''')


def factorial(n, exact=False):
    """
    The factorial of a number or array of numbers.

    The factorial of non-negative integer `n` is the product of all
    positive integers less than or equal to `n`::

        n! = n * (n - 1) * (n - 2) * ... * 1

    Parameters
    ----------
    n : int or cupy.ndarray of ints
        Input values.  If ``n < 0``, the return value is 0.
    exact : bool, optional
        If False, the result is approximated in floating point using the
        `gamma` function. The exact case is performed with arbitrary length
        integer arithmetic and has not been implemented in CuPy.

    Returns
    -------
    nf : cupy.ndarray
        Factorial of `n`, as integer or float depending on `exact`.

    See Also
    --------
    :func:`scipy.special.factorial`

    """
    if exact:
        raise NotImplementedError(
            "Exact case not available on the GPU (requires arbitrary "
            "length integers)."
        )
    return _factorial_approx(n)


factorial2_approx_implementation = """

#define NPY_PI        3.141592653589793238462643383279502884  /* pi */

__device__ double factorial2_approx(int n)
{
    double nd2, out;
    if (n >= -1) {
        nd2 = double(n) / 2;
        if (n % 2) {
            out = Gamma(nd2 + 1) / sqrt(NPY_PI) * pow(2.0, nd2 + 0.5);
        } else {
            out = Gamma(nd2 + 1) * pow(2.0, nd2);
        }
    } else {
        out = 0.0;
    }
    return out;
}
"""

_factorial2_approx = _core.create_ufunc(
    'cupyx_scipy_factorial2', ('l->d',),
    'out0 = out0_type(factorial2_approx(int(in0)))',
    preamble=gamma_definition + factorial2_approx_implementation,
    doc='''Fast approximation to factorial2 Gamma functions''')


def factorial2(n, exact=False):
    """Double factorial.

    This is the factorial with every second value skipped.  E.g., ``7!! = 7 * 5
    * 3 * 1``.  It can be approximated numerically as::

    n!! = special.gamma(n/2+1)*2**((m+1)/2)/sqrt(pi)  n odd
        = 2**(n/2) * (n/2)!                           n even

    Parameters
    ----------
    n : int or cupy.ndarray of ints
        Input values.  If ``n < 0``, the return value is 0.
    exact : bool, optional
        If False, the result is approximated in floating point using the
        `gamma` function. The exact case is performed with arbitrary length
        integer arithmetic and has not been implemented in CuPy.

    Returns
    -------
    nff : cupy.ndarray
        Double factorial of `n`, as integer or float depending on `exact`.

    See Also
    --------
    :func:`scipy.special.factorial2`

    """
    if exact:
        raise NotImplementedError(
            "Exact case not available on the GPU (requires arbitrary "
            "length integers)."
        )
    return _factorial2_approx(n)


binom_implementation = """

__noinline__ __device__ double binom(double n, double k)
{
    double kx, nx, num, den, dk, sgn;
    int i;

    if (n < 0) {
        nx = floor(n);
        if (n == nx) {
            // undefined
            return CUDART_NAN;
        }
    }

    kx = floor(k);
    if ((k == kx) && ((fabs(n) > 1e-8) || (n == 0))) {
        // Integer case: use multiplication formula for less rounding error
        // for cases where the result is an integer.
        //
        // This cannot be used for small nonzero n due to loss of
        // precision.

        nx = floor(n);
        if ((nx == n) && (kx > nx/2) && (nx > 0)) {
            // Reduce kx by symmetry
            kx = nx - kx;
        }

        if ((kx >= 0) && (kx < 20)) {
            num = 1.0;
            den = 1.0;
            for (i=1; i<=int(kx); i++) {
                num *= i + n - kx;
                den *= i;
                if (fabs(num) > 1e50) {
                    num /= den;
                    den = 1.0;
                }
            }
            return num/den;
        }
    }

    // general case:
    if ((n >= 1e10 * k) && (k > 0)){
        // avoid under/overflows in intermediate results
        return exp(-lbeta(1 + n - k, 1 + k) - log(n + 1));
    } else if (k > 1e8 * fabs(n)) {
        // avoid loss of precision
        num = Gamma(1 + n) / fabs(k) + Gamma(1 + n) * n / (2*k*k); // + ...
        num /= NPY_PI * pow(fabs(k), n);
        if (k > 0) {
            kx = floor(k);
            if (int(kx) == kx) {
                dk = k - kx;
                sgn = ((int(kx) % 2) == 0) ? 1 : -1;
            } else {
                dk = k;
                sgn = 1;
            }
            return num * sin((dk-n)*NPY_PI) * sgn;
        } else {
            kx = floor(k);
            if (int(kx) == kx) {
                return 0;
            } else {
                return num * sin(k*NPY_PI);
            }
        }
    } else {
        return 1 / (n + 1) / beta(1 + n - k, 1 + k);
    }
}

"""

binom_preamble = (
    common_beta_preamble +
    beta_definition +
    lbeta_definition +
    binom_implementation
)


binom = _core.create_ufunc(
    'cupyx_scipy_binom', ('dd->d',),
    'out0 = binom(in0, in1)',
    preamble=binom_preamble,
    doc='''Binomial coefficient

    Parameters
    ----------
    n : int or cupy.ndarray
        The number of things.
    k : int or cupy.ndarray
        The number of elements taken.

    Returns
    -------
    b : cupy.ndarray
        The number of combinations of `n` things taken `k` at a time.

    See Also
    --------
    :func:`scipy.special.binom`

    ''')


comb_double_implementation = """
__device__ double comb(double N, double k)
{
    double out;
    if ((k <= N) && (N >= 0) && (k >= 0)) {
        out = binom(N, k);
    } else {
        out = 0.0;
    }
    return out;
}

"""

_comb_double = _core.create_ufunc(
    'cupyx_scipy_comb_double', ('dd->d',),
    'out0 = comb(in0, in1)',
    preamble=binom_preamble + comb_double_implementation,
    doc='''Floating point approximation to (n choose k)''')


comb_long_implementation = """

#ifndef uint64_t
#define uint64_t unsigned long long
#endif

#define UINT64_MAX 0xFFFFFFFFFFFFFFFFULL

__noinline__ __device__
uint64_t comb_uint64(int _N, int _k)
{
    // Compute binom(N, k) for integers.
    // Returns 0 if error/overflow encountered.

    uint64_t N, k, val, j, M, nterms;

    if (_N < 0 or _k < 0) {
        return 0;
    }
    N = (unsigned long long)_N;
    k = (unsigned long long)_k;

    if ((k > N) || (N == UINT64_MAX)) {
        return 0;
    }

    M = N + 1;
    nterms = min(k, N - k);

    val = 1;

    for (j=1; j<=nterms; j++) {
        // Overflow check
        if (val > (UINT64_MAX / (M - j))){
            return 0;
        }

        val *= M - j;
        val /= j;
    }
    return val;
}
"""

_comb_uint64 = _core.create_ufunc(
    'cupyx_scipy_comb_uint64', ('ll->L', 'LL->L'),
    'out0 = _comb_uint64(in0, in1)',
    preamble=comb_uint64_implementation,
    doc='''exact unsigned long implementation of (n choose k)''')


def comb(N, k, exact=False, repetition=False):
    """The number of combinations of N things taken k at a time.

    This is often expressed as "N choose k".

    Parameters
    ----------
    N : int or cupy.ndarray
        Number of things.
    k : int or cupy.ndarray
        Number of elements taken.
    exact : bool, optional
        If `exact` is False, then floating point precision is used, otherwise
        exact long integer is computed. Only the floating point case is
        implemented in CuPy.
    repetition : bool, optional
        If `repetition` is True, then the number of combinations with
        repetition is computed.

    Returns
    -------
    val : int, float, ndarray
        The total number of combinations.

    See Also
    --------
    :func:`scipy.special.comb`

    Notes
    -----
    - Unlike SciPy, array arguments are also accepted for the exact=True case.
    - Unlike SciPy, exact=True can only produce unsigned 64-bit outputs and
      will raise a `NotImplementedError` if `val` is overflows this range.
    - If N < 0, or k < 0, then 0 is returned.
    - If k > N and repetition=False, then 0 is returned.

    """
    if repetition:
        return comb(N + k - 1, k, exact)
    if exact:
        out = comb_uint64(N, k)
        # If overflow occurred, output of _comb_int will be 0
        # We must detect this and raise NotImplementedError in this case.
        # (no fallback Python big integer code path on the GPU).
        if out.ndim == 0:
            expect_nonzero = (k <= N) and (k >= 0) and (N >= 0)
            if expect_nonzero and int(out) == 0:  # synchronize
                raise NotImplementedError(
                    "Big integer implementation not available on the GPU."
                )
        else:
            out_expect_nonzero = out[(k <= N) & (N > 0) & (k > 0)]
            if out_expect_nonzero.min() == 0:  # synchronize
                raise NotImplementedError(
                    "Big integer implementation not available on the GPU."
                )
        return out
    else:
        return _comb_double(N, k)


perm_approx_implementation = """

__device__ double perm_approx(double N, double k)
{
    double out;
    if ((k <= N) && (N >= 0) && (k >= 0)) {
        out = poch(N - k + 1, k);
    } else {
        out = 0.0;
    }
    return out;
}

"""

_perm_approx = _core.create_ufunc(
    'cupyx_scipy_factorial2', ('l->d',),
    'out0 = out0_type(perm_approx(int(in0)))',
    preamble=poch_definition + perm_approx_implementation,
    doc='''Fast approximation to permutations via the Pochhammer symbol''')


def perm(N, k, exact=False):
    """Permutations of N things taken k at a time, i.e., k-permutations of N.

    It's also known as "partial permutations".

    Parameters
    ----------
    N : int, cupy.ndarray
        Number of things.
    k : int, cupy.ndarray
        Number of elements taken.
    exact : bool, optional
        If `exact` is False, then floating point precision is used, otherwise
        an exact long integer result is computed. Only ``exact=False`` is
        implemented in CuPy.

    Returns
    -------
    val : int or cupy.ndarray
        The number of k-permutations of N.

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If k > N, N < 0, or k < 0, then a 0 is returned.

    See Also
    --------
    :func:`scipy.special.perm`

    """
    if exact:
        raise NotImplementedError(
            "Exact case not available on the GPU (requires arbitrary "
            "length integers)."
        )
    else:
        return _perm_approx(N, k)
