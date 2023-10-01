#!/usr/bin/env python3

# filename: util.py
# description: defines needed
# mathematical functions for
# inference


import jax.numpy as jnp


def log1m_exp(x):
    """
    Numerically stable calculation
    of the quantity log(1 - exp(x)),
    following the algorithm of
    Machler [1]. This is
    the algorithm used in TensorFlow Probability,
    PyMC, and Stan, but it is not provided
    yet with Numpyro.

    Currently returns NaN for x > 0,
    but may be modified in the future
    to throw a ValueError

    [1] https://cran.r-project.org/web/packages/
    Rmpfr/vignettes/log1mexp-note.pdf
    """
    # return 0. rather than -0. if
    # we get an exponent that exceeds
    # the floating point representation
    arr_x = 1.0 * jnp.array(x)
    finf = jnp.finfo(arr_x.dtype)
    oob = arr_x < jnp.log(finf.tiny)
    mask = arr_x > -0.6931472  # appox -log(2)
    more_val = jnp.log(-jnp.expm1(arr_x))
    less_val = jnp.log1p(-jnp.exp(arr_x))

    return jnp.where(
        oob,
        0.,
        jnp.where(
            mask,
            more_val,
            less_val))


def log1p_exp_scalar(x):
    """
    Valuable for checking the below
    vectorized jax.lax implementation
    of the algorithm of Machler [1]

    [1] https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    result = None
    if x <= -37:
        result = jnp.exp(x)
    elif x <= 18:
        result = jnp.log1p(jnp.exp(x))
    elif x <= 33.3:
        result = x + jnp.exp(-x)
    else:
        result = x
    return result


def log1p_exp(x):
    """
    Stably calculate log(1 + exp(x))
    according to the
    algorithm of Machler [1]

    [1] https://cran.r-project.org/web/packages/Rmpfr/
    vignettes/log1mexp-note.pdf
    """
    return jnp.where(
        x <= 18.0,
        jnp.log1p(jnp.exp(x)),
        x + jnp.exp(-x))


def log_diff_exp(a, b):
    # note that following Stan,
    # we want the log diff exp
    # of -inf, -inf to be -inf,
    # not nan, because that
    # corresponds to log(0 - 0) = -inf
    mask = a > b
    masktwo = (a == b) & (a < jnp.inf)
    return jnp.where(mask,
                     1.0 * a + log1m_exp(
                         1.0 * b - 1.0 * a),
                     jnp.where(masktwo,
                               -jnp.inf,
                               jnp.nan))


def log_abs_diff_exp(a, b):
    return jnp.where(
        a > b,
        log_diff_exp(a, b),
        log_diff_exp(b, a))


def log_prob_to_logit_prob(log_prob):
    """
    Convert a log probability
    into a logit probability (i.e.
    a log odds) in a numerically
    stable way
    """
    return log_prob - log1m_exp(log_prob)


def spearman_karber(
        total_positive,
        wells_per_row=4,
        first_dilution=0,
        dilution_factor=1):
    return (
        first_dilution + dilution_factor * (
            -0.5 + (total_positive / wells_per_row)))
