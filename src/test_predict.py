import pytest
import numpy as np
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal)

import predict as pred


def test_predict_swab_virions():
    """
    Unit tests for the predict.predict_swab_virions
    function to ensure that it is numerically
    correct with swab peaks that lie
    either before or after the air peak
    """
    air_peak_times = np.array([2, 2, 4, 4])
    swab_peak_times = [2, 1, 5, 3]
    initial_loads = [2, 3.5, 5, 1]
    variant_decay_rate_log_ratios = 0.3252

    # if no growth or decay, final is just
    # initial
    assert_array_equal(
        pred.predict_swab_virions(
            np.array([-6200.523, 500, -2.5, 20]),
            air_peak_times,
            swab_peak_times,
            initial_loads,
            0,
            0,
            variant_decay_rate_log_ratios),
        initial_loads)

    # if growth but no decay, final is air peak
    assert_array_almost_equal(
        pred.predict_swab_virions(
            np.array([99, 103, 10000.1, 592.9]),
            air_peak_times,
            swab_peak_times,
            initial_loads,
            0.35,
            0,
            variant_decay_rate_log_ratios),
        initial_loads + 0.35 * air_peak_times,
        decimal=6)

    # decay works and decay rate ratio is used
    # and offset from peak are used
    assert_array_almost_equal(
        pred.predict_swab_virions(
            np.array([99, 103, 10000.1, 592.9]),
            air_peak_times,
            swab_peak_times,
            initial_loads,
            0.35,
            0.25,
            np.log([1, 0.5, 2, 2])),
        initial_loads + 0.35 * air_peak_times - (
            np.array([0.25, 0.125, 0.5, 0.5]) *
            (np.array([99, 103, 10000.1, 592.9]) -
             swab_peak_times)),
        decimal=4)


def test_shedding_integrals():
    # shedding doesn't matter after filter removed
    right_before_stop = pred.definite_integral_viable(
        3.2, 19.95, 20, 5, 0.2325, 0.25, 0.2, 2)
    right_at_stop = pred.definite_integral_viable(
        3.2, 20, 20, 5, 0.2325, 0.25, 0.2, 2)
    a_bit_after = pred.definite_integral_viable(
        3.2, 20.15, 20, 5, 0.2325, 0.25, 0.2, 2)
    way_after = pred.definite_integral_viable(
        3.2, 2353.235, 20, 5, 0.2325, 0.25, 0.2, 2)
    assert_array_almost_equal(right_at_stop, a_bit_after)
    assert_array_almost_equal(right_at_stop, way_after)
    assert np.all(right_before_stop < right_at_stop)

    right_before_stop = pred.viable_ode(
        3.2, 19.95, 20, 5, 0.2325, 0.25, 0.2, 2)
    right_at_stop = pred.viable_ode(
        3.2, 20, 20, 5, 0.2325, 0.25, 0.2, 2)
    a_bit_after = pred.viable_ode(
        3.2, 20.15, 20, 5, 0.2325, 0.25, 0.2, 2)
    way_after = pred.viable_ode(
        3.2, 2353.235, 20, 5, 0.2325, 0.25, 0.2, 2)
    assert_array_almost_equal(right_at_stop, a_bit_after)
    assert_array_almost_equal(right_at_stop, way_after)
    assert np.all(right_before_stop < right_at_stop)

    # log version and linear scale version give same
    # answers if numerics are stable, including
    # across a variety of shapes, as do thinning
    # version versus ode version
    np.random.seed(5)
    n_rand = (200, 3, 2)
    rand_starts = np.exp(2 * np.random.random(n_rand))
    rand_peaks = np.exp(2 * np.random.random(n_rand))
    rand_filter_ends = rand_starts + np.exp(2 * np.random.random(n_rand))
    rand_shedder_ends = rand_starts + np.exp(2 * np.random.random(n_rand))
    rand_growth_rates = np.exp(0.25 * np.random.random(n_rand))
    rand_decay_rates = np.exp(0.25 * np.random.random(n_rand))
    rand_loss_rates = np.exp(0.25 * np.random.random(n_rand))
    rand_init_log_loads = 2 * np.random.random(n_rand)

    assert_array_almost_equal(
        np.log(pred.definite_integral_viable(
            rand_starts,
            rand_shedder_ends,
            rand_filter_ends,
            rand_peaks,
            rand_growth_rates,
            rand_decay_rates,
            rand_loss_rates,
            rand_init_log_loads)),
        pred.log_def_integral_viable(
            rand_starts,
            rand_shedder_ends,
            rand_filter_ends,
            rand_peaks,
            rand_growth_rates,
            rand_decay_rates,
            rand_loss_rates,
            rand_init_log_loads),
        decimal=3)

    assert_array_almost_equal(
        pred.log_def_integral_viable(
            rand_starts,
            rand_shedder_ends,
            rand_filter_ends,
            rand_peaks,
            rand_growth_rates,
            rand_decay_rates,
            rand_loss_rates,
            rand_init_log_loads),
        np.log(pred.viable_ode(
            rand_starts,
            rand_shedder_ends,
            rand_filter_ends,
            rand_peaks,
            rand_growth_rates,
            rand_decay_rates,
            rand_loss_rates,
            rand_init_log_loads)),
        decimal=3)
    
    assert_array_almost_equal(
        pred.log_def_integral_viable(
            rand_starts,
            rand_shedder_ends,
            rand_filter_ends,
            rand_peaks,
            rand_growth_rates,
            rand_decay_rates,
            rand_loss_rates,
            rand_init_log_loads),
        pred.log_viable_ode(
            rand_starts,
            rand_shedder_ends,
            rand_filter_ends,
            rand_peaks,
            rand_growth_rates,
            rand_decay_rates,
            rand_loss_rates,
            rand_init_log_loads),
        decimal=3)

    print([
        rand_starts[0, 0, 1],
        rand_shedder_ends[0, 0, 1],
        rand_filter_ends[0, 0, 1],
        rand_peaks[0, 0, 1],
        rand_growth_rates[0, 0, 1],
        rand_decay_rates[0, 0, 1],
        rand_loss_rates[0, 0, 1],
        rand_init_log_loads[0, 0, 1]])

    assert_array_almost_equal(
        pred.log_definite_integral_viable_alt(
            rand_starts,
            rand_shedder_ends,
            rand_filter_ends,
            rand_peaks,
            rand_growth_rates,
            rand_decay_rates,
            rand_loss_rates,
            rand_init_log_loads),
        pred.log_def_integral_viable(
            rand_starts,
            rand_shedder_ends,
            rand_filter_ends,
            rand_peaks,
            rand_growth_rates,
            rand_decay_rates,
            rand_loss_rates,
            rand_init_log_loads),
        decimal=3)
