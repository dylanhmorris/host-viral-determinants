import jax.numpy as jnp
import numpy as np
from util import log_diff_exp, log1m_exp


def predict_swab_virions(
        swab_times,
        air_peak_times,
        swab_peak_times,
        initial_loads,
        variant_growth_rates,
        variant_decay_rates,
        variant_decay_rate_log_ratios):
    swab_times_pivot = swab_times
    air_peaks = (initial_loads + air_peak_times *
                 variant_growth_rates)
    swab_peak_diffs = swab_times_pivot - swab_peak_times
    transformed_decay_rates = (
        variant_decay_rates *
        jnp.exp(variant_decay_rate_log_ratios))
    rates_of_change = jnp.where(
        swab_times_pivot < swab_peak_times,
        variant_growth_rates,
        -transformed_decay_rates)
    return (air_peaks + swab_peak_diffs * rates_of_change)


def integral_growth_log(growth_rate,
                        initial_log_load,
                        t_start,
                        t_end):
    return (
        log_diff_exp(
            initial_log_load + growth_rate * t_end,
            initial_log_load + growth_rate * t_start
        ) - jnp.log(growth_rate))


def integral_decay_log(decay_rate,
                       initial_log_load,
                       t_start,
                       t_end):
    return (
        log_diff_exp(
            initial_log_load - decay_rate * t_start,
            initial_log_load - decay_rate * t_end
        ) - jnp.log(decay_rate))


def log_antiderivative_shed(time,
                            growth_rate,
                            decay_rate,
                            initial_log_load,
                            t_peak):
    time_since = time - t_peak
    log_peak_load = initial_log_load + growth_rate * t_peak
    log_growth = jnp.log(growth_rate)
    log_decay = jnp.log(decay_rate)
    log_decay_term = jnp.logaddexp(
        log1m_exp(-decay_rate * time_since) - log_decay,
        -log_growth)

    return jnp.where(
        time_since < 0,
        initial_log_load + growth_rate * time - log_growth,
        log_peak_load + log_decay_term
    )


def log_definite_integral_shed(
        t_left,
        t_right,
        growth_rate,
        decay_rate,
        initial_log_load,
        t_peak):

    return log_diff_exp(
        log_antiderivative_shed(t_right, growth_rate,
                                decay_rate, initial_log_load, t_peak),
        log_antiderivative_shed(t_left, growth_rate,
                                decay_rate, initial_log_load, t_peak))


def predict_log_instantaneous_air_shedding(
        times,
        true_peak_times,
        initial_loads,
        variant_growth_rates,
        variant_decay_rates):
    """
    Compute log intantaneous rate of
    shedding into the air
    in log PFU per mL per hour
    """
    time_since_peak = times - true_peak_times
    peaks = initial_loads + (
      variant_growth_rates * true_peak_times)

    return jnp.where(
      time_since_peak < 0,
      initial_loads + (
          variant_growth_rates * times),
      peaks - (
          variant_decay_rates *
          time_since_peak))


def predict_log_air_sample(
        times,
        true_peak_times,
        initial_loads,
        log_mean_hourly_respiration_rates,
        variant_growth_rates,
        variant_decay_rates,
        sample_duration_hours=24):
    """
    Compute log cumulative virus
    sampled from the air over the
    course of a sampling window
    """
    # convert mL per hour to mL per 24h
    log_mL_per_sample_duration = (
        log_mean_hourly_respiration_rates +
        np.log(sample_duration_hours))
    peak_diffs = times - true_peak_times
    peak_loads = (
        initial_loads + variant_growth_rates *
        true_peak_times)

    # calculate cumulative
    # sample_duration shedding per
    # mL
    log_shed_per_mL_per_sample_duration = jnp.where(
        peak_diffs < -sample_duration_hours,
        integral_growth_log(variant_growth_rates,
                            initial_loads,
                            times,
                            times +
                            sample_duration_hours),
        jnp.where(
            peak_diffs >= 0,
            integral_decay_log(variant_decay_rates,
                               peak_loads,
                               peak_diffs,
                               peak_diffs +
                               sample_duration_hours),
            jnp.logaddexp(
                integral_growth_log(variant_growth_rates,
                                    initial_loads,
                                    times,
                                    true_peak_times),
                integral_decay_log(variant_decay_rates,
                                   peak_loads,
                                   0.0,
                                   peak_diffs +
                                   sample_duration_hours))))
    # multiply
    return (log_mL_per_sample_duration +
            log_shed_per_mL_per_sample_duration)


def log_antiderivative_viable(
        time_since_inoculation,
        rate_of_change,
        time_initial,
        time_of_sample,
        viability_loss_rate):
    timedelta = time_since_inoculation - time_initial

    return (-jnp.log(rate_of_change + viability_loss_rate) +
            rate_of_change * timedelta - viability_loss_rate * (
                time_of_sample - time_since_inoculation))


def antiderivative_viable(
        time_since_inoculation,
        rate_of_change,
        time_initial,
        time_of_sample,
        viability_loss_rate):

    return (jnp.exp(rate_of_change * (time_since_inoculation - time_initial) -
                    viability_loss_rate * (
                        time_of_sample - time_since_inoculation)
                    ) / (rate_of_change + viability_loss_rate))


def viable_ode(
        time_of_sample_start,
        time_shedder_removed,
        time_filter_recovered,
        time_of_peak_air_shedding,
        growth_rate,
        decay_rate,
        viability_loss_rate,
        initial_log_load):
    """
    Compute the quantity of viable virions at
    a time of filter recovery (t_2 in equations
    in the manuscript) shed since a time of sample
    start (t_1 in equations in the manuscript),
    by treating the process as a pair of coupled
    differential equations for shedding and decay.

    Parameters
    ----------
    time_of_sample_start : float
        When did air sampling begin relative to the time of
        inoculation of the shedder?

    time_shedder_removed : float
        When was the shedder removed (typically less than or
        equal to the time the filter was recovered), relative
        to the time of inoculation of the shedder?

    time_filter_recovered : float
        When was the filter recovered, relative to the
        time of inoculation of the shedder

    time_of_peak_air_shedding : float
        When did the shedder achieve peak air shedding,
        relative to its time of inoculation?

    growth_rate : float
        What was the exponential growth rate of shedding
        up to that peak?

    decay_rate : float
        What was the exponential decay rate of shedding
        down from that peak?

    viability_loss_rate : float
        Exponential decay rate of viable virus shed in
        the environment

    initial_log_load : float
        Initial log viral load for the shedder at time
        of inoculation (t=0).
    """
    if not np.all(
            time_shedder_removed > time_of_sample_start):
        raise ValueError("time of removal of shedder "
                         "must be greater than time of "
                         "sample start")
    if not np.all(
            time_filter_recovered > time_of_sample_start):
        raise ValueError("time of recovery of filter "
                         "must be greater than time of "
                         "sample start")
    delta_t_start_to_peak = (
        time_of_peak_air_shedding - time_of_sample_start)
    delta_t_peak_to_start = -delta_t_start_to_peak

    time_accumulation_stops = jnp.minimum(time_shedder_removed,
                                          time_filter_recovered)

    t_peak_log_load = (initial_log_load +
                       time_of_peak_air_shedding *
                       growth_rate)

    t_start_log_load = jnp.where(
        time_of_sample_start < time_of_peak_air_shedding,
        initial_log_load + time_of_sample_start * growth_rate,
        t_peak_log_load - decay_rate * delta_t_peak_to_start)

    def viability_loss(timedelta):
        return jnp.exp(-viability_loss_rate * timedelta)

    def growth(timedelta, log_load_init):
        growth_multiplier = (
            jnp.exp(log_load_init) /
            (growth_rate + viability_loss_rate))
        return growth_multiplier * (
            jnp.exp(growth_rate * timedelta) -
            viability_loss(timedelta))

    def decay(timedelta, log_load_init):
        decay_multiplier = (
            jnp.exp(log_load_init) /
            (-decay_rate + viability_loss_rate))
        return decay_multiplier * (
            jnp.exp(-decay_rate * timedelta) -
            viability_loss(timedelta))

    viable_at_accumlation_stop = jnp.where(
        time_accumulation_stops < time_of_peak_air_shedding,
        growth(time_accumulation_stops - time_of_sample_start,
               t_start_log_load),
        jnp.where(
            time_of_sample_start > time_of_peak_air_shedding,
            decay(time_accumulation_stops - time_of_sample_start,
                  t_start_log_load),
            decay(time_accumulation_stops - time_of_peak_air_shedding,
                  t_peak_log_load) +
            growth(time_of_peak_air_shedding - time_of_sample_start,
                   t_start_log_load) * viability_loss(
                       time_accumulation_stops - time_of_peak_air_shedding)
        ))

    result = jnp.where(time_filter_recovered > time_accumulation_stops,
                       viable_at_accumlation_stop * viability_loss(
                           time_filter_recovered - time_accumulation_stops),
                       viable_at_accumlation_stop)
    return result


def log_viable_ode(
        time_of_sample_start,
        time_shedder_removed,
        time_filter_recovered,
        time_of_peak_air_shedding,
        growth_rate,
        decay_rate,
        viability_loss_rate,
        initial_log_load):
    """
    Compute the quantity of viable virions at
    a time of filter recovery (t_2 in equations
    in the manuscript) shed since a time of sample
    start (t_1 in equations in the manuscript),
    by treating the process as a pair of coupled
    differential equations for shedding and decay.

    Parameters
    ----------
    time_of_sample_start : float
        When did air sampling begin relative to the time of
        inoculation of the shedder?

    time_shedder_removed : float
        When was the shedder removed (typically less than or
        equal to the time the filter was recovered), relative
        to the time of inoculation of the shedder?

    time_filter_recovered : float
        When was the filter recovered, relative to the
        time of inoculation of the shedder

    time_of_peak_air_shedding : float
        When did the shedder achieve peak air shedding,
        relative to its time of inoculation?

    growth_rate : float
        What was the exponential growth rate of shedding
        up to that peak?

    decay_rate : float
        What was the exponential decay rate of shedding
        down from that peak?

    viability_loss_rate : float
        Exponential decay rate of viable virus shed in
        the environment

    initial_log_load : float
        Initial log viral load for the shedder at time
        of inoculation (t=0).
    """
    if not np.all(
            time_shedder_removed > time_of_sample_start):
        raise ValueError("time of removal of shedder "
                         "must be greater than time of "
                         "sample start")
    if not np.all(
            time_filter_recovered > time_of_sample_start):
        raise ValueError("time of recovery of filter "
                         "must be greater than time of "
                         "sample start")
    delta_t_start_to_peak = (
        time_of_peak_air_shedding - time_of_sample_start)
    delta_t_peak_to_start = -delta_t_start_to_peak

    time_accumulation_stops = jnp.minimum(time_shedder_removed,
                                          time_filter_recovered)

    t_peak_log_load = (initial_log_load +
                       time_of_peak_air_shedding *
                       growth_rate)

    t_start_log_load = jnp.where(
        time_of_sample_start < time_of_peak_air_shedding,
        initial_log_load + time_of_sample_start * growth_rate,
        t_peak_log_load - decay_rate * delta_t_peak_to_start)

    def log_viability_loss(timedelta):
        return -viability_loss_rate * timedelta

    def log_growth(timedelta, log_load_init):
        log_growth_multiplier = (
            log_load_init - jnp.log(
                growth_rate + viability_loss_rate))
        return log_growth_multiplier + log_diff_exp(
            growth_rate * timedelta,
            log_viability_loss(timedelta))

    def decay(timedelta, log_load_init):
        decay_multiplier = (
            jnp.exp(log_load_init) /
            (-decay_rate + viability_loss_rate))
        return decay_multiplier * (
            jnp.exp(-decay_rate * timedelta) -
            jnp.exp(log_viability_loss(timedelta)))

    viable_at_accumlation_stop = jnp.where(
        time_accumulation_stops < time_of_peak_air_shedding,
        log_growth(time_accumulation_stops - time_of_sample_start,
                   t_start_log_load),
        jnp.where(
            time_of_sample_start > time_of_peak_air_shedding,
            jnp.log(decay(time_accumulation_stops - time_of_sample_start,
                          t_start_log_load)),
            jnp.log(
                decay(time_accumulation_stops - time_of_peak_air_shedding,
                      t_peak_log_load) +
                jnp.exp(
                    log_growth(time_of_peak_air_shedding -
                               time_of_sample_start,
                               t_start_log_load) + log_viability_loss(
                                   time_accumulation_stops -
                                   time_of_peak_air_shedding))
            )))

    result = jnp.where(time_filter_recovered > time_accumulation_stops,
                       viable_at_accumlation_stop +
                       log_viability_loss(
                           time_filter_recovered -
                           time_accumulation_stops),
                       viable_at_accumlation_stop)
    return result


def definite_integral_viable(
        time_of_sample_start,
        time_shedder_removed,
        time_filter_recovered,
        time_of_peak_air_shedding,
        growth_rate,
        decay_rate,
        viability_loss_rate,
        initial_log_load):
    """
    Definite integral of viable virus
    accumulating on a filter on a natural
    scale. Not as numerically stable,
    but more interpretable, and allows
    for testing of the log scale function

    Parameters
    ----------
    time_of_sample_start : float
        When did air sampling begin relative to the time of
        inoculation of the shedder?

    time_shedder_removed : float
        When was the shedder removed (typically less than or
        equal to the time the filter was recovered), relative
        to the time of inoculation of the shedder?

    time_filter_recovered : float
        When was the filter recovered, relative to the
        time of inoculation of the shedder

    time_of_peak_air_shedding : float
        When did the shedder achieve peak air shedding,
        relative to its time of inoculation?

    growth_rate : float
        What was the exponential growth rate of shedding
        up to that peak?

    decay_rate : float
        What was the exponential decay rate of shedding
        down from that peak?

    viability_loss_rate : float
        Exponential decay rate of viable virus shed in
        the environment

    initial_log_load : float
        Initial log viral load for the shedder at time
        of inoculation (t=0).
    """
    if not np.all(
            time_shedder_removed > time_of_sample_start):
        raise ValueError("time of removal of shedder "
                         "must be greater than time of "
                         "sample start")
    if not np.all(
            time_filter_recovered > time_of_sample_start):
        raise ValueError("time of recovery of filter "
                         "must be greater than time of "
                         "sample start")
    time_accumulation_stops = jnp.minimum(time_shedder_removed,
                                          time_filter_recovered)

    delta_t_start_to_peak = (
        time_of_peak_air_shedding - time_of_sample_start)
    delta_t_peak_to_start = -delta_t_start_to_peak

    t_peak_log_load = (initial_log_load +
                       time_of_peak_air_shedding *
                       growth_rate)

    t_start_log_load = jnp.where(
        time_of_sample_start < time_of_peak_air_shedding,
        initial_log_load + time_of_sample_start * growth_rate,
        t_peak_log_load - decay_rate * delta_t_peak_to_start)

    result = jnp.where(
        time_accumulation_stops < time_of_peak_air_shedding,
        jnp.exp(t_start_log_load) * (
            antiderivative_viable(
                time_accumulation_stops,
                growth_rate,
                time_of_sample_start,
                time_filter_recovered,
                viability_loss_rate) -
            antiderivative_viable(
                time_of_sample_start,
                growth_rate,
                time_of_sample_start,
                time_filter_recovered,
                viability_loss_rate)
        ),
        jnp.where(
            time_of_sample_start > time_of_peak_air_shedding,
            jnp.exp(t_start_log_load) * (
                antiderivative_viable(
                    time_accumulation_stops,
                    -decay_rate,
                    time_of_sample_start,
                    time_filter_recovered,
                    viability_loss_rate) -
                antiderivative_viable(
                    time_of_sample_start,
                    -decay_rate,
                    time_of_sample_start,
                    time_filter_recovered,
                    viability_loss_rate)),
            jnp.exp(t_start_log_load) * (
                antiderivative_viable(
                    time_of_peak_air_shedding,
                    growth_rate,
                    time_of_sample_start,
                    time_filter_recovered,
                    viability_loss_rate) -
                antiderivative_viable(
                    time_of_sample_start,
                    growth_rate,
                    time_of_sample_start,
                    time_filter_recovered,
                    viability_loss_rate)) +
            jnp.exp(t_peak_log_load) * (
                antiderivative_viable(
                    time_accumulation_stops,
                    -decay_rate,
                    time_of_peak_air_shedding,
                    time_filter_recovered,
                    viability_loss_rate) -
                antiderivative_viable(
                    time_of_peak_air_shedding,
                    -decay_rate,
                    time_of_peak_air_shedding,
                    time_filter_recovered,
                    viability_loss_rate))))
    return result


def log_definite_integral_viable_alt(
        time_of_sample_start,
        time_shedder_removed,
        time_filter_recovered,
        time_of_peak_air_shedding,
        growth_rate,
        decay_rate,
        viability_loss_rate,
        initial_log_load):
    """
    Alternate log definite integral of viable virus
    accumulating on a filter on a natural
    scale. Not as numerically stable,
    but more interpretable, and allows
    for testing of the log scale function

    Parameters
    ----------
    time_of_sample_start : float
        When did air sampling begin relative to the time of
        inoculation of the shedder?

    time_shedder_removed : float
        When was the shedder removed (typically less than or
        equal to the time the filter was recovered), relative
        to the time of inoculation of the shedder?

    time_filter_recovered : float
        When was the filter recovered, relative to the
        time of inoculation of the shedder

    time_of_peak_air_shedding : float
        When did the shedder achieve peak air shedding,
        relative to its time of inoculation?

    growth_rate : float
        What was the exponential growth rate of shedding
        up to that peak?

    decay_rate : float
        What was the exponential decay rate of shedding
        down from that peak?

    viability_loss_rate : float
        Exponential decay rate of viable virus shed in
        the environment

    initial_log_load : float
        Initial log viral load for the shedder at time
        of inoculation (t=0).
    """
    if not np.all(
            time_shedder_removed > time_of_sample_start):
        raise ValueError("time of removal of shedder "
                         "must be greater than time of "
                         "sample start")
    if not np.all(
            time_filter_recovered > time_of_sample_start):
        raise ValueError("time of recovery of filter "
                         "must be greater than time of "
                         "sample start")
    time_accumulation_stops = jnp.minimum(time_shedder_removed,
                                          time_filter_recovered)
    delta_t_start_to_peak = (
        time_of_peak_air_shedding - time_of_sample_start)
    delta_t_peak_to_start = -delta_t_start_to_peak

    log_load_at_t_peak = (initial_log_load +
                          time_of_peak_air_shedding *
                          growth_rate)

    log_load_at_t_start = jnp.where(
        time_of_sample_start < time_of_peak_air_shedding,
        initial_log_load + time_of_sample_start * growth_rate,
        log_load_at_t_peak - decay_rate * delta_t_peak_to_start)

    result = jnp.where(
        time_accumulation_stops < time_of_peak_air_shedding,
        log_load_at_t_start + log_diff_exp(
            log_antiderivative_viable(
                time_accumulation_stops,
                growth_rate,
                time_of_sample_start,
                time_filter_recovered,
                viability_loss_rate),
            log_antiderivative_viable(
                time_of_sample_start,
                growth_rate,
                time_of_sample_start,
                time_filter_recovered,
                viability_loss_rate)
        ),
        jnp.where(
            time_of_sample_start > time_of_peak_air_shedding,
            log_load_at_t_start + jnp.log(
                antiderivative_viable(
                    time_accumulation_stops,
                    -decay_rate,
                    time_of_sample_start,
                    time_filter_recovered,
                    viability_loss_rate) -
                antiderivative_viable(
                    time_of_sample_start,
                    -decay_rate,
                    time_of_sample_start,
                    time_filter_recovered,
                    viability_loss_rate)),
            jnp.logaddexp(
                log_load_at_t_start + log_diff_exp(
                    log_antiderivative_viable(
                        time_of_peak_air_shedding,
                        growth_rate,
                        time_of_sample_start,
                        time_filter_recovered,
                        viability_loss_rate),
                    log_antiderivative_viable(
                        time_of_sample_start,
                        growth_rate,
                        time_of_sample_start,
                        time_filter_recovered,
                        viability_loss_rate)),
                log_load_at_t_peak + jnp.log(
                    antiderivative_viable(
                        time_accumulation_stops,
                        -decay_rate,
                        time_of_peak_air_shedding,
                        time_filter_recovered,
                        viability_loss_rate) -
                    antiderivative_viable(
                        time_of_peak_air_shedding,
                        -decay_rate,
                        time_of_peak_air_shedding,
                        time_filter_recovered,
                        viability_loss_rate)))))
    return result


def log_def_integral_viable(
        t_start,
        t_end,
        t_sample,
        t_peak,
        growth_rate,
        decay_rate,
        viability_loss_rate,
        initial_log_load):

    # short aliases
    v0 = initial_log_load
    t = t_sample
    g = growth_rate
    d = decay_rate
    vl = viability_loss_rate
    p = t_peak
    v1 = v0 + g * p

    left_endpoint = jnp.where(
        t_start < t_sample,
        t_start,
        t_sample)

    right_endpoint = jnp.where(
        t_end < t_sample,
        t_end,
        t_sample)

    def q1(x):
        return v0 + (g * x - vl * (t - x)) - jnp.log(g + vl)

    def q1diff(right, left):
        return log_diff_exp(q1(right), q1(left))

    decay_diff = vl - d

    def q2(x):
        return v1 + (-d * (x - p) - vl * (t - x))

    def q2diff(right, left):
        return jnp.where(
            decay_diff > 0,
            -jnp.log(decay_diff) +
            log_diff_exp(q2(right), q2(left)),
            jnp.where(decay_diff < 0,
                      -jnp.log(-decay_diff) +
                      log_diff_exp(q2(left), q2(right)),
                      v1 + (-d * (t - p)) + jnp.log(right - left)))

    def both_diff(right, left, peak):
        return jnp.logaddexp(q1diff(peak, left),
                             q2diff(right, peak))

    return jnp.where(
        right_endpoint < t_peak,
        q1diff(right_endpoint, left_endpoint),
        jnp.where(left_endpoint >= t_peak,
                  q2diff(right_endpoint, left_endpoint),
                  both_diff(right_endpoint, left_endpoint, t_peak)))


def predict_viable_air_sample(
        times,
        true_peak_times,
        initial_loads,
        log_mean_hourly_respiration_rates,
        variant_growth_rates,
        variant_decay_rates,
        variant_environmental_decay_rates,
        sample_duration_hours=24):

    log_mL_per_sample = (
        log_mean_hourly_respiration_rates +
        np.log(sample_duration_hours))

    # calculate cumulative duration shedding per
    # mL
    log_shed_per_mL_per_sample = log_viable_ode(
        times,
        times + sample_duration_hours,
        times + sample_duration_hours,
        true_peak_times,
        variant_growth_rates,
        variant_decay_rates,
        variant_environmental_decay_rates,
        initial_loads)

    # multiply
    return (log_mL_per_sample + log_shed_per_mL_per_sample)
