#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import sys
import pickle
import jax
import numpyro.distributions as dist
import polars as pl
import string
from numpyro.infer import MCMC

import plotting as plot
from predict import (predict_log_air_sample,
                     predict_viable_air_sample,
                     predict_swab_virions,
                     predict_log_instantaneous_air_shedding)

def predict(xs,
            posterior_samples,
            model,
            rng_seed=None):

    if rng_seed is None:
        rng_seed = np.random.randint(1, 10000)

    ps = posterior_samples
    predictions = dict()
    cage_pred = dict()

    rng_key = jax.random.PRNGKey(rng_seed)

    female_peak_loads = dist.Normal(ps["mean_log_peak"],
                                    ps["sd_log_peak"])
    male_peak_loads = dist.Normal(ps["mean_log_peak"] +
                                  ps["male_peak_offset"][..., np.newaxis],
                                  ps["sd_log_peak"])
    respiration = dist.Normal(ps["log_resp_rate_mean"],
                              ps["log_resp_rate_sd"])
    male_decay = dist.Normal(ps["variant_log_decay_rate"] +
                             ps["male_decay_rate_offset"][
                                 ..., np.newaxis],
                             ps["sd_individual_log_decay_rate"])
    female_decay = dist.Normal(ps["variant_log_decay_rate"],
                               ps["sd_individual_log_decay_rate"])
    male_growth = dist.Normal(ps["variant_log_growth_rate"] +
                              ps["male_growth_rate_offset"][
                                  ..., np.newaxis],
                              ps["sd_individual_log_growth_rate"])
    female_growth = dist.Normal(ps["variant_log_growth_rate"],
                                ps["sd_individual_log_growth_rate"])
    peak_times = dist.Normal(ps["mean_log_peak_time"],
                             ps["sd_log_peak_time"])
    
    predictions = {
        "Alpha": dict(),
        "Delta": dict()
    }
    
    for cage_name, cage in model["cages"].items():
        
        indivs = cage["individuals"]
        cage_pred[cage_name] = dict()
        i_vars = model["hamster_variant_ids"][indivs]
        i_sexes = model["hamster_sexes"][indivs]
        
        air_array = predict_log_air_sample(
            (xs * 24)[..., np.newaxis, np.newaxis],
            ps["individual_peak_time"][::, indivs],
            ps["individual_initial_load"][::, indivs],
            ps["mean_respiration_rate"][::, indivs],
            ps["individual_growth_rate"][::, indivs],
            np.exp(
                ps["log_rna_decay_rate_ratio"][::, i_vars] +
                ps["individual_log_decay_rate"][::, indivs]))
        
        cage_pred[cage_name]["predicted_log10_air_rna"] = (
            logsumexp(
                ps["log_air_RNA_per_air_virions"][::, i_vars] + 
                air_array, axis=2).transpose() /
            np.log(10) # convert to log10 copies
        )
        
    
        air_pfu_array = predict_viable_air_sample(
            (xs * 24)[..., np.newaxis, np.newaxis],
            ps["individual_peak_time"][::, indivs],
            ps["individual_initial_load"][::, indivs],
            ps["mean_respiration_rate"][::, indivs],
            ps["individual_growth_rate"][::, indivs],
            ps["individual_decay_rate"][::, indivs],
            ps["environmental_decay_rate"][::, i_vars])

        cage_pred[cage_name]["predicted_air_log_PFU"] = logsumexp(
            air_pfu_array, axis=2).transpose()
    
        cage_pred[cage_name]["predicted_PFU"] = (
            jnp.exp(cage_pred[cage_name]["predicted_air_log_PFU"]))
    
        cage_pred[cage_name]["i_var"] = i_vars
        cage_pred[cage_name]["sex"] = i_sexes
        cage_pred[cage_name]["n_indiv"] = len(indivs)
        cage_pred[cage_name]["xs"] = xs


    for i_var, variant in enumerate(["Alpha", "Delta"]):
        for i_sex, sex in enumerate(["F", "M"]):
            if sex == "M":
                peak_loads = male_peak_loads
                decay = male_decay
                growth = male_growth
            elif sex == "F":
                peak_loads = female_peak_loads
                decay = female_decay
                growth = female_growth
            else:
                raise ValueError("Sex {} not modeled"
                                 "".format(sex))
            _, rng_key = jax.random.split(rng_key)
    
            sim_peak_loads = peak_loads.sample(key=rng_key)
            _, rng_key = jax.random.split(rng_key)
            sim_respiration_rates = respiration.sample(key=rng_key)
            _, rng_key = jax.random.split(rng_key)
            sim_decay_rates = np.exp(decay.sample(key=rng_key))
            _, rng_key = jax.random.split(rng_key)
            sim_growth_rates = np.exp(growth.sample(key=rng_key))
            _, rng_key = jax.random.split(rng_key)
            sim_peak_times = np.exp(peak_times.sample(key=rng_key))
            _, rng_key = jax.random.split(rng_key)

            sim_inocula = sim_peak_loads - (
                sim_growth_rates * sim_peak_times)

            predictions[variant]["air_" + sex] = jnp.exp(
                sim_respiration_rates +
                predict_log_instantaneous_air_shedding(
                    (xs * 24)[..., np.newaxis], 
                    sim_peak_times[::, i_var], 
                    sim_inocula[::, i_var], 
                    sim_growth_rates[::, i_var],
                    sim_decay_rates[::, i_var]))
        
            predictions[variant]["swab_virions_" + sex] = (
                predict_swab_virions(
                    (xs * 24)[..., np.newaxis], 
                    sim_peak_times[::, i_var], 
                    np.exp(np.log(sim_peak_times[::, i_var]) + 
                           ps["log_swab_peak_offset"][::, i_var]),
                    sim_inocula[::, i_var], 
                    sim_growth_rates[::, i_var],
                    sim_decay_rates[::, i_var],
                    ps["log_swab_decay_rate_ratio"][::, i_var])
            )

            predictions[variant]["log10_swab_rna_" + sex] = (
                (
                    ps["oral_RNA_per_shed_virions"][::, i_var] +
                    predict_swab_virions(
                        (xs * 24)[..., np.newaxis],
                        sim_peak_times[::, i_var],
                        np.exp(np.log(sim_peak_times[::, i_var]) + 
                               ps["log_swab_peak_offset"][::, i_var]),
                        sim_inocula[::, i_var],
                        sim_growth_rates[::, i_var],
                        sim_decay_rates[::, i_var],
                        ps["log_swab_decay_rate_ratio"][::, i_var] + 
                        ps["log_rna_decay_rate_ratio"][::, i_var])
                ) / np.log(10) # convert to log10 units
            )
            
            predictions[variant]["swab_tcid_" + sex] = (
                (ps["oral_TCID_per_shed_virions"][::, i_var] +
                 predictions[variant]["swab_virions_" + sex]))
            
            predictions[variant]["xs"] = xs

            pass # end loop over sexes
        pass # end loop over variants

    return (predictions, cage_pred)


def create_main_figure(
        model,
        entry,
        swabs,
        pleth,
        air_samples,
        predictions,
        cage_predictions,
        n_curves_kinetics,
        n_curves_air):

    mpl.rcParams["axes.labelsize"] = 15
    mpl.rcParams["axes.titlesize"] = 15
    mpl.rcParams["xtick.labelsize"] = 15
    mpl.rcParams["ytick.labelsize"] = 15
    mpl.rcParams["figure.titlesize"] = 15
    fig = plt.figure(figsize=[13, 11.2])
    (kinetics_fig,
     shedding_fig) = fig.subfigures(
         2, 1, height_ratios=[1, 0.8])

    letter_x = 0
    letter_y = 1
    letter_size = 40
    letter_weight = "bold"

    print("Creating kinetics subplot...\n")
    _ = plot.kinetics_subfigure(swabs,
                                predictions,
                                fig=kinetics_fig,
                                markersize=7,
                                markeredgewidth=0.5,
                                n_curves=n_curves_kinetics,
                                data_lw=1)
    cax = kinetics_fig.axes[0]
    _ = cax.text(letter_x, letter_y + 0.1,
                 "A",
                 transform=cax.transAxes,
                 fontsize=letter_size,
                 fontweight=letter_weight)
    _ = kinetics_fig.supylabel("Shedding")

    print("Creating air shedding subplot...\n")
    _ = plot.air_shedding_subfigure(
        model,
        air_samples,
        cage_predictions,
        fig=shedding_fig,
        markersize=7,
        markeredgewidth=0.5,
        n_curves=n_curves_air)

    cax = shedding_fig.axes[0]
    _ = cax.text(
        letter_x, letter_y + 0.2,
        "B",
        transform=cax.transAxes,
        fontsize=letter_size,
        fontweight=letter_weight)

    print("Styling final figure...\n")
    margin = 0.1
    plt.subplots_adjust(left=margin, right=1-margin)
    return fig


def main(entry_data_path,
         mcmc_chains_path,
         swab_data_path,
         pleth_data_path,
         air_sample_data_path,
         output_path,
         dpi=400,
         separator="\t"):

    prior_check = "prior-check" in mcmc_chains_path

    if prior_check:
        print("Plotting figure with prior checks. "
              "If this is not what you intended, check "
              "which set of MCMC chains are being passed "
              "in")

    print("Reading in entry data from {}...\n".format(entry_data_path))
    entry = pl.read_csv(entry_data_path,
                        separator=separator)

    print("Reading in swab data from {}...\n"
          "".format(swab_data_path))
    swabs = pl.read_csv(swab_data_path,
                        separator=separator)

    print("Reading in pleth data from {}...\n"
          "".format(pleth_data_path))
    pleth = pl.read_csv(pleth_data_path,
                        separator=separator)

    print("Reading in air sample data from {}...\n"
          "".format(air_sample_data_path))
    air_samples = pl.read_csv(air_sample_data_path,
                              separator=separator)

    print("Reading in model and mcmc chains from {}"
          "...\n".format(mcmc_chains_path))

    with open(mcmc_chains_path, "rb") as infile:
        o, model = pickle.load(infile)
        if isinstance(o, MCMC):
            posterior_samples = o.get_samples()
        elif isinstance(o, dict):
            posterior_samples = dict(o)
        else:
            raise ValueError("Unknown output format; "
                             "expected numpyro.infer.MCMC "
                             "object or dict of samples")

    print("Making kinetics and cage air "
          "sample predictions...\n")

    predictions, cage_predictions = predict(
            np.linspace(0, 10, 250),
            posterior_samples,
            model,
            rng_seed=4332)
    
    if prior_check:
        n_curves_air = 100
        n_curves_kinetics = 500
    else:
        n_curves_air = 100
        n_curves_kinetics = 100
    
    print("Creating plot...\n")
    fig = create_main_figure(
        model,
        entry,
        swabs,
        pleth,
        air_samples,
        predictions,
        cage_predictions,
        n_curves_kinetics=n_curves_kinetics,
        n_curves_air=n_curves_air)

    print("Saving figure to {}...\n".format(output_path))
    fig.savefig(output_path, dpi=dpi)


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("USAGE: ./figure-main.py <entry data path> "
              "<mcmc chain path> <swab data path>  "
              "<pleth data path>  <air sample data path> "
              "<output path>")
    else:
        main(*sys.argv[1:6], sys.argv[-1])
