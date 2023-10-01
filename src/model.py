import jax.numpy as jnp
import numpy as np
import numpyro as npro
import numpyro.distributions as dist

from predict import (predict_swab_virions,
                     predict_log_air_sample,
                     predict_viable_air_sample)

from distributions import CensoredNormal, PoissonSingleHit, TiterWells
from jax.scipy.special import logsumexp


def sample_base_parameters(global_hypers=None):
    """
    Sample global parameters shared across
    the entire model that are not individual
    hamster- or individual virus variant-specific
    """

    parm = dict()
    # sample global params

    parm["log_resp_rate_mean"] = npro.sample(
        "log_resp_rate_mean",
        dist.Normal(
            global_hypers["log_resp_rate_mean_prior_mean"],
            global_hypers["log_resp_rate_mean_prior_sd"]))

    parm["log_resp_rate_sd"] = npro.sample(
        "log_resp_rate_sd",
        dist.TruncatedNormal(
            global_hypers["log_resp_rate_sd_prior_mode"],
            global_hypers["log_resp_rate_sd_prior_sd"],
            low=0.0))

    parm["male_respiration_rate_offset"] = npro.sample(
        "male_respiration_rate_offset",
        dist.Normal(
            global_hypers["male_respiration_rate_offset_prior_mean"],
            global_hypers["male_respiration_rate_offset_prior_sd"]))

    parm["male_peak_offset"] = npro.sample(
            "male_peak_offset",
            dist.Normal(
                loc=global_hypers["male_peak_offset_prior_mean"],
                scale=global_hypers["male_peak_offset_prior_sd"]))

    parm["male_growth_rate_offset"] = npro.sample(
            "male_growth_rate_offset",
            dist.Normal(
                loc=global_hypers["male_growth_rate_offset_prior_mean"],
                scale=global_hypers["male_growth_rate_offset_prior_sd"]))

    parm["male_decay_rate_offset"] = npro.sample(
        "male_decay_rate_offset",
        dist.Normal(
            loc=global_hypers["male_decay_rate_offset_prior_mean"],
            scale=global_hypers["male_decay_rate_offset_prior_sd"]))

    parm["sd_obs_resp_rate"] = npro.sample(
        "sd_obs_resp_rate",
        dist.TruncatedNormal(
            global_hypers["sd_obs_resp_rate_prior_mode"],
            global_hypers["sd_obs_resp_rate_prior_sd"],
            low=0.0))

    parm["donor_copy_number_offset"] = npro.sample(
        "donor_copy_number_offset",
        dist.Normal(
            loc=global_hypers[
                "donor_copy_number_offset_prior_mean"],
            scale=global_hypers[
                "donor_copy_number_offset_prior_sd"]))

    return parm


def sample_variants(
        variant_priors,
        variant_names=[0]):
    """
    Sample variant-specific
    virological parameters characterizing
    within and between host dynamics

    :param variant_priors A dictionary of prior
    distribution parameters
    :param variant_names: a list of what to call
    the variants internally
    """

    sampled = dict()
    n_variants = len(variant_names)

    # loop over the n_variants variants
    with npro.plate("variants", n_variants):
        #################################################
        # hierarchical virological parameter
        # means/modes
        #################################################
        sampled["mean_log_peak"] = npro.sample(
            "mean_log_peak",
            dist.Normal(
                loc=variant_priors["log_peak_prior_mean"],
                scale=variant_priors["log_peak_prior_sd"]))

        sampled["variant_decay_phase_log_halflife"] = npro.sample(
            "variant_decay_phase_log_halflife",
            dist.Normal(
                variant_priors["log_decay_half_life_prior_mean"],
                variant_priors["log_decay_half_life_prior_sd"]))

        sampled["variant_log_decay_rate"] = npro.deterministic(
            "variant_log_decay_rate",
            jnp.log(jnp.log(2)) -
            sampled["variant_decay_phase_log_halflife"])

        sampled["mean_log_peak_time"] = npro.sample(
            "mean_log_peak_time",
            dist.Normal(
                loc=variant_priors["mean_log_peak_time_prior_mean"],
                scale=variant_priors["mean_log_peak_time_prior_sd"]))

        sampled["variant_log_doubling_time"] = npro.sample(
            "variant_log_doubling_time",
            dist.Normal(
                variant_priors[
                    "variant_log_doubling_time_prior_mean"],
                variant_priors[
                    "variant_log_doubling_time_prior_sd"]))

        sampled["variant_log_growth_rate"] = npro.deterministic(
            "variant_log_growth_rate",
            jnp.log(jnp.log(2)) -
            sampled["variant_log_doubling_time"])

        sampled["variant_growth_rate"] = npro.deterministic(
            "variant_growth_rate",
            jnp.exp(sampled["variant_log_growth_rate"]))

        #################################################
        # virological parameter ratios, relating
        # derived parameters to fundamental ones
        #################################################
        sampled["log_swab_decay_rate_ratio"] = npro.sample(
            "log_swab_decay_rate_ratio",
            dist.Normal(
                variant_priors["log_swab_decay_rate_ratio_prior_mean"],
                variant_priors["log_swab_decay_rate_ratio_prior_sd"]))

        sampled["log_rna_decay_rate_ratio"] = npro.sample(
            "log_rna_decay_rate_ratio",
            dist.Normal(
                variant_priors["log_rna_decay_rate_ratio_prior_mean"],
                variant_priors["log_rna_decay_rate_ratio_prior_sd"]))

        sampled["log_swab_peak_offset"] = npro.sample(
            "log_swab_peak_offset",
            dist.Normal(
                variant_priors["log_swab_peak_offset_prior_mean"],
                variant_priors["log_swab_peak_offset_prior_sd"]))

        ################################################
        # estimated hierarchical parameter SDs
        # (i.e. degree of individual variation
        # about the population mean/mode:
        # these are variant-specific in case
        # one variant has more variable within-host
        # kinetics than another
        ################################################
        sampled["sd_log_peak"] = npro.sample(
            "sd_log_peak",
            dist.TruncatedNormal(
                variant_priors["sd_log_peak_prior_mode"],
                variant_priors["sd_log_peak_prior_sd"],
                low=0.0))

        sampled["sd_log_peak_time"] = npro.sample(
            "sd_log_peak_time",
            dist.TruncatedNormal(
                loc=variant_priors["sd_log_peak_time_prior_mode"],
                scale=variant_priors["sd_log_peak_time_prior_sd"],
                low=0.0))

        sampled["sd_individual_log_decay_rate"] = npro.sample(
            "sd_individual_log_decay_rate",
            dist.TruncatedNormal(
                variant_priors["sd_individual_decay_rate_prior_mode"],
                variant_priors["sd_individual_decay_rate_prior_sd"],
                low=0.0))

        sampled["sd_individual_log_growth_rate"] = npro.sample(
            "sd_individual_log_growth_rate",
            dist.TruncatedNormal(
                variant_priors["sd_individual_growth_rate_prior_mode"],
                variant_priors["sd_individual_growth_rate_prior_sd"],
                low=0.0))

        ############################################
        # observation process scaling factors
        ############################################
        sampled["log_air_RNA_per_air_virions"] = npro.sample(
            "log_air_RNA_per_air_virions",
            dist.Normal(variant_priors["log_air_RNA_prior_mean"],
                        variant_priors["log_air_RNA_prior_sd"]))

        sampled["oral_RNA_per_shed_virions"] = npro.sample(
            "oral_RNA_per_shed_virions",
            dist.Normal(variant_priors["log_oral_RNA_prior_mean"],
                        variant_priors["log_oral_RNA_prior_sd"]))

        sampled["oral_TCID_per_shed_virions"] = npro.sample(
            "oral_TCID_per_shed_virions",
            # for titration experiments
            dist.Normal(variant_priors["log_oral_TCID_prior_mean"],
                        variant_priors["log_oral_TCID_prior_sd"]))

        sampled["log_variant_infectivity"] = npro.sample(
            "log_variant_infectivity",
            # for sentinel challenge
            dist.Normal(
                variant_priors["log_variant_infectivity_prior_mean"],
                variant_priors["log_variant_infectivity_prior_sd"]))

        # loss of infectious virus in the environment
        sampled["log_environmental_half_life"] = npro.sample(
            "log_environmental_half_life",
            dist.Normal(
                variant_priors["log_half_life_prior_mean"],
                variant_priors["log_half_life_prior_sd"]))
        sampled["environmental_half_life"] = npro.deterministic(
            "environmental_half_life",
            jnp.exp(sampled["log_environmental_half_life"]))
        sampled["environmental_decay_rate"] = npro.deterministic(
            "environmental_decay_rate",
            jnp.log(2) / sampled["environmental_half_life"])

        ##########################################
        # Normal observation error SDs
        ##########################################

        sampled["sd_obs_oral_rna"] = npro.sample(
            "sd_obs_oral_rna",
            dist.TruncatedNormal(
                loc=variant_priors["sd_obs_oral_rna_prior_mode"],
                scale=variant_priors["sd_obs_oral_rna_prior_sd"],
                low=0.0))

        sampled["sd_obs_air_rna"] = npro.sample(
            "sd_obs_air_rna",
            dist.TruncatedNormal(
                loc=variant_priors["sd_obs_air_rna_prior_mode"],
                scale=variant_priors["sd_obs_air_rna_prior_sd"],
                low=0.0))

        sampled["names"] = variant_names
        pass  # end plate

    return sampled


def expand_variant_params(
        hamster_variant_ids,
        sampled_variants):
    """
    Take variant parameter samples
    (n_variants length arrays) and
    expand to an n_individuals length
    array, such that each individual
    has a parameter value equal to
    its infecting variant. This is a
    convenience function to make indexing
    and then sampling individual-level
    virological parameters easier
    """
    parms = [
        "mean_log_peak",
        "sd_log_peak",
        "mean_log_peak_time",
        "sd_log_peak_time",
        "variant_growth_rate",
        "variant_log_growth_rate",
        "log_swab_peak_offset",
        "variant_log_decay_rate",
        "sd_individual_log_decay_rate",
        "sd_individual_log_growth_rate",
        "log_swab_decay_rate_ratio",
        "log_rna_decay_rate_ratio",
        "environmental_decay_rate",
        "log_air_RNA_per_air_virions",
        "sd_obs_oral_rna",
        "sd_obs_air_rna",
        "oral_RNA_per_shed_virions",
        "oral_TCID_per_shed_virions"]

    variant_dict = dict()

    for parm in parms:
        variant_dict[parm] = sampled_variants[parm][hamster_variant_ids]

    return variant_dict


def sample_animals(
        hamster_sex=None,
        variant_params=None,
        obs_dict=None,
        global_parameter_dict=None):
    """
    Sample individual-level parameters
    and observe each individual animal
    studied
    """
    n_hamsters = hamster_sex.size
    parms = global_parameter_dict
    sampled = dict()

    with npro.plate("hamsters", n_hamsters):
        sampled["mean_respiration_rate"] = npro.sample(
            "mean_respiration_rate",
            dist.Normal(
                loc=parms["log_resp_rate_mean"] +
                parms["male_respiration_rate_offset"] * hamster_sex,
                scale=parms["log_resp_rate_sd"]
            ))

        sampled["peak_load"] = npro.sample(
            "individual_peak_load",
            dist.Normal(loc=variant_params["mean_log_peak"] +
                        parms["male_peak_offset"] * hamster_sex,
                        scale=variant_params["sd_log_peak"]))

        sampled["individual_log_growth_rate"] = npro.sample(
            "individual_log_growth_rate",
            dist.Normal(
                loc=variant_params["variant_log_growth_rate"] +
                parms["male_growth_rate_offset"] * hamster_sex,
                scale=variant_params["sd_individual_log_growth_rate"]))

        sampled["individual_growth_rate"] = npro.deterministic(
            "individual_growth_rate",
            jnp.exp(sampled["individual_log_growth_rate"]))

        sampled["individual_log_peak_time"] = npro.sample(
            "individual_log_peak_time",
            dist.Normal(
                loc=variant_params["mean_log_peak_time"],
                scale=variant_params["sd_log_peak_time"]))

        sampled["individual_peak_time"] = npro.deterministic(
            "individual_peak_time",
            jnp.exp(sampled["individual_log_peak_time"]))

        sampled["initial_load"] = npro.deterministic(
            "individual_initial_load",
            sampled["peak_load"] - (
                sampled["individual_peak_time"] *
                sampled["individual_growth_rate"]))

        sampled["swab_peak"] = npro.deterministic(
            "swab_peak",
            jnp.exp(
                sampled["individual_log_peak_time"] +
                variant_params["log_swab_peak_offset"]))

        sampled["log_decay_rate"] = npro.sample(
            "individual_log_decay_rate",
            dist.Normal(
                loc=variant_params["variant_log_decay_rate"] +
                parms["male_decay_rate_offset"] * hamster_sex,
                scale=variant_params["sd_individual_log_decay_rate"]))

        sampled["individual_decay_rate"] = npro.deterministic(
            "individual_decay_rate",
            jnp.exp(sampled["log_decay_rate"]))

        # end plate

    # observation process for hamsters
    indivs = obs_dict["respiration_rates"]["individuals"]
    mean_resp_rates = (
                sampled["mean_respiration_rate"][indivs])
    npro.sample(
        "observed_respiration_rates",
        dist.Normal(
            loc=mean_resp_rates,
            scale=parms["sd_obs_resp_rate"]),
        obs=obs_dict["respiration_rates"]["values"])

    indivs = obs_dict["oral_rna"]["individuals"]
    sampled["log10_predicted_swab_RNA"] = (
        (variant_params["oral_RNA_per_shed_virions"][indivs] +
         (parms["donor_copy_number_offset"] *
          obs_dict["oral_rna"]["is_donor"]) +
         predict_swab_virions(
             obs_dict["oral_rna"]["times"],
             sampled["individual_peak_time"][indivs],
             sampled["swab_peak"][indivs],
             sampled["initial_load"][indivs],
             sampled["individual_growth_rate"][indivs],
             sampled["individual_decay_rate"][indivs],
             variant_params["log_rna_decay_rate_ratio"][indivs] +
             variant_params["log_swab_decay_rate_ratio"][indivs])
         ) / jnp.log(10))  # convert from log to log10

    npro.sample("observed_oral_rna",
                dist.Normal(
                    loc=sampled["log10_predicted_swab_RNA"],
                    scale=variant_params["sd_obs_oral_rna"][indivs]),
                obs=obs_dict["oral_rna"]["log10_copies_subgenomic"])

    indivs = obs_dict["oral_TCID"]["individuals"]
    sampled["predicted_oral_log10_tcid50"] = npro.deterministic(
        "predicted_oral_log10_tcid50",
        (variant_params["oral_TCID_per_shed_virions"][indivs] +
         predict_swab_virions(
             obs_dict["oral_TCID"]["times"],
             sampled["individual_peak_time"][indivs],
             sampled["swab_peak"][indivs],
             sampled["initial_load"][indivs],
             sampled["individual_growth_rate"][indivs],
             sampled["individual_decay_rate"][indivs],
             variant_params["log_swab_decay_rate_ratio"][indivs])))

    npro.sample("observed_oral_log10_tcid50",
                TiterWells(
                    log_titer_id50=sampled[
                        "predicted_oral_log10_tcid50"] - 1.0,
                    first_row_dilution=obs_dict[
                        "oral_TCID"]["first_row_dilution"],
                    wells_per_row=4,
                    log_dilution_factor=1,
                    log_base=10),
                obs=obs_dict["oral_TCID"]["n_positive_wells"])

    return sampled


def sample_single_cage(cage_id,
                       cage_data,
                       sampled_animals,
                       variant_params,
                       sampled_parameters,
                       return_dict):
    """
    function to sample all
    parameters for a single
    cage of hamsters
    """
    cage = cage_data[cage_id]
    indivs = cage["individuals"]
    cage_name = str(cage_id)
    i_var = cage["variant_id"]

    return_dict[cage_name] = dict()
    air_samples = cage.get("air_samples", {})
    air_rna_obs = air_samples.get("log10_copies", None)
    air_plaque_obs = air_samples.get("total_plaques", None)
    air_times = air_samples.get("times", jnp.zeros(1))
    return_dict[cage_name]["predicted_log10_air_rna"] = npro.deterministic(
        "predicted_air_log10_copies_" + cage_name,
        (logsumexp(
            variant_params["log_air_RNA_per_air_virions"][indivs] +
            predict_log_air_sample(
                air_times[..., np.newaxis],
                sampled_animals["individual_peak_time"][indivs],
                sampled_animals["initial_load"][indivs],
                sampled_animals["mean_respiration_rate"][indivs],
                sampled_animals["individual_growth_rate"][indivs],
                jnp.exp(
                    variant_params[
                        "log_rna_decay_rate_ratio"][indivs] +
                    sampled_animals["log_decay_rate"][indivs])),
            axis=1)
         ) / jnp.log(10)  # convert to log10 copies
    )

    return_dict[cage_name]["predicted_air_log_PFU"] = npro.deterministic(
      "predicted_air_log_PFU_" + cage_name,
      logsumexp(
          predict_viable_air_sample(
              air_times[..., np.newaxis],
              sampled_animals["individual_peak_time"][indivs],
              sampled_animals["initial_load"][indivs],
              sampled_animals["mean_respiration_rate"][indivs],
              sampled_animals["individual_growth_rate"][indivs],
              sampled_animals["individual_decay_rate"][indivs],
              variant_params["environmental_decay_rate"][indivs]),
          axis=1))

    return_dict[cage_name]["predicted_PFU"] = (
        jnp.exp(return_dict[cage_name]["predicted_air_log_PFU"]))

    npro.sample(
        "observations_" + cage_name,
        CensoredNormal(loc=return_dict[cage_name][
            "predicted_log10_air_rna"],
                       scale=variant_params["sd_obs_air_rna"][i_var],
                       lower_limit=-0.8707614,
                       upper_limit=11.92514),
        obs=air_rna_obs)
    try:
        npro.sample(
            "observed_air_plaques_" + cage_name,
            dist.Poisson(
                rate=return_dict[cage_name]["predicted_PFU"] + 1e-20),
            obs=air_plaque_obs)
        # numpyro poisson needs a strictly positive
        # rate, not just non-negative
        # for inference
    except Exception as e:
        print(return_dict[cage_name]["predicted_PFU"])
        print(e)
        raise
    return return_dict


def sample_cages(
        cage_data=None,
        sampled_animals=None,
        variant_params=None,
        sampled_parameters=None):

    n_cages = len(cage_data)
    return_dict = dict()
    for cage_id in range(n_cages):
        return_dict = sample_single_cage(
            cage_id,
            cage_data,
            sampled_animals,
            variant_params,
            sampled_parameters,
            return_dict)

    return return_dict


def sample_sentinels(sampled_animals,
                     sentinel_data,
                     log_variant_infectivities):

    result = dict()
    donor_ids = sentinel_data["donor_uid"]

    result["sentinel_log_doses"] = predict_log_air_sample(
        sentinel_data["donor_time_in"],
        sampled_animals["individual_peak_time"][donor_ids],
        sampled_animals["initial_load"][donor_ids],
        sampled_animals["mean_respiration_rate"][donor_ids],
        sampled_animals["individual_growth_rate"][donor_ids],
        sampled_animals["individual_decay_rate"][donor_ids],
        sample_duration_hours=(
            sentinel_data["donor_time_out"] -
            sentinel_data["donor_time_in"])
    )

    result["sentinel_single_hit_rate"] = npro.deterministic(
        "sentinel_single_hit_rate",
        jnp.exp(result["sentinel_log_doses"] +
                log_variant_infectivities) +
        1e-20
        # need tiny possibility in all cases for autodiff
    )

    npro.sample(
        "observed_infections",
        PoissonSingleHit(
            rate=result["sentinel_single_hit_rate"]),
        obs=sentinel_data["infection_status"]
    )

    return result


def inoculated_model(global_hypers=None,
                     variant_names=None,
                     hamster_variant_ids=None,
                     hamster_sexes=None,
                     cage_data=None,
                     obs_dict=None,
                     sentinel_data=None):

    params = sample_base_parameters(
        global_hypers=global_hypers)

    variants = sample_variants(
        global_hypers,
        variant_names=variant_names)

    variant_params = expand_variant_params(
        hamster_variant_ids,
        variants)

    animals = sample_animals(
        hamster_sex=hamster_sexes,
        variant_params=variant_params,
        obs_dict=obs_dict,
        global_parameter_dict=params)

    cages = sample_cages(
        cage_data=cage_data,
        sampled_animals=animals,
        variant_params=variant_params,
        sampled_parameters=params)

    sentinels = dict()

    if sentinel_data is not None:
        sentinels = sample_sentinels(
            animals,
            sentinel_data,
            variants["log_variant_infectivity"])

    return dict(**variants,
                **animals,
                **cages,
                **sentinels)
