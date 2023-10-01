#!/usr/bin/env python3
import sys
import jax
import pickle
import polars as pl
import numpy as np
import numpyro
from priors import priors
from numpyro.infer.reparam import LocScaleReparam
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS, Predictive

from model import inoculated_model
from config import mcmc_params


def build_air_sampling_model(
        all_hamsters,
        air_samples,
        cages_to_use=range(1, 9)):

    model = {
        "cage_length": 17.78,
        "cage_width": 27.04,
        "cage_height": 22.86,
        "air_exchange_rate": 30 / 60,
        "cages": dict()
    }

    # asign unique ids to individuals
    hamsters = all_hamsters.filter(
        ~pl.col("variant").is_null()
    ).with_columns(
      (pl.col("animal_name").rank("dense").alias("hamster_uid") - 1)
    )

    # filter out null observations
    resp_obs = hamsters.filter(
        (~pl.col("MVb").is_null()))
    rna_obs = hamsters.filter(
        (~pl.col("Ct_subgenomic").is_null()))
    TCID_obs = hamsters.filter(
        (~pl.col("n_positive_wells").is_null()))

    # build observation dictionary
    obs_dict = dict()

    obs_dict["respiration_rates"] = {
        "times": resp_obs["timepoint"].to_numpy() * 24.0,
        "values": np.log(resp_obs["MVb"].to_numpy()) + np.log(60),
        # convert resp rates from mL/min to mL/h
        "individuals": resp_obs["hamster_uid"].to_numpy()}

    obs_dict["oral_rna"] = {
        "times": rna_obs["timepoint"].to_numpy() * 24.0,
        "ct": rna_obs["Ct_subgenomic"].to_numpy(),
        "log10_copies_subgenomic": np.nan_to_num(
            rna_obs["log10_copies_subgenomic"].to_numpy(), nan=-1.0),
        "individuals": rna_obs["hamster_uid"].to_numpy(),
        "is_donor": rna_obs["animal_role"].to_numpy() == "donor"}

    obs_dict["oral_TCID"] = {
        "times": TCID_obs["timepoint"].to_numpy() * 24.0,
        "values": TCID_obs["log10_tcid50"].to_numpy(),
        "n_positive_wells": TCID_obs["n_positive_wells"].to_numpy(),
        "first_row_dilution": TCID_obs["starting_row_dilution"].to_numpy(),
        "individuals": TCID_obs["hamster_uid"].to_numpy()}

    model["obs_dict"] = obs_dict

    hamster_index = hamsters.unique(
      subset="hamster_uid"
    ).sort(
      by="hamster_uid")

    hamster_index = hamster_index.with_columns(
        pl.when(
            pl.col("animal_sex") == "M"
        ).then(
            1
        ).when(
            pl.col("animal_sex") == "F"
        ).then(
            0
        ).otherwise(
            None
        ).alias("sex_id")
    ).with_columns(
        pl.when(
            pl.col("variant") == "Alpha"
        ).then(
            0
        ).when(
            pl.col("variant") == "Delta"
        ).then(
            1
        ).otherwise(
            None
        ).alias("variant_id")
    )

    model["hamster_sexes"] = hamster_index.select(
        "sex_id"
    ).to_numpy().flatten()

    model["hamster_variant_ids"] = hamster_index.select(
        "variant_id"
    ).to_numpy().flatten()

    model["hamster_index"] = hamster_index

    for cage_id in cages_to_use:
        cage_name = "air_sample_" + str(cage_id)
        new_cage = {
            "name": cage_name,
            "id": int(cage_id),
            "length": model["cage_length"],
            "width": model["cage_width"],
            "height": model["cage_height"],
            "air_exchange_rate": model["air_exchange_rate"],
            "obs_dict": dict()
        }

        new_cage["individuals"] = hamsters.filter(
            pl.col("cage") == str(cage_id)
        ).select(
            "hamster_uid"
        ).unique().to_numpy().flatten()
        air_obs = air_samples.filter(
            pl.col("cage_id") == cage_id)
        new_cage["air_samples"] = {
            "times": (air_obs["Timepoint"].to_numpy()) * 24.0,
            "Ct_values": air_obs["Ct_subgenomic"].to_numpy(),
            "log10_copies": np.nan_to_num(
                air_obs["log10_copies_subgenomic"].to_numpy(),
                nan=-1.0),
            "total_plaques": air_obs["total_plaques"].to_numpy()}

        new_cage["variants"] = np.unique(
            model["hamster_variant_ids"][new_cage["individuals"]])
        if not new_cage["variants"].size <= 1:
            raise ValueError("Multiple variant cages not "
                             "currently supported")
        new_cage["variant_id"] = new_cage["variants"][0]

        new_cage["sexes"] = (
            model["hamster_sexes"][new_cage["individuals"]])

        model["cages"][cage_name] = new_cage

        pass

    #  build sentinel data dictionary
    sentinels = all_hamsters.filter(
        pl.col("animal_role") == "sentinel"
    ).group_by(
        "animal_name"
    ).agg(
        [
            (
                (pl.col("pct_Delta") > 5) &
                (pl.col("Ct_subgenomic") < 40)
            ).any(
            ).alias(
                "delta_infection_status"
            ),

            (
                (pl.col("pct_Delta") < 95) &
                (pl.col("Ct_subgenomic") < 40)
            ).any(
            ).alias(
                "alpha_infection_status"
            ),

            pl.col("cage").first().alias("cage")
        ]
    )
    sentinels = sentinels.sort(
        [pl.col("cage"), pl.col("animal_name")]
    ).with_row_count("sentinel_uid")

    sentinels = sentinels.join(
        model["hamster_index"].filter(
            (pl.col("variant") == "Delta") &
            (pl.col("animal_role") == "donor")
        ).select(
            [pl.col("cage"),
             pl.col("animal_name").alias("delta_donor_name"),
             (24.0 * pl.col("time_in")).alias("delta_donor_time_in"),
             (24.0 * pl.col("time_out")).alias("delta_donor_time_out"),
             pl.col("hamster_uid").alias("delta_donor_uid")
             ]),
        on="cage"
    ).join(
        model["hamster_index"].filter(
            (pl.col("variant") == "Alpha") &
            (pl.col("animal_role") == "donor")
        ).select(
            [pl.col("cage"),
             pl.col("animal_name").alias("alpha_donor_name"),
             (24.0 * pl.col("time_in")).alias("alpha_donor_time_in"),
             (24.0 * pl.col("time_out")).alias("alpha_donor_time_out"),
             pl.col("hamster_uid").alias("alpha_donor_uid")
             ]
        ),
        on="cage"
    )

    sentinel_data = dict()

    for attribute in ["donor_uid",
                      "donor_time_in",
                      "donor_time_out",
                      "infection_status"]:
        sentinel_data[attribute] = sentinels.select(
            ["alpha_" + attribute,
             "delta_" + attribute]
        ).to_numpy()

    model["sentinel_data"] = sentinel_data
    model["sentinel_index"] = sentinels

    return model


def infer_from_model(model,
                     prior_params=None,
                     num_warmup=1000,
                     num_samples=2000,
                     num_chains=1,
                     adapt_delta=0.85,
                     max_tree_depth=(9, 14),
                     jit_model_args=False,
                     random_seed=0,
                     forward_mode_differentiation=False,
                     chain_method="parallel",
                     CPU=True,
                     CPU_cores=1,
                     use_sentinels=True):
    variant_names = ["Alpha", "Delta"]
    hamster_variant_ids = model["hamster_variant_ids"]
    cage_data = list(model["cages"].values())
    obs_dict = model["obs_dict"]
    hamster_sexes = model["hamster_sexes"]

    if chain_method == "parallel" and CPU:
        numpyro.set_host_device_count(min(num_chains, CPU_cores))

    if use_sentinels:
        sentinel_data = model["sentinel_data"]
    else:
        sentinel_data = None

    hierarchical_param_stems = [
        "mean_respiration_rate",
        "individual_peak_load",
        "individual_log_growth_rate",
        "individual_log_peak_time",
        "individual_log_decay_rate",
        "single_hit_log_dose"
    ]

    config_dict = {key: LocScaleReparam(0) for
                   key in hierarchical_param_stems}
    print("Conducting non-centered reparametrization "
          "for the following variables:")
    print(config_dict.keys())
    reparam_model = reparam(inoculated_model,
                            config=config_dict)

    nuts_kernel = NUTS(
        reparam_model,
        target_accept_prob=adapt_delta,
        max_tree_depth=max_tree_depth,
        forward_mode_differentiation=forward_mode_differentiation)
    mcmc_runner = MCMC(nuts_kernel,
                       num_warmup=num_warmup,
                       num_samples=num_samples,
                       num_chains=num_chains,
                       chain_method=chain_method,
                       progress_bar=True,
                       jit_model_args=jit_model_args)
    rng_key = jax.random.PRNGKey(random_seed)

    print("Starting MCMC run...")
    mcmc_runner.run(rng_key,
                    global_hypers=prior_params,
                    variant_names=variant_names,
                    hamster_sexes=hamster_sexes,
                    hamster_variant_ids=hamster_variant_ids,
                    cage_data=cage_data,
                    obs_dict=obs_dict,
                    sentinel_data=sentinel_data)
    return mcmc_runner


def prior_check_from_model(model,
                           num_samples=1000,
                           rng_seed=0,
                           prior_params=None,
                           use_sentinels=True):
    variant_names = ["Alpha", "Delta"]
    hamster_variant_ids = model["hamster_variant_ids"]
    cage_data = list(model["cages"].values())
    obs_dict = model["obs_dict"]
    hamster_sexes = model["hamster_sexes"]
    if use_sentinels:
        sentinel_data = model["sentinel_data"]
    else:
        sentinel_data = None

    pp = Predictive(inoculated_model,
                    num_samples=num_samples)
    print(rng_seed)
    rng_key = jax.random.PRNGKey(seed=rng_seed)
    result = pp(rng_key,
                global_hypers=prior_params,
                variant_names=variant_names,
                hamster_sexes=hamster_sexes,
                hamster_variant_ids=hamster_variant_ids,
                cage_data=cage_data,
                obs_dict=obs_dict,
                sentinel_data=sentinel_data)
    return result


def main(swab_path,
         pleth_path,
         dual_donor_path,
         air_sample_path,
         outpath):

    print("\nReading in data...")
    sep = "\t"
    swabs, pleth, dd, air_samples = [
        pl.read_csv(file, separator=sep)
        for file in [swab_path,
                     pleth_path,
                     dual_donor_path,
                     air_sample_path]]

    columns_to_use = [
        "animal_name",
        "animal_role",
        "animal_sex",
        "variant",
        "timepoint",
        "log10_tcid50",
        "n_positive_wells",
        "starting_row_dilution",
        "log10_copies_subgenomic",
        "Ct_subgenomic",
        "pct_Delta",
        "MVb",
        "cage",
        "time_in",
        "time_out"]

    hamsters = swabs.join(
        pleth,
        right_on=["hamster_id", "DPI"],
        left_on=["hamster_id", "Timepoint"],
        how="left"
    ).rename(
        {"sex": "animal_sex"}
    ).with_columns(
        pl.col("cage_id").cast(pl.Utf8).alias("cage")
    ).with_columns(
        pl.lit("inoculated").alias("animal_role")
    ).with_columns(
        pl.lit("inoculated").alias("exposure_sequence")
    ).with_columns(
        pl.lit(None).cast(pl.Float64).alias("time_in")
    ).with_columns(
        pl.lit(None).cast(pl.Float64).alias("time_out")
    ).with_columns(
        pl.lit(None).cast(pl.Float64).alias("pct_Delta")
    ).with_columns(
        pl.col("Timepoint").cast(pl.Float64).alias("timepoint")
    )

    dd_dat = dd.with_columns(
        pl.lit(None).cast(pl.Float64).alias("MVb")
    )

    all_hamsters = pl.concat(
        [dd_dat.select(columns_to_use),
         hamsters.select(columns_to_use)]
    )

    print("Configuring model...")
    model = build_air_sampling_model(
        all_hamsters,
        air_samples,
        range(1, 9))

    if outpath.endswith("main.pickle"):
        model_name = "main"
    elif outpath.endswith("simple.pickle"):
        model_name = "simple"
    else:
        raise ValueError("Unknown model for outpath"
                         "{}".format(outpath))
    use_sentinels = priors.get(
        model_name + "_use_sentinels",
        None)

    mcmc = "prior-check" not in outpath
    prior_check = "prior-check" in outpath

    if mcmc:
        print("Setting up and running mcmc...")
        output = infer_from_model(model,
                                  prior_params=priors,
                                  num_warmup=mcmc_params.get(
                                      "n_warmup", None),
                                  num_samples=mcmc_params.get(
                                      "n_samples", None),
                                  num_chains=mcmc_params.get(
                                      "n_chains", None),
                                  adapt_delta=mcmc_params.get(
                                      "adapt_delta", None),
                                  max_tree_depth=mcmc_params.get(
                                      "max_tree", None),
                                  random_seed=mcmc_params.get(
                                      model_name + "_rng_seed",
                                      None),
                                  chain_method=mcmc_params.get(
                                      "chain_method", "parallel"),
                                  forward_mode_differentiation=mcmc_params.get(
                                      "forward_mode_differentiation", False),
                                  CPU=mcmc_params.get("CPU", True),
                                  CPU_cores=mcmc_params.get("CPU_cores", 1),
                                  use_sentinels=use_sentinels)
        output.print_summary(exclude_deterministic=False)
    elif prior_check:
        print("Sampling prior checks...")
        output = prior_check_from_model(
            model,
            num_samples=mcmc_params.get(
                "n_prior_check_samples", None),
            rng_seed=mcmc_params.get(
                model_name + "_prior_check_rng_seed",
                None),
            prior_params=priors,
            use_sentinels=use_sentinels)

    print("Saving output to {}...\n".format(outpath))
    with open(outpath, "wb") as outfile:
        pickle.dump((output, model), outfile)

    return True


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("\nUSAGE: ./run_model.py <swab data path> "
              "<pleth data path> <dual donor data path> "
              "<air sample data path> <output path>\n\n")
    else:
        main(*sys.argv[1:5], sys.argv[-1])
