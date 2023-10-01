#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
import sys

from analysis import load_chains
import plotting as plot
from plotting import experiment_hues


def plot_infection_prob(df_cages):
    fig, ax = plt.subplots(figsize=[8, 4])
    sns.violinplot(
        x="cage",
        y="single_hit_prob",
        hue="variant",
        data=df_cages.to_pandas(),
        ax=ax,
        orient="v",
        inner=None,
        scale="width",
        split=True,
        palette=plot.variant_colors)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylim(bottom=0, top=1e0)
    ax.legend(fancybox=True,
              frameon=True,
              framealpha=True,
              title="Variant",
              loc=[1, 0.8])
    ax.set_xlabel("Cage")
    ax.set_ylabel("Infection probability")
    fig.tight_layout()

    return (fig, ax)


def plot_coinfection_prob(co_probs):
    fig, ax = plt.subplots(figsize=[15, 5])
    sns.violinplot(
        x="cage",
        y="coinfection_prob",
        hue="experiment_type",
        dodge=False,
        legend=False,
        palette=experiment_hues,
        data=co_probs.to_pandas(),
        ax=ax
    )
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.set_xlabel("Cage")
    ax.set_ylabel("Coinfection probability")
    ax.legend(fancybox=True,
              frameon=True,
              framealpha=True,
              title="Experiment type",
              title_fontsize=15,
              fontsize=15,
              loc=[0.85, 0.8])
    fig.tight_layout()

    return (fig, ax)


def plot_expected_coinfections(co_probs):
    total_expected = co_probs.group_by(
        ["experiment_type", "draw"]
    ).agg(
        pl.col(
            "expected_coinfections"
        ).sum(
        ).alias(
            "total_expected_coinfections"
        )
    )

    fig, ax = plt.subplots(figsize=[8, 5])
    sns.violinplot(
        x="experiment_type",
        y="total_expected_coinfections",
        hue="experiment_type",
        data=total_expected.to_pandas(),
        palette=experiment_hues,
        dodge=False,
        legend=None,
        ax=ax)
    ax.set_ylim(bottom=0, top=8)
    ax.set_ylabel("Expected coinfections\n(all cages)")
    ax.set_xlabel("Experiment type")
    ax.get_legend().remove()
    fig.tight_layout()

    return (fig, ax)


def main(mcmc_output_path,
         dual_donor_path,
         outpath):
    ps, model, obj = load_chains(mcmc_output_path)
    df_alpha = pl.DataFrame(
        np.array(ps["sentinel_single_hit_rate"][..., 0]))
    df_delta = pl.DataFrame(
        np.array(ps["sentinel_single_hit_rate"][..., 1]))
    df_alpha = df_alpha.with_row_count(
        "draw"
    ).melt(id_vars="draw",
           variable_name="sentinel_uid"
           ).with_columns(
               pl.col("sentinel_uid").str.extract(
                   "([0-9]+)"
               ).cast(
                   pl.Int64
               ).alias("sentinel_uid")
           ).with_columns(
               pl.lit("Alpha").alias("variant")
           )

    df_delta = df_delta.with_row_count(
        "draw"
    ).melt(
        id_vars="draw",
        variable_name="sentinel_uid"
    ).with_columns(
        pl.col("sentinel_uid").str.extract(
            "([0-9]+)"
        ).cast(
            pl.Int64
        ).alias("sentinel_uid")
    ).with_columns(
        pl.lit("Delta").alias("variant")
    )

    df = pl.concat(
        [df_alpha,
         df_delta]
    ).with_columns(
        (1 - np.exp(-pl.col("value"))
         ).alias("single_hit_prob")
    )

    cage_reps = model["sentinel_index"].group_by("cage").agg(
        [
            pl.first(
                "sentinel_uid"
            ).cast(
                pl.Int64
            ).alias(
                "sentinel_uid"
            ),

            pl.sum(
                "alpha_infection_status"
            ).alias(
                "n_alpha"
            ),

            pl.sum(
                "delta_infection_status"
            ).alias(
                "n_delta"
            ),

            (
                pl.col("delta_infection_status") &
                pl.col("alpha_infection_status")
            ).sum(
            ).alias(
                "n_both"
            ),

            pl.count(
                "alpha_infection_status"
            ).alias("n_total")
        ]
    )

    df_cages = df.join(
        cage_reps,
        on="sentinel_uid"
    ).with_columns(
        (
            1 - pl.col("single_hit_prob")
        ).pow(
            pl.col("n_total")
        ).alias("p_none")
    ).with_columns(
        (pl.col("single_hit_prob") *
         pl.col("n_total")
         ).alias("expected_infections")
    )

    co_probs = df_cages.group_by(
        ["draw",
         "cage",
         "n_total"]
    ).agg(
        pl.col(
            "single_hit_prob"
        ).log().sum().exp(
        ).alias(
            "coinfection_prob"
        )
    ).with_columns(
        (
            pl.col("coinfection_prob") *
            pl.col("n_total")
        ).alias("expected_coinfections")
    ).with_columns(
        (
            1 - pl.col(
                "coinfection_prob")
        ).pow(
            pl.col("n_total")
        ).alias(
            "p_no_coinfections"
        )
    ).sort(
        "cage"
    ).with_columns(
        pl.when(
            pl.col("cage").is_in(
                ["A", "B", "C", "D"])
        ).then(
            pl.lit("sequential")
        ).otherwise(
            pl.lit("simultaneous")
        ).alias("experiment_type")
    )

    if "infection-probability-by-variant" in outpath:
        fig, ax = plot_infection_prob(df_cages)
    elif "coinfection-probability" in outpath:
        fig, ax = plot_coinfection_prob(co_probs)
    elif "expected-coinfections" in outpath:
        fig, ax = plot_expected_coinfections(co_probs)
    else:
        raise ValueError("unknown figure to generate"
                         "based on output path {}"
                         "".format(outpath))

    fig.savefig(outpath)

    return True


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("USAGE: ./coinfection-figures.py "
              "<mcmc chain path> "
              "<dual donor data path> "
              "<output path>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[-1])
