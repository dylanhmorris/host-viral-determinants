#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import plotting as plot


def main(air_data_path="../dat/cleaned/air-samples.tsv",
         outpath="../ms/figures/figure-plaques-per-rna.pdf"):

    lod = 10.**-0.8707614
    
    dat = pl.read_csv(
        air_data_path,
        sep="\t")

    dat = dat.with_column(
        pl.col(
            "copies_subgenomic"
        ).fill_null(
            pl.lit(lod)
        ).alias("copies_subgenomic")
    ).with_column(
        (pl.col("copies_subgenomic") /
         pl.col("total_plaques")
         ).alias("copies_per_plaque")
    )

    print(
        dat.filter(
            pl.col("total_plaques") > 0
        ).select(
            [
                "total_plaques",
                "copies_subgenomic",
                "copies_per_plaque"
            ]))
    
    fig, ax = plt.subplots(1, 2,
                           figsize=[10, 5])

    for variant in ["Alpha", "Delta"]:
        var_dat = dat.filter(
            pl.col("variant") == variant)
        xs = var_dat["copies_subgenomic"].to_numpy(),
        ys = var_dat["total_plaques"].to_numpy(),
        above_lod = (
            var_dat["copies_subgenomic"] > (lod)
        ).to_numpy().flatten().astype("bool")

        
        ax[0].plot(
            np.where(above_lod, xs, None).flatten(),
            np.where(above_lod, ys, None).flatten(),
            label = variant,
            color = plot.variant_palette.get(variant, None),
            marker = "o",
            lw=0,
            markeredgecolor="k",
            markersize=20,
            alpha=0.75)
        
        ax[0].plot(
            np.where(~above_lod, lod, None).flatten(),
            np.where(~above_lod, ys, None).flatten(),
            label = variant,
            color = plot.variant_palette.get(variant, None),
            marker = "<",
            lw=0,
            markeredgecolor="k",
            markersize=20,
            alpha=0.75)

        ax[0].set_xscale("log")
        ax[0].set_xticks(10.**np.arange(-1, 7, 1))
        ax[0].set_yticks(np.arange(0, 46, 5))
        ax[0].set_xlim([0.5e-1, 2e4])
        ax[0].set_ylim([-2.5, 27.5])
        ax[0].set_xlabel("sgRNA copies")
        ax[0].set_ylabel("Plaques")

        xs = var_dat["Timepoint"].to_numpy(),
        ys = var_dat["copies_per_plaque"].to_numpy(),
        
        ax[1].plot(
            np.where(above_lod, xs, None).flatten(),
            np.where(above_lod, ys, None).flatten(),
            label = variant,
            color = plot.variant_palette.get(variant, None),
            marker = "o",
            lw=0,
            markeredgecolor="k",
            markersize=20,
            alpha=0.75)
        ax[1].plot(
            np.where(~above_lod, xs, None).flatten(),
            np.where(~above_lod, lod, None).flatten(),
            label = variant,
            color = plot.variant_palette.get(variant, None),
            marker = "v",
            lw=0,
            markeredgecolor="k",
            markersize=20,
            alpha=0.75)

        ax[1].set_yscale("log")
        ax[1].set_xticks(np.arange(0, 6, 1))
        ax[1].set_yticks(10.**np.arange(-1, 5, 1))
        ax[1].set_xlim([-0.5, 4.5])
        ax[1].set_ylim([0.5e-1, 2e4])
        ax[1].set_xlabel("Time since inoculation")
        ax[1].set_ylabel("sgRNA copies / plaque")
        pass # end loop over variants
    
    handles, labels = ax[0].get_legend_handles_labels()

    leg = ax[0].legend(
        handles=[handles[0], handles[2]],
        labels=[labels[0], labels[2]],
        title="Variant",
        fancybox=True,
        frameon=True,
        framealpha=1,
        markerscale=0.5,
        loc="upper left")

    fig.tight_layout()
    fig.savefig(outpath)
    


if __name__ == "__main__":
    main(sys.argv[1],
         sys.argv[-1])
