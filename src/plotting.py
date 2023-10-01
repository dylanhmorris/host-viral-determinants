#!/usr/bin/env python3

"""
style and helper functions
for plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import polars as pl

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [6, 3]
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['axes.formatter.limits'] = ((-3, 3))
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['ytick.labelsize'] = 30
mpl.rcParams['xtick.labelsize'] = 30


alpha_color = "#6E6E6E"
delta_color = "#1A9863"
no_spike_color = "#6195c2"
shed_lw = 2
shed_seethru = 0.05
ct_lw = 0.5
ct_linealpha = 0.4
jitter_width = 0.1
rows = 3
LOD_tcid = 0.5
msize = 3.25
variant_colors = [alpha_color, delta_color]
variant_palette = {
    "Alpha": alpha_color,
    "Delta": delta_color,
    "No spike": no_spike_color}
experiment_hues = {"sequential": "#64a6e8",
                   "simultaneous": "#c47ddb"}
sample_n_curves = 100

def plot_data_with_lod(
        xs,
        ys,
        lld=-np.inf,
        uld=np.inf,
        standard_marker="o",
        lld_marker="v",
        uld_marker="^",
        data_lw=None,
        ax=None,
        **kwargs):

    if ax is None:
        fig, ax = plt.subplots()

    sup_uld = ys >= uld
    sub_lld = ys <= lld
    inbounds = ~(sup_uld | sub_lld)

    _ = ax.plot(
        xs,
        ys,
        lw=data_lw,
        **kwargs)

    _ = ax.plot(
        xs[inbounds],
        ys[inbounds],
        lw=0,
        marker=standard_marker,
        **kwargs)

    _ = ax.plot(
        xs[sub_lld],
        ys[sub_lld],
        lw=0,
        marker=lld_marker,
        **kwargs)

    _ = ax.plot(
        xs[sup_uld],
        ys[sup_uld],
        lw=0,
        marker=uld_marker,
        **kwargs)

    return ax


def add_pvalue(text,
               ax,
               height,
               center_point,
               width,
               bracket_length,
               text_offset=None,
               **kwargs):
    x1, x2 = center_point - width, center_point + width
    ymax = height + bracket_length
    ymin = height
    if text_offset is None:
        text_offset = bracket_length

    ax.plot([x1, x1, x2, x2], [ymin, ymax, ymax, ymin], **kwargs)
    ax.text(center_point,
            ymax + text_offset,
            text,
            ha="center",
            va="center",
            **kwargs)


def entry_subplot(dat,
                  ax=None,
                  **kwargs):

    p_dat = dat.to_pandas()
    if ax is None:
        fig, ax = plt.subplots(figsize=[15, 5])

    sns.boxplot(x="species",
                y="entry",
                hue="spike",
                dodge=True,
                data=p_dat,
                ax=ax,
                showcaps=False,
                boxprops={'facecolor': 'None'},
                showfliers=False,
                palette=variant_palette)

    ax = sns.stripplot(x="species",
                       y="entry",
                       hue="spike",
                       dodge=True,
                       data=p_dat,
                       ax=ax,
                       palette=variant_palette,
                       **kwargs)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
            handles=handles[3:],
            labels=labels[3:],
            fancybox=True,
            frameon=True,
            framealpha=1,
            title="Spike")
    ax.set_xlabel("Species")
    ax.set_ylabel("Entry (relative to no spike)")

    return ax


def air_shedding_subfigure(
        model,
        data,
        cage_predictions,
        fig=None,
        rng_seed=602,
        xlim=[-1, 9],
        n_curves=sample_n_curves,
        **kwargs):

    if fig is None:
        fig = plt.figure(figsize=[8, 3.5])

    ax = fig.subplots(nrows=2,
                      ncols=8,
                      sharex=True,
                      sharey="row")

    np.random.seed(rng_seed)

    subsample = np.random.randint(
        low=0,
        high=cage_predictions["air_sample_1"][
            "predicted_PFU"].shape[1],
        size=n_curves)

    gridspec = ax[0, 0].get_subplotspec().get_gridspec()
    alpha_fig = fig.add_subplot(gridspec[::, :4])
    delta_fig = fig.add_subplot(gridspec[::, 4:])
    alpha_fig.axis("off")
    delta_fig.axis("off")

    for i_cage, cage in enumerate(model["cages"].keys()):
        cage_id = i_cage + 1
        i_var = cage_predictions[cage]["i_var"][0]
        i_sex = cage_predictions[cage]["sex"][0]
        cage_n = cage_predictions[cage]["n_indiv"]
        cage_sex = ["F", "M"][i_sex]
        xs = cage_predictions[cage]["xs"]
        dat = data.filter(pl.col("cage_id") == cage_id)
        var_color = variant_colors[i_var]
        p_ax = ax[0, i_cage]
        c_ax = ax[1, i_cage]
        times = dat["Timepoint"].to_numpy()
        plaques = dat["total_plaques"].to_numpy()
        sgRNA = np.nan_to_num(
            dat["log10_copies_subgenomic"].to_numpy(),
            nan=-1.0)

        pred_pfu = np.swapaxes(
            cage_predictions[cage]["predicted_PFU"], 0, 1)
        pred_rna = np.swapaxes(
            cage_predictions[cage]["predicted_log10_air_rna"], 0, 1)

        _ = p_ax.plot(
            xs,
            pred_pfu[::, subsample] / cage_n,
            color=var_color,
            alpha=shed_seethru,
            lw=shed_lw)

        plot_data_with_lod(
            times,
            plaques / cage_n,
            lld=0,
            color="k",
            markeredgecolor="k",
            markerfacecolor=var_color,
            ax=p_ax,
            **kwargs)

        # handle RNA copies
        lld_per_cap = np.exp(np.log(10) * -0.8707614) / cage_n

        _ = c_ax.axhline(
            lld_per_cap,
            lw=shed_lw,
            linestyle="dashed",
            color="k")

        _ = c_ax.plot(
            xs,
            np.exp(np.log(10) * pred_rna[::, subsample]) / cage_n,
            color=var_color,
            alpha=shed_seethru,
            lw=shed_lw)

        plot_data_with_lod(
            times,
            np.exp(
                np.log(10) *
                np.clip(
                    sgRNA,
                    a_min=-0.8707614,
                    a_max=11.92514)
            ) / cage_n,
            lld=lld_per_cap,
            uld=np.exp(np.log(10) * 11) / cage_n,
            lld_marker="v",
            uld_marker="^",
            color="k",
            markeredgecolor="k",
            markerfacecolor=var_color,
            ax=c_ax,
            **kwargs)

        p_ax.set_title("{} ({}{})".format(cage_id, cage_n, cage_sex))
        p_ax.set_xticks(np.arange(0, 9, 2))
        p_ax.set_xlim(xlim)
        p_ax.set_yticks([0, 5, 10, 15, 20])
        p_ax.set_ylim([-2.5, 25])

        c_ax.set_xticks(np.arange(0, 9, 2))
        c_ax.set_xlim(xlim)
        c_ax.set_yscale("log")
        c_ax.set_ylim([0.1 * lld_per_cap, 1e4])
        pass  # end loop over cages

    _ = fig.supxlabel("Time since inoculation (days)")
    _ = fig.supylabel("24h air sample per capita")
    _ = ax[0, 0].set_ylabel("Plaques")
    _ = ax[1, 0].set_ylabel("sgRNA copies")
    _ = alpha_fig.set_title("Alpha cages", y=1.1)
    _ = delta_fig.set_title("Delta cages", y=1.1)

    return ax


def kinetics_subfigure(swabs,
                       predictions,
                       fig=None,
                       rng_seed=8324,
                       xlim=[-1, 9],
                       markersize=msize,
                       markeredgewidth=1,
                       n_curves=sample_n_curves,
                       data_lw=ct_lw):

    if fig is None:
        fig = plt.figure(figsize=[8, 5])

    ax = fig.subplots(3, 4, sharex=True, sharey='row')

    np.random.seed(rng_seed)
    sexes = ["F", "M"]
    m_subsample = np.random.randint(
        low=0,
        high=predictions["Alpha"]["air_M"].shape[1],
        size=n_curves)
    f_subsample = np.random.randint(
        low=0,
        high=predictions["Alpha"]["air_F"].shape[1],
        size=n_curves)

    gridspec = ax[0, 0].get_subplotspec().get_gridspec()
    alpha_fig = fig.add_subplot(gridspec[::, :2])
    delta_fig = fig.add_subplot(gridspec[::, 2:])
    alpha_fig.axis("off")
    delta_fig.axis("off")

    for i in range(4):
        ax[1, i].axhline(10 ** LOD_tcid,
                         lw=shed_lw,
                         color="k",
                         linestyle="dashed")

    for i_var, variant in enumerate(["Alpha", "Delta"]):
        for i_sex, sex in enumerate(sexes):
            if sex == "F":
                sex_subsample = f_subsample
            elif sex == "M":
                sex_subsample = m_subsample
            else:
                raise ValueError("Data for sex {} not found".format(sex))

            v_color = variant_colors[i_var]
            i_col = 2 * i_var + (not i_sex)

            for i_row, y_val in enumerate(["air",
                                           "swab_tcid",
                                           "log10_swab_rna"]):
                xs = predictions[variant]["xs"]

                y_val_stem = y_val + "_{}"
                ys = predictions[variant][y_val_stem.format(sex)][
                    ::, sex_subsample]

                if "swab" in y_val:
                    ys = np.exp(np.log(10) * ys)

                _ = ax[i_row, i_col].plot(
                    xs,
                    ys,
                    color=v_color,
                    alpha=shed_seethru,
                    lw=shed_lw)
                pass  # end loop over y value types

            var_sex_swabs = swabs.filter(
                (pl.col("sex") == sex) &
                (pl.col("variant") == variant))
            for _, grp in var_sex_swabs.to_pandas().groupby(
                    ["hamster_id"]):
                times = np.random.normal(
                    loc=grp["Timepoint"],
                    scale=jitter_width)

                plot_data_with_lod(
                    times,
                    np.exp(np.log(10) *
                           grp["log10_tcid50"]),
                    lld=np.exp(np.log(10) * 0.5),
                    color="black",
                    markeredgecolor="black",
                    markerfacecolor=v_color,
                    markeredgewidth=markeredgewidth,
                    markersize=markersize,
                    data_lw=data_lw,
                    ax=ax[1, i_col])

                plot_data_with_lod(
                    times,
                    np.exp(
                        np.log(10) *
                        np.nan_to_num(
                            grp["log10_copies_subgenomic"],
                            -1.0)),
                    uld=np.exp(np.log(10) * 11),
                    uld_marker="^",
                    lld=np.exp(np.log(10) * -1.0),
                    lld_marker="v",
                    markeredgecolor="black",
                    markerfacecolor=v_color,
                    color="k",
                    markeredgewidth=markeredgewidth,
                    markersize=markersize,
                    data_lw=data_lw,
                    ax=ax[2, i_col])
            pass  # end loop over sexes
        pass  # end loop over variants
    for col in range(4):
        ax[0, col].set_title(sexes[not (col % 2)])

    ax[0, 0].set_ylabel("Air PFU/h")
    ax[1, 0].set_ylabel("Swab TCID$_{50}$/mL")
    ax[2, 0].set_ylabel("Swab sgRNA copies")

    ax[0, 0].set_yscale("log")
    ax[1, 0].set_yscale("log")
    ax[2, 0].set_yscale("log")

    ax[0, 0].set_xticks(np.arange(0, 9, 2))
    ax[0, 0].set_xlim(xlim)
    ax[0, 0].set_yticks([1e-2, 1e-1, 1e0, 1e1])
    ax[0, 0].set_ylim([.5e-2, 2e1])

    ax[1, 0].set_yticks([1e1, 1e3, 1e5])
    ax[1, 0].set_ylim([1e0, 1e6])

    ax[2, 0].set_yticks([1e2, 1e4, 1e6])
    ax[2, 0].set_ylim([1e1, 1e7])

    _ = fig.supxlabel("Time since inoculation (days)")
    _ = alpha_fig.set_title("Alpha", y=1.05)
    _ = delta_fig.set_title("Delta", y=1.05)

    return ax
