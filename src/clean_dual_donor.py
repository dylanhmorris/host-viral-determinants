#!/usr/bin/env python3

import polars as pl
import pandas as pd
import numpy as np
import string
import sys


def get_time_in():
    """
    calculate time of
    placement into the exposure
    cage in days post donor
    infection
    """

    return pl.when(
        pl.col("animal_role") == "donor"
    ).then(
        pl.when(
            pl.col("variant") == "Delta"
        ).then(
            pl.when(
                pl.col("exposure_sequence") == "B.1.1.7 first"
            ).then(
                1 + 2.0 / 24
            ).otherwise(
                1 + 0.0 / 24
            )
        ).when(
            pl.col("variant") == "Alpha"
        ).then(
            pl.when(
                pl.col("exposure_sequence") == "B.617 first"
            ).then(
                1 + 2.0 / 24
            ).otherwise(
                1 + 0.0 / 24
            )
        ).otherwise(
            None
        )
    ).when(
        pl.col("animal_role") == "sentinel"
    ).then(
        1 + 0.0 / 24
    ).otherwise(
        None
    )


def get_duration():
    """
    calculate time spent in
    the exposure
    cage in days
    """
    return pl.when(
        pl.col("animal_role") == "donor"
    ).then(
        pl.when(
            pl.col("exposure_sequence").str.contains(
                "first")
        ).then(
            2.0 / 24
        ).when(
            pl.col("exposure_sequence").str.contains(
                "both")
        ).then(
            4.0 / 24
        ).otherwise(
            None
        )
    ).when(
        pl.col("animal_role") == "sentinel"
    ).then(
        4.0 / 24
    ).otherwise(
        None
    )


def clean_dual_donor_metadata(metadata_path):
    """
    function to clean up
    metadata for dual
    donor experiments
    to merge with measurement
    data
    """
    # read in data
    metadata = pl.DataFrame(
        pd.read_excel(
            metadata_path,
            sheet_name=1,
            header=None)
    )

    # fix column names
    metadata.columns = [
        "date",
        "experiment_no",
        "animal_sex",
        "animal_name",
        "animal_role",
        "animal_experiment_role",
        "animal_number",
        "empty_column",
        "exposure_sequence"
    ]

    # convert experiment numbers
    # to cage letters
    metadata = metadata.with_columns(
        pl.col("experiment_no").cast(
            pl.Int64
        ).alias(
            "experiment_no"
        )
    ).filter(
        ~pl.col("experiment_no").is_null()
    )
    letters = np.array(list(string.ascii_uppercase))
    metadata = metadata.with_columns(
        cage=pl.lit(letters[metadata["experiment_no"] - 1]))

    # clean up animal roles
    # and add times in / out
    # based on role
    metadata = metadata.with_columns(
        pl.when(
            pl.col("animal_role") == "Donor"
        ).then(
            pl.lit("donor")
        ).when(
            pl.col("animal_role") == "Sentinal"
        ).then(
            pl.lit("sentinel")
        ).otherwise(
            None
        ).alias(
            "animal_role"
        )
    ).with_columns(
        pl.when(
            (pl.col("animal_role") == "donor") &
            (pl.col("animal_experiment_role") == "B.1.617.2")
        ).then(
            pl.lit("Delta")
        ).when(
            (pl.col("animal_role") == "donor") &
            (pl.col("animal_experiment_role") == "B.1.1.7")
        ).then(
            pl.lit("Alpha")
        ).otherwise(
            None
        ).alias(
            "variant"
        )
    ).with_columns(
        pl.col(
            "exposure_sequence"
        ).fill_null(
            strategy="forward")
    ).with_columns(
        time_in=get_time_in()
    ).with_columns(
        time_out=(pl.col("time_in") +
                  get_duration())
    ).with_columns(
        pl.when(
            pl.col("exposure_sequence").str.contains("first")
        ).then(
            pl.lit("sequential")
        ).when(
            pl.col("exposure_sequence").str.contains("both")
        ).then(
            pl.lit("simultaneous")
        ).otherwise(
            None
        ).alias(
            "exposure_sequence"
        )
    ).with_columns(
        pl.when(
            pl.col("animal_role") == "donor"
        ).then(
            pl.col("cage") +
            "_" +
            pl.col("variant")
        ).when(
            pl.col("animal_role") == "sentinel"
        ).then(
            pl.col("cage") +
            pl.col("animal_experiment_role").str.extract(
                "Group [0-9].([0-9])"
            )
        ).otherwise(
            None
        ).alias(
            "animal_cage_id"
        )
    )

    return metadata.select(
        [
            "animal_cage_id",
            "cage",
            "variant",
            "animal_name",
            "animal_role",
            "animal_sex",
            "exposure_sequence",
            "time_in",
            "time_out"
        ]
    )


def clean_donors(dual_donor_path):
    """
    read in and clean
    swab data for donors
    """
    donors = pl.DataFrame(
        pd.read_excel(dual_donor_path,
                      sheet_name=1)
    )

    donors = donors.melt(
        id_vars="Donor"
    ).pivot(
        values='value',
        index='variable',
        columns='Donor'
    ).rename(
     {"variable": "donor",
      "log10 RNA copies": "log10_copies_subgenomic",
      "TCID50 log10": "log10_tcid50"}
    ).with_columns(
        pl.when(
            pl.col("donor").str.contains(".1")
        ).then(
            pl.lit("Alpha")
        ).otherwise(
            pl.lit("Delta")
        ).alias("variant")
    ).with_columns(
        pl.col("donor").str.slice(0, 1).alias("cage")
    ).with_columns(
        (
            pl.col("cage") +
            "_" +
            pl.col("variant")
        ).alias("animal_cage_id")
    ).with_columns(
        pl.lit(1.0).alias("timepoint")
    ).select(
        ["animal_cage_id",
         "cage",
         "variant",
         "timepoint",
         "log10_tcid50",
         "log10_copies_subgenomic"])

    return donors


def clean_sentinels(dual_donor_path):
    """
    Read in and clean swab
    and sequencing data for
    sentinels
    """

    sentinels = pl.DataFrame(
        pd.read_excel(
            dual_donor_path,
            sheet_name=2,
            header=1))

    sentinels = sentinels.rename(
        {
            "Sentinel ID": "animal_cage_id",
            "5 DPI oral swab TCID50/ml": "5_DPI_log10_tcid50",
            "2 DPI oral swab.1": "2_DPI_pct_Delta",
            "5 DPI oral swab": "5_DPI_pct_Delta"
        }
    ).with_columns(
        pl.col(
            "animal_cage_id"
        ).str.slice(0, 1).alias("cage")
    )

    sgRNA_copies = sentinels.melt(
        id_vars=["animal_cage_id", "cage"],
        value_vars=["2 DPI oral swab",
                    "3 DPI oral swab",
                    "5 DPI oral swab RNA copies/ml"],
        value_name="log10_copies_subgenomic",
        variable_name="measurement"
    ).with_columns(
        pl.col("measurement").str.slice(
            0, 1
        ).cast(
            pl.Float64
        ).alias("timepoint")
    ).select(
        ["animal_cage_id",
         "cage",
         "log10_copies_subgenomic",
         "timepoint"
         ]
    )

    pct_Delta = sentinels.melt(
        id_vars=["animal_cage_id", "cage"],
        value_vars=["2_DPI_pct_Delta",
                    "5_DPI_pct_Delta"],
        value_name="pct_Delta",
        variable_name="measurement"
    ).with_columns(
        pl.col("measurement").str.slice(
            0, 1
        ).cast(
            pl.Float64
        ).alias("timepoint")
    ).select(
        ["animal_cage_id",
         "cage",
         "pct_Delta",
         "timepoint"
         ]
    )

    titers = sentinels.select(
        [
            "animal_cage_id",
            "cage",
            "5_DPI_log10_tcid50"
        ]
    ).with_columns(
        pl.lit(5.0).alias("timepoint")
    ).rename(
        {"5_DPI_log10_tcid50": "log10_tcid50"}
    )

    return sgRNA_copies.join(
        pct_Delta,
        on=["animal_cage_id", "cage", "timepoint"],
        how="outer"
    ).join(
        titers,
        on=["animal_cage_id", "cage", "timepoint"],
        how="outer"
    )


def clean(measurement_path,
          metadata_path,
          outpath=None):

    d = clean_donors(measurement_path)

    s = clean_sentinels(measurement_path)

    meta = clean_dual_donor_metadata(
        metadata_path)

    print(meta)

    # add metadata to donors and sentinels
    sm = s.join(meta, on="animal_cage_id")
    dm = d.join(meta, on="animal_cage_id")
    # concatenate donors and sentinels
    # and return final dataframe

    needed_cols = [
        "animal_cage_id",
        "cage",
        "animal_name",
        "animal_sex",
        "variant",
        "animal_role",
        "timepoint",
        "log10_copies_subgenomic",
        "pct_Delta",
        "log10_tcid50",
        "exposure_sequence",
        "time_in",
        "time_out"
    ]

    sm = sm.select(needed_cols)

    # handle the fact that
    # donors don't have
    # pct_Delta measurements
    dm = dm.with_columns(
        pl.lit(
            None
        ).cast(
            pl.Float64
        ).alias(
            "pct_Delta"
        )
    ).select(needed_cols)

    df = pl.concat([sm, dm])

    df = df.with_columns(
        (40 +
         -3.3938 *
         (pl.col("log10_copies_subgenomic") - 3.331)
         ).alias("Ct_subgenomic")
    ).with_columns(
        (
            4 * (pl.col("log10_tcid50") - 0.5)
        ).alias("total_pos_wells")
    ).with_columns(
        (pl.col("total_pos_wells") % 4).alias("remainder")
    ).with_columns(
        pl.when(
            (pl.col("remainder") + 4) >
            (pl.col("total_pos_wells"))
        ).then(
            pl.col("total_pos_wells")
        ).otherwise(
            pl.col("remainder") + 4
        ).cast(
            pl.Int64
        ).alias("n_positive_wells")
    ).with_columns(
        (
            (pl.col("total_pos_wells") -
             pl.col("n_positive_wells")) / -4
        ).cast(
            pl.Int64
        ).alias("starting_row_dilution")
    )

    if outpath is not None:
        df.write_csv(outpath,
                     separator="\t")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("\nUSAGE: ./clean_dual_donor.py "
              "<donor data path> <metadata path> "
              "<output path>\n\n")
    else:
        clean(sys.argv[1],
              sys.argv[2],
              outpath=sys.argv[3])
