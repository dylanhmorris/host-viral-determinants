#!/usr/bin/env python3
import pandas as pd
import polars as pl
import sys


def clean_shedding(pleth=None,
                   shedding_rna=None,
                   shedding_virus=None,
                   shedding_meta=None,
                   cage_plaque=None,
                   cage_assignments=None,
                   outpath=None):

    print("Reading in data...")

    cage_assignments = pl.read_csv(cage_assignments)

    pleth = pl.DataFrame(
        pd.read_excel(pleth))

    shedding_rna = pd.read_excel(shedding_rna,
                                 skiprows=1)
    shedding_rna.loc[shedding_rna["Ct"] == "Undetermined", "Ct"] = 40
    shedding_rna.loc[shedding_rna["Ct.1"] == "Undetermined", "Ct.1"] = 40
    shed_dat = pl.DataFrame(shedding_rna)

    shedding_virus = pd.read_excel(shedding_virus,
                                   skiprows=2)
    shedding_virus = shedding_virus.fillna(0.0)

    shedding_virus = pl.DataFrame(shedding_virus)
    shedding_virus = shedding_virus.melt(
        id_vars=["Unnamed: 0"]
    ).rename(
        {"Unnamed: 0": "Timepoint",
         "variable": "animal_name",
         "value": "log10_tcid50"}
    ).with_columns(
        pl.col("animal_name").str.extract(
            "([0-9]+)", 1).cast(
                pl.Int64).alias("hamster_id")
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

    shedding_metadata = pd.read_excel(
        shedding_meta,
        sheet_name=3,
        header=None)
    shedding_metadata.columns = ["date",
                                 "experimenter",
                                 "study",
                                 "sex",
                                 "breed",
                                 "animal_name",
                                 "sacrifice_day",
                                 "variant",
                                 "animal_number"]
    shedding_meta = pl.DataFrame(shedding_metadata)

    cage_plaque = pl.read_csv(cage_plaque)

    # data processing and cleaning
    print("pre-processing and cleaning data...")

    # fix incorrectly recorded donor titer

    pleth = pleth.select(
        [pl.col("All Subjects").str.slice(4, 4).cast(
            pl.Int64).alias("hamster_id"),
         pl.col("DPI"),
         pl.col("MVb(ml/min)").alias("MVb")])

    shedding_meta = shedding_meta.with_columns(
        hamster_id=pl.col("animal_name").str.extract(
            "([0-9]+)", 1).cast(
                pl.Int64)
    )

    shedding_meta = shedding_meta.select(
        [
            pl.col("animal_name"),
            pl.col("hamster_id"),
            pl.col("sex"),
            pl.col("sacrifice_day"),
            pl.when(
                pl.col("variant") == "B.1.1.7"
            ).then(
                pl.lit("Alpha")
            ).when(
                pl.col("variant") == "B.1617.2"
            ).then(
                pl.lit("Delta")
            ).otherwise(
                None
            ).alias("variant")
        ])

    shedding_meta = shedding_meta.join(
        cage_assignments,
        left_on="animal_name",
        right_on="animal"
    ).rename(
        {"cage": "cage_id"}
    )
    shed_dat = shed_dat.rename(
            {"copies": "copies_genomic",
             "Ct": "Ct_genomic",
             "copies.1": "copies_subgenomic",
             "Ct.1": "Ct_subgenomic"}
        ).drop(
            "virus"
        ).with_columns(
            pl.col(
                "copies_subgenomic"
            ).log10(
            ).alias(
                "log10_copies_subgenomic"
            )
        )

    swabs = shed_dat.filter(
        pl.col("Tissue") == "oral swab"
    ).with_columns(
        pl.col(
            "Animal #"
        ).str.extract(
            "([0-9]+)", 1
        ).cast(
            pl.Int64
        ).alias("hamster_id")
    ).join(
        shedding_meta,
        on="hamster_id"
    ).join(
        shedding_virus.select(
            ["hamster_id",
             "Timepoint",
             "log10_tcid50",
             "n_positive_wells",
             "starting_row_dilution"]
        ),
        on=["hamster_id", "Timepoint"]
    )

    pleth = pleth.join(
        shedding_meta,
        on="hamster_id")

    air_samples = shed_dat.filter(
        pl.col("Tissue") == "cage air"
    ).with_columns(
        pl.col("Animal #").str.extract(
            "([0-9]+)", 1
        ).cast(
            pl.Int64
        ).alias("cage_id")
    ).with_columns(
        (pl.col("Timepoint")
         ).alias("Timepoint")
    ).filter(
        pl.col("Timepoint") >= 0
    )

    cage_plaque = cage_plaque.with_columns(
        pl.col("cage").str.extract(
                "([0-9]+)", 1
            ).cast(
                pl.Int64
            ).alias("cage_id")
    ).with_columns(
        (pl.col("plaques/ml") * 6.0 / 5.0
         ).round(
            0
        ).cast(
            pl.Int64
        ).alias("total_plaques")
    ).rename(
        {"day": "Timepoint"}
    )

    cage_variant = swabs[["variant", "cage_id"]].unique()
    air_samples = air_samples.join(
        cage_plaque,
        on=["cage_id", "Timepoint"]
    ).join(
        cage_variant,
        on="cage_id"
    )

    if outpath is not None:
        if "swab" in outpath:
            swabs.write_csv(outpath, separator="\t")
        elif "pleth" in outpath:
            pleth.write_csv(outpath, separator="\t")
        elif "air" in outpath:
            air_samples.write_csv(outpath, separator="\t")
        else:
            raise ValueError("Unsupported output path {}.\n"
                             "path must contain 'pleth', "
                             "'swab', or 'air'".format(outpath))

    return (swabs, pleth, air_samples)


if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("\nUSAGE: ./clean_shedding.py "
              "<pleth path> "
              "<shed rna path> "
              "<shed virus path> "
              "<shedding metadata path> "
              "<cage plaque path> "
              "<cage assignments path> "
              "<output path>\n\n")
    else:
        clean_shedding(*sys.argv[1:])
