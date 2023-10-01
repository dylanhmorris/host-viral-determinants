#!/usr/bin/env python3

import polars as pl
import pandas as pd
import sys


def main(input_path,
         output_path):
    # read in and manually
    # clean up spanning columns
    dat = pl.DataFrame(
        pd.read_excel(input_path))

    col_names = (["species"] +
                 ["{}_{}".format(spike, x) for
                  spike in ["Alpha", "Delta", "No spike"]
                  for x in range(8)])

    dat.columns = col_names

    # pivot to long format
    dat = dat.melt(
        id_vars="species",
        variable_name="spike",
        value_name="entry"
    ).with_columns(
        pl.col("spike").str.extract(
            r"([aA-zZ\s]+)_"
        ).alias("spike")
    )

    if output_path.endswith(".tsv"):
        separator = "\t"
    else:
        separator = ","

    # save to outpath
    dat.write_csv(output_path,
                  separator=separator)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("USAGE: ./clean_entry.py "
              "<input path> <output path>")
    else:
        main(sys.argv[1],
             sys.argv[2])
