#!/usr/bin/env python3

# filename: priors.py
# description: specification of prior
# distributions for model


import jinja2
import os
import sys
import csv

import numpy as np


def half_life_to_decay_rate(half_life):
    return np.log(2) / half_life


log_priors = {
    "log_half_life_prior_mean_log_arg": 1.5,
    "variant_log_doubling_time_prior_mean_log_arg": 5,
    "mean_log_peak_time_prior_mean_log_arg": 24,
    "log_decay_half_life_prior_mean_log_arg": 15,
    "log_rna_decay_rate_ratio_prior_mean_log_arg": 1,
    "log_resp_rate_mean_prior_mean_log_arg": 80 * 60,
    "log_day_peak_prior_mean_log_arg": 1,
    "day_peak_logten_copies_prior_mean": 2,
}

priors = {
    "log_variant_infectivity_prior_mean": 0,
    "log_variant_infectivity_prior_sd": 3,
    "log_half_life_prior_mean": np.log(log_priors[
        "log_half_life_prior_mean_log_arg"]),
    "log_half_life_prior_sd": 0.5,


    # in log virions per 24 hours at
    # a resp rate of 4800 mL/h
    "log_peak_prior_mean": (
        np.log(
            log_priors["log_day_peak_prior_mean_log_arg"]
        ) - np.log(
            24
            # convert to per hour
        ) - np.log(
            log_priors["log_resp_rate_mean_prior_mean_log_arg"]
            # convert to per exhaled mL
        )),

    "log_peak_prior_sd": 3,

    "sd_log_peak_prior_mode": 0,
    "sd_log_peak_prior_sd": 2,

    "variant_log_doubling_time_prior_mean": np.log(
        log_priors["variant_log_doubling_time_prior_mean_log_arg"]),

    "variant_log_doubling_time_prior_sd": 0.5,

    "male_respiration_rate_offset_prior_mean": 0,
    "male_respiration_rate_offset_prior_sd": 0.25,

    "male_peak_offset_prior_mean": 0,
    "male_peak_offset_prior_sd": 0.25,

    "male_growth_rate_offset_prior_mean": 0,
    "male_growth_rate_offset_prior_sd": 0.25,

    "male_decay_rate_offset_prior_mean": 0,
    "male_decay_rate_offset_prior_sd": 0.25,

    "sd_log_peak_time_prior_mode": 0,
    "sd_log_peak_time_prior_sd": 0.15,

    "mean_log_peak_time_prior_mean": np.log(
        log_priors["mean_log_peak_time_prior_mean_log_arg"]
    ),
    "mean_log_peak_time_prior_sd": 0.5,

    "log_decay_half_life_prior_mean": np.log(
        log_priors["log_decay_half_life_prior_mean_log_arg"]),
    "log_decay_half_life_prior_sd": 0.75,

    "log_swab_decay_rate_ratio_prior_mean": 0,
    "log_swab_decay_rate_ratio_prior_sd": 1,

    "log_rna_decay_rate_ratio_prior_mean": np.log(
        log_priors[
            "log_rna_decay_rate_ratio_prior_mean_log_arg"]
    ),
    "log_rna_decay_rate_ratio_prior_sd": 1,

    "log_swab_peak_offset_prior_mean": 0,
    "log_swab_peak_offset_prior_sd": 0.25,

    "sd_individual_growth_rate_prior_mode": 0,
    "sd_individual_growth_rate_prior_sd": 0.2,
    "sd_individual_decay_rate_prior_mode": 0,
    "sd_individual_decay_rate_prior_sd": 0.2,

    "sd_obs_oral_rna_prior_mode": 0,
    "sd_obs_oral_rna_prior_sd": 0.5,
    "sd_obs_air_rna_prior_mode": 0,
    "sd_obs_air_rna_prior_sd": 0.5,
    "donor_copy_number_offset_prior_mean": 0,
    "donor_copy_number_offset_prior_sd": 1.5,

    "log_oral_RNA_prior_mean": 0,
    "log_oral_RNA_prior_sd": 10,
    "log_oral_TCID_prior_mean": 0,
    "log_oral_TCID_prior_sd": 10,
    "log_air_RNA_prior_mean": 0,
    "log_air_RNA_prior_sd": 10,
    "log_resp_rate_mean_prior_mean": np.log(
        log_priors["log_resp_rate_mean_prior_mean_log_arg"]),
    "log_resp_rate_mean_prior_sd": 0.25,
    "log_resp_rate_sd_prior_mode": 0,
    "log_resp_rate_sd_prior_sd": 0.25,
    "sd_obs_resp_rate_prior_mode": 0,
    "sd_obs_resp_rate_prior_sd": 0.2,

    "main_use_sentinels": True,
    "simple_use_sentinels": False
}


def get_template(template_filepath):
    """Get a jinja template with latex tags per
    http://eosrei.net/articles/2015/11/latex-templates-python-and-jinja2-generate-pdfs
    """
    path, filename = os.path.split(template_filepath)
    latex_jinja_env = jinja2.Environment(
    	block_start_string = '\BLOCK{',
    	block_end_string = '}',
    	variable_start_string = '\VAR{',
    	variable_end_string = '}',
    	comment_start_string = '\#{',
    	comment_end_string = '}',
    	line_statement_prefix = '%%',
    	line_comment_prefix = '%#',
    	trim_blocks = True,
    	autoescape = False,
    	loader = jinja2.FileSystemLoader(path)
    )
    template = latex_jinja_env.get_template(filename)
    return template


def underscores_to_caps(word):
    """
    Replace underscores with capitalization
    (for taking env variables to LaTeX macros)
    """
    return ''.join(x.capitalize() for x in word.split('_'))

def digits_to_names(word):
    trans_dict = {
        "0": "nought",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine"}
    return word.translate(str.maketrans(trans_dict))

def add_prefix_and_suffix(word, prefix, suffix):
    if prefix is not None:
        word = prefix + word
    if suffix is not None:
        word = word + suffix
    return word

def escape_word(word, prefix=None, suffix=None):
    word = digits_to_names(word)

    word = add_prefix_and_suffix(word,
                                 prefix,
                                 suffix)
    word = underscores_to_caps(word)
    return word

def escape_contents(input_dict, prefix=None):
    return {escape_word(key, prefix=prefix): entry
            for key, entry in input_dict.items()}


def isnum(param_value):
    """
    Function to separate out numerical 
    parameter values from others, as the
    template will render numerical values as
    LaTeX numbers (\num{value})
    """
    if type(param_value) is str:
        return param_value.replace(
            '.', '').replace(
            'e', '').replace(
            '-', '').isdigit()
    else:
        return True

def dict_to_list(input_dict):
    return [{"name": key, "value": value,
             "isnum": isnum(value)}
            for key, value
            in input_dict.items()]

def main(template,
         outpath,
         prefix=None):
    
    contents = escape_contents(dict(**priors,
                                    **log_priors),
                               prefix=prefix)
    contents = dict_to_list(contents)
    print(contents)
    template = get_template(template)
    with open(outpath, 'w') as output:
        output.write(
            template.render(
                envvars=contents
            )
        )
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Supply template file and output path, in that order"
              "optionally supply prefix to append to all macros")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3:
        main(sys.argv[1],
             sys.argv[2],
             sys.argv[3])
