#!/usr/bin/env python3

import pickle 

def load_chains(
        chain_path):

    with open(chain_path, "rb") as infile:
        o, model = pickle.load(infile)
        posterior_samples = o.get_samples()

    return (posterior_samples, model, o)


def load_checks(
        check_path):

    with open(check_path, "rb") as infile:
        output, model = pickle.load(infile)

    return (output, model)
