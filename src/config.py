#!/usr/bin/env python3

"""
Configuration file containing
analysis parameters that are
not prior distribution parameters
(those are in priors.py),
such as MCMC defaults
and various constants
"""

mcmc_params = dict(
    n_warmup=1000,
    n_samples=1000,
    n_prior_check_samples=4000,
    n_chains=3,
    chain_method="parallel",
    CPU=True,
    CPU_cores=1,
    adapt_delta=0.9,
    max_tree=(10, 10),
    forward_mode_differentiation=True,
    main_rng_seed=1234,
    simple_rng_seed=5678,
    main_prior_check_rng_seed=13252
)
