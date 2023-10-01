#####################################
# name: Makefile
# author: Dylan H. Morris (dylanhmorris.com)
#
# Makefile to generate analyses
# of Alpha and Delta SARS-CoV-2
# kinetics and shedding in hamsters 
####################################

#####################################
# Directory structure
####################################

default: all

#####################################
# Directory structure
####################################

default: all

SRC = src
OUT = out
DATA = dat
MS = ms

RAW = $(DATA)/raw
CLEAN = $(DATA)/cleaned
FIGS = $(MS)/figures

#####################################
# Expected terminal settings
#
# Check these vs your local
# machine setup if you are having
# difficulty reproducing the
# analysis
#####################################

MKDIR := @mkdir -p
CP := @cp
RM := rm -f


#####################################
# Clean data
#####################################
ENTRY_DATA = $(CLEAN)/entry.tsv
DUAL_DONOR_DATA = $(CLEAN)/dual-donor.tsv
SWAB_DATA = $(CLEAN)/swabs.tsv
PLETH_DATA = $(CLEAN)/pleth.tsv
AIR_DATA = $(CLEAN)/air-samples.tsv

METADATA_FILE = $(RAW)/hamster-metadata.xlsx
DUAL_DONOR_FILE = $(RAW)/dual-donor.xlsx
#####################################
# Results
#####################################

MCMC_OUT = $(OUT)/mcmc-output-main.pickle
PRIOR_CHECK_OUT_MAIN = $(OUT)/prior-check-output-main.pickle
MCMC_SIMPLE = $(OUT)/mcmc-output-simple.pickle
MACROS = $(MS)/parameters.sty

FIGURE_NAMES = figure-main.pdf figure-no-sentinels.pdf figure-comparison.pdf figure-coinfection-probability.pdf figure-expected-coinfections.pdf figure-infection-probability-by-variant.pdf figure-main-prior-check.pdf

FIGURE_PATHS = $(addprefix $(FIGS)/, $(FIGURE_NAMES))

$(MCMC_OUT): $(SRC)/run_model.py $(SWAB_DATA) $(PLETH_DATA) $(DUAL_DONOR_DATA) $(AIR_DATA) $(SRC)/model.py $(SRC)/predict.py $(SRC)/priors.py
	$(MKDIR) $(OUT)
	./$^ $@

$(PRIOR_CHECK_OUT_MAIN): $(SRC)/run_model.py $(SWAB_DATA) $(PLETH_DATA) $(DUAL_DONOR_DATA) $(AIR_DATA) $(SRC)/model.py $(SRC)/predict.py $(SRC)/priors.py
	$(MKDIR) $(OUT)
	./$^ $@

$(PRIOR_CHECK_OUT_SIMPLE): $(SRC)/run_model.py $(SWAB_DATA) $(PLETH_DATA) $(DUAL_DONOR_DATA) $(AIR_DATA) $(SRC)/model.py $(SRC)/predict.py $(SRC)/priors.py
	$(MKDIR) $(OUT)
	./$^ $@


$(MCMC_SIMPLE): $(SRC)/run_model.py $(SWAB_DATA) $(PLETH_DATA) $(DUAL_DONOR_DATA) $(AIR_DATA) $(SRC)/model.py $(SRC)/predict.py $(SRC)/priors.py
	$(MKDIR) $(OUT)
	./$^ $@

$(MCMC_INTERACTION): $(SRC)/run_interaction_model.py $(SWAB_DATA) $(PLETH_DATA) $(DUAL_DONOR_DATA) $(AIR_DATA) $(SRC)/interactionmodel.py $(SRC)/predict.py $(SRC)/priors.py
	$(MKDIR) $(OUT)
	./$^ $@

$(FIGS)/figure-main.pdf: $(SRC)/figure-main.py $(ENTRY_DATA) $(MCMC_OUT) $(SWAB_DATA) $(PLETH_DATA) $(AIR_DATA) $(SRC)/plotting.py
	$(MKDIR) $(FIGS)
	./$^ $@

$(FIGS)/figure-main-prior-check.pdf: $(SRC)/figure-main.py $(ENTRY_DATA) $(PRIOR_CHECK_OUT_MAIN) $(SWAB_DATA) $(PLETH_DATA) $(AIR_DATA) $(SRC)/plotting.py
	$(MKDIR) $(FIGS)
	./$^ $@

$(FIGS)/figure-no-sentinels.pdf: $(SRC)/figure-main.py $(ENTRY_DATA) $(MCMC_SIMPLE) $(SWAB_DATA) $(PLETH_DATA) $(AIR_DATA) $(SRC)/plotting.py
	$(MKDIR) $(FIGS)
	./$^ $@

$(FIGS)/figure-comparison.pdf: $(SRC)/figure-comparison.py $(MCMC_OUT) $(MCMC_SIMPLE) $(ENTRY_DATA) $(SWAB_DATA) $(PLETH_DATA) $(AIR_DATA) $(SRC)/plotting.py
	$(MKDIR) $(FIGS)
	./$^ $@

$(FIGS)/figure-coinfection-probability.pdf: $(SRC)/coinfection-figures.py $(MCMC_OUT) $(DUAL_DONOR_DATA) $(SRC)/plotting.py
	$(MKDIR) $(FIGS)
	./$^ $@

$(FIGS)/figure-expected-coinfections.pdf: $(SRC)/coinfection-figures.py $(MCMC_OUT) $(DUAL_DONOR_DATA) $(SRC)/plotting.py
	$(MKDIR) $(FIGS)
	./$^ $@

$(FIGS)/figure-infection-probability-by-variant.pdf: $(SRC)/coinfection-figures.py $(MCMC_OUT) $(DUAL_DONOR_DATA) $(SRC)/plotting.py
	$(MKDIR) $(FIGS)
	./$^ $@

$(FIGS)/figure-plaques-per-rna.pdf: $(SRC)/figure-plaques-per-rna.py $(AIR_DATA) $(SRC)/plotting.py
	$(MKDIR) $(FIGS)
	./$^ $@


$(ENTRY_DATA): $(SRC)/clean_entry.py $(RAW)/entry-data.xlsx
	$(MKDIR) $(CLEAN)
	./$^ $@

$(DUAL_DONOR_DATA): $(SRC)/clean_dual_donor.py $(DUAL_DONOR_FILE) $(METADATA_FILE)
	$(MKDIR) $(CLEAN)
	./$^ $@

$(SWAB_DATA): $(SRC)/clean_shedding.py $(RAW)/pleth.xlsx $(RAW)/shedding-RNA.xlsx $(RAW)/shedding-virus.xlsx $(RAW)/hamster-metadata.xlsx $(RAW)/cage-plaque.csv $(RAW)/cage-assignments.csv
	$(MKDIR) $(CLEAN)
	./$^ $@

$(PLETH_DATA): $(SRC)/clean_shedding.py $(RAW)/pleth.xlsx $(RAW)/shedding-RNA.xlsx $(RAW)/shedding-virus.xlsx $(RAW)/hamster-metadata.xlsx $(RAW)/cage-plaque.csv $(RAW)/cage-assignments.csv
	$(MKDIR) $(CLEAN)
	./$^ $@

$(AIR_DATA): $(SRC)/clean_shedding.py $(RAW)/pleth.xlsx $(RAW)/shedding-RNA.xlsx $(RAW)/shedding-virus.xlsx $(RAW)/hamster-metadata.xlsx $(RAW)/cage-plaque.csv $(RAW)/cage-assignments.csv
	$(MKDIR) $(CLEAN)
	./$^ $@

$(MACROS): $(SRC)/priors.py $(DATA)/templates/parameters.sty.tmpl
	$(MKDIR) $(MS)
	./$^ $@


all: install $(PRIOR_CHECK_OUT_MAIN) $(MCMC_OUT) $(MCMC_SIMPLE) $(FIGURE_PATHS) $(MACROS)


#####################################
# Convenience shortcuts

.PHONY: all 

.PHONY: checks
checks: $(PRIOR_CHECK_OUT_MAIN)

.PHONY: chains
chains: checks $(MCMC_OUT) $(MCMC_SIMPLE)

.PHONY: test

test:
	pytest src

.PHONY: figures
figures: $(FIGURE_PATHS)

.PHONY: deltemp
deltemp:
	$(RM) src/*#
	$(RM) src/*~

.PHONY: install
install: requirements.txt
	pip install -r $^

.PHONY: clean
clean: deltemp
	$(RM) dat/cleaned/*
	rmdir dat/cleaned
	$(RM) out/*
	rmdir out
