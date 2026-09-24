# Long COVID Paxlovid analysis repository

This repository contains the analysis code for the Long COVID follow-up study in the CanTreatCOVID Paxlovid arm.

## Scope

- baseline summaries and randomization checks
- symptom analyses at Day 90 and Week 36
- long COVID incidence and symptom-burden modeling
- plots for adjusted associations and CONSORT-style summaries

## Files

- `paxlovid_baseline.py` — baseline/demographic summaries
- `longcovid_symptoms.py` — symptom-level outcome extraction and tests
- `longcovid_secondary_analysis.py` — secondary regression and outcome modeling
- `longcovid_plots.py` — plot generation (forest plot, bar plots, etc.)
- `longcovid_repo_utils.py` — shared helper functions kept locally for portability

## Setup

```bash
cd /workspaces/CTC_covid/py_src/longcovid_ctc_pax
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the scripts

From the repository directory:

```bash
python paxlovid_baseline.py
python longcovid_symptoms.py
python longcovid_secondary_analysis.py
```

## Notes

This repo was deliberately made self-contained so the analysis can be run without importing helper functions from the external antioxidant analysis folder. Shared logic was consolidated into `longcovid_repo_utils.py`.
