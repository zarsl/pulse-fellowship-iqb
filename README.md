# Infrastructure Heterogeneity and the Stability of Internet Quality Scores

Research notebooks for the Pulse project *"Infrastructure Heterogeneity and the Stability of Internet Quality Scores"* (Arslan, 2026). The project builds on the IQB Phase 2 report (Sermpezis & Arslan, arXiv:2606.11040) and asks how far a country's [Internet Quality Barometer (IQB)](https://github.com/m-lab/iqb) score can be trusted: how many tests it needs, how much it moves with the chosen percentile and thresholds, and whether it tracks other indicators.

All analysis uses M-Lab NDT speed-test data, aggregated by M-Lab to monthly percentile tables (P1, P5, P10, P25, P50, P75, P90, P95, P99 of download, upload, latency and packet loss), January 2024 – December 2025.

## Notebooks

`0-setup` must run first; the others can be read independently once the data is in place.

| Notebook | Question | Proposal |
|---|---|---|
| [0-setup](0-setup.ipynb) | Download the M-Lab percentile tables into `data/` | — |
| [1a-sample-size](1a-sample-size.ipynb) | How many tests are needed to display a country's IQB score with confidence? | RQ 1a |
| [1b-sensitivity](1b-sensitivity.ipynb) | Does percentile sensitivity vary by development context? | RQ 1b |
| [2a-thresholds](2a-thresholds.ipynb) | At which percentile does a country flip from meeting to missing a use case's thresholds? | RQ 2a |
| [2b-random-forest](2b-random-forest.ipynb) | Which metric drives the IQB score, and is each weighted as intended? | RQ 2b |
| [3-use-cases](3-use-cases.ipynb) | Do context-specific use-case profiles give more meaningful scores? | RQ 3 |
| [4a-urban-rural](4a-urban-rural.ipynb) | Are city-level scores more stable than national ones? | RQ 4a |
| [4b-correlations](4b-correlations.ipynb) | Do IQB scores track GDP, broadband penetration and market concentration? | RQ 4b |
| [5-map](5-map.ipynb) | Interactive country map of IQB scores with their confidence | — |


## Key idea: pass rates

IQB scores each network requirement as pass/fail at a chosen percentile. Because M-Lab's percentiles rank quality (P95 is the best 5% of tests for every metric), a requirement passes at percentile *q* exactly when more than 1 − *q* of tests meet its threshold. A handful of **pass rates** (the share of tests meeting each threshold) therefore determines the score at every percentile, and pass rates from different months or places combine as a test-weighted average.

M-Lab does not publish pass rates. The notebooks recover them from the nine reported percentiles by interpolating between them on a log scale.

## Setup

**Requirements:** Python 3.12 or later, and the IQB library from M-Lab (`mlab-iqb`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# IQB library (from a local clone of https://github.com/m-lab/iqb)
pip install -e path/to/iqb/library

pip install numpy pandas scipy matplotlib pyarrow requests pycountry scikit-learn jupyter
```

Then open `0-setup.ipynb` and run it. It downloads the country-level download and upload tables from M-Lab's IQB data manifest into `data/v1/`, skipping files that are already there.

## Repository layout

```
0-setup.ipynb … 5-map.ipynb   analysis notebooks (this structure)
data/v1/                       M-Lab percentile tables, one folder per month (not tracked by git)
outputs/tables/                tables written by the notebooks
```

## Data

- **M-Lab NDT** percentile tables via the [IQB data manifest](https://github.com/m-lab/iqb), country level, monthly. Each row is one country-month with test counts (`sample_count`) and the nine percentiles of each metric.
- **External indicators** (used in 1b and 4b): World Bank income groups and GDP, ITU broadband indicators, APNIC ISP market shares. <!-- TODO: confirm sources and add download instructions. -->

## Limitations

- Scores describe **tests, not users**: people who test often count many times, and M-Lab tables do not identify users.
- Pass rates are interpolated from nine percentiles, so values between reported percentiles are estimates.

## Citation

Sermpezis, P. & Arslan, Z. (2026). IQB Phase 2 report. arXiv:2606.11040.
