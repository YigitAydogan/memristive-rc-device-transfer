# What survives a device change in memristive reservoir computing?

Code, results and figure scripts for the manuscript

> Y. Aydoğan and E. Minayi, *What survives a device change in memristive reservoir computing? Readout transfer across
> device families, device-to-device spread and measurement-calibrated TiO<sub>x</sub> devices*, submitted to
> Neuromorphic Computing and Engineering.

The study asks which parts of a trained memristive reservoir computer can be reused when the physical device changes,
and what each reuse strategy requires. Transfer is simulated with six published compact memristor models and with
TiO<sub>x</sub> models calibrated to published pulse measurements, on 28 UCR time-series classification datasets and
the NARMA-10 and Mackey-Glass forecasting tasks.

All transfer results in this repository are simulations. The TiO<sub>x</sub> models are calibrated to published
measurements, but no task was run end to end on physical hardware.

## Repository layout

```
scripts/     simulation and analysis scripts, one per analysis, in the order they were run
results/     every table, configuration file and log produced by those scripts
analysis/    scripts that regenerate the manuscript figures and the statistics cited in the text
figures/     the manuscript figures as PDF and PNG
data/        instructions for obtaining the measured device data
```

## Scripts and what they produce

Each script in `scripts/` is self-contained: it holds every device model, the readout, the transfer methods and the
statistics, so it runs on its own in a single Python process or a single Colab cell. Later scripts read the outputs of
earlier ones, so run them in this order.

| Script | Content | Manuscript |
|---|---|---|
| `memristive_portability_v2.py` | Development experiments: 18 datasets, device draws from seeds 11, 22, 33; operating-point landscape, absolute performance, baselines | Table 5, figure 6a |
| `memristive_portability_v3_diagnostics.py` | Development diagnostics: probe maps, spread sweep, feature correlation, readout cosine | Figure 4a |
| `memristive_portability_v4_main_evaluation.py` | Main evaluation: 28 datasets, new device draws from seeds 44, 55, 66, the nine prespecified hypotheses H1 to H9 | Figures 2 to 5, table 6 |
| `memristive_portability_v5_operating_point.py` | Universal normalized operating point chosen on the development datasets | Figure 6 |
| `memristive_portability_v6_measured_validation.py` | TiO<sub>x</sub> calibration, 3-bit check, measured noise, TX as source and target | Figures 7 and 8a, 8b |
| `memristive_portability_v7_measured_pair.py` | Transfer between the models of the two TiO<sub>x</sub> measurement series | Figure 8c |
| `memristive_portability_v8_corrected_correspondence.py` | Repeatability-corrected correspondence across noise levels | Figure 9 |
| `memristive_portability_v9_generality.py` | Multi-device reservoirs, NARMA-10 and Mackey-Glass forecasting | Figure 10 |
| `memristive_portability_v10_additional_analyses.py` | Teacher-label data budget, source-accuracy analysis, robustness over all draws, decision rule, separate TiO<sub>x</sub> calibration | Figure 4b, tables 4, 8, 9 |
| `collect_results_bundle.py` | Collects tables, configurations and logs of all runs into one zip | |

The prespecified protocol of the main evaluation is in
`results/portability_v4_main_evaluation/config/protocol_v4.json`. It records the hypotheses, tests, correction, device
draws and unit of replication, with a timestamp of 2026-10-20T02:40:50Z and hash `98fd8d185f36`, written before the main
evaluation was carried out. All later analyses are exploratory, as stated in the manuscript.

### Running the scripts

The scripts were developed for Google Colab with Google Drive mounted, and by default write to
`/content/drive/MyDrive/memristive_device_generalization`. To run elsewhere, set the project root:

```bash
pip install -r requirements.txt
export MEMRISTIVE_PROJECT_ROOT=/path/to/work
python scripts/memristive_portability_v2.py
```

Each script creates `<project root>/<run name>/` with `tables/`, `config/`, `units/`, `figures/` and `logs/`. Work is
checkpointed per dataset and device draw, so an interrupted run resumes where it stopped. The UCR datasets are
downloaded automatically through `aeon` on first use. The TiO<sub>x</sub> scripts need the measured data described in
`data/README.md`. Optional environment variables:

* `PORTV2_SMOKE=1` runs a quick test on three synthetic datasets with one device draw.
* `PORTV2_DATASETS=Coffee,Trace` restricts a run to the named datasets.
* `PORTV10_SKIP_ROBUST=1` and `PORTV10_SKIP_PAIR=1` skip parts of the last script.

The full sequence took about ten hours on a standard Colab CPU runtime; the logs in `results/*/logs/run.log` give the
runtime of each script.

### Regenerating the figures and cited statistics

The scripts in `analysis/` read only `results/` and `data/`, so they reproduce the figures without rerunning the
simulations:

```bash
cd analysis
python fig_transfer.py      # figures 2 and 3
python fig_teacher.py       # figures 4 and 5
python fig_rest.py          # figures 6 and 7
python fig_more.py          # figures 1, 8, 9 and 10
python cited_stats.py       # statistics cited in the text, written to analysis/numbers.json
python additional_stats.py  # held-out breakdown, decision-rule errors, family-level bootstrap, written to analysis/review_numbers.json
python make_tables.py       # table bodies, written to analysis/tables/
```

Figures 1 and 7 need the measured TiO<sub>x</sub> files in `data/measured/`. Every figure script also writes
`figures/figure_audit.txt`, which checks axis bounds, data within bounds and font sizes.

## Results

`results/<run>/tables/` holds every table produced by each run, including per-unit results, where a unit is one
dataset, device draw and ordered device pair. `results/<run>/config/` holds the protocols, the device instances, the
universal operating point and the TiO<sub>x</sub> calibration parameters, and `results/<run>/logs/` the run logs. The
largest files are the per-unit tables of the main evaluation and of the additional analyses, about 30 to 40 MB each.

## Data

* UCR time-series classification archive, downloaded through `aeon` with the official training and test splits.
* TiO<sub>x</sub> pulse measurements of Wang et al., *J. Phys. D: Appl. Phys.* 59 015101 (2026), dataset
  [10.5258/SOTON/D3702](https://doi.org/10.5258/SOTON/D3702), licensed CC BY.
* Mesoporous silica memristor measurements of Jaafar et al., *Nanoscale* 14 17170 (2022), dataset
  [10.5258/SOTON/D2447](https://doi.org/10.5258/SOTON/D2447).

The measured data are not redistributed here; `data/README.md` explains where to place them.

## Use of generative AI

Claude by Anthropic and ChatGPT by OpenAI were used to assist with writing and debugging the code. All code and results
were reviewed and verified by the authors.

## Licence

The code is released under the MIT licence, see `LICENSE`. The result tables and figures are released under the
Creative Commons Attribution 4.0 licence, see `LICENSE-DATA.md`.

## Citation

If you use this code or these results, please cite the manuscript and this repository, see `CITATION.cff`.

## Contact

Yiğit Aydoğan, Department of Computer Science, Aberystwyth University, yia@aber.ac.uk
