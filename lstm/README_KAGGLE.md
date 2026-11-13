# Copper LSTM–SA research on Kaggle

## Files

- `Copper_Research_Kaggle.ipynb`: phase controller.
- `research.py`: data, protocol, LSTM, SA, TPE, logging and report code.
- `experiment_config.json`: central settings and budgets.
- `audit_report.md`: comparison with the original paper and old notebooks.
- `tests/test_protocol.py`: protocol checks.

The old `Copper_LSTM_SA.ipynb` and `copper-lstm-sa_result_lasted.ipynb` remain intact. Their stored results are historical evidence; the new notebook does not claim to have rerun them.

## Kaggle setup

1. Create a Kaggle Dataset with `research.py` and `experiment_config.json`. Attach it to `Copper_Research_Kaggle.ipynb`. Enable **GPU T4** and **Internet** if downloading Yahoo data.
2. Run `QUICK_TEST` with `PHASES = ["phase1", "phase2", "phase3", "phase4", "phase5"]` to check the pipeline. This uses 800 recent rows, 1–2 epochs, and two search steps. Its numbers are diagnostic only.
3. Change `MODE` to `FULL_RESEARCH`. Run `phase1` first. Inspect `/kaggle/working/full_research/data_quality_report.csv`, `sanity_check.json`, `benchmark.json`, and `copper_data_snapshot.csv`.
4. Run `phase2` to screen output activation, lookback and feature rule with one seed/split. The paper preset uses 39/111 and 5e-5; the baseline uses 32/32 and .001. Screening uses 15 epochs, so its MSE is not the 100 epoch final result.
5. Run `phase3`. It writes a measured pilot projection to `compute_estimate.json`. If the projection exceeds four hours, set `ALLOW_LARGE = True` after reviewing it. FULL mode runs one SA trajectory to 300 steps, a fixed LR SA trajectory to 300 steps, and TPE with 301 model evaluations. Prefixes at 30, 100 and 300 are saved. This can take many GPU hours; the 8/8-unit pilot is only a rough estimate of larger searched models.
6. Run `phase4` for five training seeds, five **30-step** SA search seeds, three train/test ratios and three expanding validation folds. All these use validation; they do not read test labels for selection.
7. Run `phase5` once. It chooses the champion by mean primary-split validation MSE across seeds, freezes `final_selection.json`, and evaluates persistence, vanilla LSTM, paper configuration rerun, SA and TPE on each test period. SA/TPE hyperparameters were optimized on the primary 90% split and are transferred to the other ratios. `summary_report.md` is generated from actual CSV records. It does not contain invented scores.

To use an existing frozen dataset instead of downloading Yahoo, set `data.csv` in `experiment_config.json` to a CSV with `Date`, `Copper`, and all eight candidate columns. Reattach the edited configuration. A run reuses its own `copper_data_snapshot.csv` on later phases, so Yahoo revisions cannot silently change the sample. Use a fresh output directory if changing data or config. QUICK and FULL outputs are separated under `/kaggle/working/`.

## Resuming and artifacts

SA stores its current state, best state and RNG state after every proposal. TPE uses `optuna.db`. Screening and final runs skip completed experiment IDs. Preserve the **whole** `/kaggle/working/full_research/` folder between fresh Kaggle sessions by saving it as a Dataset and restoring it before rerunning a phase. The manifest refuses a different data/config fingerprint in the same output directory.

Main artifacts: `experiment_results.csv`, `sa_search_history.csv`, `optuna_trials.csv`, `best_hyperparameters.json`, `data_quality_report.csv`, `experiment_manifest.json`, `summary_report.md`, `seed_summary.csv`, prediction CSVs and PNG charts. `validation_mse` and `test_mse` are on the MinMax target scale; `mse_price2` is on the original copper price scale. The original paper's MSE is reported separately in the audit, never treated as the same benchmark.

## Local protocol tests

Run `python -m unittest discover -s lstm/tests -v` from the repository root. The optional prediction test runs if TensorFlow is installed. These tests use synthetic dates and values and do not train a research model.

## Known limits

The task predicts the next **joint trading session** after the final input close. WTI, Gold and Silver are taken from the previous joint date. Actual market close/publication times may differ, so the forecast decision time must be after all input closes. The Yahoo continuous futures series is a proxy for the paper's Investing.com history. One validation block can still favor a noisy hyperparameter choice; the expanding folds and seeds describe that sensitivity. Test periods differ across 90/10, 92.5/7.5 and 95/5; their MSE values should not be read as an isolated effect of train fraction.
