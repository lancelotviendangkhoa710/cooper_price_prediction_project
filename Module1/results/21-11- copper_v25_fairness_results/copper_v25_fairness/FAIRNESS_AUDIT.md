# Fairness coverage after V25

- **FS06 WTI Expanding WF, 542 matched test dates, 5 neural seeds**: all 12 fitted model families if and only if V23 ZIP is supplied and `wf_all_models_ranking.csv` was generated; Ridge is deterministic and fit only once/fold.
- **Walk-forward FS06 under Rolling/Recency**: incomplete for most architectures. Optional targeted audit 3 models × 3 strategies × 1 seed is not comprehensive.
- **Walk-forward FS02/FS04**: incomplete for most architectures on matched 2024–26 dates.
- **SA search protocol**: SA HP found on a prior validation split; other architecture HP fixed, so tuning budgets aren't equal.
- **Train/validation/test**: within folds, scaler fitted using subtrain and hyperparameters held fixed, but no untouched final time period following extensive test comparisons.
- **Future-data availability**: same-date commodity alignment across markets requires exchange-time/as-of audit.
- **Statistical inference**: DM-HAC per seed uses paired daily absolute losses; multiple models/seeds are not independent, and multiplicity/selection bias should be considered.
- **Conclusion**: rank descriptively within observed settings only. Don't claim universally best model or comprehensive methodological fairness.
