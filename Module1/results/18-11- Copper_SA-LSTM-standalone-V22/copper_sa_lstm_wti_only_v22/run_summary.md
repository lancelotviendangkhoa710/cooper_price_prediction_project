# Copper forecasting — V22 fresh SA, WTI-only
Generated: 2026-11-18 17:29:44
MODE: FULL_MODE | HORIZON: 1 | LOOKBACK: 22
Dataset SHA256: 9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600
Feature sets: {'wti_replacement_with_copper': ['Copper', 'Aluminium', 'Tin', 'WTI']}
Models: Naive, LSTM, SimpleRNN, Paper-HP LSTM, Fresh SA tuned LSTM
Strategies: ('expanding', 'rolling_756', 'recency_weighted')
Fixed paper HP: {'units_1': 39, 'units_2': 111, 'learning_rate': 5e-05}
Fresh SA tuned separately for every feature variant: {'wti_replacement_with_copper': {'units_1': 47, 'units_2': 127, 'learning_rate': 5.0478806754686805e-05}}
SA budget/variant: 16 x 15 max epochs
HP search uses earliest 80/20 Expanding TRAIN/VALIDATION only; test was not consulted.
Spearman ranking of PRICE LEVELS motivated hypothesis, not evidence of predictive value.
No legacy/Paper3 retraining. Compare with historical results in the later report.
Paper configuration is partially reconstructed (Dense(32) is a study assumption).
Results are chronological holdout, NOT retrained walk-forward.
Experiment jobs: 30
Runtime minutes: 11.6