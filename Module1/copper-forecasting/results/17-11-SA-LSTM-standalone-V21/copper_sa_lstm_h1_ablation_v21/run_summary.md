# Copper Forecasting SA-LSTM Comparison — Horizon 1 Only
Generated: 2026-11-18 13:51:29
MODE: FULL_MODE | HORIZON: 1 | LOOKBACK: 22
Feature variants: paper3_with_copper, paper3_without_copper, legacy14_with_copper, legacy14_without_copper
Models: Naive, baseline LSTM, baseline RNN, Paper SA-LSTM, Prior-SA Reoptimized LSTM
Strategies: expanding, rolling_756, recency_weighted
Fixed HP paper: {'units_1': 39, 'units_2': 111, 'learning_rate': 5e-05}
Fixed HP prior SA (searched WITH Copper): {'paper3': {'units_1': 22, 'units_2': 91, 'learning_rate': 0.000922}, 'legacy14': {'units_1': 22, 'units_2': 83, 'learning_rate': 0.000655}}
Important: Without-Copper runs TRANSFER existing SA HP, and do NOT re-optimize SA.
Important: Paper configuration is partially reconstructed (Dense(32) is an implementation assumption).
Important: Values are HOLDOUT comparisons, NOT walk-forward retraining.
Samples/rows: 100
Runtime minutes: 36.2