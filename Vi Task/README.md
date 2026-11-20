# Vi Task — iTransformer vs SA-LSTM

This folder contains the executed Colab notebook and the rendered outputs available inside that notebook.

## Contents

- `iTransformer_vs_SA_LSTM_K5_Fair_FINAL_Colab.ipynb` — executed notebook, including its saved cell outputs.
- `outputs/figures/` — the three PNG figures embedded in the notebook: actual/predicted/naive prices, Copper changes, and MAE comparison.
- `outputs/tables/` — HTML exports of the seven tables displayed by the notebook.
- `outputs/execution_log.txt` — captured cell output logs, including the run completion message.

## Run record

The notebook records a `FULL_MODE` run with **84/84 jobs complete** and `missing_jobs=[]`. It used 2,710 rows from 2016-01-13 through 2026-11-13, five inputs (Aluminium, Tin, Silver, Gold, and historical Copper only), lookback 22, seed 42, and TensorFlow 2.21.0 on a Colab GPU.

The notebook reports that the full CSV/JSON results and ZIP archive were written to the user's Google Drive under `Copper_iTransformer_vs_SA_LSTM_K5_Fair/itransformer_vs_sa_lstm/FULL_MODE_K5_lb22_s42_dataa761a908_cfgd1b751120e44/`. Those Drive files were not present in `E:\Task\iTransformer`, which contained only the notebook, so this folder exports the figures, rendered tables, and logs embedded in that notebook. The HTML tables preserve the rendered notebook tables; they are not a replacement for the complete result CSV files.

## Reproducibility note

The notebook expects `dataset_9_commodities_daily.csv` in Colab/Drive or an uploaded CSV. Add that dataset as a separate GitHub/Kaggle input if collaborators need to rerun training; it is not copied into this folder.
