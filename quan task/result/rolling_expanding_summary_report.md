# Báo cáo thực nghiệm Phương pháp 4: Rolling & Expanding Window Forecasting

- **Chế độ chạy:** `FULL_RESEARCH_MODE` | **Dữ liệu:** 5231 phiên (2006-01-03 -> 2026-10-29)
- **Chính sách Walk-Forward chọn từ Validation:** `W* = 504` phiên, `S* = 20` phiên, `update_mode* = warm_start`
- **Kiến trúc cơ sở cố định:** 2-layer LSTM ({'lstm_units_1': 9, 'lstm_units_2': 37, 'learning_rate': 0.00040267575055105107}), Lookback = 22, Horizon = 1

## 1. Bảng kết quả chính trên tập Test (FEATURES_4: Copper, WTI, Gold, Silver)

| Chiến lược | MAE ($/lb) | RMSE ($/lb) | MASE | R² | Số lần Retrain | Thời gian (s) |
|---|---:|---:|---:|---:|---:|---:|
| naive_persistence | 0.0693 | 0.1092 | 1.7593 | 0.9827 | 0 | 0.0 |
| static_train_only | 0.7563 | 0.9953 | 19.3730 | -0.4356 | 1 | 31.7 |
| static_train_val | 0.5393 | 0.7165 | 13.6871 | 0.2562 | 1 | 35.0 |
| expanding_window | 0.1172 | 0.1625 | 2.9759 | 0.9617 | 27 | 236.1 |
| rolling_window | 0.2259 | 0.2923 | 5.7325 | 0.8762 | 27 | 77.4 |

## 2. So sánh cải thiện tương đối (RMSE)
- Rolling Window vs. Static (Train-Only): **-70.63%**
- Rolling Window vs. Static (Train+Val): **-59.20%**
- Expanding Window vs. Static (Train-Only): **-83.67%**
- Rolling Window vs. Expanding Window: **+79.88%**

## 3. Kiểm định ý nghĩa thống kê Diebold-Mariano (DM Test)

| So sánh | Loss | DM Stat | p-value | Có ý nghĩa (5%) |
|---|---|---:|---:|:---:|
| Rolling vs Static (Train-Only) | L1 | -19.868 | 0.0000 | Có |
| Rolling vs Static (Train-Only) | L2 | -17.927 | 0.0000 | Có |
| Rolling vs Static (Train+Val) | L1 | -16.476 | 0.0000 | Có |
| Rolling vs Static (Train+Val) | L2 | -16.451 | 0.0000 | Có |
| Expanding vs Static (Train-Only) | L1 | -22.882 | 0.0000 | Có |
| Expanding vs Static (Train-Only) | L2 | -18.557 | 0.0000 | Có |
| Expanding vs Static (Train+Val) | L1 | -20.914 | 0.0000 | Có |
| Expanding vs Static (Train+Val) | L2 | -17.758 | 0.0000 | Có |
| Rolling vs Expanding | L1 | 16.195 | 0.0000 | Có |
| Rolling vs Expanding | L2 | 12.258 | 0.0000 | Có |
| Rolling vs Naive Persistence | L1 | 20.406 | 0.0000 | Có |
| Rolling vs Naive Persistence | L2 | 13.122 | 0.0000 | Có |
| Rolling (4 Features) vs Rolling (3 Features) | L1 | -11.348 | 0.0000 | Có |
| Rolling (4 Features) vs Rolling (3 Features) | L2 | -10.066 | 0.0000 | Có |