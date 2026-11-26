# Báo cáo thực nghiệm Phương pháp 4 (Full Parity): Rolling & Expanding Window Forecasting

- **Chế độ chạy:** `FULL_RESEARCH_MODE` | **Dữ liệu:** 2710 phiên (2016-01-13 -> 2026-11-13)
- **Tỷ lệ chia:** `[0.8, 0.9, 0.925, 0.95]` (Chính: `PRIMARY_RATIO = 0.9`)
- **Chính sách Walk-Forward chọn từ Validation:** `W* = 756`, `S* = 40`, `update_mode* = cold_start`
- **Kiến trúc chuẩn số đông (`PAPER_CONFIG`):** 2-layer LSTM ({'lstm_units_1': 39, 'lstm_units_2': 111, 'learning_rate': 5e-05}), Lookback = 22, Horizon = 1

## 1. Bảng chỉ số chính trên tập Test (FEATURES_4 Direct vs. Residual)

| Mode | Bộ biến | Chiến lược | MAE ($/lb) | RMSE ($/lb) | MAPE (%) | MASE | R² | DA (%) | ΔMAE vs Naive |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| direct | FS01_CU | static_train_only | 0.4928 | 0.5342 | 7.94% | 14.3635 | -0.1782 | 46.84% | +539.36% |
| direct | FS01_CU | static_train_val | 0.3355 | 0.3776 | 5.40% | 9.0626 | 0.4112 | 47.58% | +335.34% |
| direct | FS01_CU | expanding_window | 0.2026 | 0.2595 | 3.38% | 5.4732 | 0.7219 | 53.90% | +162.92% |
| direct | FS01_CU | rolling_window | 0.2739 | 0.3337 | 4.56% | 7.3987 | 0.5402 | 50.56% | +255.42% |
| direct | FEATURES_4 | naive_persistence | 0.0771 | 0.1010 | 1.28% | 2.0817 | 0.9579 | — | +0.00% |
| direct | FEATURES_4 | static_train_only | 1.5516 | 1.6156 | 25.53% | 45.2235 | -9.7766 | 46.84% | +1913.04% |
| direct | FEATURES_4 | static_train_val | 0.2869 | 0.3308 | 4.66% | 7.7477 | 0.5481 | 48.33% | +272.18% |
| direct | FEATURES_4 | expanding_window | 0.2370 | 0.2943 | 3.84% | 6.3999 | 0.6423 | 51.30% | +207.43% |
| direct | FEATURES_4 | rolling_window | 0.2837 | 0.3404 | 4.54% | 7.6626 | 0.5215 | 47.21% | +268.09% |
| direct | FEATURES_3 | static_train_only | 0.3199 | 0.3853 | 5.29% | 9.3241 | 0.3871 | 52.79% | +315.05% |
| direct | FEATURES_3 | static_train_val | 0.3663 | 0.4842 | 6.09% | 9.8923 | 0.0318 | 55.02% | +375.20% |
| direct | FEATURES_3 | expanding_window | 0.3861 | 0.4770 | 6.22% | 10.4269 | 0.0604 | 51.30% | +400.88% |
| direct | FEATURES_3 | rolling_window | 0.4608 | 0.5522 | 7.31% | 12.4441 | -0.2592 | 48.33% | +497.78% |
| direct | FS04_METALS_CU | static_train_only | 1.3227 | 1.3893 | 21.56% | 38.5519 | -6.9697 | 46.84% | +1616.07% |
| direct | FS04_METALS_CU | static_train_val | 0.7246 | 0.7804 | 11.64% | 19.5695 | -1.5143 | 47.21% | +840.07% |
| direct | FS04_METALS_CU | expanding_window | 0.3384 | 0.3829 | 5.45% | 9.1407 | 0.3948 | 50.19% | +339.10% |
| direct | FS04_METALS_CU | rolling_window | 0.3497 | 0.3999 | 5.60% | 9.4455 | 0.3396 | 50.19% | +353.74% |
| direct | FS05_METALS_EXO | static_train_only | 0.5638 | 0.6678 | 8.91% | 16.4321 | -0.8415 | 48.70% | +631.44% |
| direct | FS05_METALS_EXO | static_train_val | 0.5762 | 0.6902 | 9.08% | 15.5630 | -0.9670 | 46.84% | +647.61% |
| direct | FS05_METALS_EXO | expanding_window | 0.3618 | 0.4315 | 5.75% | 9.7721 | 0.2312 | 48.33% | +369.43% |
| direct | FS05_METALS_EXO | rolling_window | 0.4141 | 0.4737 | 6.64% | 11.1841 | 0.0736 | 49.44% | +437.26% |
| direct | FS06_WTI_CU | static_train_only | 0.5585 | 0.6267 | 8.89% | 16.2772 | -0.6218 | 48.33% | +624.55% |
| direct | FS06_WTI_CU | static_train_val | 0.5257 | 0.5891 | 8.37% | 14.1992 | -0.4331 | 47.21% | +582.09% |
| direct | FS06_WTI_CU | expanding_window | 0.2192 | 0.2721 | 3.57% | 5.9205 | 0.6942 | 49.81% | +184.41% |
| direct | FS06_WTI_CU | rolling_window | 0.2742 | 0.3282 | 4.40% | 7.4062 | 0.5553 | 48.70% | +255.77% |
| residual | FS01_CU | static_train_only | 0.0788 | 0.1021 | 1.30% | 2.2962 | 0.9570 | 46.84% | +2.21% |
| residual | FS01_CU | static_train_val | 0.0778 | 0.1013 | 1.29% | 2.1019 | 0.9576 | 46.47% | +0.97% |
| residual | FS01_CU | expanding_window | 0.0774 | 0.1015 | 1.28% | 2.0893 | 0.9575 | 50.56% | +0.37% |
| residual | FS01_CU | rolling_window | 0.0773 | 0.1017 | 1.28% | 2.0883 | 0.9573 | 50.93% | +0.31% |
| residual | FEATURES_4 | static_train_only | 0.1097 | 0.1358 | 1.80% | 3.1964 | 0.9238 | 46.84% | +42.28% |
| residual | FEATURES_4 | static_train_val | 0.0781 | 0.1026 | 1.29% | 2.1099 | 0.9566 | 48.33% | +1.35% |
| residual | FEATURES_4 | expanding_window | 0.0785 | 0.1022 | 1.30% | 2.1206 | 0.9568 | 47.21% | +1.87% |
| residual | FEATURES_4 | rolling_window | 0.0803 | 0.1038 | 1.33% | 2.1676 | 0.9555 | 47.58% | +4.13% |
| residual | FEATURES_3 | static_train_only | 0.0882 | 0.1143 | 1.47% | 2.5707 | 0.9461 | 53.16% | +14.43% |
| residual | FEATURES_3 | static_train_val | 0.0800 | 0.1060 | 1.33% | 2.1602 | 0.9536 | 52.79% | +3.77% |
| residual | FEATURES_3 | expanding_window | 0.0769 | 0.1011 | 1.28% | 2.0777 | 0.9578 | 50.93% | -0.19% |
| residual | FEATURES_3 | rolling_window | 0.0774 | 0.1017 | 1.28% | 2.0893 | 0.9573 | 52.04% | +0.36% |
| residual | FS04_METALS_CU | static_train_only | 0.0834 | 0.1056 | 1.37% | 2.4301 | 0.9540 | 46.47% | +8.17% |
| residual | FS04_METALS_CU | static_train_val | 0.0775 | 0.1007 | 1.28% | 2.0921 | 0.9581 | 48.33% | +0.50% |
| residual | FS04_METALS_CU | expanding_window | 0.0777 | 0.1023 | 1.29% | 2.0983 | 0.9568 | 48.70% | +0.80% |
| residual | FS04_METALS_CU | rolling_window | 0.0791 | 0.1029 | 1.31% | 2.1369 | 0.9563 | 52.04% | +2.65% |
| residual | FS05_METALS_EXO | static_train_only | 0.0794 | 0.1054 | 1.32% | 2.3149 | 0.9542 | 54.28% | +3.04% |
| residual | FS05_METALS_EXO | static_train_val | 0.0806 | 0.1071 | 1.34% | 2.1773 | 0.9526 | 53.90% | +4.59% |
| residual | FS05_METALS_EXO | expanding_window | 0.0772 | 0.1015 | 1.28% | 2.0850 | 0.9574 | 51.67% | +0.16% |
| residual | FS05_METALS_EXO | rolling_window | 0.0779 | 0.1024 | 1.29% | 2.1033 | 0.9567 | 48.70% | +1.04% |
| residual | FS06_WTI_CU | static_train_only | 0.0794 | 0.1024 | 1.31% | 2.3134 | 0.9567 | 46.84% | +2.98% |
| residual | FS06_WTI_CU | static_train_val | 0.0812 | 0.1036 | 1.34% | 2.1932 | 0.9557 | 46.84% | +5.35% |
| residual | FS06_WTI_CU | expanding_window | 0.0776 | 0.1015 | 1.29% | 2.0969 | 0.9575 | 50.56% | +0.73% |
| residual | FS06_WTI_CU | rolling_window | 0.0792 | 0.1030 | 1.31% | 2.1397 | 0.9562 | 49.07% | +2.78% |

## 2. Độ ổn định qua 5 Seeds ([41, 42, 43, 44, 45], Expanding Window, FEATURES_4)

| Mode | Seed | MAE | RMSE | DA (%) | ΔMAE vs Naive | DM(L1) p-value |
|---|---:|---:|---:|---:|---:|---:|
| direct | 41 | 0.29207 | 0.35185 | 49.81% | +278.94% | 0.0000 |
| direct | 42 | 0.23696 | 0.29435 | 51.30% | +207.43% | 0.0000 |
| direct | 43 | 0.23867 | 0.28423 | 50.19% | +209.65% | 0.0000 |
| direct | 44 | 0.29922 | 0.34450 | 49.44% | +288.21% | 0.0000 |
| direct | 45 | 0.32497 | 0.37067 | 49.44% | +321.62% | 0.0000 |
| residual | 41 | 0.07742 | 0.10136 | 50.93% | +0.45% | 0.6734 |
| residual | 42 | 0.07852 | 0.10224 | 47.21% | +1.87% | 0.0583 |
| residual | 43 | 0.07913 | 0.10326 | 49.07% | +2.67% | 0.0420 |
| residual | 44 | 0.07798 | 0.10190 | 49.07% | +1.17% | 0.2185 |
| residual | 45 | 0.07834 | 0.10231 | 47.96% | +1.64% | 0.2138 |


## 2B. Độ ổn định qua 5 Seeds mở rộng ([41, 42, 43, 44, 45], Rolling FS02 & Expanding FS03, Phase 2C)

| Bộ biến | Chiến lược | Mode | Seed | MAE | RMSE | DA (%) | ΔMAE vs Naive | DM(L1) p-value |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FEATURES_4 | rolling_window | direct | 41 | 0.31027 | 0.37492 | 49.07% | +302.55% | 0.0000 |
| FEATURES_4 | rolling_window | direct | 42 | 0.28371 | 0.34043 | 47.21% | +268.09% | 0.0000 |
| FEATURES_4 | rolling_window | direct | 43 | 0.29525 | 0.34960 | 46.47% | +283.05% | 0.0000 |
| FEATURES_4 | rolling_window | direct | 44 | 0.37650 | 0.43297 | 48.33% | +388.48% | 0.0000 |
| FEATURES_4 | rolling_window | direct | 45 | 0.36138 | 0.41878 | 49.81% | +368.85% | 0.0000 |
| FEATURES_4 | rolling_window | residual | 41 | 0.07862 | 0.10211 | 47.58% | +2.01% | 0.1015 |
| FEATURES_4 | rolling_window | residual | 42 | 0.08026 | 0.10382 | 47.58% | +4.13% | 0.0071 |
| FEATURES_4 | rolling_window | residual | 43 | 0.07936 | 0.10279 | 49.44% | +2.96% | 0.0751 |
| FEATURES_4 | rolling_window | residual | 44 | 0.07766 | 0.10153 | 52.42% | +0.76% | 0.4535 |
| FEATURES_4 | rolling_window | residual | 45 | 0.07728 | 0.10089 | 51.67% | +0.27% | 0.7668 |
| FEATURES_3 | expanding_window | residual | 41 | 0.07967 | 0.10491 | 50.56% | +3.36% | 0.0969 |
| FEATURES_3 | expanding_window | residual | 42 | 0.07693 | 0.10107 | 50.93% | -0.19% | 0.7862 |
| FEATURES_3 | expanding_window | residual | 43 | 0.07760 | 0.10156 | 48.33% | +0.68% | 0.3970 |
| FEATURES_3 | expanding_window | residual | 44 | 0.07839 | 0.10381 | 51.67% | +1.71% | 0.3775 |
| FEATURES_3 | expanding_window | residual | 45 | 0.07691 | 0.10137 | 51.30% | -0.22% | 0.8433 |

### Tổng hợp Mean ± Std qua 5 Seeds (5 cấu hình Phase 2B + 2C)

| Bộ biến | Chiến lược | Mode | Mean MAE ± Std | Mean RMSE | Mean DA (%) | Mean ΔMAE vs Naive | Số seed DM p<0.05 |
|---|---|---|---:|---:|---:|---:|---:|
| FEATURES_3 | expanding_window | residual | 0.07790 ± 0.00116 | 0.10254 | 50.56% | +1.07% | 0/5 |
| FEATURES_4 | expanding_window | direct | 0.27838 ± 0.03900 | 0.32912 | 50.04% | +261.17% | 5/5 |
| FEATURES_4 | expanding_window | residual | 0.07828 ± 0.00063 | 0.10221 | 48.85% | +1.56% | 1/5 |
| FEATURES_4 | rolling_window | direct | 0.32542 ± 0.04117 | 0.38334 | 48.18% | +322.21% | 5/5 |
| FEATURES_4 | rolling_window | residual | 0.07864 ± 0.00122 | 0.10223 | 49.74% | +2.02% | 1/5 |
## 3. Độ bền vững qua 4 mốc chia thời gian [0.8, 0.9, 0.925, 0.95] (FEATURES_4)

| Mode | Ratio | n_test | Chiến lược | MAE ($/lb) | RMSE ($/lb) | R² | DA (%) |
|---|---|---:|---|---:|---:|---:|---:|
| direct | 0.800 | 542 | naive_persistence | 0.0690 | 0.1080 | 0.9833 | — |
| direct | 0.800 | 542 | static_train_only | 1.0993 | 1.3312 | -1.5368 | 46.85% |
| direct | 0.800 | 542 | static_train_val | 1.0210 | 1.2836 | -1.3588 | 48.15% |
| direct | 0.800 | 542 | expanding_window | 0.2592 | 0.3369 | 0.8375 | 51.48% |
| direct | 0.800 | 542 | rolling_window | 0.2745 | 0.3488 | 0.8258 | 49.07% |
| direct | 0.900 | 271 | naive_persistence | 0.0771 | 0.1010 | 0.9579 | — |
| direct | 0.900 | 271 | static_train_only | 1.5516 | 1.6156 | -9.7766 | 46.84% |
| direct | 0.900 | 271 | static_train_val | 0.2869 | 0.3308 | 0.5481 | 48.33% |
| direct | 0.900 | 271 | expanding_window | 0.2370 | 0.2943 | 0.6423 | 51.30% |
| direct | 0.900 | 271 | rolling_window | 0.2837 | 0.3404 | 0.5215 | 47.21% |
| direct | 0.925 | 204 | naive_persistence | 0.0751 | 0.0984 | 0.9238 | — |
| direct | 0.925 | 204 | static_train_only | 1.5157 | 1.5739 | -18.4912 | 47.03% |
| direct | 0.925 | 204 | static_train_val | 0.3644 | 0.4134 | -0.3449 | 49.01% |
| direct | 0.925 | 204 | expanding_window | 0.2770 | 0.3351 | 0.1167 | 51.98% |
| direct | 0.925 | 204 | rolling_window | 0.4018 | 0.4653 | -0.7038 | 48.51% |
| direct | 0.950 | 136 | naive_persistence | 0.0728 | 0.0966 | 0.7359 | — |
| direct | 0.950 | 136 | static_train_only | 1.2845 | 1.3466 | -50.3169 | 45.19% |
| direct | 0.950 | 136 | static_train_val | 0.2559 | 0.2915 | -1.4045 | 46.67% |
| direct | 0.950 | 136 | expanding_window | 0.3444 | 0.3802 | -3.0899 | 45.93% |
| direct | 0.950 | 136 | rolling_window | 0.4493 | 0.4837 | -5.6202 | 45.19% |
| residual | 0.800 | 542 | static_train_only | 0.0767 | 0.1152 | 0.9810 | 53.70% |
| residual | 0.800 | 542 | static_train_val | 0.0750 | 0.1147 | 0.9812 | 49.81% |
| residual | 0.800 | 542 | expanding_window | 0.0711 | 0.1108 | 0.9824 | 51.30% |
| residual | 0.800 | 542 | rolling_window | 0.0734 | 0.1125 | 0.9819 | 51.48% |
| residual | 0.900 | 271 | static_train_only | 0.1097 | 0.1358 | 0.9238 | 46.84% |
| residual | 0.900 | 271 | static_train_val | 0.0781 | 0.1026 | 0.9566 | 48.33% |
| residual | 0.900 | 271 | expanding_window | 0.0785 | 0.1022 | 0.9568 | 47.21% |
| residual | 0.900 | 271 | rolling_window | 0.0803 | 0.1038 | 0.9555 | 47.58% |
| residual | 0.925 | 204 | static_train_only | 0.0959 | 0.1182 | 0.8901 | 47.03% |
| residual | 0.925 | 204 | static_train_val | 0.0768 | 0.1007 | 0.9202 | 52.97% |
| residual | 0.925 | 204 | expanding_window | 0.0771 | 0.1014 | 0.9192 | 45.54% |
| residual | 0.925 | 204 | rolling_window | 0.0803 | 0.1036 | 0.9156 | 45.05% |
| residual | 0.950 | 136 | static_train_only | 0.1017 | 0.1206 | 0.5885 | 45.19% |
| residual | 0.950 | 136 | static_train_val | 0.0721 | 0.0965 | 0.7367 | 54.81% |
| residual | 0.950 | 136 | expanding_window | 0.0757 | 0.0990 | 0.7224 | 51.85% |
| residual | 0.950 | 136 | rolling_window | 0.0786 | 0.1002 | 0.7158 | 47.41% |

## 4. Kiểm định ý nghĩa thống kê Diebold-Mariano (DM Test - HLN 1997)

| So sánh | Loss | DM Stat | p-value | Có ý nghĩa (5%) |
|---|---|---:|---:|:---:|
| Rolling vs Static (Train-Only) | L1 | -45.521 | 0.0000 | Có |
| Rolling vs Static (Train-Only) | L2 | -29.198 | 0.0000 | Có |
| Rolling vs Static (Train+Val) | L1 | -0.304 | 0.7612 | Không |
| Rolling vs Static (Train+Val) | L2 | 0.927 | 0.3550 | Không |
| Expanding vs Static (Train-Only) | L1 | -51.359 | 0.0000 | Có |
| Expanding vs Static (Train-Only) | L2 | -30.092 | 0.0000 | Có |
| Expanding vs Static (Train+Val) | L1 | -5.755 | 0.0000 | Có |
| Expanding vs Static (Train+Val) | L2 | -4.153 | 0.0000 | Có |
| Rolling vs Expanding | L1 | 6.632 | 0.0000 | Có |
| Rolling vs Expanding | L2 | 6.277 | 0.0000 | Có |
| Rolling vs Naive Persistence | L1 | 17.000 | 0.0000 | Có |
| Rolling vs Naive Persistence | L2 | 12.903 | 0.0000 | Có |
| Expanding vs Naive Persistence | L1 | 14.277 | 0.0000 | Có |
| Expanding vs Naive Persistence | L2 | 10.335 | 0.0000 | Có |
| Rolling (FS02 Paper3+Cu) vs Rolling (FS03 Paper3-Cu) | L1 | -15.819 | 0.0000 | Có |
| Rolling (FS02 Paper3+Cu) vs Rolling (FS03 Paper3-Cu) | L2 | -12.910 | 0.0000 | Có |
| Expanding (FS02 Paper3+Cu) vs Expanding (FS03 Paper3-Cu) | L1 | -7.681 | 0.0000 | Có |
| Expanding (FS02 Paper3+Cu) vs Expanding (FS03 Paper3-Cu) | L2 | -8.150 | 0.0000 | Có |
| Expanding (FS04 5 Metals) vs Expanding (FS02 Paper3+Cu) | L1 | 10.827 | 0.0000 | Có |
| Expanding (FS04 5 Metals) vs Expanding (FS02 Paper3+Cu) | L2 | 9.059 | 0.0000 | Có |
| Rolling (FS04 5 Metals) vs Rolling (FS02 Paper3+Cu) | L1 | 8.345 | 0.0000 | Có |
| Rolling (FS04 5 Metals) vs Rolling (FS02 Paper3+Cu) | L2 | 7.355 | 0.0000 | Có |
| Expanding (FS02 Paper3+Cu) vs Expanding (FS01 Only Copper) | L1 | 3.365 | 0.0009 | Có |
| Expanding (FS02 Paper3+Cu) vs Expanding (FS01 Only Copper) | L2 | 2.994 | 0.0030 | Có |
| Expanding (FS06 WTI+Cu) vs Expanding (FS04 5 Metals) | L1 | -14.499 | 0.0000 | Có |
| Expanding (FS06 WTI+Cu) vs Expanding (FS04 5 Metals) | L2 | -12.275 | 0.0000 | Có |
| Expanding (FS04 5 Metals+Cu) vs Expanding (FS05 4 Metals-Cu) | L1 | -2.969 | 0.0033 | Có |
| Expanding (FS04 5 Metals+Cu) vs Expanding (FS05 4 Metals-Cu) | L2 | -6.374 | 0.0000 | Có |
| [Residual FS01] Expanding vs Naive Persistence | L1 | 0.761 | 0.4475 | Không |
| [Residual FS01] Expanding vs Naive Persistence | L2 | 1.242 | 0.2154 | Không |
| [Residual FS02] Expanding vs Naive Persistence | L1 | 1.898 | 0.0587 | Không |
| [Residual FS02] Expanding vs Naive Persistence | L2 | 1.400 | 0.1627 | Không |
| [Residual FS02] Rolling vs Naive Persistence | L1 | 2.709 | 0.0072 | Có |
| [Residual FS02] Rolling vs Naive Persistence | L2 | 2.208 | 0.0281 | Có |
| [Residual FS04] Expanding vs Naive Persistence | L1 | 0.600 | 0.5491 | Không |
| [Residual FS04] Expanding vs Naive Persistence | L2 | 0.930 | 0.3531 | Không |
| [Residual FS06] Expanding vs Naive Persistence | L1 | 0.939 | 0.3485 | Không |
| [Residual FS06] Expanding vs Naive Persistence | L2 | 0.735 | 0.4629 | Không |
| Residual Expanding vs Direct Expanding (FS02) | L1 | -14.166 | 0.0000 | Có |
| Residual Expanding vs Direct Expanding (FS02) | L2 | -10.301 | 0.0000 | Có |