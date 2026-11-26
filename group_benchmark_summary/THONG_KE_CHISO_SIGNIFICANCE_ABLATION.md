# BÁO CÁO TỔNG HỢP TOÀN NHÓM: THỐNG KÊ CHỈ SỐ, KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (DIEBOLD–MARIANO), NGHIÊN CỨU CẮT LỚP (ABLATION STUDY) & ĐÁNH GIÁ ƯU/NHƯỢC ĐIỂM 5 PHƯƠNG PHÁP

- **Học phần:** IS403.R11 — Phân tích dữ liệu kinh doanh  
- **Đề tài:** Dự báo giá đồng (*Copper Futures Price Forecasting*)  
- **Phạm vi:** Đối chuẩn độc lập, chuẩn hóa đồng nhất và trung lập giữa **5 hướng tiếp cận mô hình hóa (`PP1–PP5`)** trên cùng bộ dữ liệu và cùng các mốc kiểm định  
- **Ngày cập nhật:** 26/11/2026 *(Đã đồng bộ đầy đủ 6 bộ biến `FS01–FS06`, Walk-Forward V25 của `PP1`, và kiểm định 5-seed mở rộng cho Rolling & `FS03` của `PP4`)*

---

## 1. Xác minh Tính Đồng nhất Dữ liệu & Giao thức Thực nghiệm (`5/5 Phương pháp`)

### 1.1. Xác minh mã băm SHA-256 giữa môi trường Cục bộ và Kaggle
Môi trường Windows cục bộ ghi nhận SHA-256 của tệp `data/raw/dataset_9_commodities_daily.csv` là `9580ede1...`, trong khi snapshot trên Kaggle Linux ghi nhận `a761a908...`. Kiểm định đối chiếu cho thấy nguyên nhân duy nhất là ký tự xuống dòng `CRLF (\r\n)` so với `LF (\n)`:

| Phiên bản tệp | Ký tự xuống dòng | Mã băm SHA-256 | Số phiên ($N$) | Khoảng thời gian | Sai lệch số học cực đại ($\max |x_i - y_i|$) |
|---|---|---|---:|---|---:|
| **Local Git (`data/raw/`)** | Windows `CRLF (\r\n)` | `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600` | 2.710 | `13/01/2016 → 13/11/2026` | **`0.00e+00`** |
| **Kaggle Dataset Snapshot** | Linux `LF (\n)` | `a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83` | 2.710 | `13/01/2016 → 13/11/2026` | **`0.00e+00`** |

Cả 5 phương pháp dùng chung chuỗi 2.710 phiên và khớp đường cơ sở **Naive Persistence ($\hat{y}_t = y_{t-1}$)** đến 6 chữ số thập phân ở cả 4 tỷ lệ chia:
- **Split 80/20 ($N_{\text{test}} = 542$):** $\text{MAE}_{\text{Naive}} = 0.069001\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.108000\text{ \$/lb}$
- **Split 90/10 ($N_{\text{test}} = 271$):** $\text{MAE}_{\text{Naive}} = 0.077077\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.101033\text{ \$/lb}$
- **Split 92.5/7.5 ($N_{\text{test}} = 204$):** $\text{MAE}_{\text{Naive}} = 0.075116\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.098399\text{ \$/lb}$
- **Split 95/5 ($N_{\text{test}} = 136$):** $\text{MAE}_{\text{Naive}} = 0.072799\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.096612\text{ \$/lb}$

### 1.2. Phạm vi thực nghiệm của 5 phương pháp (`PP1–PP5`)

| Mã nhóm | Phương pháp thực nghiệm | Tổ hợp Bộ biến (`FS01–FS06`) | Tỷ lệ chia & Chiến lược | Mục tiêu | Đa hạt giống (`Multi-Seed`) | Dữ liệu dự báo chi tiết (Raw Preds) |
|---|---|---|---|---|---|---|
| **PP1** | **Non-SA Baselines & SA-LSTM Standalone** (`GRU`, `LSTM`, `CNN-LSTM`, `CNN-GRU`, `Ridge`, `SA-LSTM V21–V24`) | Đủ 6 bộ (Non-SA: `FS01, 02, 04, 06`; SA: `FS02–06`) | 4 mốc × 3 schemes (`Expanding`, `Rolling 756`, `Recency`) | `Direct` & `Residual` | 5 seeds (`41–45`) cho Walk-Forward (`V20`, `V23`, `V25`) | Có (`grid_predictions.csv` 103.770+ dòng + `wf_predictions.csv`) |
| **PP2** | **Tối ưu siêu tham số SA–LSTM vs. TPE–LSTM (Optuna)** (192 search trials + 72 final fits) | Đủ 6 bộ (`FS01–FS06`) | 4 mốc (`Expanding`) + `Rolling 756` & `Recency` ở `90/10` | `Residual` | 1 seed (`42`) | Có (`test_predictions.csv`, 20.340 dòng) |
| **PP3** | **iTransformer (Inverted Attention)** + Đối chuẩn SA-LSTM Fixed/Tuned | Đủ 6 bộ (`FS01–FS06`) | 4 mốc × 3 schemes | `Direct` & `Residual` | 1 seed (`42`) | Chưa xuất file predictions chi tiết (có bảng 216 dòng + DM ở `K=5`) |
| **PP4** | **Rolling & Expanding Window Walk-Forward LSTM** (`Static T`, `Static T+V`, `Expanding`, `Rolling`, Ablation $W, S$, Cold/Warm) | Đủ 6 bộ (`FS01–FS06`) ở `90/10` | 4 mốc (`Cross-Ratio` trên `FS02`) | `Direct` & `Residual` | 5 seeds (`41–45`) cho `Expanding FS02`, `Rolling FS02` & `Expanding FS03` | Có (49 file `pred_*.csv`) |
| **PP5** | **Hybrid ARIMA–GRU** (`auto_arima` + GRU 2 tầng chuẩn nhóm `64→32→16→1` học phần dư, 360 lượt train GRU) | Đủ 6 bộ (`FS01–FS06`) | 4 mốc `Expanding` + `Rolling 756` & `Recency` ở `90/10` | `Residual` (Hybrid/ARIMA) & `Direct` (GRU) | 5 seeds (`41–45`) cho toàn bộ lưới | Có (36 file `predictions_*.csv`) |

---

## 2. Thống kê Chỉ số Tổng hợp Toàn Nhóm (Benchmarking Leaderboard)

### 2.1. Bảng xếp hạng nhánh Residual trên cấu hình chuẩn `Split 90/10 — Expanding` ($N_{\text{test}} = 271$, `29/10/2025 → 13/11/2026`)

Biểu đồ đối chiếu cân bằng: [fig16_group_benchmark_all_5_members.png](fig16_group_benchmark_all_5_members.png) · Bảng dữ liệu CSV: [group_ablation_6_feature_sets_all_members.csv](group_ablation_6_feature_sets_all_members.csv)

| Hạng (Best MAE) | Nhóm PP | Phương pháp / Mô hình | Bộ biến đạt đỉnh (Best FS) | Best MAE (\$/lb) | RMSE tại Best FS (\$/lb) | DA (%) tại Best FS | % Cải thiện vs. Naive (`0.07708`) | MAE Trung bình qua các bộ biến (\$/lb) | Độ lệch chuẩn MAE giữa các bộ biến (\$/lb) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| **1** | **PP3** | **iTransformer Residual** | `FS01` (Copper) | **0.07681** | 0.10097 | 53.14% | **+0.35%** | 0.07969 *(6 bộ)* | 0.00527 |
| **2** | **PP4** | **WF Expanding Residual (`W=756, S=40`)** | `FS03` (WTI, Gold, Silver) | **0.07693** | 0.10107 | 50.93% | **+0.19%** | 0.07756 *(6 bộ)* | 0.00055 |
| **3** | **PP1** | **GRU Residual (Non-SA V20)** | `FS04` (5 Kim loại) | **0.07695** | **0.10081** | 52.77% | **+0.16%** | 0.07793 *(4 bộ)* | 0.00151 |
| **4** | **PP2** | **SA–LSTM Residual (Per-FS Search)** | `FS01` (Copper) | **0.07695** | 0.10084 | 52.77% | **+0.16%** | 0.07975 *(6 bộ)* | 0.00283 |
| **5** | **PP1** | **LSTM Residual (Non-SA V20)** | `FS04` (5 Kim loại) | **0.07698** | 0.10095 | 52.77% | **+0.13%** | **0.07701 *(4 bộ có Cu)*** | **0.00002** |
| **—** | **Baseline** | **Naive Persistence ($\hat{y}_t = y_{t-1}$)** | `FS01` (Copper) | **0.07708** | **0.10103** | — | **0.00%** | **0.07708** | **0.00000** |
| **6** | **PP1** | **CNN–LSTM Residual (Non-SA V20)** | `FS04` (5 Kim loại) | 0.07725 | 0.10086 | 46.86% | -0.22% | 0.07810 *(4 bộ)* | 0.00134 |
| **7** | **PP2** | **TPE–LSTM Residual (Optuna Per-FS)** | `FS01` (Copper) | 0.07727 | 0.10094 | 48.71% | -0.25% | 0.08212 *(6 bộ)* | 0.00896 |
| **8** | **PP1** | **Ridge Residual ($\alpha = 1.0$)** | `FS01` (Copper) | 0.07731 | 0.10119 | 47.60% | -0.30% | 0.07937 *(4 bộ)* | 0.00217 |
| **9** | **PP4** | **WF Rolling Residual (`W=756, S=40`)** | `FS01` (Copper) | 0.07732 | 0.10167 | 50.93% | -0.32% | 0.07853 *(6 bộ)* | 0.00119 |
| **10** | **PP5** | **Hybrid ARIMA(0,1,0)–GRU** *(TB 5 seeds, GRU 2 tầng)* | `FS01` (Copper) | 0.07745 | 0.10142 | 48.71% | -0.48% | 0.08048 *(6 bộ)* | 0.00218 |
| **11** | **PP4** | **Static (Train+Val) Residual** | `FS04` (5 Kim loại) | 0.07746 | 0.10070 | 48.33% | -0.50% | 0.07920 *(6 bộ)* | 0.00160 |

> **Nhận xét khách quan Bảng 2.1:**
> - **Độ chính xác điểm đơn (`Best MAE`, Seed 42):** `PP3 (iTransformer Residual FS01 = 0.07681)` dẫn đầu (`+0.35%` vs Naive), theo sau sát bởi `PP4 WF-Expanding (FS03 = 0.07693)`, `PP1 GRU-Res (FS04 = 0.07695)`, `PP2 SA-LSTM (FS01 = 0.07695)` và `PP1 LSTM-Res (FS04 = 0.07698)`. Khoảng cách giữa hạng 1 và Naive chỉ là `0.00027 $/lb`. Khi kiểm tra thêm 5 seeds (`41–45`) cho `PP4 WF-Expanding FS03`, trung bình 5 seeds là `0.07790 ± 0.00116` (`+1.07%` vs Naive), cho thấy mức `0.07693` ở Seed 42 chịu ảnh hưởng của dao động khởi tạo ngẫu nhiên.
> - **Tính ổn định qua các bộ biến:** Xét trên 4 bộ biến có lịch sử Copper của `PP1` (`FS01, FS02, FS04, FS06`), **`PP1 LSTM Residual` (`0.07701 ± 0.00002 $/lb`)** đạt trung bình thấp nhất và độ phân tán hẹp nhất nhóm (cả 4/4 bộ đều vượt nhẹ Naive). Xét riêng trong 4 phương pháp chạy đủ cả 6 bộ biến bao gồm 2 bộ bỏ Copper (`FS03, FS05`), **`PP4 WF Expanding Residual` (`0.07756 ± 0.00055 $/lb`)** có trung bình và độ lệch chuẩn thấp nhất.

### 2.2. Đối chiếu nhánh Direct (Dự báo mức giá trực tiếp) toàn nhóm (`Split 90/10`)

| Nhóm PP | Mô hình (Nhánh Direct) | `FS01` MAE | `FS02` MAE | `FS03` MAE | `FS04` MAE | `FS05` MAE | `FS06` MAE | MAE Trung bình (\$/lb) | So với Naive (`0.07708`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **PP4** | **Walk-Forward Expanding Direct (`W=756, S=40`)** | **0.20265** | **0.23696** | **0.38606** | 0.33844 | **0.36182** | 0.21921 | **0.29086** *(6 bộ)* | Kém Naive +277% |
| **PP1** | **GRU Direct (Non-SA V20/V22)** | 0.43711 | 0.24789 | — | **0.24082** | — | 0.24321 | **0.29226** *(4 bộ)* | Kém Naive +279% |
| **PP4** | **Walk-Forward Rolling Direct (`W=756, S=40`)** | 0.27394 | 0.28371 | 0.46075 | 0.34973 | 0.41410 | 0.27422 | 0.34274 *(6 bộ)* | Kém Naive +345% |
| **PP5** | **GRU Direct (TB 5 seeds, chuẩn nhóm)** | 0.33809 | 0.29023 | 0.49914 | 0.29322 | 0.56456 | 0.30653 | 0.38196 *(6 bộ)* | Kém Naive +396% |
| **PP1** | **Paper SA-LSTM Direct (V21/V22)** | — | 0.97807 | 0.54072 | 0.47093 | 0.46499 | **0.18463** | 0.52787 *(5 bộ)* | Kém Naive +585% |
| **PP1** | **LSTM Direct (Non-SA V20/V22)** | 0.44672 | 1.16396 | — | 0.24741 | — | 0.32494 | 0.54576 *(4 bộ)* | Kém Naive +608% |
| **PP1** | **Reopt SA-LSTM Direct (V21/V22)** | — | 0.87742 | 0.59561 | 0.51047 | 0.46660 | 0.35892 | 0.56180 *(5 bộ)* | Kém Naive +629% |
| **PP4** | **Static (Train-Only) Direct** | 0.49280 | 1.55159 | 0.31990 | 1.32269 | 0.56377 | 0.55846 | 0.80154 *(6 bộ)* | Kém Naive +940% |
| **PP3** | **iTransformer Direct** | 1.75765 | 1.14381 | 1.54427 | 1.63235 | 1.81230 | 1.38313 | 1.54558 *(6 bộ)* | Kém Naive +1905% |

> **Phát hiện xuyên suốt số 1 (Direct vs. Residual):** **100% mô hình Direct** ở cả 4 nhóm thực hiện (`PP1, PP3, PP4, PP5`) đều kém xa Naive (`MAE = 0.185 – 1.812 $/lb` vs `0.07708 $/lb`) do giá đồng giai đoạn Test (`5.0 – 6.89 $/lb`) vượt ngoài biên chuẩn hóa `MinMaxScaler` của tập Train. Trên nhánh Direct, `PP4 WF Expanding Direct` (`TB = 0.29086`) và `PP1 GRU Direct` tĩnh (`TB = 0.29226`) xấp xỉ nhau ở vị trí dẫn đầu trung bình, trong khi `PP1 Paper SA-LSTM Direct` đạt điểm đơn thấp nhất trên `FS06` (`0.18463`), còn tái huấn luyện định kỳ của `PP4` giảm `64%–85%` sai số so với `Static Direct`.

### 2.3. Thống kê xuyên suốt 4 tỷ lệ chia dữ liệu (`Cross-Ratio 80/20, 90/10, 92.5/7.5, 95/5`)

Bảng dữ liệu chi tiết: [group_cross_ratio_4_splits_all_members.csv](group_cross_ratio_4_splits_all_members.csv)

| Phương pháp đại diện | Nhóm PP | Bộ biến | `80/20` ($N=542$) MAE | `90/10` ($N=271$) MAE | `92.5/7.5` ($N=204$) MAE | `95/5` ($N=136$) MAE | Trung bình 4 mốc (\$/lb) | Số mốc vượt Naive |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Naive Persistence ($\hat{y}_t = y_{t-1}$)** | Baseline | `FS01` | **0.06900** | **0.07708** | **0.07512** | **0.07280** | **0.07350** | — |
| **ARIMA Standalone (`auto_arima`)** | PP5 | `FS01` | 0.06900 (0.00%) | 0.07708 (0.00%) | **0.07456 (+0.75%)** | **0.07258 (+0.31%)** | **0.07331** | 2 / 4 |
| **SA–LSTM Residual (Expanding)** | PP2 | `FS01` | **0.06883 (+0.25%)** | **0.07695 (+0.16%)** | **0.07506 (+0.08%)** | **0.07252 (+0.38%)** | **0.07334** | **4 / 4** |
| **GRU Residual (Expanding)** | PP1 | `FS02` | **0.06878 (+0.32%)** | 0.07714 (-0.09%) | 0.07521 (-0.12%) | **0.07277 (+0.04%)** | **0.07348** | 2 / 4 |
| **LSTM Residual (Expanding)** | PP1 | `FS01` | 0.06920 (-0.29%) | **0.07701 (+0.09%)** | 0.07536 (-0.33%) | **0.07273 (+0.09%)** | **0.07358** | 2 / 4 |
| **iTransformer Residual (Expanding)** | PP3 | `FS01` | **0.06885 (+0.22%)** | **0.07681 (+0.35%)** | 0.07595 (-1.11%) | 0.07289 (-0.12%) | 0.07363 | 2 / 4 |
| **iTransformer Residual (Expanding)** | PP3 | `FS02` | 0.06945 (-0.65%) | **0.07701 (+0.09%)** | 0.07568 (-0.75%) | **0.07253 (+0.37%)** | 0.07367 | 2 / 4 |
| **TPE–LSTM Residual (Expanding)** | PP2 | `FS01` | 0.06955 (-0.79%) | 0.07727 (-0.24%) | 0.07517 (-0.07%) | 0.07314 (-0.47%) | 0.07378 | 0 / 4 |
| **TPE–LSTM Residual (Expanding)** | PP2 | `FS02` | 0.06946 (-0.66%) | 0.07798 (-1.17%) | 0.07564 (-0.69%) | 0.07337 (-0.79%) | 0.07411 | 0 / 4 |
| **Hybrid ARIMA–GRU (TB 5 seeds)** | PP5 | `FS01` | 0.07275 (-5.43%) | 0.07745 (-0.48%) | **0.07475 (+0.48%)** | 0.07284 (-0.06%) | 0.07445 | 1 / 4 |
| **Static (Train+Val) Residual** | PP4 | `FS02` | 0.07505 (-8.76%) | 0.07812 (-1.35%) | 0.07684 (-2.30%) | **0.07213 (+0.91%)** | 0.07553 | 1 / 4 |
| **WF Expanding Residual (`W=756, S=40`)** | PP4 | `FS02` | 0.07108 (-3.02%) | 0.07852 (-1.87%) | 0.07714 (-2.70%) | 0.07573 (-4.02%) | 0.07562 | 0 / 4 |

> **Nhận xét Bảng 2.3:** Qua 4 mốc chia thời gian lồng nhau, **`PP2 SA–LSTM (FS01)`** là mô hình nơ-ron duy nhất vượt nhẹ Naive ở cả **4/4 mốc** (`TB = 0.07334`), **`PP5 ARIMA` thuần** đạt TB 4 mốc thấp nhất (`0.07331`), theo sát là **`PP1 GRU/LSTM Residual`** (`0.07348 / 0.07358`) và **`PP3 iTransformer`** (`0.07363`). Ngược lại, trên bộ biến `FS02`, phương pháp tái huấn luyện định kỳ **`PP4 WF Expanding Residual`** (`TB = 0.07562`, `0/4` mốc vượt Naive) có sai số trung bình cao nhất bảng Cross-Ratio (do liên tục retrain mạng LSTM `lr = 5e-5` làm tăng nhẹ phương sai sai số trên chuỗi sai phân đã dừng sẵn).

---

## 3. Kiểm định Ý nghĩa Thống kê Toàn Nhóm (Diebold–Mariano Significance Tests)

Biểu đồ kiểm định đối xứng toàn nhóm: [fig18_group_dm_significance_and_pros_cons.png](fig18_group_dm_significance_and_pros_cons.png) · Bảng CSV đầy đủ (232 dòng kiểm định): [group_significance_dm_tests_complete.csv](group_significance_dm_tests_complete.csv)

Toàn bộ các phép kiểm định được chuẩn hóa theo **Diebold–Mariano (1995)** với ước lượng phương sai vững **Newey–West HAC** ($\text{lag} = 5$) và hiệu chỉnh mẫu nhỏ **Harvey–Leybourne–Newbold (HLN)** trên cả hàm mất mát $L_1$ ($|e_t|$) và $L_2$ ($e_t^2$). *(Lưu ý: `PP3` chưa xuất tệp dự báo chi tiết theo ngày nên chưa tính DM-vs-Naive cho `FS01–FS06` của `PP3`; `PP3` có kết quả DM riêng ở benchmark `K=5`.)*

### 3.1. Kiểm định giả thuyết $H_0^{(1)}$: *"Có mô hình Residual nào vượt Naive Persistence có ý nghĩa thống kê không?"*

Bảng kiểm kê đầy đủ trên **toàn bộ 548 lượt chạy Residual** có dữ liệu dự báo từng phiên từ 4 nhóm phương pháp (`PP1, PP2, PP4, PP5`):

| Nhóm PP | Tập lượt chạy kiểm định với Naive Persistence | Số lượt (Tổng / Expanding) | Thắng Naive ý nghĩa 5% ($L_1$) | Thắng Naive ý nghĩa 5% ($L_2$) | Kém Naive ý nghĩa 5% ($L_1$) | Thống kê $\text{DM}_{L1}$ & $p$-value tiêu biểu |
|---|---|---:|---:|---:|---:|---|
| **PP1 (Grid V20+V22)** | 5 mô hình Residual (`lstm, gru, cnn_lstm, cnn_gru, ridge`) $\times$ 4 FS $\times$ 4 splits $\times$ 3 schemes | `240` (`80` Expanding) | **0 / 240** (`0 / 80` Exp) | **2 / 240** *(chỉ ở `gru_residual`)* | `102 / 240` (`18 / 80` Exp; `lstm_res` Exp: **`0/16`**) | `lstm_res FS04`: $\text{DM} = -0.28, p = 0.780$; `gru_res FS04`: $\text{DM} = -0.49, p = 0.622$ |
| **PP1 (WF V20+V23+V25)** | Walk-Forward `gru_res`, `lstm_res`, `cnn_gru_res`, `cnn_lstm_res`, `ridge_res` (`5 seeds 41–45`, `Ridge 1 seed`) | `26` (`26` Expanding) | **0 / 26** | — | `5 / 26` | V20 `gru_res` s41 ($N=2.214$): $p = 0.704$; V23 `gru_res` s41: $p = 0.493$; V25 `cnn_lstm_res` s44: $p = 0.727$ |
| **PP2 (`SA–LSTM`)** | `SA–LSTM Residual` (6 FS $\times$ 6 bối cảnh) | `36` (`24` Expanding) | **0 / 36** (`0 / 24` Exp) | **1 / 36** (`0 / 24` Exp) | `19 / 36` (`11 / 24` Exp) | `FS01` (90/10 Exp): $\text{DM} = -0.58, p = 0.560$; `FS06`: $\text{DM} = +0.02, p = 0.983$ |
| **PP2 (`TPE–LSTM`)** | `TPE–LSTM Residual` (6 FS $\times$ 6 bối cảnh) | `36` (`24` Expanding) | **0 / 36** (`0 / 24` Exp) | **0 / 36** | `18 / 36` (`8 / 24` Exp) | `FS01` (90/10 Exp): $\text{DM} = +1.25, p = 0.212$; `FS02`: $\text{DM} = +1.69, p = 0.093$ |
| **PP4 (`Walk-Forward`)** | `90/10` (`6 FS × 3 chiến lược`) + `4 seeds (41, 43–45)` Exp `FS02` + `8 seeds` (`Roll FS02` & `Exp FS03`) | `30` (`14` Expanding) | **0 / 30** (`0 / 14` Exp) | **0 / 30** | `5 / 30` (`1 / 14` Exp Seed 43) | `FS03` Exp (s42): $\text{DM} = -0.34, p = 0.735$; `FS01` Exp: $\text{DM} = +0.78, p = 0.438$ |
| **PP5 (`Hybrid ARIMA–GRU`)** | `Hybrid ARIMA–GRU`: `Expanding` (4 splits × 6 FS × 5 seeds = `120`) + `Recency/Rolling 756` ở 90/10 (`60`) | `180` (`120` Expanding) | **1 / 180** (`0 / 120` Exp; duy nhất `Recency FS01 s42`, $p=0.0385$) | **0 / 180** (`0 / 120` Exp) | `57 / 120` ở Exp; `69 / 90` ở 90/10 cả 3 schemes | `90/10 Exp FS01`: $\text{DM}_{tb} = +0.70, p_{tb} = 0.355$; `FS06`: $\text{DM}_{tb} = +1.90, p_{tb} = 0.100$ |

> **Kết luận Kiểm định $H_0^{(1)}$ (Đồng thuận toàn nhóm):**
> - Trên **tổng cộng 288 lượt chạy chuẩn `Expanding` độc nhất** (`80` PP1 Grid + `26` PP1 WF + `48` PP2 + `14` PP4 + `120` PP5): **`0 / 288 (0.0%)` lượt chạy thắng được Naive Persistence có ý nghĩa thống kê dưới hàm mất mát $L_1$**!
> - Tính chung trên **toàn bộ 548 lượt chạy Residual** (gồm cả `Rolling` và `Recency`), có **`547 / 548 lượt (99.8%)` KHÔNG vượt qua được Naive dưới $L_1$**.
> - Kết quả này chứng minh thực nghiệm đồng nhất trên mọi kiến trúc: chuỗi sai phân giá đồng ngày hành xử như quá trình **Martingale Difference Sequence (MDS)** với tỷ lệ tín hiệu trên nhiễu cực thấp ($|r_{\text{lag-1}}| \le 0.091$), khiến nghiệm tối ưu dưới MSE bị co phương sai (**Variance Shrinkage**), hội tụ tiệm cận về Naive $\hat{y}_t \approx y_{t-1}$.

### 3.2. Kiểm định đối đầu nội bộ `PP2: TPE–LSTM vs. SA–LSTM` (36 cặp)

| Bộ biến (`Split 90/10 Expanding`, $N=271$) | SA–LSTM MAE (\$/lb) | TPE–LSTM MAE (\$/lb) | Chênh lệch $\bar{d}_{L1} = \text{TPE} - \text{SA}$ | $\text{DM}_{L1}$ ($p_{L1}$) | $\text{DM}_{L2}$ ($p_{L2}$) | Kết luận ở mức ý nghĩa $\alpha = 5\%$ |
|---|---:|---:|---:|---:|---:|---|
| **`FS01` (Copper)** | **0.07695** | 0.07727 | +0.00031 | `+1.032` (`0.3029`) | `+0.320` (`0.7489`) | Không khác biệt ý nghĩa |
| **`FS02` (Paper3 + Cu)** | 0.08453 | **0.07798** | **-0.00655** | **`-3.159` (`0.0018`)** | **`-3.562` (`0.0004`)** | **TPE–LSTM thắng có ý nghĩa thống kê ($p < 0.01$)** |
| **`FS03` (Paper3 - Cu)** | 0.07999 | **0.07812** | **-0.00187** | **`-3.135` (`0.0019`)** | **`-3.301` (`0.0011`)** | **TPE–LSTM thắng có ý nghĩa thống kê ($p < 0.01$)** |
| **`FS04` (5 Kim loại)** | **0.08097** | 0.10006 | **+0.01910** | **`+3.685` (`0.0003`)** | **`+3.635` (`0.0003`)** | **SA–LSTM thắng có ý nghĩa thống kê ($p < 0.001$)** |
| **`FS05` (4 Kim loại - Cu)** | **0.07896** | 0.08198 | +0.00302 | `+1.182` (`0.2384`) | `+1.993` (`0.0473`) | Không khác biệt $L_1$ (SA tốt hơn $L_2$) |
| **`FS06` (WTI + Cu)** | **0.07709** | 0.07731 | +0.00023 | `+0.661` (`0.5095`) | `-0.215` (`0.8300`) | Không khác biệt ý nghĩa |

- **Trên cả 36 cặp bối cảnh (`6 FS × 6 bối cảnh`) của `PP2`:** `18/36` (50.0%) tương đương thống kê; **SA–LSTM thắng ý nghĩa `11/36` (30.6%)** (tập trung ở `FS04` và `FS01`, nơi SA ít bị overfit vào Validation); **TPE–LSTM thắng ý nghĩa `7/36` (19.4%)** (tập trung ở `FS02` và `FS03`).

### 3.3. Kiểm định DM chéo đối xứng giữa các phương pháp (`90/10 Expanding`, $N = 271$)

Bảng dưới đây tổng hợp **đối đầu từng đôi một không thiên vị** (từ [group_significance_dm_tests_complete.csv](group_significance_dm_tests_complete.csv), khớp cùng `Seed = 42` trên các bộ biến dùng chung):

| Cặp so sánh (`A vs B`, Seed 42) | Số FS chung | Số bộ `A` thắng ý nghĩa | Số bộ `B` thắng ý nghĩa | Hòa thống kê ($p \ge 0.05$) | $\text{DM}_{L1}$ trung bình | Các bộ biến có ý nghĩa thống kê ($p_{L1} < 0.05$) |
|---|---:|---:|---:|---:|---:|---|
| **`PP1: LSTM-Res` vs `PP1: GRU-Res`** | 4 | **1** | 0 | 3 | -1.15 | `PP1: LSTM-Res` thắng ở `FS01` ($p=0.0071$) |
| **`PP1: LSTM-Res` vs `PP2: SA-LSTM`** | 4 | **2** | 0 | 2 | -1.26 | `PP1: LSTM-Res` thắng ở `FS02` ($p=0.0024$) & `FS04` ($p=0.0051$) |
| **`PP1: LSTM-Res` vs `PP2: TPE-LSTM`** | 4 | **1** | 0 | 3 | -1.98 | `PP1: LSTM-Res` thắng ở `FS04` ($p<0.0001$) |
| **`PP1: LSTM-Res` vs `PP4: WF-Expanding`** | 4 | 0 | 0 | **4** | -1.08 | **Hòa cả 4/4** ($p \in [0.110, 0.475]$, `PP1` có MAE thấp hơn cả 4/4) |
| **`PP1: LSTM-Res` vs `PP5: Hybrid (s42)`** | 4 | **3** | 0 | 1 | -3.12 | `PP1: LSTM-Res` thắng ở `FS02` ($p=0.0128$), `FS04` ($p<0.0001$), `FS06` ($p=0.0045$) |
| **`PP1: GRU-Res` vs `PP4: WF-Expanding`** | 4 | 0 | **1** | 3 | -0.02 | `PP4` thắng ở `FS01` ($p=0.0054$); `PP1` nhỉnh điểm ở `FS02, 04, 06` |
| **`PP2: SA-LSTM` vs `PP4: WF-Expanding`** | 6 | 0 | **3** | 3 | +1.28 | `PP4` thắng ở `FS02` ($p=0.0018$), `FS03` ($p=0.0169$), `FS04` ($p=0.0351$) |
| **`PP2: TPE-LSTM` vs `PP4: WF-Expanding`** | 6 | 0 | **2** | 4 | +1.28 | `PP4` thắng ở `FS04` ($p<0.0001$) & `FS05` ($p=0.0179$) |
| **`PP2: SA-LSTM` vs `PP5: Hybrid (s42)`** | 6 | **2** | **1** | 3 | -2.27 | `PP2 SA` thắng `FS04, FS06` ($p \le 0.0005$); `PP5 s42` thắng `FS02` ($p=0.0426$) |
| **`PP2: TPE-LSTM` vs `PP5: Hybrid (s42)`** | 6 | **3** | 0 | 3 | -2.02 | `PP2 TPE` thắng `FS03, FS05, FS06` ($p \le 0.0007$) |
| **`PP4: WF-Expanding` vs `PP5: Hybrid (s42)`** | 6 | **4** | 0 | 2 | -2.53 | `PP4` thắng `FS03, FS04, FS05, FS06` ($p \in [<0.0001, 0.0195]$) |

- **Kiểm định độ nhạy khi dùng Ensemble 5 seeds của `PP5` (`mean(Hybrid_s41..s45)`):** Khi trung bình hóa 5 hạt giống ngẫu nhiên của `PP5`, phương sai sai số giảm mạnh:
  - **`PP5` 5-seed Ensemble thắng có ý nghĩa** `PP2 SA-LSTM` trên `FS02` ($p = 0.0021$), `PP2 TPE-LSTM` trên `FS04` ($p < 0.0001$) & `FS05` ($p = 0.0011$), và `PP1 GRU-Res` trên `FS01` ($p = 0.0043$); đồng thời hòa thống kê với `PP4 WF-Expanding` trên cả 6/6 bộ ($p \in [0.176, 0.941]$).
  - Duy nhất **`PP1 LSTM-Res`** vẫn thắng có ý nghĩa `PP5` 5-seed Ensemble trên `FS04` ($p = 0.0055$) & `FS06` ($p = 0.0346$) và không thua trên bất kỳ bộ biến nào (`mean_DM = -2.00`).
- **Kiểm định DM của `PP3` (`iTransformer vs SA-LSTM`, benchmark `K=5`):** `iTransformer Residual (MAE = 0.07812)` thắng có ý nghĩa thống kê cả `SA-LSTM Fixed (0.08293, p = 0.0248)` lẫn `SA-LSTM Tuned (0.08535, p = 0.00014)`.
- **Kiểm định DM & Multi-Seed nội bộ của `PP4`:** Trên nhánh Direct, `Expanding` & `Rolling` thắng `Static (Train-Only)` tuyệt đối ($\text{DM}_{L1} = -51.36$ & $-43.54, p < 0.0001$). Trên nhánh Residual (`FS02`, 5 seeds `41–45`), `Expanding` đạt `0.07828 ± 0.00063` (ổn định gấp ~2 lần so với `Rolling` `0.07864 ± 0.00122`).

---

## 4. Nghiên cứu Cắt lớp Toàn Nhóm (Group-Wide Ablation Studies)

Biểu đồ Heatmap & đối chiếu Direct/Residual 6 bộ biến: [fig17_group_ablation_6_feature_sets.png](fig17_group_ablation_6_feature_sets.png)

### 4.1. Ablation 1 — Vai trò 6 Bộ biến (`FS01` → `FS06`) trên cả 5 phương pháp (`Split 90/10 — Expanding`)

| Mã bộ biến | Các biến trong tensor $X_{t-22:t-1}$ | $K$ | `PP1` (`LSTM-Res`) | `PP1` (`Ridge-Res`) | `PP2` (`SA-LSTM`) | `PP2` (`TPE-LSTM`) | `PP3` (`iTrans-Res`) | `PP4` (`WF-Exp-Res`) | `PP5` (`Hybrid-Res`) | `PP4` (`WF-Exp-Dir`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`FS01_CU`** | `Copper` | 1 | 0.07701 | **0.07731** | **0.07695** | **0.07727** | **0.07681** | 0.07736 | **0.07745** | **0.20265** |
| **`FS02_PAPER_CU`** | `Copper, WTI, Gold, Silver` | 4 | 0.07701 | 0.08071 | 0.08453 | 0.07798 | 0.07701 | 0.07852 | 0.08110 | 0.23696 |
| **`FS03_PAPER_EXO`** | `WTI, Gold, Silver` *(Bỏ Copper)* | 3 | — | — | 0.07999 | 0.07812 | *0.09027* | **0.07693** | *0.08334* | *0.38606* |
| **`FS04_METALS_CU`** | `Copper, Al, Tin, Silver, Gold` | 5 | **0.07698** | *0.08171* | 0.08097 | *0.10006* | 0.07687 | 0.07769 | 0.08185 | 0.33844 |
| **`FS05_METALS_EXO`** | `Al, Tin, Silver, Gold` *(Bỏ Copper)* | 4 | — | — | 0.07896 | 0.08198 | 0.07924 | 0.07720 | 0.08071 | *0.36182* |
| **`FS06_WTI_CU`** | `Copper, Al, Tin, WTI` | 4 | 0.07704 | 0.07774 | 0.07709 | 0.07731 | 0.07794 | 0.07764 | 0.07844 | 0.21921 |
| **Trung bình** | — | — | **0.07701** | 0.07937 | 0.07975 | 0.08212 | 0.07969 | 0.07756 | 0.08048 | 0.29086 |

**3 phát hiện chung của toàn nhóm từ Ablation 6 bộ biến:**
1. **Nhiễu chiều từ biến ngoại sinh (*Exogenous Noise Inflation*):** Tương quan trễ bậc 1 ngày trên Train (`Gold, Silver, WTI, Al, Tin` vs `Copper`) đều rất yếu ($|r| \le 0.091$). Do đó, mô hình đơn biến `FS01` dẫn đầu trên phần lớn phương pháp (`PP3 0.07681`, `PP2 SA-LSTM 0.07695`, `PP2 TPE-LSTM 0.07727`, `PP1 Ridge 0.07731`, `PP5 0.07745`) và ngang đỉnh trên `PP1 LSTM (0.07701)`. Khi thêm nhiều biến ngoại sinh ở `FS02` hoặc `FS04`, hầu hết kiến trúc bị nhiễu: `PP1 Ridge` giảm `4.7%–6.0%`, `PP2 TPE-LSTM` bị overfit lên `0.10006`, `PP2 SA-LSTM` ở `FS02` lên `0.08453`, còn `PP5 Hybrid ARIMA–GRU` xấu đi trên cả `FS02–FS06`.
2. **Tác động khi loại bỏ `Copper` khỏi tensor đầu vào (`FS03`, `FS05`):** Trên nhánh Direct, mất `Copper` làm sai số tăng `1.5–2.0` lần. Trên nhánh Residual (dù bước cộng ngược vẫn dùng $y_{t-1}$), `PP3 iTransformer` mất token neo trung tâm khiến MAE `FS03` vọt lên `0.09027` (`-17.12%` vs Naive), và `PP5 Hybrid` tăng lên `0.08334` (`-8.13%`). Riêng `PP4 WF Expanding` (`0.07693`, `0.07720`) ít bị dao động nhất giữa các bộ biến.
3. **Nhóm kim loại + năng lượng (`FS06`):** Trong các bộ đa biến, `FS06 (Copper, Al, Tin, WTI)` cho kết quả ổn định nhất trên `PP1` (`LSTM 0.07704`, `Paper SA-LSTM Direct 0.18463`), `PP2` (`0.07709`), `PP4` (`0.07764`) và `PP5` (`0.07844`).

### 4.2. Ablation 2 — Chiến lược Cửa sổ (`Expanding` vs. `Rolling 756` vs. `Recency`) & Tham số Walk-Forward (`W, S, Cold/Warm`)

1. **So sánh 3 chiến lược cửa sổ trên cả 5 phương pháp:** **`Expanding Window`** ổn định hơn rõ rệt so với `Rolling 756` (~3 năm) ở `80/20` và `90/10` trên mọi kiến trúc, vì mạng học sâu cần tập mẫu đủ lớn (~2.173 phiên). Đặc biệt ở nhánh `PP5` (`90/10`), sai số TB 6 FS của `Expanding` là `+4.42%`, `Recency` là `+6.18%`, còn `Rolling 756` vọt lên **`+32.45%`** (`FS03 Rolling 756` lên tới `+101.22%`). Tuy vậy, **`Recency Weighted` (half-life 252)** trên `FS01` giúp `PP3 iTransformer` đạt đỉnh `0.07645` (`+0.82%` vs Naive) và giúp `PP5 Hybrid (Seed 42)` đạt `0.07652` ở `90/10`.
2. **Ablation tham số Walk-Forward (`W`, `S`, `Cold/Warm`) của `PP4`:** Window $W=756$ (`0.28371`) tốt hơn $W=252$ (`0.29233`) và $W=504$ (`0.29514`) trên `Rolling Direct`; giảm Stride $S$ từ `60 → 10` hạ MAE `Expanding Direct` từ `0.32841 → 0.22756` nhưng tăng thời gian train từ `64.4s → 364.2s` ($S^*=40$ cân bằng tốt nhất). `Warm-Start` giúp tăng tốc huấn luyện nhưng ở `Rolling` làm tăng MAE Direct (`0.37394` vs `0.28371` của `Cold-Start`).

---

## 5. Đánh giá Ưu / Nhược điểm & Tính Có Ý Nghĩa của 5 Phương Pháp (`PP1–PP5`)

| Mã & Phương pháp | Ưu điểm nổi bật (Strengths) | Nhược điểm & Giới hạn (Weaknesses) | Kết luận thống kê cốt lõi |
|---|---|---|---|
| **PP1 — Non-SA Baselines (`LSTM, GRU, Ridge`) & SA-LSTM V21–V25** | - Khung kiểm chứng bài bản nhất nhóm: `240` lượt Grid + Walk-Forward 5 seeds (`V20`, `V23`, `V25` phủ đủ 12 họ mô hình) + hệ thống kiểm định kinh tế lượng đầy đủ (ADF, KPSS, Zivot–Andrews, Brown–Forsythe, Spearman FDR) và giải thích toán học *Variance Shrinkage*.<br>- `LSTM Residual` đạt trung bình **tốt nhất trên 4 bộ có Copper (`0.07701 ± 0.00002`)**, hòa/thắng DM chéo trước hầu hết mô hình khác (`0/16` lượt Expanding Grid thua Naive). | - Các mô hình Non-SA dùng siêu tham số cố định (`lr=1e-3`), chưa HPO riêng từng FS; bộ Non-SA Grid chưa chạy 2 bộ bỏ Copper (`FS03, FS05`).<br>- `Ridge Residual` nhạy cảm nhiễu ngoại sinh (`40/48` lượt thua Naive có ý nghĩa, `FS04` giảm `-6.01%`). | `0/240` (Grid L1) & `0/26` (WF L1) lượt thắng Naive; là chuẩn baseline chặt chẽ nhất giải thích tính chất Random Walk của chuỗi giá đồng ngày. |
| **PP2 — Tối ưu Siêu tham số SA–LSTM vs. TPE–LSTM (Optuna)** | - Đối chuẩn HPO công bằng cùng ngân sách (16 trials/FS trên `80/20`, 192 search + 72 final fits) và chạy đủ 6 FS × 6 bối cảnh.<br>- **`SA-LSTM (FS01)` là mô hình nơ-ron duy nhất vượt nhẹ Naive ở cả 4/4 mốc Cross-Ratio** (`TB = 0.07334`).<br>- `TPE-LSTM` thắng `SA-LSTM` có ý nghĩa ($p < 0.01$) trên `FS02` & `FS03`. | - Chạy 1 seed (`42`), mô hình tĩnh trên Test.<br>- `TPE-LSTM` dễ **overfit Validation** trên tập đa cộng tuyến cao (`FS04 = 0.10006`, thua SA với $p=0.0003$), còn `SA-LSTM` dao động mạnh trên `FS02` (`0.08453`, thua `PP1 LSTM` & `PP4 WF` với $p < 0.003$). | `18/36` cặp SA vs TPE hòa thống kê; SA thắng `11/36`, TPE thắng `7/36`; cả hai đều có `0/36` lượt thắng Naive dưới $L_1$. |
| **PP3 — iTransformer (Inverted Attention)** | - Kiến trúc Transformer hiện đại đảo trục biến thành variable-tokens (`Dense 22→48`).<br>- Đạt **MAE thấp nhất toàn nhóm ở `90/10 Expanding` (`FS01 = 0.07681`, `FS04 = 0.07687`)** và đạt `0.07645 (+0.82%)` khi dùng `Recency Weighted`.<br>- Thắng có ý nghĩa `SA-LSTM Fixed/Tuned` ($p < 0.025$) ở benchmark `K=5`. | - Chạy 1 seed (`42`) và **chưa có tệp dự báo từng ngày chi tiết**, nên chưa tham gia ma trận DM chéo `5×5` theo ngày với 4 nhóm PP còn lại.<br>- Phụ thuộc nặng token `Copper`: bỏ `Copper` ở `FS03` làm MAE Residual rớt xuống `0.09027 (-17.12%)`; nhánh Direct kém nhất nhóm (`1.14–1.81`). | Đạt độ chính xác điểm dẫn đầu khi có `Copper` (`FS01, FS02, FS04`) + Residual, nhưng không phù hợp cho bài toán Direct Level hoặc thiếu biến mục tiêu (`FS03`). |
| **PP4 — Rolling & Expanding Window Walk-Forward LSTM** | - Giải quyết tốt bài toán **Regime Shift** ở nhánh Direct: giảm `64%–85%` sai số so với Static Direct ($\text{DM}_{L1} = -51.36, p < 0.0001$).<br>- Có đầy đủ kiểm định 5 seeds (`41–45`) cho cả **Expanding `FS02` (`0.07828 ± 0.00063`)**, **Rolling `FS02` (`0.07864 ± 0.00122`)** và **Expanding `FS03` (`0.07790 ± 0.00116`)**, chứng minh Expanding ổn định gấp ~2 lần Rolling và giữ TB 6 FS thấp nhất (`0.07756 ± 0.00055`). | - Đứng **cuối bảng Cross-Ratio 4 mốc chia (`FS02`, TB `0.07562`, `0/4` mốc vượt Naive)**.<br>- Trên 2 bộ biến cốt lõi **`FS01` (`0.07736`) & `FS02` (`0.07852`)** ở `90/10`, `WF-Expanding` có MAE cao hơn `PP1 LSTM-Res` (`0.07701`), `PP3 iTransformer`, `PP2 SA-LSTM (FS01)` và kém cả `Static (T+V)` của chính `PP4` ở `FS02` (`0.07812`).<br>- Tốn thời gian GPU gấp `7–28` lần mô hình tĩnh. | `0/30` lượt Residual thắng Naive (`p = 0.438` ở `FS01`, `p = 0.735` ở `FS03`, `0/5` seed `FS03` thắng Naive), hòa DM `PP1 LSTM-Res` (`4/4`) & `PP5 5-seed ensemble` (`6/6`). |
| **PP5 — Hybrid ARIMA–GRU (`FS01–FS06`, 5 seeds)** | - Hoàn thiện đủ `6 FS × 4 splits Expanding × 5 seeds` (`360` lượt train GRU 2 tầng chuẩn nhóm) + xuất đủ 36 file predictions.<br>- Đóng góp đối chuẩn kinh tế lượng **`auto_arima`**: đứng **#1 trung bình Cross-Ratio 4 mốc (`0.07331`)** ở nhánh tuyến tính thuần (`92.5/7.5: 0.07456`; `95/5: 0.07258`).<br>- Khi lấy **Ensemble 5 seeds**, triệt nhiễu tốt và thắng `PP2 SA/TPE` ở một số bộ đa biến. | - Khi đồng bộ GRU 2 tầng chuẩn nhóm, việc ghép thêm GRU vào phần dư ARIMA lại làm **MAE kém hơn cả ARIMA thuần & Naive** trên cả 6/6 bộ ở `90/10 Expanding` (`FS01 = 0.07745`, TB `0.08048`) và bùng sai số nặng ở `Rolling 756` (`+32.45%` TB, `FS03 = +101.22%`).<br>- Ở `80/20` và `90/10`, `auto_arima` chọn `(0,1,0)` trùng hệt Naive; thêm biến ngoại sinh luôn làm xấu đi. | `0/120` lượt `Expanding` (`1/180` tính cả `Recency/Rolling`) thắng Naive $L_1$, `57/120` Exp thua Naive có ý nghĩa. Chứng minh rõ việc xếp chồng GRU phi tuyến lên phần dư ARIMA không vượt được bước ngẫu nhiên. |

---

## 6. Danh mục Tệp Tổng hợp Chung (`group_benchmark_summary/`)

- Báo cáo tổng hợp: [THONG_KE_CHISO_SIGNIFICANCE_ABLATION.md](THONG_KE_CHISO_SIGNIFICANCE_ABLATION.md)
- Bảng Ablation 6 bộ biến toàn nhóm: [group_ablation_6_feature_sets_all_members.csv](group_ablation_6_feature_sets_all_members.csv) (`21` dòng)
- Bảng Cross-Ratio 4 mốc chia toàn nhóm: [group_cross_ratio_4_splits_all_members.csv](group_cross_ratio_4_splits_all_members.csv) (`60` dòng)
- Bảng kiểm định Diebold–Mariano toàn nhóm: [group_significance_dm_tests_complete.csv](group_significance_dm_tests_complete.csv) (`232` dòng)
- 3 biểu đồ tổng hợp cân bằng toàn nhóm:
  - [fig16_group_benchmark_all_5_members.png](fig16_group_benchmark_all_5_members.png)
  - [fig17_group_ablation_6_feature_sets.png](fig17_group_ablation_6_feature_sets.png)
  - [fig18_group_dm_significance_and_pros_cons.png](fig18_group_dm_significance_and_pros_cons.png)
