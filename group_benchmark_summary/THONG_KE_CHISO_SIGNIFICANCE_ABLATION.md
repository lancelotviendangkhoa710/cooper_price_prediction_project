# BÁO CÁO TỔNG HỢP TOÀN NHÓM: THỐNG KÊ CHỈ SỐ, KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (DIEBOLD–MARIANO), NGHIÊN CỨU CẮT LỚP (ABLATION STUDY) & ĐÁNH GIÁ ƯU/NHƯỢC ĐIỂM 5 PHƯƠNG PHÁP

- **Học phần:** IS403.R11 — Phân tích dữ liệu kinh doanh  
- **Đề tài:** Dự báo giá đồng (*Copper Futures Price Forecasting*)  
- **Tổng hợp số liệu & kiểm định chéo toàn nhóm:** Lê Anh Quân *(thực hiện trên cơ sở đối chuẩn độc lập và trung lập giữa toàn bộ 5 thành viên)*  
- **Ngày cập nhật:** 23/11/2026 *(Đã đồng bộ đầy đủ 6 bộ biến `FS01–FS06` từ Khoa, Công, Vĩ, Quân, Huy)*

---

## 1. Xác minh Tính Đồng nhất Dữ liệu & Giao thức Thực nghiệm Toàn Nhóm (5/5 Thành Viên)

### 1.1. Xác minh mã băm SHA-256 giữa môi trường Cục bộ và Kaggle
Báo cáo của **Khoa (`Module1/`)** và **Quân (`quan task/`)** ghi nhận SHA-256 của tệp `data/raw/dataset_9_commodities_daily.csv` là `9580ede1...`, trong khi manifest trên Kaggle của **Công (`TPE+LSTM/`)** và **Vĩ (`Vi Task/`)** ghi nhận `a761a908...`. Kiểm định đối chiếu cho thấy nguyên nhân duy nhất là ký tự xuống dòng Windows (`CRLF`) so với Linux (`LF`):

| Phiên bản tệp | Ký tự xuống dòng | Mã băm SHA-256 | Số phiên ($N$) | Khoảng thời gian | Sai lệch số học cực đại ($\max |x_i - y_i|$) |
|---|---|---|---:|---|---:|
| **Local Git (`data/raw/`)** | Windows `CRLF (\r\n)` | `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600` | 2.710 | `13/01/2016 → 13/11/2026` | **`0.00e+00`** |
| **Kaggle Dataset Snapshot** | Linux `LF (\n)` | `a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83` | 2.710 | `13/01/2016 → 13/11/2026` | **`0.00e+00`** |

Cả 5 thành viên dùng chung dữ liệu 2.710 phiên và khớp đường cơ sở **Naive Persistence ($\hat{y}_t = y_{t-1}$)** đến 6 chữ số thập phân ở cả 4 tỷ lệ chia:
- **Split 80/20 ($N_{\text{test}} = 542$):** $\text{MAE}_{\text{Naive}} = 0.069001\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.108000\text{ \$/lb}$
- **Split 90/10 ($N_{\text{test}} = 271$):** $\text{MAE}_{\text{Naive}} = 0.077077\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.101033\text{ \$/lb}$
- **Split 92.5/7.5 ($N_{\text{test}} = 204$):** $\text{MAE}_{\text{Naive}} = 0.075116\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.098399\text{ \$/lb}$
- **Split 95/5 ($N_{\text{test}} = 136$):** $\text{MAE}_{\text{Naive}} = 0.072799\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.096612\text{ \$/lb}$

### 1.2. Phạm vi thực nghiệm của 5 thành viên

| Thành viên | Thư mục mã nguồn | Phương pháp phụ trách | Tổ hợp Bộ biến (`FS01–FS06`) | Tỷ lệ chia & Chiến lược | Mục tiêu | Đa hạt giống (`Multi-Seed`) | Dữ liệu dự báo chi tiết (Raw Preds) |
|---|---|---|---|---|---|---|---|
| **1. Viên Đăng Khoa** | `Module1/` | **Non-SA Baselines & SA-LSTM Standalone** (`GRU`, `LSTM`, `CNN-LSTM`, `CNN-GRU`, `Ridge`, `SA-LSTM V21–V23`) | Đủ 6 bộ (V20+V22 Non-SA: `FS01, 02, 04, 06`; V21+V22 SA: `FS02–06`) | 4 mốc × 3 schemes (`Expanding`, `Rolling 756`, `Recency`) | `Direct` & `Residual` | 5 seeds (`41–45`) cho Walk-Forward (V20, V23) | Có (`grid_predictions.csv` 103.770+ dòng + `wf_predictions.csv`) |
| **2. Nguyễn Thành Công** | `TPE+LSTM/` | **Tối ưu siêu tham số SA–LSTM vs. TPE–LSTM (Optuna)** (192 search trials + 72 final fits) | Đủ 6 bộ (`FS01–FS06`) | 4 mốc (`Expanding`) + `Rolling 756` & `Recency` ở `90/10` | `Residual` | 1 seed (`42`) | Có (`test_predictions.csv`, 20.340 dòng) |
| **3. Vĩ** | `Vi Task/` | **iTransformer (Inverted Attention)** + Đối chuẩn SA-LSTM Fixed/Tuned | Đủ 6 bộ (`FS01–FS06`) | 4 mốc × 3 schemes | `Direct` & `Residual` | 1 seed (`42`) | Chưa đẩy file predictions lên GitHub (chỉ có bảng 216 dòng + DM ở K5) |
| **4. Lê Anh Quân** | `quan task/` | **Rolling & Expanding Window Walk-Forward Backtesting** (`Static T`, `Static T+V`, `Expanding`, `Rolling`, Ablation $W, S$, Cold/Warm) | Đủ 6 bộ (`FS01–FS06`) ở `90/10` | 4 mốc (`Cross-Ratio` trên `FS02`) | `Direct` & `Residual` | 5 seeds (`41–45`) trên `FS02 90/10` | Có (49 file `pred_*.csv`) |
| **5. Phan Ngọc Đức Huy** | `huy task/` | **Hybrid ARIMA–GRU** (`auto_arima` + GRU 2 tầng chuẩn nhóm `64→32→16→1` học phần dư, 360 lượt train GRU) | Đủ 6 bộ (`FS01–FS06`) | 4 mốc `Expanding` + `Rolling 756` & `Recency` ở `90/10` | `Residual` (Hybrid/ARIMA) & `Direct` (GRU) | 5 seeds (`41–45`) cho toàn bộ lưới | Có (36 file `predictions_*.csv`) |

---

## 2. Thống kê Chỉ số Tổng hợp Toàn Nhóm (Benchmarking Leaderboard)

### 2.1. Bảng xếp hạng nhánh Residual trên cấu hình chuẩn `Split 90/10 — Expanding` ($N_{\text{test}} = 271$, `29/10/2025 → 13/11/2026`)

Biểu đồ đối chiếu cân bằng: [fig16_group_benchmark_all_5_members.png](fig16_group_benchmark_all_5_members.png) · Bảng dữ liệu CSV: [group_ablation_6_feature_sets_all_members.csv](group_ablation_6_feature_sets_all_members.csv)

| Hạng (Best MAE) | Thành viên | Phương pháp / Mô hình | Bộ biến đạt đỉnh (Best FS) | Best MAE (\$/lb) | RMSE tại Best FS (\$/lb) | DA (%) tại Best FS | % Cải thiện vs. Naive (`0.07708`) | MAE Trung bình qua các bộ biến (\$/lb) | Độ lệch chuẩn MAE giữa các bộ biến (\$/lb) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| **1** | **Vĩ (PP3)** | **iTransformer Residual** | `FS01` (Copper) | **0.07681** | 0.10097 | 53.14% | **+0.35%** | 0.07969 *(6 bộ)* | 0.00527 |
| **2** | **Quân (PP4)** | **WF Expanding Residual (`W=756, S=40`)** | `FS03` (WTI, Gold, Silver) | **0.07693** | 0.10107 | 50.93% | **+0.19%** | 0.07756 *(6 bộ)* | 0.00055 |
| **3** | **Khoa (Module 1)** | **GRU Residual (Non-SA V20)** | `FS04` (5 Kim loại) | **0.07695** | **0.10081** | 52.77% | **+0.16%** | 0.07793 *(4 bộ)* | 0.00151 |
| **4** | **Công (PP2)** | **SA–LSTM Residual (Per-FS Search)** | `FS01` (Copper) | **0.07695** | 0.10084 | 52.77% | **+0.16%** | 0.07975 *(6 bộ)* | 0.00283 |
| **5** | **Khoa (Module 1)** | **LSTM Residual (Non-SA V20)** | `FS04` (5 Kim loại) | **0.07698** | 0.10095 | 52.77% | **+0.13%** | **0.07701 *(4 bộ có Cu)*** | **0.00002** |
| **—** | **Baseline** | **Naive Persistence ($\hat{y}_t = y_{t-1}$)** | `FS01` (Copper) | **0.07708** | **0.10103** | — | **0.00%** | **0.07708** | **0.00000** |
| **6** | **Khoa (Module 1)** | **CNN–LSTM Residual (Non-SA V20)** | `FS04` (5 Kim loại) | 0.07725 | 0.10086 | 46.86% | -0.22% | 0.07810 *(4 bộ)* | 0.00134 |
| **7** | **Công (PP2)** | **TPE–LSTM Residual (Optuna Per-FS)** | `FS01` (Copper) | 0.07727 | 0.10094 | 48.71% | -0.25% | 0.08212 *(6 bộ)* | 0.00896 |
| **8** | **Khoa (Module 1)** | **Ridge Residual ($\alpha = 1.0$)** | `FS01` (Copper) | 0.07731 | 0.10119 | 47.60% | -0.30% | 0.07937 *(4 bộ)* | 0.00217 |
| **9** | **Quân (PP4)** | **WF Rolling Residual (`W=756, S=40`)** | `FS01` (Copper) | 0.07732 | 0.10167 | 50.93% | -0.32% | 0.07853 *(6 bộ)* | 0.00119 |
| **10** | **Huy (PP5)** | **Hybrid ARIMA(0,1,0)–GRU** *(TB 5 seeds, GRU 2 tầng)* | `FS01` (Copper) | 0.07745 | 0.10142 | 48.71% | -0.48% | 0.08048 *(6 bộ)* | 0.00218 |
| **11** | **Quân (PP4)** | **Static (Train+Val) Residual** | `FS04` (5 Kim loại) | 0.07746 | 0.10070 | 48.33% | -0.50% | 0.07920 *(6 bộ)* | 0.00160 |

> **Nhận xét khách quan Bảng 2.1:**
> - **Độ chính xác điểm cao nhất (`Best MAE`):** **Vĩ (`iTransformer Residual FS01 = 0.07681`)** dẫn đầu (`+0.35%` vs Naive), theo sau rất sát bởi `Quân WF-Expanding (FS03 = 0.07693)`, `Khoa GRU-Res (FS04 = 0.07695)`, `Công SA-LSTM (FS01 = 0.07695)` và `Khoa LSTM-Res (FS04 = 0.07698)`. Khoảng cách giữa hạng 1 và Naive chỉ là `0.00027 $/lb`.
> - **Tính ổn định qua các bộ biến:** Xét trên 4 bộ biến có lịch sử Copper mà Khoa chạy (`FS01, FS02, FS04, FS06`), **`LSTM Residual` của Khoa (`0.07701 ± 0.00002 $/lb`)** đạt trung bình thấp nhất và độ phân tán hẹp nhất nhóm (cả 4/4 bộ đều vượt nhẹ Naive). Xét riêng trong 4 phương pháp chạy đủ cả 6 bộ biến bao gồm 2 bộ bỏ Copper (`FS03, FS05`), **`WF Expanding Residual` của Quân (`0.07756 ± 0.00055 $/lb`)** có trung bình và độ lệch chuẩn thấp nhất.

### 2.2. Đối chiếu nhánh Direct (Dự báo mức giá trực tiếp) toàn nhóm (`Split 90/10`)

| Thành viên | Mô hình (Nhánh Direct) | `FS01` MAE | `FS02` MAE | `FS03` MAE | `FS04` MAE | `FS05` MAE | `FS06` MAE | MAE Trung bình (\$/lb) | So với Naive (`0.07708`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Quân (PP4)** | **Walk-Forward Expanding Direct (`W=756, S=40`)** | **0.20265** | **0.23696** | **0.38606** | 0.33844 | **0.36182** | 0.21921 | **0.29086** *(6 bộ)* | Kém Naive +277% |
| **Khoa (Module 1)** | **GRU Direct (Non-SA V20/V22)** | 0.43711 | 0.24789 | — | **0.24082** | — | 0.24321 | **0.29226** *(4 bộ)* | Kém Naive +279% |
| **Quân (PP4)** | **Walk-Forward Rolling Direct (`W=756, S=40`)** | 0.27394 | 0.28371 | 0.46075 | 0.34973 | 0.41410 | 0.27422 | 0.34274 *(6 bộ)* | Kém Naive +345% |
| **Huy (PP5)** | **GRU Direct (TB 5 seeds, chuẩn nhóm)** | 0.33809 | 0.29023 | 0.49914 | 0.29322 | 0.56456 | 0.30653 | 0.38196 *(6 bộ)* | Kém Naive +396% |
| **Khoa (Module 1)** | **Paper SA-LSTM Direct (V21/V22)** | — | 0.97807 | 0.54072 | 0.47093 | 0.46499 | **0.18463** | 0.52787 *(5 bộ)* | Kém Naive +585% |
| **Khoa (Module 1)** | **LSTM Direct (Non-SA V20/V22)** | 0.44672 | 1.16396 | — | 0.24741 | — | 0.32494 | 0.54576 *(4 bộ)* | Kém Naive +608% |
| **Khoa (Module 1)** | **Reopt SA-LSTM Direct (V21/V22)** | — | 0.87742 | 0.59561 | 0.51047 | 0.46660 | 0.35892 | 0.56180 *(5 bộ)* | Kém Naive +629% |
| **Quân (PP4)** | **Static (Train-Only) Direct** | 0.49280 | 1.55159 | 0.31990 | 1.32269 | 0.56377 | 0.55846 | 0.80154 *(6 bộ)* | Kém Naive +940% |
| **Vĩ (PP3)** | **iTransformer Direct** | 1.75765 | 1.14381 | 1.54427 | 1.63235 | 1.81230 | 1.38313 | 1.54558 *(6 bộ)* | Kém Naive +1905% |

> **Phát hiện xuyên suốt số 1 (Direct vs. Residual):** **100% mô hình Direct** của cả 4 thành viên thực hiện đều kém xa Naive (`MAE = 0.185 – 1.812 $/lb` vs `0.07708 $/lb`) do giá đồng giai đoạn Test (`5.0 – 6.89 $/lb`) vượt ngoài biên chuẩn hóa `MinMaxScaler` của tập Train. Trên nhánh Direct, `WF Expanding Direct` của Quân (`TB = 0.29086`) và `GRU Direct` tĩnh của Khoa (`TB = 0.29226`) xấp xỉ nhau ở vị trí dẫn đầu trung bình, trong khi `Paper SA-LSTM Direct` của Khoa đạt điểm đơn thấp nhất trên `FS06` (`0.18463`), còn tái huấn luyện định kỳ của Quân giảm 64%–85% sai số so với `Static Direct`.

### 2.3. Thống kê xuyên suốt 4 tỷ lệ chia dữ liệu (`Cross-Ratio 80/20, 90/10, 92.5/7.5, 95/5`)

Bảng dữ liệu chi tiết: [group_cross_ratio_4_splits_all_members.csv](group_cross_ratio_4_splits_all_members.csv)

| Phương pháp đại diện | Thành viên | Bộ biến | `80/20` ($N=542$) MAE | `90/10` ($N=271$) MAE | `92.5/7.5` ($N=204$) MAE | `95/5` ($N=136$) MAE | Trung bình 4 mốc (\$/lb) | Số mốc vượt Naive |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Naive Persistence ($\hat{y}_t = y_{t-1}$)** | Baseline | `FS01` | **0.06900** | **0.07708** | **0.07512** | **0.07280** | **0.07350** | — |
| **ARIMA Standalone (`auto_arima`)** | Huy (PP5) | `FS01` | 0.06900 (0.00%) | 0.07708 (0.00%) | **0.07456 (+0.75%)** | **0.07258 (+0.31%)** | **0.07331** | 2 / 4 |
| **SA–LSTM Residual (Expanding)** | Công (PP2) | `FS01` | **0.06883 (+0.25%)** | **0.07695 (+0.16%)** | **0.07506 (+0.08%)** | **0.07252 (+0.38%)** | **0.07334** | **4 / 4** |
| **GRU Residual (Expanding)** | Khoa (M1) | `FS02` | **0.06878 (+0.32%)** | 0.07714 (-0.09%) | 0.07521 (-0.12%) | **0.07277 (+0.04%)** | **0.07348** | 2 / 4 |
| **LSTM Residual (Expanding)** | Khoa (M1) | `FS01` | 0.06920 (-0.29%) | **0.07701 (+0.09%)** | 0.07536 (-0.33%) | **0.07273 (+0.09%)** | **0.07358** | 2 / 4 |
| **iTransformer Residual (Expanding)** | Vĩ (PP3) | `FS01` | **0.06885 (+0.22%)** | **0.07681 (+0.35%)** | 0.07595 (-1.11%) | 0.07289 (-0.12%) | 0.07363 | 2 / 4 |
| **iTransformer Residual (Expanding)** | Vĩ (PP3) | `FS02` | 0.06945 (-0.65%) | **0.07701 (+0.09%)** | 0.07568 (-0.75%) | **0.07253 (+0.37%)** | 0.07367 | 2 / 4 |
| **TPE–LSTM Residual (Expanding)** | Công (PP2) | `FS01` | 0.06955 (-0.79%) | 0.07727 (-0.24%) | 0.07517 (-0.07%) | 0.07314 (-0.47%) | 0.07378 | 0 / 4 |
| **TPE–LSTM Residual (Expanding)** | Công (PP2) | `FS02` | 0.06946 (-0.66%) | 0.07798 (-1.17%) | 0.07564 (-0.69%) | 0.07337 (-0.79%) | 0.07411 | 0 / 4 |
| **Hybrid ARIMA–GRU (TB 5 seeds)** | Huy (PP5) | `FS01` | 0.07275 (-5.43%) | 0.07745 (-0.48%) | **0.07475 (+0.48%)** | 0.07284 (-0.06%) | 0.07445 | 1 / 4 |
| **Static (Train+Val) Residual** | Quân (PP4) | `FS02` | 0.07505 (-8.76%) | 0.07812 (-1.35%) | 0.07684 (-2.30%) | **0.07213 (+0.91%)** | 0.07553 | 1 / 4 |
| **WF Expanding Residual (`W=756, S=40`)** | Quân (PP4) | `FS02` | 0.07108 (-3.02%) | 0.07852 (-1.87%) | 0.07714 (-2.70%) | 0.07573 (-4.02%) | 0.07562 | 0 / 4 |

> **Nhận xét Bảng 2.3:** Qua 4 mốc chia thời gian lồng nhau, **`SA–LSTM (FS01)` của Công** là mô hình nơ-ron duy nhất vượt nhẹ Naive ở cả **4/4 mốc** (`TB = 0.07334`), **`ARIMA` thuần của Huy** đạt TB 4 mốc thấp nhất (`0.07331`), theo sát là **`GRU/LSTM Residual` của Khoa** (`0.07348 / 0.07358`) và **`iTransformer` của Vĩ** (`0.07363`). Ngược lại, trên bộ biến `FS02`, phương pháp tái huấn luyện định kỳ **`WF Expanding Residual` của Quân (`TB = 0.07562`, `0/4` mốc vượt Naive)** có sai số trung bình cao nhất bảng Cross-Ratio (do liên tục retrain mạng LSTM `lr = 5e-5` làm tăng nhẹ phương sai sai số trên chuỗi sai phân đã dừng sẵn).

---

## 3. Kiểm định Ý nghĩa Thống kê Toàn Nhóm (Diebold–Mariano Significance Tests)

Biểu đồ kiểm định đối xứng toàn nhóm: [fig18_group_dm_significance_and_pros_cons.png](fig18_group_dm_significance_and_pros_cons.png) · Bảng CSV đầy đủ (232 dòng kiểm định): [group_significance_dm_tests_complete.csv](group_significance_dm_tests_complete.csv)

Toàn bộ các phép kiểm định được chuẩn hóa theo **Diebold–Mariano (1995)** với ước lượng phương sai vững **Newey–West HAC** ($\text{lag} = 5$) và hiệu chỉnh mẫu nhỏ **Harvey–Leybourne–Newbold (HLN)** trên cả hàm mất mát $L_1$ ($|e_t|$) và $L_2$ ($e_t^2$). *(Lưu ý: Vĩ chưa đưa raw predictions lên GitHub nên chưa tính DM-vs-Naive cho `FS01–FS06` của Vĩ; Vĩ có kết quả DM riêng ở notebook benchmark `K=5`.)*

### 3.1. Kiểm định giả thuyết $H_0^{(1)}$: *"Có mô hình Residual nào vượt Naive Persistence có ý nghĩa thống kê không?"*

Bảng kiểm kê đầy đủ trên **toàn bộ 529 lượt chạy Residual** có dữ liệu dự báo từng phiên từ 4 thành viên (Khoa, Công, Quân, Huy):

| Thành viên | Tập lượt chạy kiểm định với Naive Persistence | Số lượt (Tổng / Expanding) | Thắng Naive ý nghĩa 5% ($L_1$) | Thắng Naive ý nghĩa 5% ($L_2$) | Kém Naive ý nghĩa 5% ($L_1$) | Thống kê $\text{DM}_{L1}$ & $p$-value tiêu biểu |
|---|---|---:|---:|---:|---:|---|
| **1. Khoa (`Module1/` — Grid V20+V22)** | 5 mô hình Residual (`lstm, gru, cnn_lstm, cnn_gru, ridge`) $\times$ 4 FS $\times$ 4 splits $\times$ 3 schemes | `240` (`80` Expanding) | **0 / 240** (`0 / 80` Exp) | **2 / 240** *(chỉ ở `gru_residual`)* | `102 / 240` (`18 / 80` Exp; `lstm_res` Exp: **`0/16`**) | `lstm_res FS04`: $\text{DM} = -0.28, p = 0.780$; `gru_res FS04`: $\text{DM} = -0.49, p = 0.622$ |
| **1b. Khoa (`Module1/` — WF V20+V23)** | Walk-Forward `gru_residual` & `lstm_residual` qua 5 seeds `41–45` | `15` (`15` Expanding) | **0 / 15** | — | `2 / 15` | V20 `gru_res` Seed 41 ($N=2.214$): $p = 0.704$; V23 `gru_res` Seed 41: $p = 0.493$ |
| **2a. Công (`TPE+LSTM/`)** | `SA–LSTM Residual` (6 FS $\times$ 6 bối cảnh) | `36` (`24` Expanding) | **0 / 36** (`0 / 24` Exp) | **1 / 36** (`0 / 24` Exp) | `19 / 36` (`11 / 24` Exp) | `FS01` (90/10 Exp): $\text{DM} = -0.58, p = 0.564$; `FS06`: $\text{DM} = +0.02, p = 0.983$ |
| **2b. Công (`TPE+LSTM/`)** | `TPE–LSTM Residual` (6 FS $\times$ 6 bối cảnh) | `36` (`24` Expanding) | **0 / 36** (`0 / 24` Exp) | **0 / 36** | `18 / 36` (`8 / 24` Exp) | `FS01` (90/10 Exp): $\text{DM} = +1.25, p = 0.212$; `FS02`: $\text{DM} = +1.69, p = 0.093$ |
| **3. Quân (`quan task/`)** | `90/10` (`6 FS × 3 chiến lược: Exp, Roll, Static T+V`) + `4 seeds bổ sung (41, 43–45)` ở `FS02` | `22` (`10` Expanding) | **0 / 22** (`0 / 10` Exp) | **0 / 22** | `5 / 22` (`1 / 10` Exp Seed 43) | `FS03` Exp: $\text{DM} = -0.34, p = 0.735$; `FS01` Exp: $\text{DM} = +0.78, p = 0.438$ |
| **4. Huy (`huy task/`)** | `Hybrid ARIMA–GRU`: `Expanding` (4 splits × 6 FS × 5 seeds = `120`) + `Recency/Rolling 756` ở 90/10 (`60`) | `180` (`120` Expanding) | **1 / 180** (`0 / 120` Exp; duy nhất `Recency FS01 s42`, $p=0.0385$) | **0 / 180** (`0 / 120` Exp) | `57 / 120` ở Exp; `69 / 90` ở 90/10 cả 3 schemes | `90/10 Exp FS01`: $\text{DM}_{tb} = +0.70, p_{tb} = 0.355$; `FS06`: $\text{DM}_{tb} = +1.90, p_{tb} = 0.100$ |

> **Kết luận Kiểm định $H_0^{(1)}$ (Đồng thuận toàn nhóm):**
> - Trên **tổng cộng 273 lượt chạy chuẩn `Expanding` độc nhất** (`80` Khoa grid + `15` Khoa WF + `48` Công + `10` Quân + `120` Huy): **`0 / 273 (0.0%)` lượt chạy thắng được Naive Persistence có ý nghĩa thống kê dưới hàm mất mát $L_1$**!
> - Tính chung trên **toàn bộ 529 lượt chạy Residual** (gồm cả `Rolling` và `Recency`), có **`528 / 529 lượt (99.8%)` KHÔNG vượt qua được Naive dưới $L_1$**.
> - Kết quả này củng cố bằng chứng thực nghiệm đồng nhất của Khoa (`Module1/`) và cả nhóm: chuỗi sai phân giá đồng ngày hành xử như quá trình **Martingale Difference Sequence (MDS)** với tỷ lệ tín hiệu trên nhiễu cực thấp ($|r_{\text{lag-1}}| \le 0.091$), khiến nghiệm tối ưu dưới MSE bị co phương sai (**Variance Shrinkage**), hội tụ tiệm cận về Naive $\hat{y}_t \approx y_{t-1}$.

### 3.2. Kiểm định đối đầu nội bộ `TPE–LSTM vs. SA–LSTM` của Công (36 cặp)

| Bộ biến (`Split 90/10 Expanding`, $N=271$) | SA–LSTM MAE (\$/lb) | TPE–LSTM MAE (\$/lb) | Chênh lệch $\bar{d}_{L1} = \text{TPE} - \text{SA}$ | $\text{DM}_{L1}$ ($p_{L1}$) | $\text{DM}_{L2}$ ($p_{L2}$) | Kết luận ở mức ý nghĩa $\alpha = 5\%$ |
|---|---:|---:|---:|---:|---:|---|
| **`FS01` (Copper)** | **0.07695** | 0.07727 | +0.00031 | `+1.032` (`0.3029`) | `+0.320` (`0.7489`) | Không khác biệt ý nghĩa |
| **`FS02` (Paper3 + Cu)** | 0.08453 | **0.07798** | **-0.00655** | **`-3.159` (`0.0018`)** | **`-3.562` (`0.0004`)** | **TPE–LSTM thắng có ý nghĩa thống kê ($p < 0.01$)** |
| **`FS03` (Paper3 - Cu)** | 0.07999 | **0.07812** | **-0.00187** | **`-3.135` (`0.0019`)** | **`-3.301` (`0.0011`)** | **TPE–LSTM thắng có ý nghĩa thống kê ($p < 0.01$)** |
| **`FS04` (5 Kim loại)** | **0.08097** | 0.10006 | **+0.01910** | **`+3.685` (`0.0003`)** | **`+3.635` (`0.0003`)** | **SA–LSTM thắng có ý nghĩa thống kê ($p < 0.001$)** |
| **`FS05` (4 Kim loại - Cu)** | **0.07896** | 0.08198 | +0.00302 | `+1.182` (`0.2384`) | `+1.993` (`0.0473`) | Không khác biệt $L_1$ (SA tốt hơn $L_2$) |
| **`FS06` (WTI + Cu)** | **0.07709** | 0.07731 | +0.00023 | `+0.661` (`0.5095`) | `-0.215` (`0.8300`) | Không khác biệt ý nghĩa |

- **Trên cả 36 cặp bối cảnh (`6 FS × 6 bối cảnh`) của Công:** `18/36` (50.0%) tương đương thống kê; **SA–LSTM thắng ý nghĩa `11/36` (30.6%)** (tập trung ở `FS04` và `FS01`, nơi SA ít bị overfit vào Validation); **TPE–LSTM thắng ý nghĩa `7/36` (19.4%)** (tập trung ở `FS02` và `FS03`).

### 3.3. Kiểm định DM chéo đối xứng giữa các phương pháp của 5 thành viên (`90/10 Expanding`, $N = 271$)

Bảng dưới đây tổng hợp **đối đầu từng đôi một không thiên vị** (từ [group_significance_dm_tests_complete.csv](group_significance_dm_tests_complete.csv), khớp cùng `Seed = 42` trên các bộ biến dùng chung):

| Cặp so sánh (`A vs B`, Seed 42) | Số FS chung | Số bộ `A` thắng ý nghĩa | Số bộ `B` thắng ý nghĩa | Hòa thống kê ($p \ge 0.05$) | $\text{DM}_{L1}$ trung bình | Các bộ biến có ý nghĩa thống kê ($p_{L1} < 0.05$) |
|---|---:|---:|---:|---:|---:|---|
| **`Khoa LSTM-Res` vs `Khoa GRU-Res`** | 4 | **1** | 0 | 3 | -1.15 | Khoa LSTM thắng ở `FS01` ($p=0.0071$) |
| **`Khoa LSTM-Res` vs `Công SA-LSTM`** | 4 | **2** | 0 | 2 | -1.26 | Khoa LSTM thắng ở `FS02` ($p=0.0024$) & `FS04` ($p=0.0051$) |
| **`Khoa LSTM-Res` vs `Công TPE-LSTM`** | 4 | **1** | 0 | 3 | -1.98 | Khoa LSTM thắng ở `FS04` ($p<0.0001$) |
| **`Khoa LSTM-Res` vs `Quân WF-Expanding`** | 4 | 0 | 0 | **4** | -1.08 | **Hòa cả 4/4** ($p \in [0.110, 0.475]$, nhưng Khoa có MAE thấp hơn cả 4/4) |
| **`Khoa LSTM-Res` vs `Huy Hybrid (s42)`** | 4 | **3** | 0 | 1 | -3.12 | Khoa LSTM thắng ở `FS02` ($p=0.0128$), `FS04` ($p<0.0001$), `FS06` ($p=0.0045$) |
| **`Khoa GRU-Res` vs `Quân WF-Expanding`** | 4 | 0 | **1** | 3 | -0.02 | Quân thắng ở `FS01` ($p=0.0054$); Khoa nhỉnh điểm ở `FS02, 04, 06` ($p \in [0.099, 0.767]$) |
| **`Công SA-LSTM` vs `Quân WF-Expanding`** | 6 | 0 | **3** | 3 | +1.28 | Quân thắng ở `FS02` ($p=0.0018$), `FS03` ($p=0.0169$), `FS04` ($p=0.0351$) |
| **`Công TPE-LSTM` vs `Quân WF-Expanding`** | 6 | 0 | **2** | 4 | +1.28 | Quân thắng ở `FS04` ($p<0.0001$) & `FS05` ($p=0.0179$) |
| **`Công SA-LSTM` vs `Huy Hybrid (s42)`** | 6 | **2** | **1** | 3 | -2.27 | Công SA thắng `FS04, FS06` ($p \le 0.0005$); **Huy s42 thắng `FS02`** ($p=0.0426$) |
| **`Công TPE-LSTM` vs `Huy Hybrid (s42)`** | 6 | **3** | 0 | 3 | -2.02 | Công TPE thắng `FS03, FS05, FS06` ($p \le 0.0007$) |
| **`Quân WF-Expanding` vs `Huy Hybrid (s42)`** | 6 | **4** | 0 | 2 | -2.53 | Quân thắng `FS03, FS04, FS05, FS06` ($p \in [<0.0001, 0.0195]$) |

- **Kiểm định độ nhạy khi dùng Ensemble 5 seeds của Huy (`mean(Hybrid_s41..s45)`):** Khi trung bình hóa 5 hạt giống ngẫu nhiên của Huy, sai số ngẫu nhiên giảm mạnh, làm đảo ngược một số kết quả so với `s42` đơn lẻ:
  - **Huy 5-seed Ensemble thắng có ý nghĩa** `Công SA-LSTM` trên `FS02` ($p = 0.0021$), `Công TPE-LSTM` trên `FS04` ($p < 0.0001$) & `FS05` ($p = 0.0011$), và `Khoa GRU-Res` trên `FS01` ($p = 0.0043$); đồng thời hòa thống kê với `Quân WF-Expanding` trên cả 6/6 bộ ($p \in [0.176, 0.941]$).
  - Duy nhất **`Khoa LSTM-Res`** vẫn thắng có ý nghĩa Huy 5-seed Ensemble trên `FS04` ($p = 0.0055$) & `FS06` ($p = 0.0346$) và không thua trên bất kỳ bộ biến nào (`mean_DM = -2.00`).
- **Kiểm định DM của Vĩ (`iTransformer vs SA-LSTM`, benchmark `K=5`):** `iTransformer Residual (MAE = 0.07812)` thắng có ý nghĩa thống kê cả `SA-LSTM Fixed (0.08293, p = 0.0248)` lẫn `SA-LSTM Tuned (0.08535, p = 0.00014)`.
- **Kiểm định DM nội bộ Walk-Forward của Quân:** Trên nhánh Direct, `Expanding` & `Rolling` thắng `Static (Train-Only)` tuyệt đối ($\text{DM}_{L1} = -51.36$ & $-43.54, p < 0.0001$). Trên nhánh Residual (`FS02`), `Expanding` thắng `Static Train-Only` ($p < 0.0001$), nhưng hòa `Rolling` ($p = 0.412$).

---

## 4. Nghiên cứu Cắt lớp Toàn Nhóm (Group-Wide Ablation Studies)

Biểu đồ Heatmap & đối chiếu Direct/Residual 6 bộ biến: [fig17_group_ablation_6_feature_sets.png](fig17_group_ablation_6_feature_sets.png)

### 4.1. Ablation 1 — Vai trò 6 Bộ biến (`FS01` → `FS06`) trên cả 5 thành viên (`Split 90/10 — Expanding`)

| Mã bộ biến | Các biến trong tensor $X_{t-22:t-1}$ | $K$ | Khoa (`LSTM-Res`) | Khoa (`Ridge-Res`) | Công (`SA-LSTM`) | Công (`TPE-LSTM`) | Vĩ (`iTrans-Res`) | Quân (`WF-Exp-Res`) | Huy (`Hybrid-Res`) | Quân (`WF-Exp-Dir`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`FS01_CU`** | `Copper` | 1 | 0.07701 | **0.07731** | **0.07695** | **0.07727** | **0.07681** | 0.07736 | **0.07745** | **0.20265** |
| **`FS02_PAPER_CU`** | `Copper, WTI, Gold, Silver` | 4 | 0.07701 | 0.08071 | 0.08453 | 0.07798 | 0.07701 | 0.07852 | 0.08110 | 0.23696 |
| **`FS03_PAPER_EXO`** | `WTI, Gold, Silver` *(Bỏ Copper)* | 3 | — | — | 0.07999 | 0.07812 | *0.09027* | **0.07693** | *0.08334* | *0.38606* |
| **`FS04_METALS_CU`** | `Copper, Al, Tin, Silver, Gold` | 5 | **0.07698** | *0.08171* | 0.08097 | *0.10006* | 0.07687 | 0.07769 | 0.08185 | 0.33844 |
| **`FS05_METALS_EXO`** | `Al, Tin, Silver, Gold` *(Bỏ Copper)* | 4 | — | — | 0.07896 | 0.08198 | 0.07924 | 0.07720 | 0.08071 | *0.36182* |
| **`FS06_WTI_CU`** | `Copper, Al, Tin, WTI` | 4 | 0.07704 | 0.07774 | 0.07709 | 0.07731 | 0.07794 | 0.07764 | 0.07844 | 0.21921 |
| **Trung bình** | — | — | **0.07701** | 0.07937 | 0.07975 | 0.08212 | 0.07969 | 0.07756 | 0.08048 | 0.29086 |

**3 phát hiện chung của toàn nhóm từ Ablation 6 bộ biến:**
1. **Nhiễu chiều từ biến ngoại sinh (*Exogenous Noise Inflation*):** Tương quan trễ bậc 1 ngày trên Train (`Gold, Silver, WTI, Al, Tin` vs `Copper`) đều rất yếu ($|r| \le 0.091$). Do đó, mô hình đơn biến `FS01` dẫn đầu trên **4/5 thành viên** (`Vĩ 0.07681`, `Công SA-LSTM 0.07695`, `Công TPE-LSTM 0.07727`, `Khoa Ridge 0.07731`, `Huy 0.07745`) và ngang đỉnh trên `Khoa LSTM (0.07701)`. Khi thêm nhiều biến ngoại sinh ở `FS02` hoặc `FS04`, hầu hết kiến trúc bị nhiễu: `Ridge` của Khoa giảm `4.7%–6.0%`, `TPE-LSTM` của Công bị overfit lên `0.10006`, `SA-LSTM` ở `FS02` lên `0.08453`, còn `Hybrid ARIMA–GRU` của Huy xấu đi trên cả `FS02–FS06`.
2. **Tác động khi loại bỏ `Copper` khỏi tensor đầu vào (`FS03`, `FS05`):** Trên nhánh Direct, mất `Copper` làm sai số tăng `1.5–2.0` lần. Trên nhánh Residual (dù bước cộng ngược vẫn dùng $y_{t-1}$), `iTransformer` của Vĩ mất token neo trung tâm khiến MAE `FS03` vọt lên `0.09027` (`-17.12%` vs Naive), và `Hybrid` của Huy tăng lên `0.08334` (`-8.13%`). Riêng `WF Expanding` của Quân (`0.07693`, `0.07720`) ít bị dao động nhất khi vắng Copper.
3. **Nhóm kim loại + năng lượng (`FS06`):** Trong các bộ đa biến, `FS06 (Copper, Al, Tin, WTI)` cho kết quả ổn định nhất trên Khoa (`LSTM 0.07704`, `Paper SA-LSTM Direct 0.18463`), Công (`0.07709`), Quân (`0.07764`) và Huy (`0.07844`).

### 4.2. Ablation 2 — Chiến lược Cửa sổ (`Expanding` vs. `Rolling 756` vs. `Recency`) & Tham số Walk-Forward (`W, S, Cold/Warm`)

1. **So sánh 3 chiến lược cửa sổ trên cả 5 thành viên:** **`Expanding Window`** ổn định hơn rõ rệt so với `Rolling 756` (~3 năm) ở `80/20` và `90/10` trên mọi kiến trúc, vì mạng học sâu cần tập mẫu đủ lớn (~2.173 phiên). Đặc biệt ở `schemes_90_10.csv` của **Huy**, sai số TB 6 FS của `Expanding` là `+4.42%`, `Recency` là `+6.18%`, còn `Rolling 756` vọt lên **`+32.45%`** (`FS03 Rolling 756` lên tới `+101.22%`). Tuy vậy, **`Recency Weighted` (half-life 252)** trên `FS01` lại giúp `iTransformer` (Vĩ) đạt đỉnh `0.07645` (`+0.82%` vs Naive) và giúp `Hybrid` (Huy, `Seed 42`) đạt `0.07652` ở `90/10`.
2. **Ablation tham số Walk-Forward (`W`, `S`, `Cold/Warm`) của Quân ([phase3_ablation_study.csv](../quan%20task/result/phase3_ablation_study.csv)):** Window $W=756$ (`0.28371`) tốt hơn $W=252$ (`0.33899`) và $W=504$ (`0.35547`); giảm Stride $S$ từ `60 → 10` hạ MAE Direct từ `0.36000 → 0.23003` nhưng tăng thời gian train từ `66.8s → 333.7s` ($S^*=40$ cân bằng tốt nhất). `Warm-Start` hợp nhánh **Direct** (`0.21942` vs `0.28371`), còn `Cold-Start` hợp nhánh **Residual** (`0.08026` vs `0.09777` tránh *gradient drift*).

---

## 5. Đánh giá Ưu / Nhược điểm & Tính Có Ý Nghĩa của 5 Phương Pháp (Khách quan & Cân bằng)

| Thành viên & Phương pháp | Ưu điểm nổi bật (Strengths) | Nhược điểm & Giới hạn (Weaknesses) | Kết luận thống kê cốt lõi |
|---|---|---|---|
| **1. Khoa (`Module1/`)** — *Non-SA Baselines (`LSTM, GRU, Ridge`) & SA-LSTM V21–V23* | - Khung kiểm chứng bài bản nhất nhóm: `240` lượt Grid + Walk-Forward 2.214 phiên $\times$ 5 seeds + hệ thống kiểm định kinh tế lượng đầy đủ (ADF, KPSS, Zivot–Andrews, Granger, Spearman FDR) và giải thích toán học *Variance Shrinkage*.<br>- `LSTM Residual` đạt trung bình **tốt nhất trên 4 bộ có Copper (`0.07701 ± 0.00002`)**, hòa/thắng DM chéo trước hầu hết mô hình khác trong nhóm (`0/16` lượt Expanding thua Naive). | - Các mô hình Non-SA dùng siêu tham số cố định (`lr=1e-3`), chưa HPO riêng từng FS; bộ Non-SA chưa chạy 2 bộ bỏ Copper (`FS03, FS05`).<br>- `Ridge Residual` nhạy cảm nhiễu ngoại sinh (`40/48` lượt thua Naive có ý nghĩa, `FS04` giảm `-6.01%`). | `0/240` (Grid L1) & `0/15` (WF) lượt thắng Naive; là chuẩn baseline chặt chẽ nhất giải thích tính chất Random Walk của chuỗi giá đồng ngày. |
| **2. Công (`TPE+LSTM/`)** — *SA–LSTM vs. TPE–LSTM (Optuna)* | - Đối chuẩn HPO công bằng cùng ngân sách (16 trials/FS trên `80/20`, 192 search + 72 final fits) và chạy đủ 6 FS × 6 bối cảnh.<br>- **`SA-LSTM (FS01)` là mô hình nơ-ron duy nhất vượt nhẹ Naive ở cả 4/4 mốc Cross-Ratio** (`TB = 0.07334`).<br>- `TPE-LSTM` thắng `SA-LSTM` có ý nghĩa ($p < 0.01$) trên `FS02` & `FS03`. | - Chạy 1 seed (`42`), mô hình tĩnh trên Test.<br>- `TPE-LSTM` dễ **overfit Validation** trên tập đa cộng tuyến cao (`FS04 = 0.10006`, thua SA với $p=0.0003$), còn `SA-LSTM` dao động mạnh trên `FS02` (`0.08453`, thua Khoa LSTM & Quân WF với $p < 0.003$). | `18/36` cặp SA vs TPE hòa thống kê; SA thắng `11/36`, TPE thắng `7/36`; cả hai đều có `0/36` lượt thắng Naive dưới $L_1$. |
| **3. Vĩ (`Vi Task/`)** — *iTransformer (Inverted Attention)* | - Kiến trúc Transformer hiện đại đảo trục biến thành variable-tokens (`Dense 22→48`).<br>- Đạt **MAE thấp nhất toàn nhóm ở `90/10 Expanding` (`FS01 = 0.07681`, `FS04 = 0.07687`)** và đạt `0.07645 (+0.82%)` khi dùng `Recency Weighted`.<br>- Thắng có ý nghĩa `SA-LSTM Fixed/Tuned` ($p < 0.025$) ở benchmark `K=5`. | - Chạy 1 seed (`42`) và **chưa đưa file raw predictions lên GitHub**, nên chưa tham gia bảng DM chéo `5×5` theo ngày với 4 thành viên kia.<br>- Phụ thuộc nặng token `Copper`: bỏ `Copper` ở `FS03` làm MAE Residual rớt xuống `0.09027 (-17.12%)`; nhánh Direct kém nhất nhóm (`1.14–1.81`). | Đạt độ chính xác điểm dẫn đầu khi có `Copper` (`FS01, FS02, FS04`) + Residual, nhưng không phù hợp cho bài toán Direct Level hoặc thiếu biến mục tiêu (`FS03`). |
| **4. Quân (`quan task/`)** — *Rolling & Expanding Window Walk-Forward* | - Giải quyết tốt bài toán **Regime Shift** ở nhánh Direct: giảm 64%–85% sai số so với Static Direct ($\text{DM}_{L1} = -51.36, p < 0.0001$).<br>- Trong 4 phương pháp chạy đủ 6 bộ biến (gồm `FS03, FS05`), `WF-Expanding Residual` ổn định nhất (`0.07756 ± 0.00055`), thắng DM đơn seed `42` của Công (`FS02–05`) & Huy (`FS03–06`). | - Đứng **cuối bảng Cross-Ratio 4 mốc chia (`FS02`, TB `0.07562`, `0/4` mốc vượt Naive)**.<br>- Trên 2 bộ biến cốt lõi **`FS01` (`0.07736`) & `FS02` (`0.07852`)** ở `90/10`, `WF-Expanding` có MAE cao hơn Khoa `LSTM-Res` (`0.07701`), Vĩ `iTransformer`, Công `SA-LSTM (FS01)` và kém cả `Static (T+V)` của chính Quân ở `FS02` (`0.07812`).<br>- Tốn thời gian GPU gấp `7–28` lần mô hình tĩnh. | `0/22` lượt Residual thắng Naive (`p = 0.438` ở `FS01`, `p = 0.735` ở `FS03`), hòa DM Khoa `LSTM-Res` (`4/4`) & Huy `5-seed ensemble` (`6/6`). Chỉ thật sự vượt trội khi chạy Direct dài hạn hoặc giữ ổn định trên bộ biến bỏ Copper. |
| **5. Huy (`huy task/`)** — *Hybrid ARIMA–GRU (`FS01–FS06`, 5 seeds)* | - Hoàn thiện đủ `6 FS × 4 splits Expanding × 5 seeds` (`360` lượt train GRU 2 tầng chuẩn nhóm) + xuất đủ 36 file predictions.<br>- Đóng góp đối chuẩn kinh tế lượng **`auto_arima`**: đứng **#1 trung bình Cross-Ratio 4 mốc (`0.07331`)** ở nhánh tuyến tính thuần (`92.5/7.5: 0.07456`; `95/5: 0.07258`).<br>- Khi lấy **Ensemble 5 seeds**, triệt nhiễu tốt và thắng `Công SA/TPE` ở một số bộ đa biến. | - Khi đồng bộ GRU 2 tầng chuẩn nhóm, việc ghép thêm GRU vào phần dư ARIMA lại làm **MAE kém hơn cả ARIMA thuần & Naive** trên cả 6/6 bộ ở `90/10 Expanding` (`FS01 = 0.07745`, TB `0.08048`) và bùng sai số nặng ở `Rolling 756` (`+32.45%` TB, `FS03 = +101.22%`).<br>- Ở `80/20` và `90/10`, `auto_arima` chọn `(0,1,0)` trùng hệt Naive; thêm biến ngoại sinh luôn làm xấu đi. | `0/120` lượt `Expanding` (`1/180` tính cả `Recency/Rolling`) thắng Naive $L_1$, `57/120` Exp thua Naive có ý nghĩa. Chứng minh rõ việc xếp chồng GRU phi tuyến lên phần dư ARIMA không vượt được bước ngẫu nhiên. |

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
