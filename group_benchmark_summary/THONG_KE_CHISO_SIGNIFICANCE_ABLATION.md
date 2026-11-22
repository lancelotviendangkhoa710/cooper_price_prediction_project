# BÁO CÁO TỔNG HỢP TOÀN NHÓM: THỐNG KÊ CHỈ SỐ, KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (DIEBOLD–MARIANO), NGHIÊN CỨU CẮT LỚP (ABLATION STUDY) & ĐÁNH GIÁ ƯU/NHƯỢC ĐIỂM TỪNG PHƯƠNG PHÁP

- **Học phần:** IS403.R11 — Phân tích dữ liệu kinh doanh  
- **Đề tài:** Dự báo giá đồng (*Copper Futures Price Forecasting*)  
- **Phụ trách tổng hợp & kiểm định toàn nhóm:** **Lê Anh Quân** (Phương pháp 4 + Tổng hợp Thống kê / Significance Test / Ablation Study)  
- **Ngày cập nhật:** 23/11/2026 *(Đã tích hợp bản cập nhật đầy đủ 6 bộ biến `FS01–FS06` của Huy ngày 23/11)*

---

## 1. Xác minh Tính Đồng nhất Dữ liệu & Giao thức Thực nghiệm Toàn Nhóm (5/5 Thành Viên)

### 1.1. Giải quyết triệt để câu hỏi mã băm SHA-256 giữa môi trường Cục bộ và Kaggle
Trong quá trình độc lập chạy thực nghiệm, báo cáo của **Khoa (`Module1/`)** và **Quân (`quan task/`)** ghi nhận mã băm SHA-256 của tệp `data/raw/dataset_9_commodities_daily.csv` là `9580ede1...`, trong khi manifest trên Kaggle của **Công (`TPE+LSTM/`)** và **Vĩ (`Vi Task/`)** ghi nhận `a761a908...`. Kiểm định đối chiếu byte và giá trị số học trên toàn bộ 2.710 dòng $\times$ 9 cột hàng hóa xác nhận:

| Phiên bản tệp | Ký tự xuống dòng | Mã băm SHA-256 | Số phiên ($N$) | Khoảng thời gian | Sai lệch số học cực đại ($\max |x_i - y_i|$) |
|---|---|---|---:|---|---:|
| **Local Git (`data/raw/`)** | Windows `CRLF (\r\n)` | `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600` | 2.710 | `13/01/2016 → 13/11/2026` | **`0.00e+00`** |
| **Kaggle Dataset Snapshot** | Linux `LF (\n)` | `a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83` | 2.710 | `13/01/2016 → 13/11/2026` | **`0.00e+00`** |

> **Kết luận kiểm chứng:** Khi chuẩn hóa `CRLF → LF`, mã băm của tệp cục bộ trùng khớp **100%** với `a761a908...` trên Kaggle, và sai lệch giá trị số học giữa `dataset_snapshot.csv` của Công/Vĩ với CSV gốc của Khoa/Huy/Quân bằng **`0.00e+00` tuyệt đối**. Cả 5 thành viên có chung mốc cơ sở **Naive Persistence ($\hat{y}_t = y_{t-1}$)** trùng khớp đến 6 chữ số thập phân ở cả 4 tỷ lệ chia:
> - **Split 80/20 ($N_{\text{test}} = 542$):** $\text{MAE}_{\text{Naive}} = 0.069001\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.108000\text{ \$/lb}$
> - **Split 90/10 ($N_{\text{test}} = 271$):** $\text{MAE}_{\text{Naive}} = 0.077077\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.101033\text{ \$/lb}$
> - **Split 92.5/7.5 ($N_{\text{test}} = 204$):** $\text{MAE}_{\text{Naive}} = 0.075116\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.098399\text{ \$/lb}$
> - **Split 95/5 ($N_{\text{test}} = 136$):** $\text{MAE}_{\text{Naive}} = 0.072799\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.096612\text{ \$/lb}$

### 1.2. Bảng phân công phương pháp và phạm vi thực nghiệm của 5 thành viên (Full Parity)

| Thành viên | Thư mục mã nguồn | Phương pháp phụ trách | Tổ hợp Bộ biến (`FS01–FS06`) | Tỷ lệ chia (`80/90/92.5/95`) | Mục tiêu (`Direct` / `Residual`) | Đa hạt giống (`Multi-Seed`) | Kiểm định DM Test |
|---|---|---|---|---|---|---|---|
| **1. Viên Đăng Khoa** | `Module1/` | **Non-SA Baselines & SA-LSTM Standalone** (`GRU`, `LSTM`, `CNN-LSTM`, `CNN-GRU`, `Ridge`, `XGBoost`, `SA-LSTM V21/V22/V23`) | Đủ 6 bộ (`FS01–FS06` qua V20/V21/V22) | Đủ 4 mốc | Cả `Direct` & `Residual` | 5 seeds (`41–45`) cho Walk-Forward | Có (`dm_hac_by_seed.csv`, `wf_dm_hac_by_seed.csv`) |
| **2. Nguyễn Thành Công** | `TPE+LSTM/` | **Tối ưu siêu tham số SA–LSTM vs. TPE–LSTM (Optuna)** (192 search trials + 72 final fits) | Đủ 6 bộ (`FS01–FS06`) | Đủ 4 mốc + `rolling_756`, `recency` | `Residual` | 1 seed (`42`) | Đã được Quân tính bổ sung đủ **36 cặp DM-HAC** từ `test_predictions.csv` |
| **3. Vĩ** | `Vi Task/` | **Kiến trúc iTransformer (Inverted Attention)** + Đối chuẩn SA-LSTM Fixed/Tuned | Đủ 6 bộ (`FS01–FS06`) | Đủ 4 mốc + `rolling_756`, `recency` | Cả `Direct` & `Residual` | 1 seed (`42`) | Có (DM-HAC giữa iTransformer vs SA-LSTM Fixed/Tuned) |
| **4. Lê Anh Quân** | `quan task/` | **Rolling & Expanding Window Walk-Forward Backtesting** (`Static T`, `Static T+V`, `Expanding`, `Rolling`, Ablation $W, S$, Cold/Warm) | Đủ 6 bộ (`FS01–FS06`) | Đủ 4 mốc | Cả `Direct` & `Residual` | 5 seeds (`41–45`) | Có (**20 cặp / 40 dòng DM** + **DM chéo giữa các thành viên**) |
| **5. Phan Ngọc Đức Huy** | `huy task/` | **Mô hình lai Hybrid ARIMA–GRU** (`auto_arima` + GRU 2 tầng chuẩn nhóm `64→32→16→1` học phần dư, 360 lượt train GRU) | **Đủ 6 bộ (`FS01–FS06`)** | Đủ 4 mốc `Expanding` + `rolling_756` & `recency` ở `90/10` | `Residual` (Hybrid/ARIMA) & `Direct` (GRU Direct) | 5 seeds (`41–45`) | Có (DM-HAC $L_1/L_2$ từng seed & TB vs Naive + **36 file raw predictions để DM chéo**) |

---

## 2. Thống kê Chỉ số Tổng hợp Toàn Nhóm (Benchmarking Leaderboard)

### 2.1. Bảng xếp hạng tổng hợp nhánh Residual trên cấu hình chuẩn `Split 90/10 — Expanding` ($N_{\text{test}} = 271$, `29/10/2025 → 13/11/2026`)

Biểu đồ đối chiếu trực quan: [fig16_group_benchmark_all_5_members.png](fig16_group_benchmark_all_5_members.png) · Bảng dữ liệu CSV: [group_ablation_6_feature_sets_all_members.csv](group_ablation_6_feature_sets_all_members.csv)

| Hạng (theo Best MAE) | Thành viên | Phương pháp / Mô hình | Bộ biến đạt đỉnh (Best FS) | Best MAE (\$/lb) | RMSE tại Best FS (\$/lb) | DA (%) tại Best FS | % Cải thiện vs. Naive (`0.07708`) | MAE Trung bình qua các bộ biến (\$/lb) | Độ lệch chuẩn MAE giữa các bộ biến (\$/lb) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| **1** | **Vĩ (PP3)** | **iTransformer Residual** | `FS01` (Copper) | **0.07681** | 0.10097 | 53.14% | **+0.35%** | 0.07969 *(6 bộ)* | 0.00527 |
| **2** | **Quân (PP4)** | **Walk-Forward Expanding Residual (`W=756, S=40`)** | `FS03` (WTI, Gold, Silver) | **0.07693** | 0.10107 | 50.93% | **+0.19%** | **0.07756 *(6 bộ)*** | **0.00055 (Ổn định nhất 6 bộ)** |
| **3** | **Khoa (Module 1)** | **GRU Residual (Non-SA V20)** | `FS04` (5 Kim loại) | **0.07695** | **0.10081** | 52.77% | **+0.16%** | 0.07793 *(4 bộ)* | 0.00151 |
| **4** | **Công (PP2)** | **SA–LSTM Residual (Per-FS Search)** | `FS01` (Copper) | **0.07695** | 0.10084 | 52.77% | **+0.16%** | 0.07975 *(6 bộ)* | 0.00283 |
| **5** | **Khoa (Module 1)** | **LSTM Residual (Non-SA V20)** | `FS04` (5 Kim loại) | **0.07698** | 0.10095 | 52.77% | **+0.13%** | **0.07701 *(4 bộ)*** | **0.00002** |
| **—** | **Baseline** | **Naive Persistence ($\hat{y}_t = y_{t-1}$)** | `FS01` (Copper) | **0.07708** | **0.10103** | — | **0.00%** | **0.07708** | **0.00000** |
| **6** | **Khoa (Module 1)** | **CNN–LSTM Residual (Non-SA V20)** | `FS04` (5 Kim loại) | 0.07725 | 0.10086 | 46.86% | -0.22% | 0.07810 *(4 bộ)* | 0.00134 |
| **7** | **Công (PP2)** | **TPE–LSTM Residual (Optuna Per-FS)** | `FS01` (Copper) | 0.07727 | 0.10094 | 48.71% | -0.25% | 0.08212 *(6 bộ)* | 0.00896 |
| **8** | **Khoa (Module 1)** | **Ridge Residual ($\alpha = 1.0$)** | `FS01` (Copper) | 0.07731 | 0.10119 | 47.60% | -0.30% | 0.07937 *(4 bộ)* | 0.00217 |
| **9** | **Quân (PP4)** | **Walk-Forward Rolling Residual (`W=756, S=40`)** | `FS01` (Copper) | 0.07732 | 0.10167 | 50.93% | -0.32% | 0.07853 *(6 bộ)* | 0.00119 |
| **10** | **Huy (PP5)** | **Hybrid ARIMA(0,1,0)–GRU** *(TB 5 seeds, GRU 2 tầng)* | `FS01` (Copper) | 0.07745 | 0.10142 | 48.71% | -0.48% | 0.08048 *(6 bộ)* | 0.00218 |
| **11** | **Quân (PP4)** | **Static (Train+Val) Residual** | `FS04` (5 Kim loại) | 0.07746 | 0.10070 | 48.33% | -0.50% | 0.07920 *(6 bộ)* | 0.00160 |

### 2.2. Đối chiếu nhánh Direct (Dự báo mức giá trực tiếp) vs. Nhánh Residual toàn nhóm (`Split 90/10`)

| Thành viên | Mô hình (Nhánh Direct) | `FS01` MAE | `FS02` MAE | `FS03` MAE | `FS04` MAE | `FS05` MAE | `FS06` MAE | MAE Trung bình 6 bộ (\$/lb) | So với Naive (`0.07708`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Quân (PP4)** | **Walk-Forward Expanding Direct (`W=756, S=40`)** | **0.20265** | **0.23696** | **0.38606** | 0.33844 | **0.36182** | 0.21921 | **0.29086** | Kém Naive +277% *(nhưng giảm 64%–85% sai số so với Static Direct)* |
| **Khoa (Module 1)** | **GRU Direct (Non-SA V20/V22)** | 0.43711 | 0.24789 | — | **0.24082** | — | 0.24321 | **0.29226** *(4 bộ)* | Kém Naive +279% |
| **Quân (PP4)** | **Walk-Forward Rolling Direct (`W=756, S=40`)** | 0.27394 | 0.28371 | 0.46075 | 0.34973 | 0.41410 | 0.27422 | **0.34274** | Kém Naive +345% |
| **Huy (PP5)** | **GRU Direct (TB 5 seeds, chuẩn nhóm)** | 0.33809 | 0.29023 | 0.49914 | 0.29322 | 0.56456 | 0.30653 | **0.38196** | Kém Naive +396% |
| **Khoa (Module 1)** | **Paper SA-LSTM Direct (V21/V22)** | — | 0.97807 | 0.54072 | 0.47093 | 0.46499 | **0.18463** | **0.52787** *(5 bộ)* | Kém Naive +585% |
| **Khoa (Module 1)** | **LSTM Direct (Non-SA V20/V22)** | 0.44672 | 1.16396 | — | 0.24741 | — | 0.32494 | **0.54576** *(4 bộ)* | Kém Naive +608% |
| **Khoa (Module 1)** | **Reopt SA-LSTM Direct (V21/V22)** | — | 0.87742 | 0.59561 | 0.51047 | 0.46660 | 0.35892 | **0.56180** *(5 bộ)* | Kém Naive +629% |
| **Quân (PP4)** | **Static (Train-Only) Direct** | 0.49280 | 1.55159 | 0.31990 | 1.32269 | 0.56377 | 0.55846 | **0.80154** | Kém Naive +940% |
| **Vĩ (PP3)** | **iTransformer Direct** | 1.75765 | 1.14381 | 1.54427 | 1.63235 | 1.81230 | 1.38313 | **1.54558** | Kém Naive +1905% |

> **Phát hiện xuyên suốt số 1 (Direct vs. Residual):**
> - **100% mô hình dự báo trực tiếp mức giá (Direct Level)** của cả Khoa, Vĩ, Quân, Huy đều thất bại nặng nề trước Naive Persistence (`MAE` từ `0.185` đến `1.812 $/lb` so với `0.07708 $/lb` của Naive). Nguyên nhân gốc rễ là giá đồng trong giai đoạn Test (2025–2026) bứt phá lên vùng đỉnh lịch sử `5.0 – 6.89 $/lb`, vượt ngoài miền giá `[0, 1]` của `MinMaxScaler` fit trên tập Train.
> - Tuy nhiên, trên nhánh Direct, **cơ chế tái huấn luyện định kỳ Walk-Forward Expanding của Quân (`MAE TB = 0.29086 $/lb`)** đã kéo giảm sai số từ **64% đến 85%** so với mô hình tĩnh `Static Train-Only Direct (0.80154 $/lb)` và thấp hơn hơn 5 lần so với `iTransformer Direct tĩnh (1.54558 $/lb)`, nhờ liên tục cập nhật lại biên `MinMaxScaler` theo mặt bằng giá mới mỗi $S = 40$ phiên.
> - Khi chuyển sang **nhánh Residual ($\Delta y_t = y_t - y_{t-1}$ trên thang chuẩn hóa của Copper)**, toàn bộ 5 thành viên đều đưa sai số MAE hội tụ sát mốc `0.07681 – 0.08048 $/lb` (xấp xỉ Naive `0.07708 $/lb`).

### 2.3. Thống kê xuyên suốt 4 tỷ lệ chia dữ liệu (`Cross-Ratio 80/20, 90/10, 92.5/7.5, 95/5`)

Bảng dữ liệu chi tiết: [group_cross_ratio_4_splits_all_members.csv](group_cross_ratio_4_splits_all_members.csv)

| Phương pháp đại diện | Thành viên | Bộ biến | `80/20` ($N=542$) MAE | `90/10` ($N=271$) MAE | `92.5/7.5` ($N=204$) MAE | `95/5` ($N=136$) MAE | Trung bình 4 mốc (\$/lb) | Số mốc vượt Naive |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Naive Persistence ($\hat{y}_t = y_{t-1}$)** | Baseline | `FS01` | **0.06900** | **0.07708** | **0.07512** | **0.07280** | **0.07350** | — |
| **ARIMA Standalone (`auto_arima`)** | Huy (PP5) | `FS01` | 0.06900 (0.00%) | 0.07708 (0.00%) | **0.07456 (+0.75%)** | **0.07258 (+0.31%)** | **0.07331** | 2 / 4 |
| **SA–LSTM Residual (Expanding)** | Công (PP2) | `FS01` | **0.06883 (+0.25%)** | **0.07695 (+0.16%)** | **0.07506 (+0.08%)** | **0.07252 (+0.38%)** | **0.07334** | **4 / 4** |
| **GRU Residual (Expanding)** | Khoa (M1) | `FS02` | **0.06878 (+0.32%)** | 0.07714 (-0.09%) | 0.07521 (-0.12%) | **0.07277 (+0.04%)** | **0.07348** | 2 / 4 |
| **LSTM Residual (Expanding)** | Khoa (M1) | `FS01` | 0.06920 (-0.29%) | **0.07701 (+0.09%)** | 0.07536 (-0.33%) | **0.07273 (+0.09%)** | **0.07358** | 2 / 4 |
| **iTransformer Residual (Expanding)** | Vĩ (PP3) | `FS01` | **0.06885 (+0.22%)** | **0.07681 (+0.35%)** | 0.07595 (-1.11%) | 0.07289 (-0.12%) | **0.07363** | 2 / 4 |
| **iTransformer Residual (Expanding)** | Vĩ (PP3) | `FS02` | 0.06945 (-0.65%) | **0.07701 (+0.09%)** | 0.07568 (-0.75%) | **0.07253 (+0.37%)** | **0.07367** | 2 / 4 |
| **TPE–LSTM Residual (Expanding)** | Công (PP2) | `FS01` | 0.06955 (-0.79%) | 0.07727 (-0.24%) | 0.07517 (-0.07%) | 0.07314 (-0.47%) | **0.07378** | 0 / 4 |
| **TPE–LSTM Residual (Expanding)** | Công (PP2) | `FS02` | 0.06946 (-0.66%) | 0.07798 (-1.17%) | 0.07564 (-0.69%) | 0.07337 (-0.79%) | **0.07411** | 0 / 4 |
| **Hybrid ARIMA–GRU (TB 5 seeds)** | Huy (PP5) | `FS01` | 0.07275 (-5.43%) | 0.07745 (-0.48%) | **0.07475 (+0.48%)** | 0.07284 (-0.06%) | **0.07445** | 1 / 4 |
| **Static (Train+Val) Residual** | Quân (PP4) | `FS02` | 0.07505 (-8.76%) | 0.07812 (-1.35%) | 0.07684 (-2.30%) | **0.07213 (+0.91%)** | **0.07553** | 1 / 4 |
| **WF Expanding Residual (`W=756, S=40`)** | Quân (PP4) | `FS02` | 0.07108 (-3.02%) | 0.07852 (-1.87%) | 0.07714 (-2.70%) | 0.07573 (-4.02%) | **0.07562** | 0 / 4 |

---

## 3. Kiểm định Ý nghĩa Thống kê Toàn Nhóm (Diebold–Mariano Significance Tests)

Biểu đồ bản đồ kiểm định DM toàn nhóm: [fig18_group_dm_significance_and_pros_cons.png](fig18_group_dm_significance_and_pros_cons.png) · Bảng CSV đầy đủ (144 dòng kiểm định): [group_significance_dm_tests_complete.csv](group_significance_dm_tests_complete.csv)

Để tránh kết luận cảm tính từ các chênh lệch thập phân thứ 4–5 (`0.0001 – 0.0003 $/lb`), toàn bộ các phép so sánh trong nhóm được chuẩn hóa bằng **kiểm định Diebold–Mariano (1995)** với ước lượng phương sai vững **Newey–West HAC** ($\text{lag} = 5$) và hiệu chỉnh mẫu nhỏ **Harvey–Leybourne–Newbold (HLN)** trên cả hai hàm mất mát $L_1$ ($|e_t|$) và $L_2$ ($e_t^2$).

### 3.1. Kiểm định giả thuyết trung tâm $H_0^{(1)}$: *"Có mô hình Residual nào vượt Naive Persistence có ý nghĩa thống kê không?"*

| Thành viên | Phép so sánh với Naive Persistence ($\hat{y}_t = y_{t-1}$) | Số cấu hình / lượt chạy kiểm định | Số lượt mô hình có MAE thấp hơn Naive ($\bar{d}_{L1} < 0$) | Số lượt thắng Naive có ý nghĩa thống kê 5% ($\text{DM}_{L1} < -1.96, p < 0.05$) | Số lượt kém Naive có ý nghĩa thống kê 5% ($\text{DM}_{L1} > +1.96, p < 0.05$) | Thống kê $\text{DM}_{L1}$ & $p$-value tiêu biểu |
|---|---|---:|---:|---:|---:|---|
| **1. Khoa (`Module1/`)** | `Walk-Forward GRU Residual` vs. `Naive` ($N=2.214$ phiên, 2018–2026, 5 seeds `41–45`) | 5 seeds | 1 / 5 (Seed 41) | **0 / 5 (0%)** *(L2: 0 / 5)* | 2 / 5 (Seeds 43, 44) | Seed 41: $\text{DM} = -0.379, p = 0.7043$; Seed 42: $\text{DM} = +0.905, p = 0.3656$ |
| **2. Công (`TPE+LSTM/`)** *(Quân tính từ raw preds)* | `SA–LSTM Residual` vs. `Naive` (6 bộ biến $\times$ 6 bối cảnh) | 36 cấu hình | 6 / 36 (`FS01`, `FS06`) | **0 / 36 (0%)** *(L2: 1 / 36)* | 19 / 36 | `FS01` (90/10 Exp): $\text{DM} = -0.577, p = 0.5644$; `FS06` (90/10 Exp): $\text{DM} = +0.022, p = 0.9825$ |
| **2. Công (`TPE+LSTM/`)** *(Quân tính từ raw preds)* | `TPE–LSTM Residual` vs. `Naive` (6 bộ biến $\times$ 6 bối cảnh) | 36 cấu hình | 1 / 36 (`FS01` Roll) | **0 / 36 (0%)** *(L2: 0 / 36)* | 18 / 36 | `FS01` (90/10 Exp): $\text{DM} = +1.251, p = 0.2120$; `FS02` (90/10 Exp): $\text{DM} = +1.688, p = 0.0926$ |
| **3. Quân (`quan task/`)** | `WF Expanding Residual` vs. `Naive` (6 bộ biến `FS01–FS06` ở seed 42 + 4 seeds bổ sung `41, 43–45` ở `FS02`) | 10 lượt độc nhất *(11 mục qua 2 bảng)* | 1 / 10 (`FS03`) | **0 / 10 (0%)** *(L2: 0 / 10)* | 1 / 10 (Seed 43) | `FS03`: $\text{DM} = -0.339, p = 0.7351$; `FS01`: $\text{DM} = +0.776, p = 0.4382$; `FS04`: $\text{DM} = +0.775, p = 0.4391$ |
| **4. Huy (`huy task/`) — Chuẩn Expanding** | `Hybrid ARIMA–GRU` vs. `Naive` (4 splits $\times$ 6 bộ biến $\times$ 5 seeds `41–45`) | **120 lượt** | 17 / 120 | **0 / 120 (0%)** *(L2: 0 / 120)* | **57 / 120** *(L2: 67 / 120)* | 90/10 `FS01`: $\text{DM}_{tb} = +0.697, p_{tb} = 0.355$; `FS06`: $\text{DM}_{tb} = +1.902, p_{tb} = 0.100$ |
| **4b. Huy (`huy task/`) — Cả 3 schemes ở 90/10** | `Hybrid ARIMA–GRU` (`Expanding`, `Recency`, `Rolling 756` $\times$ 6 FS $\times$ 5 seeds) | 90 lượt *(gồm 60 lượt scheme phụ)* | — | **1 / 90** *(duy nhất `Recency FS01 s42`, MAE `0.07652`, $p_{L1}=0.0385$, $p_{L2}=0.2059$)* | 69 / 90 | `Expanding FS01`: 0/5; `Recency FS01`: 1/5 thắng L1; `Rolling 756`: 0/30 thắng L1/L2 |

> **Kết luận Kiểm định $H_0^{(1)}$ (Đóng góp học thuật cốt lõi của toàn nhóm):**
> Tổng hợp trên toàn bộ **207 lần chạy chuẩn `Expanding` độc nhất** (`5` Khoa + `72` Công + `10` Quân + `120` Huy; `208` lượt nếu liệt kê tách seed 42 trong bảng multi-seed của Quân) so với `Naive Persistence`:
> - **`0 / 207 (0.0%)` lần chạy chuẩn `Expanding` thắng được Naive Persistence ở mức ý nghĩa thống kê $\alpha = 5\%$ dưới hàm mất mát $L_1$** (`206 / 207 = 99.5%` dưới $L_2$).
> - Ngay cả khi cộng gộp thêm **60 lượt chạy phụ `Rolling 756` & `Recency Weighted` của Huy ở 90/10** (tổng cộng **267 lần chạy độc nhất**), có tới **266 / 267 lượt (99.6%) không thể vượt qua Naive dưới $L_1$** — chỉ duy nhất 1 seed riêng lẻ (`Recency, FS01, Seed 42` của Huy, $p_{L1} = 0.0385$) đạt ngưỡng $p < 0.05$ nhưng vẫn không có ý nghĩa dưới $L_2$ ($p_{L2} = 0.2059$).
> - **Giải thích kinh tế lượng (Khoa V20 & Quân Phase 2):** Do chuỗi sai phân giá đồng ngày tuân theo quá trình **Martingale Difference Sequence (MDS)** với tỷ lệ tín hiệu trên nhiễu cực thấp ($|r_{\text{lag-1}}| \le 0.091$), nghiệm tối ưu của mạng nơ-ron dưới hàm mất mát MSE hội tụ về $\hat{\Delta}_t \approx 0$ (**Variance Shrinkage**), biến mọi mô hình Residual hội tụ tiệm cận về đường Random Walk $\hat{y}_t \approx y_{t-1}$.

### 3.2. Kiểm định giả thuyết $H_0^{(2)}$: *"TPE–LSTM có vượt trội có ý nghĩa thống kê so với SA–LSTM không?"* (Kiểm định mới do Quân thực hiện trên dữ liệu của Công)

Từ 20.340 bản ghi dự báo theo ngày trong `TPE+LSTM/.../test_predictions.csv`, Quân đã thực hiện kiểm định **DM-HAC cho toàn bộ 36 cặp đối đầu trực tiếp `TPE–LSTM vs. SA–LSTM`**:

| Bộ biến (`Split 90/10 Expanding`, $N=271$) | SA–LSTM MAE (\$/lb) | TPE–LSTM MAE (\$/lb) | Chênh lệch $\bar{d}_{L1} = \text{TPE} - \text{SA}$ | Thống kê $\text{DM}_{L1}$ | Giá trị $p_{L1}$ | Thống kê $\text{DM}_{L2}$ | Giá trị $p_{L2}$ | Kết luận ở mức ý nghĩa $\alpha = 5\%$ |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`FS01` (Copper)** | **0.07695** | 0.07727 | +0.00031 | +1.0321 | 0.3029 | +0.3204 | 0.7489 | Không khác biệt ý nghĩa ($p = 0.303$) |
| **`FS02` (Paper3 + Cu)** | 0.08453 | **0.07798** | **-0.00655** | **-3.1593** | **0.0018** | **-3.5620** | **0.0004** | **TPE–LSTM thắng có ý nghĩa thống kê ($p < 0.01$)** |
| **`FS03` (Paper3 - Cu)** | 0.07999 | **0.07812** | **-0.00187** | **-3.1352** | **0.0019** | **-3.3012** | **0.0011** | **TPE–LSTM thắng có ý nghĩa thống kê ($p < 0.01$)** |
| **`FS04` (5 Kim loại)** | **0.08097** | 0.10006 | **+0.01910** | **+3.6848** | **0.0003** | **+3.6353** | **0.0003** | **SA–LSTM thắng có ý nghĩa thống kê ($p < 0.001$)** |
| **`FS05` (4 Kim loại - Cu)** | **0.07896** | 0.08198 | +0.00302 | +1.1815 | 0.2384 | +1.9930 | 0.0473 | Không khác biệt ý nghĩa $L_1$ (SA tốt hơn $L_2$) |
| **`FS06` (WTI + Cu)** | **0.07709** | 0.07731 | +0.00023 | +0.6605 | 0.5095 | -0.2149 | 0.8300 | Không khác biệt ý nghĩa ($p = 0.510$) |

- **Tổng kết trên toàn bộ 36 cặp bối cảnh (`6 bộ biến × 6 bối cảnh`) của Công:**
  - **18 / 36 cặp (50.0%)** không có sự khác biệt có ý nghĩa thống kê ($p_{L1} \ge 0.05$).
  - **11 / 36 cặp (30.6%)** **SA–LSTM thắng có ý nghĩa thống kê** ($p_{L1} < 0.05$, tập trung ở `FS04` và `FS01`).
  - **7 / 36 cặp (19.4%)** **TPE–LSTM thắng có ý nghĩa thống kê** ($p_{L1} < 0.05$, tập trung ở `FS02` và `FS03`).
  - **Nguyên nhân:** Trên bộ biến đa cộng tuyến cao như `FS04` (5 kim loại), Bayesian TPE dễ chọn cấu hình **overfit vào tập Validation** (`80/20`), trong khi Simulated Annealing (SA) giữ được tính chính quy hóa an toàn hơn.

### 3.3. Kiểm định giả thuyết $H_0^{(3)}$: *"Tái huấn luyện Walk-Forward (Quân) vs. Mô hình Tĩnh (Quân, Công, Vĩ, Huy)"*

1. **Trong nội bộ phương pháp Walk-Forward của Quân ([phase5_diebold_mariano_tests.csv](../quan%20task/result/phase5_diebold_mariano_tests.csv)):**
   - **Trên nhánh Direct:** `Expanding Window` và `Rolling Window` thắng `Static (Train-Only)` với ý nghĩa thống kê tuyệt đối (`FEATURES_4 (FS02)`: `Expanding Direct (0.2370)` vs `Static (1.5516)` $\rightarrow \text{DM}_{L1} = -51.359, p < 0.0001$; `Rolling Direct (0.2837)` vs `Static (1.5516)` $\rightarrow \text{DM}_{L1} = -43.541, p < 0.0001$).
   - **Trên nhánh Residual (`FS02`):** `Expanding Window Residual (0.07852)` thắng `Static Train-Only Residual (0.10969)` với $\text{DM}_{L1} = -4.938, p < 0.0001$.
   - **So sánh `Expanding` vs `Rolling` (`W=756`):** Trên nhánh Direct, `Expanding` thắng `Rolling` có ý nghĩa thống kê ($\text{DM}_{L1} = -6.632, p < 0.0001$); trên nhánh Residual, hai chiến lược tương đương thống kê ($\text{DM}_{L1} = -0.822, p = 0.4119$).
2. **Kiểm định DM chéo giữa `Quân (WF Expanding)`, `Công (SA/TPE-LSTM)` và `Huy (Hybrid ARIMA-GRU)` trên đúng 271 phiên Test của Split 90/10 Expanding:**

| Bộ biến | Quân WF-Exp vs Công SA-LSTM ($\text{DM}_{L1}$ / $p$) | Quân WF-Exp vs Công TPE-LSTM ($\text{DM}_{L1}$ / $p$) | Quân WF-Exp vs Huy Hybrid `s42` ($\text{DM}_{L1}$ / $p$) | Quân WF-Exp vs Huy TB 5 seeds ($\text{DM}_{L1}$ / $p$) | Công SA-LSTM vs Huy `s42` ($\text{DM}_{L1}$ / $p$) | Công TPE-LSTM vs Huy `s42` ($\text{DM}_{L1}$ / $p$) |
|---|---|---|---|---|---|---|
| **`FS01`** | `+0.84` (`0.400`) | `+0.23` (`0.817`) | `-1.17` (`0.243`) | `+0.07` (`0.941`) | `-1.56` (`0.120`) | `-1.88` (`0.061`) |
| **`FS02`** | **`-3.16` (`0.0018`)** ★Quân | `+0.62` (`0.536`) | `-0.31` (`0.758`) | `+0.65` (`0.514`) | **`+2.04` (`0.0426`)** ★Huy | `-0.72` (`0.472`) |
| **`FS03`** | **`-2.40` (`0.0169`)** ★Quân | `-1.42` (`0.157`) | **`-3.33` (`0.0010`)** ★Quân | `-1.36` (`0.176`) | `-1.90` (`0.058`) | **`-3.72` (`0.0002`)** ★TPE |
| **`FS04`** | **`-2.12` (`0.0351`)** ★Quân | **`-5.16` (`0.0000`)** ★Quân | **`-5.09` (`0.0000`)** ★Quân | `-1.31` (`0.190`) | **`-6.73` (`0.0000`)** ★SA | `+1.31` (`0.193`) |
| **`FS05`** | `-1.75` (`0.081`) | **`-2.38` (`0.0179`)** ★Quân | **`-2.94` (`0.0036`)** ★Quân | `-0.58` (`0.562`) | `-1.90` (`0.059`) | **`-3.70` (`0.0003`)** ★TPE |
| **`FS06`** | `+0.90` (`0.368`) | `+0.41` (`0.683`) | **`-2.35` (`0.0195`)** ★Quân | `+0.19` (`0.846`) | **`-3.55` (`0.0005`)** ★SA | **`-3.42` (`0.0007`)** ★TPE |

> **Phát hiện xuyên suốt từ Kiểm định DM chéo (Quân – Công – Huy):**
> - **Khi so khớp cùng seed đơn `42` trên các bộ đa biến:** Cơ chế tái huấn luyện định kỳ mỗi 40 phiên của **Quân (`WF-Expanding`)** thắng `Công SA-LSTM` ở `3/4` bộ đa biến (`FS02, FS03, FS04`), thắng `Công TPE-LSTM` ở `FS04, FS05`, và **thắng `Huy Hybrid s42` có ý nghĩa thống kê trên `4/6` bộ biến (`FS03, FS04, FS05, FS06`)**! Tương tự, `SA-LSTM` và `TPE-LSTM` của Công cũng thắng `Huy Hybrid s42` có ý nghĩa ở nhiều bộ đa biến (`FS03, FS04, FS05, FS06`).
> - **Hiệu ứng triệt nhiễu Ensemble 5 seeds:** Khi lấy **trung bình dự báo 5 seeds (`Hybrid_s41..s45`)** của Huy thay cho 1 seed đơn lẻ, phương sai ngẫu nhiên giữa các lần khởi tạo GRU được triệt tiêu, làm `Quân WF-Expanding` và `Huy 5-seed Ensemble` **tương đương thống kê trên cả 6/6 bộ biến** ($p \in [0.176, 0.941]$).
3. **Kiểm định DM của Vĩ (`iTransformer vs. SA-LSTM` ở `K=5`, Split 90/10 Expanding):**
   - `iTransformer Residual (MAE = 0.07812)` thấp hơn có ý nghĩa thống kê so với `SA-LSTM Fixed Residual (0.08293, p = 0.0248)` và `SA-LSTM Tuned Residual (0.08535, p = 0.00014)`, nhờ cơ chế LayerNorm và Inverted Attention kiểm soát nhiễu đa biến tốt hơn khi không tái huấn luyện định kỳ.

---

## 4. Nghiên cứu Cắt lớp Toàn Nhóm (Group-Wide Ablation Studies)

Biểu đồ Heatmap & Log-scale Ablation 6 bộ biến: [fig17_group_ablation_6_feature_sets.png](fig17_group_ablation_6_feature_sets.png)

### 4.1. Ablation 1 — Vai trò của 6 Bộ biến đầu vào (`FS01` → `FS06`) trên cả 5 thành viên (`Split 90/10 — Expanding`)

| Mã bộ biến | Các biến đưa vào tensor $X_{t-22:t-1}$ | $K$ | Khoa (`LSTM-Res`) | Khoa (`Ridge-Res`) | Công (`SA-LSTM`) | Công (`TPE-LSTM`) | Vĩ (`iTrans-Res`) | Quân (`WF-Exp-Res`) | Huy (`Hybrid-Res`) | Quân (`WF-Exp-Dir`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`FS01_CU`** | `Copper` | 1 | 0.07701 | **0.07731** | **0.07695** | **0.07727** | **0.07681** | 0.07736 | **0.07745** | **0.20265** |
| **`FS02_PAPER_CU`** | `Copper, WTI, Gold, Silver` | 4 | 0.07701 | 0.08071 | 0.08453 | 0.07798 | 0.07701 | 0.07852 | 0.08110 | 0.23696 |
| **`FS03_PAPER_EXO`** | `WTI, Gold, Silver` *(Bỏ Copper)* | 3 | — | — | 0.07999 | 0.07812 | *0.09027* | **0.07693** | *0.08334* | *0.38606* |
| **`FS04_METALS_CU`** | `Copper, Al, Tin, Silver, Gold` | 5 | **0.07698** | *0.08171* | 0.08097 | *0.10006* | 0.07687 | 0.07769 | 0.08185 | 0.33844 |
| **`FS05_METALS_EXO`** | `Al, Tin, Silver, Gold` *(Bỏ Copper)* | 4 | — | — | 0.07896 | 0.08198 | 0.07924 | 0.07720 | 0.08071 | *0.36182* |
| **`FS06_WTI_CU`** | `Copper, Al, Tin, WTI` | 4 | 0.07704 | 0.07774 | 0.07709 | 0.07731 | 0.07794 | 0.07764 | 0.07844 | 0.21921 |
| **Trung bình** | — | — | **0.07701** | **0.07937** | **0.07975** | **0.08212** | **0.07969** | **0.07756** | **0.08048** | **0.29086** |

**3 kết luận quan trọng từ Ablation 6 bộ biến:**
1. **Tác động của việc loại bỏ chuỗi `Copper` khỏi tensor đầu vào (`FS02 vs. FS03` và `FS04 vs. FS05`):**
   - **Trên nhánh Direct (Quân, Khoa V21 & Huy):** Khi bỏ `Copper` khỏi $X$ (`FS03`, `FS05`), mọi mô hình mất điểm neo mức giá mục tiêu, khiến MAE Direct của Quân tăng +62.9% (`0.23696 → 0.38606`) và GRU Direct của Huy tăng vọt lên `0.49914 – 0.56456 $/lb`.
   - **Trên nhánh Residual:** Mặc dù phép hoàn nguyên $\hat{y}_t = y_{t-1} + \hat{\Delta}_t$ vẫn dùng giá Copper phiên trước, các kiến trúc phụ thuộc tương quan chéo hoặc phần dư dài hạn bị ảnh hưởng rõ rệt khi bỏ Copper: **`iTransformer` của Vĩ tăng MAE vọt lên `0.09027 $/lb` (-17.12% vs Naive) ở `FS03`**, và **`Hybrid ARIMA–GRU` của Huy đạt đỉnh xấu nhất ở `FS03` (`0.08334 $/lb`, -8.13% vs Naive)**. Ngược lại, **Quân (`WF-Expanding`)** giữ ổn định toàn bộ `FS03` (`0.07693`) và `FS05` (`0.07720`).
2. **Hiện tượng Nhiễu chiều dữ liệu (Exogenous Noise Inflation) đồng thuận cả 5 thành viên:**
   - Do tương quan bậc trễ 1 ngày giữa các hàng hóa ngoại sinh (`Gold, Silver, WTI, Al, Tin`) và `Copper` đều rất thấp ($|r| \le 0.091$), việc thêm biến ngoại sinh vào **`Hybrid ARIMA–GRU` (Huy)** làm MAE xấu đi trên **tất cả `FS02–FS06`** so với `FS01` (`0.07745 → 0.07844–0.08334`), làm **`Ridge` (Khoa)** tăng sai số 4.4%–6.0%, và khiến **`TPE-LSTM` (Công)** overfit mạnh ở `FS04` (`0.10006 $/lb`).
   - `FS01 (Chỉ dùng Copper)` đạt hạng nhất hoặc sát đỉnh trên Vĩ (`0.07681`), Công (`0.07695`), Khoa (`0.07701`) và Huy (`0.07745`).
3. **Tính ổn định của `FS06 (Copper, Aluminium, Tin, WTI)`:**
   - Trong nhóm đa biến, `FS06` đạt độ ổn định tốt nhất trên hầu hết kiến trúc (`Khoa LSTM 0.07704`, `Công SA-LSTM 0.07709`, `Quân WF-Exp 0.07764`, `Huy Hybrid 0.07844` — tốt nhì chỉ sau `FS01`), và giúp `Paper SA-LSTM Direct` của Khoa đạt MAE Direct thấp nhất (`0.18463 $/lb`).

### 4.2. Ablation 2 — Độ dài cửa sổ huấn luyện (`Window Size W`), Bước trượt (`Stride S`), `Recency Weighting` & `Cold vs. Warm Start`

1. **Ablation Chiến lược Cửa sổ (`Expanding` vs. `Rolling 756` vs. `Recency Weighted`) trên Khoa, Công, Vĩ, Quân & Huy:**
   - Cả 5 thành viên đều xác nhận **`Expanding Window`** ổn định hơn hẳn `Rolling 756` (~3 năm gần nhất): đặc biệt rõ rệt trong `schemes_90_10.csv` của **Huy** ở Split 90/10, khi trung bình 6 bộ biến $\Delta\text{MAE}$ so với Naive của `Expanding` là **`+4.42%`**, `Recency Weighted` là **`+6.18%`**, còn `Rolling 756` vọt lên tới **`+32.45%`** (riêng `FS03 Rolling 756` tăng sai số tới **`+101.22%`** vì cửa sổ 756 phiên quá ngắn không đủ mẫu học phần dư ARIMA).
   - Khi có đủ biến Copper (`FS01`) và kết hợp đúng kiến trúc, **`Recency Weighted` (half-life 252 phiên)** phát huy tác dụng ở vùng giá đỉnh 2025–2026: `iTransformer Residual (Vĩ, FS01)` giảm MAE xuống **`0.07645 $/lb` (+0.82% vs Naive)** ở `90/10`, còn `Hybrid ARIMA–GRU (Huy, FS01)` đạt duy nhất 1 seed thắng Naive dưới $L_1$ ở `Recency Seed 42` (`0.07652 $/lb`, $p_{L1} = 0.0385$).
2. **Ablation Tham số Walk-Forward (`W`, `S`, `Cold-Start vs. Warm-Start`) của Quân ([phase3_ablation_study.csv](../quan%20task/result/phase3_ablation_study.csv)):**
   - **Window Size $W \in \{252, 504, 756\}$ (cố định $S=40$):** $W=756$ đạt MAE thấp nhất (`0.28371 $/lb`), vượt trội so với $W=252$ (`0.33899 $/lb`) và $W=504$ (`0.35547 $/lb`).
   - **Retrain Stride $S \in \{10, 20, 40, 60\}$ (cố định $W=756$):** Giảm $S$ từ `60 → 10` giúp giảm liên tục sai số Direct (`0.36000 → 0.23003 $/lb`), nhưng thời gian tính toán tăng từ `66.8s` lên `333.7s`. Mốc $S^* = 40$ (7 lần retrain, `88.7s`) là điểm cân bằng Pareto tối ưu.
   - **Cơ chế cập nhật trọng số (`Cold-Start` vs. `Warm-Start`):** Nhánh **Direct** hợp `Warm-Start` (`MAE = 0.21942` vs `0.28371` của Cold-Start); nhánh **Residual** hợp `Cold-Start` (`MAE = 0.08026` vs `0.09777` của Warm-Start do tránh tích lũy *gradient drift* qua các fold nhiễu cao).

---

## 5. Phân tích Ưu / Nhược điểm & Tính Có Ý Nghĩa của Từng Phương Pháp (5 Thành Viên)

| Phương pháp & Thành viên | Ưu điểm nổi bật (Strengths) | Nhược điểm & Giới hạn (Weaknesses) | Tính có ý nghĩa thống kê & Giá trị khoa học |
|---|---|---|---|
| **1. Non-SA Baselines & Residual Protocol V20–V26** *(Viên Đăng Khoa — `Module1/`)* | - Khung kiểm chứng khoa học mẫu mực: Full Factorial Grid (324 runs), Walk-Forward 2.214 phiên (2018–2026) $\times$ 5 seeds.<br>- Chứng minh toán học cơ chế **Variance Shrinkage** ($\text{std}(\hat{\Delta}_t) \ll \text{std}(\Delta_t)$) và kiểm định kinh tế lượng toàn diện (ADF, KPSS, Zivot–Andrews, Granger, Spearman FDR).<br>- `LSTM Residual` (`0.07701 ± 0.00002`) và `GRU Residual` (`0.07695`) cực kỳ ổn định. | - Các mô hình Non-SA dùng siêu tham số cố định (`lr = 1e-3`), chưa tối ưu hóa riêng cho từng cấu trúc đặc trưng.<br>- Mô hình tuyến tính `Ridge Residual` nhạy cảm mạnh với nhiễu đa biến (`Noise Inflation` làm tăng sai số 4.4%–18.9% khi thêm biến ngoại sinh). | - **Ý nghĩa học thuật cao nhất nhóm:** Chứng minh chuỗi giá đồng ngày tuân theo quy luật tiệm cận bước ngẫu nhiên (*Near-Random-Walk*), `0/5` seeds Walk-Forward thắng Naive có ý nghĩa thống kê. |
| **2. Tối ưu Siêu tham số SA–LSTM vs. TPE–LSTM (Optuna)** *(Nguyễn Thành Công — `TPE+LSTM/`)* | - Thiết kế đối chuẩn công bằng tuyệt đối: cùng ngân sách 16 trials/bộ biến trên `80/20`, khóa siêu tham số trước khi đánh giá trên 6 bối cảnh $\times$ 6 bộ biến (192 search fits + 72 final fits).<br>- `SA-LSTM (FS01)` vượt nhẹ Naive ở cả **4/4 mốc chia** (`80/20, 90/10, 92.5/7.5, 95/5`).<br>- `TPE-LSTM` hội tụ nhanh và thắng `SA-LSTM` có ý nghĩa ($p < 0.01$) trên bộ biến `Paper3` (`FS02`, `FS03`). | - Chạy trên 1 seed (`42`) và giữ mô hình tĩnh suốt giai đoạn test (không retrain định kỳ).<br>- `TPE-LSTM` dễ bị **overfit vào tập Validation** khi không gian đặc trưng có đa cộng tuyến cao (`FS04`: MAE vọt lên `0.10006 $/lb`, thua SA-LSTM với $p = 0.0003$). | - **Ý nghĩa phương pháp luận rõ rệt:** `18/36` cặp không khác biệt ý nghĩa; SA thắng ý nghĩa `11/36` cặp (an toàn với biến nhiễu), TPE thắng ý nghĩa `7/36` cặp (`FS02, FS03`), nhưng cả hai đều có `0/36` lượt thắng Naive dưới $L_1$. |
| **3. Kiến trúc iTransformer (Inverted Attention)** *(Vĩ — `Vi Task/`)* | - Kiến trúc Transformer hiện đại đảo trục biến thành token (`Dense 22 → 48`) học tương quan chéo bằng Multi-Head Attention.<br>- Đạt **MAE thấp nhất toàn nhóm trên `90/10 Expanding` (`FS01: 0.07681`, `FS04: 0.07687`)** và đạt **`0.07645 $/lb` (+0.82% vs Naive)** khi kết hợp `Recency Weighted`.<br>- Thắng `SA-LSTM Fixed` ($p = 0.0248$) và `SA-LSTM Tuned` ($p = 0.00014$) trong benchmark đối đầu `K=5`. | - Khi $K=1$ (`FS01`), self-attention suy biến trên trục kích thước 1.<br>- Phụ thuộc nặng vào token `Copper` trong input: khi bỏ `Copper` ở `FS03`, MAE Residual tăng lên `0.09027 $/lb` (-17.12% vs Naive).<br>- Nhánh Direct có sai số cao nhất nhóm (`1.14 – 1.81 $/lb`). | - **Ý nghĩa kiến trúc:** Chứng minh iTransformer + LayerNorm + Residual + Recency Weighting đạt độ chính xác điểm hàng đầu khi có `Copper` (`FS01/FS02/FS04`), nhưng dễ bị phá vỡ cấu trúc attention khi thiếu token mục tiêu (`FS03`). |
| **4. Rolling & Expanding Window Walk-Forward Backtesting** *(Lê Anh Quân — `quan task/`)* | - Giải quyết trực diện **Concept Drift & Regime Shift**: định kỳ cập nhật lại cả `MinMaxScaler` lẫn trọng số mỗi $S=40$ phiên không rò rỉ dữ liệu.<br>- **Giảm 64%–85% sai số nhánh Direct** ($\text{DM}_{L1} = -51.36, p < 0.0001$).<br>- **Đạt MAE trung bình qua đủ 6 bộ biến thấp nhất & ổn định nhất toàn nhóm (`0.07756 ± 0.00055`)**, thắng `SA-LSTM`/`TPE-LSTM` của Công và `Huy Hybrid s42` có ý nghĩa thống kê ($p < 0.05$) trên các bộ đa biến.<br>- Đầy đủ 5 seeds, Ablation $W, S, \text{Cold/Warm}$ & Regime breakdown. | - Chi phí tính toán gấp $7 - 28$ lần mô hình tĩnh do retrain định kỳ.<br>- Ở nhánh Residual trên bộ đơn giản (`FS01`), tái huấn luyện nhiều lần chưa vượt được Naive về mặt thống kê ($p = 0.438$). | - **Ý nghĩa vận hành thực tế (Production):** Ngăn chặn sự sụp đổ của mô hình đa biến và mô hình Direct trước biến động mặt bằng giá 2025–2026 (`0/10` thắng Naive nhưng ổn định nhất qua toàn bộ `FS01–FS06`). |
| **5. Mô hình Lai Hybrid ARIMA–GRU (`FS01–FS06`, Chuẩn Nhóm 23/11)** *(Phan Ngọc Đức Huy — `huy task/`)* | - Hoàn thiện đồng bộ đầy đủ **6 bộ biến × 4 mốc chia × 3 chiến lược × 5 seeds (`360` lần train GRU chuẩn 2 tầng)** và xuất **36 file dự báo chi tiết**.<br>- Đóng góp đối chuẩn **ARIMA thuần (`auto_arima`)**: ở `92.5/7.5` (`0.07456`, `-0.75%`) và `95/5` (`0.07258`, `-0.31%`), mô hình tuyến tính `ARIMA(2,1,2)` cho sai số tốt hơn cả việc ghép thêm GRU.<br>- Cho thấy hiệu ứng triệt nhiễu của **5-seed ensemble** khi đối chiếu DM chéo với Quân. | - Khi đồng bộ dùng GRU 2 tầng chuẩn nhóm thay cho 1 tầng, phần dư GRU cộng thêm làm MAE **tăng thêm nhiễu** trên cả 6 bộ ở `90/10 Expanding` (`FS01 = 0.07745`, TB 6 bộ = `0.08048`), và bùng sai số ở `Rolling 756` (`+32.45%` TB, `FS03` lên tới `+101.22%`).<br>- Ở `80/20` và `90/10`, `auto_arima` chọn `(0,1,0)` trùng Naive; thêm biến ngoại sinh vào GRU đều làm kết quả xấu đi so với `FS01`. | - **Ý nghĩa đối chuẩn kinh tế lượng mạnh mẽ:** `0/120` lượt chạy `Expanding` (và chỉ `1/90` ở `Recency` 90/10 $L_1$, `0/90` $L_2$) thắng Naive; `57/120` lượt Expanding kém Naive có ý nghĩa. Khẳng định ghép thêm GRU phi tuyến chồng lên phần dư ARIMA không vượt được Random Walk. |

---

## 6. Danh mục Tệp Tổng hợp Chung (`group_benchmark_summary/`)

- Báo cáo tổng hợp: [THONG_KE_CHISO_SIGNIFICANCE_ABLATION.md](THONG_KE_CHISO_SIGNIFICANCE_ABLATION.md)
- Bảng Ablation 6 bộ biến toàn nhóm: [group_ablation_6_feature_sets_all_members.csv](group_ablation_6_feature_sets_all_members.csv)
- Bảng Cross-Ratio 4 mốc chia toàn nhóm: [group_cross_ratio_4_splits_all_members.csv](group_cross_ratio_4_splits_all_members.csv)
- Bảng kiểm định Diebold–Mariano toàn nhóm (144 dòng): [group_significance_dm_tests_complete.csv](group_significance_dm_tests_complete.csv)
- 3 biểu đồ tổng hợp toàn nhóm:
  - [fig16_group_benchmark_all_5_members.png](fig16_group_benchmark_all_5_members.png)
  - [fig17_group_ablation_6_feature_sets.png](fig17_group_ablation_6_feature_sets.png)
  - [fig18_group_dm_significance_and_pros_cons.png](fig18_group_dm_significance_and_pros_cons.png)
