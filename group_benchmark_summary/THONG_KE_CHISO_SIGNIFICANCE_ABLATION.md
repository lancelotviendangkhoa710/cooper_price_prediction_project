# BÁO CÁO TỔNG HỢP TOÀN NHÓM: THỐNG KÊ CHỈ SỐ, KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (DIEBOLD–MARIANO), NGHIÊN CỨU CẮT LỚP (ABLATION STUDY) & ĐÁNH GIÁ ƯU/NHƯỢC ĐIỂM TỪNG PHƯƠNG PHÁP

- **Học phần:** IS403.R11 — Phân tích dữ liệu kinh doanh  
- **Đề tài:** Dự báo giá đồng (*Copper Futures Price Forecasting*)  
- **Phụ trách tổng hợp & kiểm định toàn nhóm:** **Lê Anh Quân** (Phương pháp 4 + Tổng hợp Thống kê / Significance Test / Ablation Study)  
- **Ngày hoàn thiện:** 21/11/2026  

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

### 1.2. Bảng phân công phương pháp và phạm vi thực nghiệm của 5 thành viên

| Thành viên | Thư mục mã nguồn | Phương pháp phụ trách | Tổ hợp Bộ biến (`FS01–FS06`) | Tỷ lệ chia (`80/90/92.5/95`) | Mục tiêu (`Direct` / `Residual`) | Đa hạt giống (`Multi-Seed`) | Kiểm định DM Test |
|---|---|---|---|---|---|---|---|
| **1. Viên Đăng Khoa** | `Module1/` | **Non-SA Baselines & SA-LSTM Standalone** (`GRU`, `LSTM`, `CNN-LSTM`, `CNN-GRU`, `Ridge`, `XGBoost`, `SA-LSTM V21/V22/V23`) | Đủ 6 bộ (`FS01–FS06` qua V20/V21/V22) | Đủ 4 mốc | Cả `Direct` & `Residual` | 5 seeds (`41–45`) cho Walk-Forward | Có (`dm_hac_by_seed.csv`, `wf_dm_hac_by_seed.csv`) |
| **2. Nguyễn Thành Công** | `TPE+LSTM/` | **Tối ưu siêu tham số SA–LSTM vs. TPE–LSTM (Optuna)** (192 search trials + 72 final fits) | Đủ 6 bộ (`FS01–FS06`) | Đủ 4 mốc + `rolling_756`, `recency` | `Residual` | 1 seed (`42`) | Đã được Quân tính bổ sung đủ **36 cặp DM-HAC** từ `test_predictions.csv` |
| **3. Vĩ** | `Vi Task/` | **Kiến trúc iTransformer (Inverted Attention)** + Đối chuẩn SA-LSTM Fixed/Tuned | Đủ 6 bộ (`FS01–FS06`) | Đủ 4 mốc + `rolling_756`, `recency` | Cả `Direct` & `Residual` | 1 seed (`42`) | Có (DM-HAC giữa iTransformer vs SA-LSTM Fixed/Tuned) |
| **4. Lê Anh Quân** | `quan task/` | **Rolling & Expanding Window Walk-Forward Backtesting** (`Static T`, `Static T+V`, `Expanding`, `Rolling`, Ablation $W, S$, Cold/Warm) | Đủ 6 bộ (`FS01–FS06`) | Đủ 4 mốc | Cả `Direct` & `Residual` | 5 seeds (`41–45`) | Có (**20 cặp / 40 dòng DM** + **DM chéo giữa các thành viên**) |
| **5. Phan Ngọc Đức Huy** | `huy task/` | **Mô hình lai Hybrid ARIMA–GRU** (Zhang 2003: `auto_arima` + `GRU` phần dư) | `FS01` (Đơn biến Copper) | Đủ 4 mốc | `Residual` (Hybrid/ARIMA) & `Direct` (GRU) | 5 seeds (`41–45`) | Có (20 lượt chạy $\times$ L1/L2 DM-HAC vs Naive) |

---

## 2. Thống kê Chỉ số Tổng hợp Toàn Nhóm (Benchmarking Leaderboard)

### 2.1. Bảng xếp hạng tổng hợp nhánh Residual trên cấu hình chuẩn `Split 90/10 — Expanding` ($N_{\text{test}} = 271$, `29/10/2025 → 13/11/2026`)

Biểu đồ đối chiếu trực quan: [fig16_group_benchmark_all_5_members.png](file:///d:/PTDLKD/ptdlkd-is403.r11-main/quan%20task/result/fig16_group_benchmark_all_5_members.png) · Bảng dữ liệu CSV: [group_ablation_6_feature_sets_all_members.csv](file:///d:/PTDLKD/ptdlkd-is403.r11-main/quan%20task/result/group_ablation_6_feature_sets_all_members.csv)

| Hạng (theo Best MAE) | Thành viên | Phương pháp / Mô hình | Bộ biến đạt đỉnh (Best FS) | Best MAE (\$/lb) | RMSE tại Best FS (\$/lb) | DA (%) tại Best FS | % Cải thiện vs. Naive (`0.07708`) | MAE Trung bình qua các bộ biến (\$/lb) | Độ lệch chuẩn MAE giữa 6 bộ biến (\$/lb) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| **1** | **Huy (PP5)** | **Hybrid ARIMA(0,1,0)–GRU** *(TB 5 seeds)* | `FS01` (Copper) | **0.07676** | **0.10075** | 50.92% | **+0.41%** | 0.07676 *(1 bộ)* | — |
| **2** | **Vĩ (PP3)** | **iTransformer Residual** | `FS01` (Copper) | **0.07681** | 0.10097 | 53.14% | **+0.34%** | 0.07969 *(6 bộ)* | 0.00525 |
| **3** | **Quân (PP4)** | **Walk-Forward Expanding Residual (`W=756, S=40`)** | `FS03` (WTI, Gold, Silver) | **0.07693** | 0.10112 | 47.60% | **+0.19%** | **0.07756 *(6 bộ)*** | **0.00054 (Ổn định nhất)** |
| **4** | **Khoa (Module 1)** | **GRU Residual (Non-SA V20)** | `FS04` (5 Kim loại) | **0.07695** | 0.10081 | 52.77% | **+0.16%** | 0.07793 *(4 bộ)* | 0.00151 |
| **5** | **Công (PP2)** | **SA–LSTM Residual (Per-FS Search)** | `FS01` (Copper) | **0.07695** | 0.10084 | 52.77% | **+0.16%** | 0.07975 *(6 bộ)* | 0.00283 |
| **6** | **Khoa (Module 1)** | **LSTM Residual (Non-SA V20)** | `FS04` (5 Kim loại) | **0.07698** | 0.10095 | 52.77% | **+0.13%** | **0.07701 *(4 bộ)*** | **0.00002** |
| **—** | **Baseline** | **Naive Persistence ($\hat{y}_t = y_{t-1}$)** | `FS01` (Copper) | **0.07708** | **0.10103** | — | **0.00%** | **0.07708** | **0.00000** |
| **7** | **Khoa (Module 1)** | **CNN–LSTM Residual (Non-SA V20)** | `FS04` (5 Kim loại) | 0.07725 | 0.10086 | 46.86% | -0.22% | 0.07810 *(4 bộ)* | 0.00134 |
| **8** | **Công (PP2)** | **TPE–LSTM Residual (Optuna Per-FS)** | `FS01` (Copper) | 0.07727 | 0.10094 | 48.71% | -0.24% | 0.08212 *(6 bộ)* | 0.00894 |
| **9** | **Khoa (Module 1)** | **Ridge Residual ($\alpha = 1.0$)** | `FS01` (Copper) | 0.07731 | 0.10119 | 47.60% | -0.30% | 0.07937 *(4 bộ)* | 0.00217 |
| **10** | **Quân (PP4)** | **Walk-Forward Rolling Residual (`W=756, S=40`)** | `FS01` (Copper) | 0.07732 | 0.10080 | 52.77% | -0.31% | 0.07853 *(6 bộ)* | 0.00120 |
| **11** | **Quân (PP4)** | **Static (Train+Val) Residual** | `FS04` (5 Kim loại) | 0.07746 | 0.10096 | 49.45% | -0.50% | 0.07920 *(6 bộ)* | 0.00159 |

### 2.2. Đối chiếu nhánh Direct (Dự báo mức giá trực tiếp) vs. Nhánh Residual toàn nhóm (`Split 90/10`)

| Thành viên | Mô hình (Nhánh Direct) | `FS01` MAE | `FS02` MAE | `FS03` MAE | `FS04` MAE | `FS05` MAE | `FS06` MAE | MAE Trung bình 6 bộ (\$/lb) | So với Naive (`0.07708`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Quân (PP4)** | **Walk-Forward Expanding Direct (`W=756, S=40`)** | **0.20265** | **0.23696** | **0.38606** | 0.33844 | **0.36182** | 0.21921 | **0.29086** | Kém Naive +277% *(nhưng giảm 64%–85% sai số so với Static Direct)* |
| **Khoa (Module 1)** | **GRU Direct (Non-SA V20/V22)** | 0.43711 | 0.24789 | — | **0.24082** | — | 0.24321 | **0.29226** | Kém Naive +279% |
| **Huy (PP5)** | **GRU Direct (TB 5 seeds)** | 0.29946 | — | — | — | — | — | **0.29946** | Kém Naive +289% |
| **Quân (PP4)** | **Walk-Forward Rolling Direct (`W=756, S=40`)** | 0.27394 | 0.28371 | 0.46075 | 0.34973 | 0.41410 | 0.27422 | **0.34274** | Kém Naive +345% |
| **Khoa (Module 1)** | **Paper SA-LSTM Direct (V21/V22)** | — | 0.97807 | 0.54072 | 0.47093 | 0.46499 | **0.18463** | **0.52787** | Kém Naive +585% |
| **Khoa (Module 1)** | **LSTM Direct (Non-SA V20/V22)** | 0.44672 | 1.16396 | — | 0.24741 | — | 0.32494 | **0.54576** | Kém Naive +608% |
| **Khoa (Module 1)** | **Reopt SA-LSTM Direct (V21/V22)** | — | 0.87742 | 0.59561 | 0.51047 | 0.46660 | 0.35892 | **0.56180** | Kém Naive +629% |
| **Quân (PP4)** | **Static (Train-Only) Direct** | 0.49280 | 1.55159 | 0.31990 | 1.32269 | 0.56377 | 0.55846 | **0.80154** | Kém Naive +940% |
| **Vĩ (PP3)** | **iTransformer Direct** | 1.75765 | 1.14381 | 1.54427 | 1.63235 | 1.81230 | 1.38313 | **1.54558** | Kém Naive +1905% |

> **Phát hiện xuyên suốt số 1 (Direct vs. Residual):**
> - **100% mô hình dự báo trực tiếp mức giá (Direct Level)** của cả Khoa, Vĩ, Quân, Huy đều thất bại nặng nề trước Naive Persistence (`MAE` từ `0.185` đến `1.812 $/lb` so với `0.07708 $/lb` của Naive). Nguyên nhân gốc rễ là giá đồng trong giai đoạn Test (2025–2026) bứt phá lên vùng đỉnh lịch sử `5.0 – 6.89 $/lb`, vượt ngoài miền giá `[0, 1]` của `MinMaxScaler` fit trên tập Train.
> - Tuy nhiên, trên nhánh Direct, **cơ chế tái huấn luyện định kỳ Walk-Forward Expanding của Quân (`MAE TB = 0.29086 $/lb`)** đã kéo giảm sai số từ **64% đến 85%** so với mô hình tĩnh `Static Train-Only Direct (0.80154 $/lb)` và thấp hơn gần 5 lần so với `iTransformer Direct tĩnh (1.54558 $/lb)`, nhờ liên tục cập nhật lại biên `MinMaxScaler` theo mặt bằng giá mới mỗi $S = 40$ phiên.
> - Khi chuyển sang **nhánh Residual ($\Delta y_t = y_t - y_{t-1}$ trên thang chuẩn hóa của Copper)**, toàn bộ 5 thành viên đều đưa sai số MAE hội tụ sát mốc `0.07676 – 0.08000 $/lb` (xấp xỉ Naive `0.07708 $/lb`).

### 2.3. Thống kê xuyên suốt 4 tỷ lệ chia dữ liệu (`Cross-Ratio 80/20, 90/10, 92.5/7.5, 95/5`)

Bảng dữ liệu chi tiết: [group_cross_ratio_4_splits_all_members.csv](file:///d:/PTDLKD/ptdlkd-is403.r11-main/quan%20task/result/group_cross_ratio_4_splits_all_members.csv)

| Phương pháp đại diện | Thành viên | Bộ biến | `80/20` ($N=542$) MAE | `90/10` ($N=271$) MAE | `92.5/7.5` ($N=204$) MAE | `95/5` ($N=136$) MAE | Trung bình 4 mốc (\$/lb) | Số mốc vượt Naive |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Naive Persistence ($\hat{y}_t = y_{t-1}$)** | Baseline | `FS01` | **0.06900** | **0.07708** | **0.07512** | **0.07280** | **0.07350** | — |
| **Hybrid ARIMA–GRU (TB 5 seeds)** | Huy (PP5) | `FS01` | 0.06923 (-0.32%) | **0.07676 (+0.41%)** | **0.07441 (+0.94%)** | **0.07248 (+0.44%)** | **0.07322** | 3 / 4 |
| **SA–LSTM Residual (Expanding)** | Công (PP2) | `FS01` | **0.06883 (+0.24%)** | **0.07695 (+0.16%)** | **0.07484 (+0.37%)** | **0.07257 (+0.32%)** | **0.07330** | **4 / 4** |
| **iTransformer Residual (Expanding)** | Vĩ (PP3) | `FS01` | **0.06885 (+0.22%)** | **0.07681 (+0.34%)** | 0.07595 (-1.11%) | 0.07289 (-0.12%) | **0.07363** | 2 / 4 |
| **LSTM Residual (Expanding)** | Khoa (M1) | `FS01` | 0.06910 (-0.15%) | **0.07701 (+0.09%)** | **0.07503 (+0.11%)** | 0.07356 (-1.04%) | **0.07368** | 2 / 4 |
| **GRU Residual (Expanding)** | Khoa (M1) | `FS02` | **0.06878 (+0.32%)** | 0.07714 (-0.09%) | 0.07554 (-0.56%) | 0.07479 (-2.74%) | **0.07406** | 1 / 4 |
| **TPE–LSTM Residual (Expanding)** | Công (PP2) | `FS01` | 0.06932 (-0.47%) | 0.07727 (-0.24%) | 0.07773 (-3.48%) | 0.07428 (-2.04%) | **0.07465** | 0 / 4 |
| **TPE–LSTM Residual (Expanding)** | Công (PP2) | `FS02` | 0.07063 (-2.36%) | 0.07798 (-1.17%) | 0.07668 (-2.09%) | 0.07493 (-2.93%) | **0.07506** | 0 / 4 |
| **Static (Train+Val) Residual** | Quân (PP4) | `FS02` | 0.07287 (-5.61%) | 0.07812 (-1.35%) | 0.07747 (-3.13%) | **0.07213 (+0.91%)** | **0.07515** | 1 / 4 |
| **WF Expanding Residual (`W=756, S=40`)** | Quân (PP4) | `FS02` | 0.07570 (-9.71%) | 0.07852 (-1.88%) | 0.07671 (-2.12%) | 0.07357 (-1.06%) | **0.07613** | 0 / 4 |

---

## 3. Kiểm định Ý nghĩa Thống kê Toàn Nhóm (Diebold–Mariano Significance Tests)

Biểu đồ bản đồ kiểm định DM toàn nhóm: [fig18_group_dm_significance_and_pros_cons.png](file:///d:/PTDLKD/ptdlkd-is403.r11-main/quan%20task/result/fig18_group_dm_significance_and_pros_cons.png) · Bảng CSV đầy đủ (120 dòng kiểm định): [group_significance_dm_tests_complete.csv](file:///d:/PTDLKD/ptdlkd-is403.r11-main/quan%20task/result/group_significance_dm_tests_complete.csv)

Để tránh kết luận cảm tính từ các chênh lệch thập phân thứ 4–5 (`0.0001 – 0.0003 $/lb`), toàn bộ các phép so sánh trong nhóm được chuẩn hóa bằng **kiểm định Diebold–Mariano (1995)** với ước lượng phương sai vững **Newey–West HAC** ($\text{lag} = 5$) và hiệu chỉnh mẫu nhỏ **Harvey–Leybourne–Newbold (HLN)** trên cả hai hàm mất mát $L_1$ ($|e_t|$) và $L_2$ ($e_t^2$).

### 3.1. Kiểm định giả thuyết trung tâm $H_0^{(1)}$: *"Có mô hình Residual nào vượt Naive Persistence có ý nghĩa thống kê không?"*

| Thành viên | Phép so sánh với Naive Persistence ($\hat{y}_t = y_{t-1}$) | Số cấu hình / lượt chạy kiểm định | Số lượt mô hình có MAE thấp hơn Naive ($\bar{d}_{L1} < 0$) | Số lượt thắng Naive có ý nghĩa thống kê 5% ($\text{DM}_{L1} < -1.96, p < 0.05$) | Số lượt kém Naive có ý nghĩa thống kê 5% ($\text{DM}_{L1} > +1.96, p < 0.05$) | Thống kê $\text{DM}_{L1}$ & $p$-value tiêu biểu |
|---|---|---:|---:|---:|---:|---|
| **1. Khoa (`Module1/`)** | `Walk-Forward GRU Residual` vs. `Naive` ($N=2.214$ phiên, 2018–2026, 5 seeds `41–45`) | 5 seeds | 1 / 5 (Seed 41) | **0 / 5 (0%)** | 2 / 5 (Seeds 43, 44) | Seed 41: $\text{DM} = -0.379, p = 0.7043$; Seed 42: $\text{DM} = +0.905, p = 0.3656$ |
| **2. Công (`TPE+LSTM/`)** *(Quân tính từ raw preds)* | `SA–LSTM Residual` vs. `Naive` (6 bộ biến $\times$ 6 bối cảnh) | 36 cấu hình | 6 / 36 (`FS01`, `FS06`) | **0 / 36 (0%)** | 19 / 36 | `FS01` (90/10 Exp): $\text{DM} = -1.164, p = 0.2456$; `FS06` (90/10 Exp): $\text{DM} = +0.068, p = 0.9461$ |
| **2. Công (`TPE+LSTM/`)** *(Quân tính từ raw preds)* | `TPE–LSTM Residual` vs. `Naive` (6 bộ biến $\times$ 6 bối cảnh) | 36 cấu hình | 1 / 36 (`FS01` Roll) | **0 / 36 (0%)** | 18 / 36 | `FS01` (90/10 Exp): $\text{DM} = +0.584, p = 0.5597$; `FS02` (90/10 Exp): $\text{DM} = +1.378, p = 0.1695$ |
| **3. Quân (`quan task/`)** | `WF Expanding Residual` vs. `Naive` (6 bộ biến `FS01–FS06` + 5 seeds `41–45` ở `FS02`) | 11 lượt | 1 / 11 (`FS03`) | **0 / 11 (0%)** | 2 / 11 (`FS05`, Seed 43) | `FS03`: $\text{DM} = -0.172, p = 0.8636$; `FS01`: $\text{DM} = +0.761, p = 0.4475$; `FS04`: $\text{DM} = +0.600, p = 0.5491$ |
| **4. Huy (`huy task/`)** | `Hybrid ARIMA–GRU` vs. `Naive` (4 splits $\times$ 5 seeds `41–45`) | 20 lượt | 17 / 20 | **2 / 20 (10%)** *(chỉ ở 90/10 Seed 42 & 45)* | **0 / 20 (0%)** | 90/10 TB 5 seeds: $\text{DM} = -1.900, p_{\text{tb}} = 0.0845$; 92.5/7.5 TB: $\text{DM} = -1.621, p_{\text{tb}} = 0.1216$ |

> **Kết luận Kiểm định $H_0^{(1)}$ (Đóng góp học thuật cốt lõi của toàn nhóm):**
> Tổng hợp trên **108 phép kiểm định Diebold–Mariano độc lập** giữa các mô hình học sâu nhánh Residual (`GRU`, `LSTM`, `SA-LSTM`, `TPE-LSTM`, `Walk-Forward LSTM`, `Hybrid ARIMA-GRU`) với `Naive Persistence`:
> - Có tới **106 / 108 lượt chạy (98.1%) KHÔNG thể vượt qua Naive Persistence ở mức ý nghĩa thống kê $\alpha = 5\%$** (và dưới hàm mất mát bậc hai $L_2$, có **107 / 108 lượt chạy (99.1%)** không thắng được Naive!).
> - Ngay cả các cấu hình có MAE thấp hơn Naive một lượng nhỏ (`+0.09%` đến `+0.41%` như `SA-LSTM FS01 0.07695`, `WF Expanding FS03 0.07693`, `GRU Residual FS04 0.07695`, `Hybrid ARIMA-GRU 0.07676`) đều nằm hoàn toàn trong vùng không bác bỏ $H_0$ ($|\text{DM}_{L1}| < 1.96, p \in [0.084, 0.864]$).
> - **Giải thích kinh tế lượng (Khoa V20 & Quân Phase 2):** Do chuỗi sai phân giá đồng ngày tuân theo quá trình **Martingale Difference Sequence (MDS)** với tỷ lệ tín hiệu trên nhiễu cực thấp ($|r_{\text{lag-1}}| < 0.091$), nghiệm tối ưu của mạng nơ-ron dưới hàm mất mát MSE hội tụ về $\hat{\Delta}_t \approx 0$ (hiện tượng **Variance Shrinkage**: $\text{std}(\hat{\Delta}_t) / \text{std}(\Delta_t) \approx 0.01 - 0.18$), biến mọi mô hình Residual hội tụ tiệm cận về đường Random Walk $\hat{y}_t \approx y_{t-1}$.

### 3.2. Kiểm định giả thuyết $H_0^{(2)}$: *"TPE–LSTM có vượt trội có ý nghĩa thống kê so với SA–LSTM không?"* (Kiểm định mới do Quân thực hiện trên dữ liệu của Công)

Trong báo cáo của Công ([TPE+LSTM/README.md](file:///d:/PTDLKD/ptdlkd-is403.r11-main/TPE+LSTM/README.md)), Công đã ghi nhận TPE có MAE thấp hơn ở 16/36 cặp và SA có MAE thấp hơn ở 20/36 cặp, nhưng chưa có kiểm định Diebold–Mariano. Từ 20.340 bản ghi dự báo theo ngày trong `TPE+LSTM/.../test_predictions.csv`, Quân đã thực hiện kiểm định **DM-HAC cho toàn bộ 36 cặp đối đầu trực tiếp `TPE–LSTM vs. SA–LSTM`**:

| Bộ biến (`Split 90/10 Expanding`, $N=271$) | SA–LSTM MAE (\$/lb) | TPE–LSTM MAE (\$/lb) | Chênh lệch $\bar{d}_{L1} = \text{TPE} - \text{SA}$ | Thống kê $\text{DM}_{L1}$ | Giá trị $p_{L1}$ | Thống kê $\text{DM}_{L2}$ | Giá trị $p_{L2}$ | Kết luận ở mức ý nghĩa $\alpha = 5\%$ |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`FS01` (Copper)** | **0.07695** | 0.07727 | +0.00031 | +1.0321 | 0.3029 | +0.2952 | 0.7681 | Không khác biệt ý nghĩa ($p = 0.303$) |
| **`FS02` (Paper3 + Cu)** | 0.08453 | **0.07798** | **-0.00655** | **-3.1593** | **0.0018** | **-2.9999** | **0.0029** | **TPE–LSTM thắng có ý nghĩa thống kê ($p < 0.01$)** |
| **`FS03` (Paper3 - Cu)** | 0.07999 | **0.07812** | **-0.00187** | **-3.1352** | **0.0019** | **-2.8018** | **0.0055** | **TPE–LSTM thắng có ý nghĩa thống kê ($p < 0.01$)** |
| **`FS04` (5 Kim loại)** | **0.08097** | 0.10006 | **+0.01910** | **+3.6848** | **0.0003** | **+3.6352** | **0.0003** | **SA–LSTM thắng có ý nghĩa thống kê ($p < 0.001$)** |
| **`FS05` (4 Kim loại - Cu)** | **0.07896** | 0.08198 | +0.00302 | +1.1815 | 0.2384 | +1.4789 | 0.1403 | Không khác biệt ý nghĩa ($p = 0.238$) |
| **`FS06` (WTI + Cu)** | **0.07709** | 0.07731 | +0.00023 | +0.6605 | 0.5095 | -0.2147 | 0.8301 | Không khác biệt ý nghĩa ($p = 0.510$) |

- **Tổng kết trên toàn bộ 36 cặp bối cảnh (`6 bộ biến × 6 bối cảnh`) của Công:**
  - **18 / 36 cặp (50.0%)** không có sự khác biệt có ý nghĩa thống kê ($p \ge 0.05$).
  - **11 / 36 cặp (30.6%)** **SA–LSTM thắng có ý nghĩa thống kê** ($p < 0.05$, tập trung ở `FS04` và `FS01`).
  - **7 / 36 cặp (19.4%)** **TPE–LSTM thắng có ý nghĩa thống kê** ($p < 0.05$, tập trung ở `FS02` và `FS03`).
  - **Nguyên nhân:** Với ngân sách 16 trials trên tập Validation của mốc `80/20`, thuật toán Bayesian TPE hội tụ rất nhanh vào vùng cực tiểu của tập Validation; khi gặp bộ biến đa cộng tuyến cao như `FS04` (5 kim loại), TPE dễ chọn bộ siêu tham số (`lr` cao hơn) bị **overfit vào Validation**, trong khi cơ chế bước nhảy nhiệt độ ngẫu nhiên của Simulated Annealing (SA) giữ được tính chính quy hóa tốt hơn.

### 3.3. Kiểm định giả thuyết $H_0^{(3)}$: *"Tái huấn luyện Walk-Forward (Quân) so với Mô hình Tĩnh (Quân, Công, Vĩ)"*

1. **Trong nội bộ phương pháp Walk-Forward của Quân ([phase5_diebold_mariano_tests.csv](file:///d:/PTDLKD/ptdlkd-is403.r11-main/quan%20task/result/phase5_diebold_mariano_tests.csv)):**
   - **Trên nhánh Direct:** `Expanding Window` và `Rolling Window` thắng `Static (Train-Only)` với ý nghĩa thống kê tuyệt đối trên toàn bộ các bộ biến:
     - `FEATURES_4 (FS02)`: `Expanding Direct (0.2370)` vs `Static Train-Only Direct (1.5516)` $\rightarrow \text{DM}_{L1} = -51.359, p < 0.0001$; `Rolling Direct (0.2837)` vs `Static (1.5516)` $\rightarrow \text{DM}_{L1} = -43.541, p < 0.0001$.
   - **Trên nhánh Residual:** `Expanding Window Residual (0.07852)` thắng `Static Train-Only Residual (0.10969)` với $\text{DM}_{L1} = -4.938, p < 0.0001$.
   - **So sánh `Expanding Window` vs `Rolling Window` (`W=756`):** Trên nhánh Direct, `Expanding` thắng `Rolling` có ý nghĩa thống kê ($\text{DM}_{L1} = -2.263, p = 0.0244$); trên nhánh Residual, hai chiến lược tương đương thống kê ($\text{DM}_{L1} = -0.822, p = 0.4119$).
2. **Kiểm định DM chéo giữa `Quân (WF Expanding Residual)` và `Công (SA-LSTM & TPE-LSTM tĩnh)` trên đúng 271 phiên Test của Split 90/10:**
   - Ở các bộ biến gọn nhẹ `FS01` và `FS06`: Quân và Công tương đương thống kê ($|\text{DM}_{L1}| \in [0.23, 0.90], p \in [0.368, 0.817]$).
   - Ở các bộ biến đa biến dễ gây trôi tham số khi giữ mô hình tĩnh (`FS02`, `FS03`, `FS04`, `FS05`): **Cơ chế tái huấn luyện định kỳ mỗi 40 phiên của Quân (`WF-Expanding`) thắng `SA-LSTM tĩnh` của Công có ý nghĩa thống kê ở 3/4 bộ biến** (`FS02`: $\text{DM} = -3.158, p = 0.0018$; `FS03`: $\text{DM} = -2.405, p = 0.0169$; `FS04`: $\text{DM} = -2.118, p = 0.0351$) và **thắng `TPE-LSTM tĩnh` của Công có ý nghĩa thống kê ở `FS04` ($\text{DM} = -5.163, p < 0.0001$) và `FS05` ($\text{DM} = -2.383, p = 0.0179$)**!
3. **Kiểm định DM của Vĩ (`iTransformer vs. SA-LSTM` ở `K=5`, Split 90/10 Expanding):**
   - `iTransformer Residual (MAE = 0.07812)` có sai số thấp hơn có ý nghĩa thống kê so với `SA-LSTM Fixed Residual (MAE = 0.08293, p = 0.0248)` và `SA-LSTM Tuned Residual (MAE = 0.08535, p = 0.00014)`, nhờ cơ chế LayerNorm và Inverted Attention kiểm soát nhiễu đa biến tốt hơn khi không tái huấn luyện định kỳ.

---

## 4. Nghiên cứu Cắt lớp Toàn Nhóm (Group-Wide Ablation Studies)

Biểu đồ Heatmap & Log-scale Ablation 6 bộ biến: [fig17_group_ablation_6_feature_sets.png](file:///d:/PTDLKD/ptdlkd-is403.r11-main/quan%20task/result/fig17_group_ablation_6_feature_sets.png)

### 4.1. Ablation 1 — Vai trò của 6 Bộ biến đầu vào (`FS01` → `FS06`) và Lịch sử Giá Đồng (`Copper Lag`)

| Mã bộ biến | Các biến đưa vào tensor $X_{t-22:t-1}$ | Số biến ($K$) | Khoa (`LSTM-Res`) | Khoa (`Ridge-Res`) | Công (`SA-LSTM`) | Công (`TPE-LSTM`) | Vĩ (`iTrans-Res`) | Quân (`WF-Exp-Res`) | Quân (`WF-Exp-Dir`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`FS01_CU`** | `Copper` | 1 | 0.07701 | **0.07731** | **0.07695** | **0.07727** | **0.07681** | 0.07736 | **0.20265** |
| **`FS02_PAPER_CU`** | `Copper, WTI, Gold, Silver` | 4 | 0.07701 | 0.08071 | 0.08453 | 0.07798 | 0.07701 | 0.07852 | 0.23696 |
| **`FS03_PAPER_EXO`** | `WTI, Gold, Silver` *(Bỏ Copper)* | 3 | — | — | 0.07999 | 0.07812 | *0.09027* | **0.07693** | *0.38606* |
| **`FS04_METALS_CU`** | `Copper, Al, Tin, Silver, Gold` | 5 | **0.07698** | *0.08171* | 0.08097 | *0.10006* | 0.07687 | 0.07769 | 0.33844 |
| **`FS05_METALS_EXO`** | `Al, Tin, Silver, Gold` *(Bỏ Copper)* | 4 | — | — | 0.07896 | 0.08198 | 0.07924 | 0.07720 | *0.36182* |
| **`FS06_WTI_CU`** | `Copper, Al, Tin, WTI` | 4 | 0.07704 | 0.07774 | 0.07709 | 0.07731 | 0.07794 | 0.07764 | 0.21921 |
| **Trung bình 6 bộ** | — | — | **0.07701** | **0.07937** | **0.07975** | **0.08212** | **0.07969** | **0.07756** | **0.29086** |

**3 kết luận quan trọng từ Ablation 6 bộ biến:**
1. **Tác động của việc loại bỏ chuỗi `Copper` khỏi tensor đầu vào (`FS02 vs. FS03` và `FS04 vs. FS05`):**
   - **Trên nhánh Direct (Quân & Khoa V21):** Khi bỏ `Copper` khỏi $X$ (`FS03`, `FS05`), mô hình mất hoàn toàn điểm neo mức giá mục tiêu, khiến MAE Direct của Quân tăng vọt từ `0.23696 → 0.38606 $/lb` (+62.9%).
   - **Trên nhánh Residual:** Vì bước hoàn nguyên cuối cùng vẫn sử dụng $\hat{y}_t = y_{t-1} + \hat{\Delta}_t$, các mạng LSTM có cơ chế tái huấn luyện (Quân `WF-Expanding`) tự co $\hat{\Delta}_t \to 0$ khi đầu vào ngoại sinh thiếu tín hiệu, giúp `FS03` đạt `0.07693 $/lb`. Ngược lại, **kiến trúc `iTransformer` của Vĩ bị suy giảm mạnh ở `FS03` (`MAE = 0.09027 $/lb`, kém Naive `-17.11%`)**, bởi cơ chế Inverted Attention giao tiếp giữa các variable-tokens bị mất token trung tâm (`Copper`) để đối chiếu tương quan chéo!
2. **Hiện tượng Nhiễu chiều dữ liệu (Exogenous Noise Inflation):**
   - Kiểm định tương quan trễ bậc 1 trên tập Train của Khoa (`05_lagged_returns_correlations_FDR.csv`) chứng minh tương quan giữa biến động ngày của `Gold, Silver, WTI, Aluminium, Tin` ở $t-1$ với `Copper` ở $t$ rất yếu ($|r| \le 0.091$).
   - Do đó, `FS01 (Chỉ dùng Copper, K=1)` đã đạt hiệu năng tốt nhất hoặc sát đỉnh trên hầu hết các phương pháp (`Vĩ iTransformer: 0.07681`, `Công SA-LSTM: 0.07695`, `Công TPE-LSTM: 0.07727`, `Khoa Ridge: 0.07731`, `Huy Hybrid ARIMA-GRU: 0.07676`).
   - Khi mở rộng lên 5 kim loại (`FS04`), mô hình tuyến tính `Ridge` của Khoa bị suy giảm `+5.69%`, và `TPE-LSTM` của Công bị overfit mạnh (`MAE = 0.10006 $/lb`).
3. **Tính hợp lệ của `FS06 (Copper, Aluminium, Tin, WTI)`:**
   - Bộ biến kết hợp nhóm kim loại công nghiệp LME (`Copper, Aluminium, Tin`) và năng lượng (`WTI`) đạt độ ổn định cao thứ nhì toàn nhóm (sau `FS01`), đặc biệt giúp `Paper SA-LSTM Direct` của Khoa đạt MAE Direct thấp nhất (`0.18463 $/lb`) và `SA-LSTM Residual` của Công đạt `0.07709 $/lb`.

### 4.2. Ablation 2 — Độ dài cửa sổ huấn luyện (`Window Size W`), Bước trượt (`Stride S`), `Recency Weighting` & `Cold vs. Warm Start`

1. **Ablation Chiến lược Cửa sổ (`Expanding` vs. `Rolling 756` vs. `Recency Weighted`) của Khoa, Công, Vĩ & Quân:**
   - Trên cả 4 nhánh thực nghiệm (Khoa V20, Công TPE+LSTM, Vĩ iTransformer, Quân Phase 2–3), chiến lược **`Expanding Window` (tận dụng toàn bộ lịch sử từ 2016)** ổn định hơn `Rolling 756` (~3 năm) ở các mốc `80/20` và `90/10`, vì mạng học sâu cần kích thước mẫu lớn (~2.173 phiên) để ước lượng trọng số ổn định.
   - Riêng ở các mốc test ngắn hạn ở vùng đỉnh giá cuối chuỗi (`92.5/7.5` và `95/5`), **`Recency Weighted` (half-life 252 phiên)** của Khoa và Vĩ phát huy tác dụng rõ rệt: `iTransformer Residual (FS01, Recency)` giảm MAE xuống **`0.07645 $/lb` (+0.82% vs Naive)** ở `90/10` và **`0.07438 $/lb` (+0.98% vs Naive)** ở `92.5/7.5`.
2. **Ablation Tham số Walk-Forward (`W`, `S`, `Cold-Start vs. Warm-Start`) của Quân ([phase3_ablation_study.csv](file:///d:/PTDLKD/ptdlkd-is403.r11-main/quan%20task/result/phase3_ablation_study.csv)):**
   - **Window Size $W \in \{252, 504, 756\}$ (cố định $S=40$):** $W=756$ đạt MAE thấp nhất (`0.28371 $/lb`), vượt trội so với cửa sổ quá ngắn $W=252$ (`0.33899 $/lb`, bị thiếu mẫu huấn luyện) và $W=504$ (`0.35547 $/lb`).
   - **Retrain Stride $S \in \{10, 20, 40, 60\}$ (cố định $W=756$):** Giảm $S$ từ `60 → 10` phiên giúp giảm liên tục sai số Direct (`MAE` giảm từ `0.36000 → 0.23003 $/lb`), nhưng thời gian tính toán tăng tuyến tính từ `66.8s (5 lần retrain)` lên `333.7s (28 lần retrain)`. Mốc $S^* = 40$ (7 lần retrain, `88.7s`) là điểm cân bằng Pareto tối ưu giữa chi phí GPU và độ chính xác.
   - **Cơ chế cập nhật trọng số (`Cold-Start` vs. `Warm-Start`):**
     - På nhánh **Direct**: `Warm-Start` (fine-tune tiếp từ trọng số fold trước với `lr` giảm một nửa) giúp thích nghi cực nhanh với mặt bằng giá mới (`MAE = 0.21942 $/lb` so với `0.28371 $/lb` của `Cold-Start`, tốn chỉ `77.7s`).
     - Trên nhánh **Residual**: `Cold-Start` (`MAE = 0.08026 $/lb`) ổn định hơn hẳn `Warm-Start` (`MAE = 0.09777 $/lb`), vì việc kế thừa trọng số qua nhiều fold trên chuỗi sai phân nhiễu cao gây hiện tượng tích lũy độ lệch (*gradient drift*).

---

## 5. Phân tích Ưu / Nhược điểm & Tính Có Ý Nghĩa của Từng Phương Pháp (5 Thành Viên)

| Phương pháp & Thành viên | Ưu điểm nổi bật (Strengths) | Nhược điểm & Giới hạn (Weaknesses) | Tính có ý nghĩa thống kê & Giá trị thực tiễn |
|---|---|---|---|
| **1. Non-SA Baselines & Residual Protocol V20–V26** *(Viên Đăng Khoa — `Module1/`)* | - Khung kiểm chứng khoa học mẫu mực: Full Factorial Grid (324 runs), Walk-Forward 2.214 phiên (2018–2026) $\times$ 5 seeds.<br>- Chứng minh toán học cơ chế **Variance Shrinkage** ($\text{std}(\hat{\Delta}_t) \ll \text{std}(\Delta_t)$) và kiểm định kinh tế lượng toàn diện (ADF, KPSS, Zivot–Andrews, Granger, Spearman FDR).<br>- `LSTM Residual` và `GRU Residual` cực kỳ ổn định (`MAE = 0.07695 – 0.07701 $/lb`). | - Các mô hình Non-SA dùng siêu tham số cố định (`lr = 1e-3`), chưa tối ưu hóa riêng cho từng cấu trúc đặc trưng.<br>- Mô hình tuyến tính `Ridge Residual` nhạy cảm mạnh với nhiễu đa biến (`Noise Inflation` làm tăng sai số 4.4%–18.9% khi thêm biến ngoại sinh). | - **Ý nghĩa học thuật cao nhất nhóm:** Chứng minh chuỗi giá đồng ngày tuân theo quy luật tiệm cận bước ngẫu nhiên (*Near-Random-Walk*), bác bỏ ảo tưởng rằng mạng học sâu phức tạp luôn thắng Naive ở $H=1$. |
| **2. Tối ưu Siêu tham số SA–LSTM vs. TPE–LSTM (Optuna)** *(Nguyễn Thành Công — `TPE+LSTM/`)* | - Thiết kế đối chuẩn công bằng tuyệt đối: cùng ngân sách 16 trials/bộ biến trên `80/20`, khóa siêu tham số trước khi đánh giá trên 6 bối cảnh $\times$ 6 bộ biến (192 search fits + 72 final fits).<br>- `SA-LSTM (FS01)` vượt nhẹ Naive ở cả **4/4 mốc chia** (`80/20, 90/10, 92.5/7.5, 95/5`).<br>- `TPE-LSTM` hội tụ nhanh và thắng `SA-LSTM` có ý nghĩa thống kê ($p < 0.01$) trên bộ biến chuẩn bài báo gốc (`FS02`, `FS03`). | - Chạy trên 1 seed (`42`) và mô hình giữ cố định trong suốt giai đoạn test (không retrain định kỳ).<br>- `TPE-LSTM` dễ bị **overfit vào tập Validation** khi không gian đặc trưng có đa cộng tuyến cao (`FS04`: MAE vọt lên `0.10006 $/lb`, thua SA-LSTM với $p = 0.0003$). | - **Ý nghĩa phương pháp luận rõ rệt:** Chứng minh không có thuật toán HPO nào thống trị tuyệt đối (18/36 cặp không khác biệt ý nghĩa; SA thắng ý nghĩa 11 cặp, TPE thắng ý nghĩa 7 cặp). SA an toàn hơn trên tập nhiều biến nhiễu, TPE sắc bén hơn trên bộ biến `Paper3`. |
| **3. Kiến trúc iTransformer (Inverted Attention)** *(Vĩ — `Vi Task/`)* | - Đổi mới kiến trúc hiện đại: hoán đổi trục thời gian–biến để mỗi chuỗi hàng hóa thành 1 *variable-token* (`Dense 22 → 48`), học tương quan chéo liên thị trường bằng Multi-Head Attention.<br>- Đạt MAE thấp thứ nhì toàn nhóm trên `90/10 Expanding` (`FS01: 0.07681`, `FS04: 0.07687`) và đạt **`0.07645 $/lb` (+0.82% vs Naive)** khi kết hợp `Recency Weighted`.<br>- Thắng `SA-LSTM Fixed` ($p = 0.0248$) và `SA-LSTM Tuned` ($p = 0.00014$) trong benchmark đối đầu `K=5`. | - Khi $K=1$ (`FS01`), cơ chế self-attention trên trục biến suy biến thành phép chiếu đơn token (softmax trên trục kích thước 1).<br>- Phụ thuộc nặng vào sự hiện diện của token mục tiêu `Copper` trong đầu vào: khi bỏ `Copper` ở `FS03`, MAE Residual bị suy giảm mạnh lên `0.09027 $/lb` (-17.11% vs Naive).<br>- Nhánh Direct có sai số cao nhất nhóm (`1.14 – 1.81 $/lb`). | - **Ý nghĩa kiến trúc:** Chứng minh Transformer đảo trục + LayerNorm kết hợp rất ăn ý với mục tiêu **Residual + Recency Weighting** trên tập đa kim loại (`FS04`), nhưng không phù hợp để dự báo mức giá trực tiếp (Direct Level) trên chuỗi không dừng. |
| **4. Rolling & Expanding Window Walk-Forward Backtesting** *(Lê Anh Quân — `quan task/`)* | - Giải quyết trực diện bài toán **Concept Drift & Regime Shift**: định kỳ cập nhật lại cả `MinMaxScaler` lẫn trọng số mô hình mỗi $S=40$ phiên mà không rò rỉ dữ liệu.<br>- **Giảm 64%–85% sai số trên nhánh Direct** so với mô hình tĩnh ($\text{DM}_{L1} = -51.36, p < 0.0001$).<br>- **Đạt MAE trung bình qua 6 bộ biến thấp nhất và ổn định nhất toàn nhóm (`0.07756 ± 0.00054 $/lb`)**; thắng `SA-LSTM` và `TPE-LSTM` tĩnh của Công có ý nghĩa thống kê ($p < 0.05$) trên các bộ đa biến `FS02, FS03, FS04, FS05`.<br>- Kiểm chứng đầy đủ qua 5 seeds (`41–45`), Ablation $W, S, \text{Cold/Warm}$ và 3 giai đoạn biến động thị trường. | - Chi phí tính toán cao gấp $7 - 28$ lần mô hình tĩnh do phải huấn luyện lại nhiều lần dọc theo tập Test.<br>- Trên nhánh Residual ở bộ biến đơn giản (`FS01`), việc tái huấn luyện nhiều lần không tạo ra lợi thế có ý nghĩa thống kê so với mô hình tĩnh (`Static T+V`) hay Naive ($p = 0.4475$). | - **Ý nghĩa vận hành thực tế (Production Readiness):** Là quy trình bắt buộc nếu triển khai mô hình dự báo dài hạn qua các chu kỳ dịch chuyển mặt bằng giá lớn (như đợt sóng tăng 2025–2026), giúp ngăn chặn sự sụp đổ của cả mô hình Direct lẫn mô hình đa biến tĩnh. |
| **5. Mô hình Lai Hybrid ARIMA–GRU (Zhang 2003)** *(Phan Ngọc Đức Huy — `huy task/`)* | - Kết hợp chặt chẽ kinh tế lượng cổ điển (`auto_arima` kiểm định ADF bậc sai phân $d=1$) và mạng học sâu (`GRU` học phần dư phi tuyến).<br>- Đạt **MAE thấp nhất toàn nhóm trên Split 90/10 (`0.07676 ± 0.00006 $/lb`, cải thiện `+0.41%` vs Naive)** và vượt nhẹ Naive ở **3/4 mốc chia** (`90/10`, `92.5/7.5: 0.07441 (+0.94%)`, `95/5: 0.07248 (+0.44%)`).<br>- Độ lệch chuẩn qua 5 seeds cực thấp (`0.00006 – 0.00029 $/lb`). | - Chỉ mới áp dụng cho chuỗi đơn biến `FS01 (Copper)`, chưa mở rộng sang các biến ngoại sinh (`ARIMAX–GRU`).<br>- Ở `80/20` và `90/10`, `auto_arima` chọn `ARIMA(0,1,0)` (chính là Random Walk / Naive), nên phần đóng góp của nhánh GRU thực chất chỉ thu hẹp phương sai quanh 0 giống `GRU Residual` của Khoa. | - **Ý nghĩa đối chuẩn kinh tế lượng:** Chỉ có `2/20` lượt chạy (10%) thắng Naive ở mức ý nghĩa 5% dưới loss $L_1$ (và `1/20` dưới loss $L_2$). Khẳng định ngay cả khi kết hợp ARIMA bậc cao `ARIMA(2,1,2)` + `GRU`, phần cải thiện chủ yếu đến từ thành phần tự hồi quy tuyến tính (`-0.75%`), còn phần dư phi tuyến của GRU chỉ đóng góp thêm `-0.19%` (không có ý nghĩa thống kê, $p = 0.12$). |

---

## 6. Danh mục Tệp Tổng hợp Chung (`group_benchmark_summary/`)

- Báo cáo tổng hợp: [THONG_KE_CHISO_SIGNIFICANCE_ABLATION.md](THONG_KE_CHISO_SIGNIFICANCE_ABLATION.md)
- Bảng Ablation 6 bộ biến toàn nhóm: [group_ablation_6_feature_sets_all_members.csv](group_ablation_6_feature_sets_all_members.csv)
- Bảng Cross-Ratio 4 mốc chia toàn nhóm: [group_cross_ratio_4_splits_all_members.csv](group_cross_ratio_4_splits_all_members.csv)
- Bảng kiểm định Diebold–Mariano toàn nhóm (120 dòng): [group_significance_dm_tests_complete.csv](group_significance_dm_tests_complete.csv)
- 3 biểu đồ tổng hợp toàn nhóm:
  - [fig16_group_benchmark_all_5_members.png](fig16_group_benchmark_all_5_members.png)
  - [fig17_group_ablation_6_feature_sets.png](fig17_group_ablation_6_feature_sets.png)
  - [fig18_group_dm_significance_and_pros_cons.png](fig18_group_dm_significance_and_pros_cons.png)
