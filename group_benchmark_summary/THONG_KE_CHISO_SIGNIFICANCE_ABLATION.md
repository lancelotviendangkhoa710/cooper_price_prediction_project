# BÁO CÁO TỔNG HỢP TOÀN NHÓM: THỐNG KÊ CHỈ SỐ, KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (DIEBOLD–MARIANO), NGHIÊN CỨU CẮT LỚP (ABLATION STUDY) & ĐÁNH GIÁ ƯU/NHƯỢC ĐIỂM 5 PHƯƠNG PHÁP

- **Học phần:** IS403.R11 — Phân tích dữ liệu kinh doanh  
- **Đề tài:** Dự báo giá đồng (*Copper Futures Price Forecasting*)  
- **Phạm vi:** Đối chuẩn độc lập, chuẩn hóa đồng nhất và trung lập giữa **5 hướng tiếp cận mô hình hóa (`PP1–PP5`)** trên cùng bộ dữ liệu và cùng các mốc kiểm định  
- **Ngày cập nhật:** 27/11/2026 *(Đã đồng bộ đầy đủ kết quả **5 seeds `41–45`** trên toàn bộ 6 bộ biến `FS01–FS06` của `PP2` (720 lượt + Direct/Residual), `PP3` (720 lượt + raw predictions `FS06`), `PP4` (Phase 2D 120 lượt `90/10` & `80/20`), `PP5` (360 lượt), cùng toàn bộ Grid + Walk-Forward 5-seed của `PP1`)*

---

## 1. Xác minh Tính Đồng nhất Dữ liệu & Giao thức Thực nghiệm (`5/5 Phương pháp`)

### 1.1. Xác minh mã băm SHA-256 giữa môi trường Cục bộ và Kaggle
Môi trường Windows cục bộ ghi nhận SHA-256 của tệp dữ liệu gốc là `9580ede1...`, trong khi snapshot trên Kaggle Linux ghi nhận `a761a908...`. Kiểm định đối chiếu từng phần tử số cho thấy nguyên nhân duy nhất là ký tự xuống dòng `CRLF (\r\n)` so với `LF (\n)` (áp dụng tương tự cho bộ chạy `FS06` của `PP3` dùng `9580ede1...` so với `FS01–FS05` dùng `a761a908...`):

| Phiên bản tệp | Ký tự xuống dòng | Mã băm SHA-256 | Số phiên ($N$) | Khoảng thời gian | Sai lệch số học cực đại ($\max |x_i - y_i|$) |
|---|---|---|---:|---|---:|
| **Local Git (`CRLF`)** | Windows `\r\n` | `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600` | 2.710 | `13/01/2016 → 13/11/2026` | **`0.00e+00`** |
| **Kaggle Snapshot (`LF`)** | Linux `\n` | `a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83` | 2.710 | `13/01/2016 → 13/11/2026` | **`0.00e+00`** |

Kiểm tra chéo tất cả tệp dự báo chi tiết của `PP1`, `PP2` (`FS01–FS06`), `PP3` (`FS06`), `PP4` và `PP5` xác nhận **100% trùng khớp từng ngày giao dịch (`Date`) và từng giá trị thực tế (`Actual`, sai lệch `< 10⁻⁶`)**, với chuẩn cơ sở **Naive Persistence ($\hat{y}_t = y_{t-1}$)** đồng nhất:
- **Split 80/20 ($N_{\text{test}} = 542$, `07/10/2024 → 13/11/2026`):** $\text{MAE}_{\text{Naive}} = 0.069001\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.108000\text{ \$/lb}$
- **Split 90/10 ($N_{\text{test}} = 271$, `29/10/2025 → 13/11/2026`):** $\text{MAE}_{\text{Naive}} = 0.077077\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.101033\text{ \$/lb}$
- **Split 92.5/7.5 ($N_{\text{test}} = 204$):** $\text{MAE}_{\text{Naive}} = 0.075116\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.098399\text{ \$/lb}$
- **Split 95/5 ($N_{\text{test}} = 136$):** $\text{MAE}_{\text{Naive}} = 0.072799\text{ \$/lb}$, $\text{RMSE}_{\text{Naive}} = 0.096612\text{ \$/lb}$

### 1.2. Phạm vi thực nghiệm của 5 phương pháp (`PP1–PP5`)

| Mã nhóm | Phương pháp thực nghiệm | Tổ hợp Bộ biến (`FS01–FS06`) | Tỷ lệ chia & Chiến lược | Mục tiêu | Đa hạt giống (`Multi-Seed`) | Dữ liệu dự báo chi tiết (Raw Preds) |
|---|---|---|---|---|---|---|
| **PP1** | **Non-SA Baselines & SA-LSTM Standalone** (`LSTM`, `GRU`, `CNN-LSTM`, `CNN-GRU`, `Ridge`, `SA-LSTM`) | Grid Non-SA: `FS01, 02, 04, 06`; SA: `FS02–06` | 4 mốc × 3 schemes (`Expanding`, `Rolling 756`, `Recency`) + Walk-Forward (`80/20`) | `Direct` & `Residual` | Holdout Grid: `1 seed (42)`; **Walk-Forward: `5 seeds (41–45)`** (`26` Res + `15` Dir) | Đầy đủ Grid (`480` lượt) & WF (`V20/V23/V25`) |
| **PP2** | **Tối ưu siêu tham số SA–LSTM vs. TPE–LSTM (Optuna)** (1.920 search trials + 720 final runs) | **Đủ 6 bộ (`FS01–FS06`)** | 4 mốc `Expanding` + `Rolling 756` & `Recency` ở `90/10` | **`Direct` & `Residual`** | **Full `5 seeds (41–45)`** toàn lưới (`720` lượt) | **Đầy đủ `6 FS × 5 seeds` (`203.400` dòng)** |
| **PP3** | **iTransformer (Inverted Attention)** + Đối chuẩn SA-LSTM (`K=5`) | **Đủ 6 bộ (`FS01–FS06`)** | 4 mốc × 3 schemes | **`Direct` & `Residual`** | **Full `5 seeds (41–45)`** toàn lưới (`720` lượt neural + `72` Naive) | **`FS06`: Đầy đủ `38.049` dòng + DM**; `FS01–FS05`: bảng MAE `5 seeds` |
| **PP4** | **Rolling & Expanding Window Walk-Forward LSTM** (`Static T`, `Static T+V`, `WF Exp`, `WF Roll`, Ablation $W,S$) | **Đủ 6 bộ (`FS01–FS06`)** ở `90/10` & `80/20`; `Cross-Ratio` 4 mốc trên `FS02` | `Walk-Forward (W=756, S=40)` + `Static Holdout` | **`Direct` & `Residual`** | **Full `5 seeds (41–45)`** đủ 6 FS × Exp/Roll ở `90/10` & `80/20` (`120` lượt); Direct/Static `1 seed` | **Đầy đủ `90/10` & `80/20` 6-FS 5-seed (`48.780` dòng)** + `49` tệp 1-seed |
| **PP5** | **Hybrid ARIMA–GRU** (`auto_arima` + GRU 2 tầng `64→32→16→1` học phần dư, 360 lượt GRU) | **Đủ 6 bộ (`FS01–FS06`)** | 4 mốc `Expanding` + `Rolling 756` & `Recency` ở `90/10` | `Residual` (Hybrid/ARIMA) & `Direct` (GRU) | **Full `5 seeds (41–45)`** toàn lưới (`360` lượt GRU/Hybrid) | **Đầy đủ `36` tệp (`6 FS × 6 bối cảnh`)** |

---

## 2. Thống kê Chỉ số Tổng hợp Toàn Nhóm (Benchmarking Leaderboard)

### 2.1. Bảng xếp hạng nhánh Residual trên cấu hình chuẩn `Split 90/10 — Expanding` ($N_{\text{test}} = 271$, `29/10/2025 → 13/11/2026`)

Biểu đồ đối chiếu cân bằng: [fig16_group_benchmark_all_5_members.png](fig16_group_benchmark_all_5_members.png) · Bảng dữ liệu CSV: [group_ablation_6_feature_sets_all_members.csv](group_ablation_6_feature_sets_all_members.csv)

> [!NOTE]
> **Minh bạch Giao thức So sánh (Protocol Transparency):** Bảng 2.1 đối chiếu các cấu hình **trên cùng tập kiểm định `90/10` ($N=271$)** nhưng phân tách rõ 3 giao thức: **(i) Holdout `TB 5 seeds (41–45)`** (`PP2, PP3, PP5`), **(ii) Walk-Forward `TB 5 seeds (41–45)`** tái huấn luyện định kỳ mỗi `S=40` phiên (`PP4`), và **(iii) Holdout `1 seed (42)`** (`PP1` Grid, `PP4 Static`). Không gộp đánh giá 1-seed với trung bình 5-seed khi suy diễn độ ổn định.

| Hạng (Best MAE) | Nhóm PP | Phương pháp / Mô hình | Giao thức | Bộ biến đạt đỉnh (Best FS) | Best FS MAE (\$/lb) | MAE_std (5-seed) | RMSE tại Best FS | DA (%) tại Best FS | % Cải thiện vs. Naive (`0.07708`) | MAE TB qua các FS (\$/lb) |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| **1** | **PP1** | **GRU Residual (Non-SA)** | Holdout 1-seed (42) | `FS04` (5 Kim loại) | **0.07695** | — | **0.10081** | **52.77%** | **+0.16%** | 0.07793 *(4 bộ)* |
| **2** | **PP1** | **LSTM Residual (Non-SA)** | Holdout 1-seed (42) | `FS04` (5 Kim loại) | **0.07698** | — | 0.10095 | **52.77%** | **+0.13%** | **0.07701 *(4 bộ)*** |
| **—** | **Baseline** | **Naive Persistence ($\hat{y}_t = y_{t-1}$)** | Deterministic | `FS01` (Copper) | **0.07708** | **0.00000** | **0.10103** | — | **0.00%** | **0.07708** |
| **3** | **PP1** | **CNN–LSTM Residual (Non-SA)** | Holdout 1-seed (42) | `FS04` (5 Kim loại) | 0.07725 | — | 0.10086 | 46.86% | -0.22% | 0.07810 *(4 bộ)* |
| **4** | **PP4** | **WF Rolling Residual (`W=756, S=40`)** | WF TB 5-seed (41–45) | `FS01` (Copper) | **0.07730** | **0.00013** | 0.10142 | 49.59% | -0.29% | 0.07798 *(6 bộ)* |
| **5** | **PP1** | **Ridge Residual ($\alpha = 1.0$)** | Deterministic (1-run) | `FS01` (Copper) | 0.07731 | — | 0.10119 | 47.60% | -0.30% | 0.07937 *(4 bộ)* |
| **6** | **PP4** | **WF Expanding Residual (`W=756, S=40`)** | WF TB 5-seed (41–45) | `FS01` (Copper) | 0.07732 | **0.00018** | 0.10140 | 49.96% | -0.31% | **0.07784 *(6 bộ)*** |
| **7** | **PP3** | **iTransformer Residual (TB 5 seeds)** | Holdout TB 5-seed (41–45) | `FS06` (WTI + Cu) | 0.07738 | 0.00041 | 0.10132 | 52.47% | -0.39% | 0.08178 *(6 bộ)* |
| **8** | **PP5** | **Hybrid ARIMA–GRU (TB 5 seeds)** | Holdout TB 5-seed (41–45) | `FS01` (Copper) | 0.07745 | 0.00063 | 0.10142 | 48.71% | -0.48% | 0.08048 *(6 bộ)* |
| **9** | **PP2** | **SA–LSTM Residual (TB 5 seeds)** | Holdout TB 5-seed (41–45) | `FS01` (Copper) | 0.07745 | 0.00061 | 0.10116 | 48.86% | -0.49% | 0.08393 *(6 bộ)* |
| **10** | **PP4** | **Static (Train+Val) Residual** | Holdout 1-seed (42) | `FS04` (5 Kim loại) | 0.07746 | — | **0.10070** | 48.33% | -0.50% | 0.07920 *(6 bộ)* |
| **11** | **PP2** | **TPE–LSTM Residual (TB 5 seeds)** | Holdout TB 5-seed (41–45) | `FS01` (Copper) | 0.07801 | 0.00082 | 0.10148 | 47.23% | -1.21% | 0.08167 *(6 bộ)* |

> **Phát hiện trọng tâm Bảng 2.1 (Tác động của kiểm chứng 5-seed đồng bộ):**
> - **Không có phương pháp 5-seed nào vượt Naive ở giá trị trung bình:** Khi toàn bộ `PP2, PP3, PP4, PP5` cùng hoàn thành kiểm chứng 5 seeds (`41–45`) trên 6 bộ biến ở `90/10 Expanding`, **`0 / 4` phương pháp 5-seed có cấu hình nào đạt TB 5-seed tốt hơn Naive (`0.07708`)** — đỉnh `5-seed` cao nhất lần lượt là `PP4 WF-Rolling FS01 = 0.07730`, `PP4 WF-Expanding FS01 = 0.07732`, `PP3 iTransformer FS06 = 0.07738`, `PP5 Hybrid FS01 = 0.07745`, `PP2 SA-LSTM FS01 = 0.07745`.
> - **Hiện tượng lệch hạt giống đơn (`Seed 42 bias`):** Nhiều kết quả đơn lẻ ở Seed 42 từng vượt nhẹ Naive (như `PP3 iTransformer FS01 s42 = 0.07681`, `PP4 WF-Expanding FS03 s42 = 0.07693`, hoặc `PP1 GRU/LSTM FS04 s42 = 0.07695/0.07698`). Tuy nhiên khi lấy trung bình 5 seeds, `PP3 iTransformer FS01` tăng lên `0.07853 ± 0.00268` và `PP4 WF-Expanding FS03` lên `0.07790 ± 0.00116`.
> - **Độ ổn định trên cả 6 bộ biến (`FS01–FS06`):** Trong 4 nhóm chạy đủ 6 bộ biến ở chế độ 5-seed, **`PP4 WF Expanding Residual` (`0.07784`, std giữa các FS `0.00035`)** và **`PP4 WF Rolling Residual` (`0.07798`, std `0.00048`)** ổn định nhất giữa các tập biến. Xét riêng 4 bộ biến có Copper (1 seed), **`PP1 LSTM Residual`** giữ `0.07701 ± 0.00002`.

### 2.2. Đối chiếu nhánh Direct (Dự báo mức giá trực tiếp) đủ cả `5/5 Phương pháp` (`Split 90/10`)

| Nhóm PP | Mô hình (Nhánh Direct) | Giao thức | `FS01` MAE | `FS02` MAE | `FS03` MAE | `FS04` MAE | `FS05` MAE | `FS06` MAE | MAE TB (\$/lb) | So với Naive (`0.07708`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **PP4** | **WF Expanding Direct (`W=756, S=40`)** | WF 1-seed (42) | 0.20265 | **0.23696** | **0.38606** | 0.33844 | **0.36182** | 0.21921 | **0.29086** *(6 bộ)* | Kém Naive +277% |
| **PP1** | **GRU Direct (Non-SA)** | Holdout 1-seed (42) | 0.43711 | 0.24789 | — | **0.24082** | — | 0.24321 | **0.29226** *(4 bộ)* | Kém Naive +279% |
| **PP4** | **WF Rolling Direct (`W=756, S=40`)** | WF 1-seed (42) | 0.27394 | 0.28371 | 0.46075 | 0.34973 | 0.41410 | 0.27422 | 0.34274 *(6 bộ)* | Kém Naive +345% |
| **PP5** | **GRU Direct (TB 5 seeds)** | Holdout TB 5-seed | 0.33809 | 0.29023 | 0.49914 | 0.29322 | 0.56456 | 0.30653 | 0.38196 *(6 bộ)* | Kém Naive +396% |
| **PP2** | **TPE–LSTM Direct (TB 5 seeds)** | Holdout TB 5-seed | **0.19139** | 0.46929 | 0.62901 | 0.43094 | 0.49614 | 0.40812 | 0.43748 *(6 bộ)* | Kém Naive +468% |
| **PP2** | **SA–LSTM Direct (TB 5 seeds)** | Holdout TB 5-seed | 0.26750 | 0.38674 | 0.40847 | 0.82644 | 0.48140 | 0.35898 | 0.45492 *(6 bộ)* | Kém Naive +490% |
| **PP1** | **Paper SA–LSTM Direct (V21/V22)** | Holdout 1-seed (42) | — | 0.97807 | 0.54072 | 0.47093 | 0.46499 | **0.18463** | 0.52787 *(5 bộ)* | Kém Naive +585% |
| **PP1** | **LSTM Direct (Non-SA)** | Holdout 1-seed (42) | 0.44672 | 1.16396 | — | 0.24741 | — | 0.32494 | 0.54576 *(4 bộ)* | Kém Naive +608% |
| **PP1** | **Reopt SA–LSTM Direct (V21/V22)** | Holdout 1-seed (42) | — | 0.87742 | 0.59561 | 0.51047 | 0.46660 | 0.35892 | 0.56180 *(5 bộ)* | Kém Naive +629% |
| **PP4** | **Static (Train-Only) Direct** | Holdout 1-seed (42) | 0.49280 | 1.55159 | 0.31990 | 1.32269 | 0.56377 | 0.55846 | 0.80154 *(6 bộ)* | Kém Naive +940% |
| **PP3** | **iTransformer Direct (TB 5 seeds)** | Holdout TB 5-seed | 1.65695 | 1.27131 | 1.40987 | 1.66197 | 1.70735 | 1.07955 | 1.46450 *(6 bộ)* | Kém Naive +1800% |

> **Phát hiện xuyên suốt số 1 (Direct vs. Residual):** **100% mô hình Direct ở đủ cả 5/5 phương pháp** (`PP1–PP5`) đều kém Naive bậc độ lớn (`MAE = 0.185 – 1.707 $/lb` so với `0.07708 $/lb`) vì miền giá Test (`5.0 – 6.89 $/lb`) vượt khỏi biên chuẩn hóa `MinMaxScaler` của Train. Khi cập nhật kết quả 5-seed mới của `PP2` và `PP3`: `PP2 TPE-LSTM Direct` đạt TB 5-seed tốt nhất trên `FS01` (`0.19139`, vượt mốc tĩnh khác) nhưng vẫn dao động lớn trên biến đa ngoại sinh (`0.43–0.63`), còn `PP4 WF Expanding Direct` dẫn đầu trung bình 6 bộ (`0.29086`, giảm `63.7%` sai số so với `Static Direct = 0.80154`).

### 2.3. Thống kê xuyên suốt 4 tỷ lệ chia dữ liệu (`Cross-Ratio 80/20, 90/10, 92.5/7.5, 95/5`)

Bảng dữ liệu chi tiết: [group_cross_ratio_4_splits_all_members.csv](group_cross_ratio_4_splits_all_members.csv)

| Phương pháp đại diện | Nhóm PP | Giao thức | Bộ biến | `80/20` ($N=542$) MAE | `90/10` ($N=271$) MAE | `92.5/7.5` ($N=204$) MAE | `95/5` ($N=136$) MAE | TB 4 mốc (\$/lb) | Số mốc vượt Naive |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| **Naive Persistence ($\hat{y}_t = y_{t-1}$)** | Baseline | Deterministic | `FS01` | **0.06900** | **0.07708** | **0.07512** | **0.07280** | **0.07350** | — |
| **ARIMA Standalone (`auto_arima`)** | PP5 | Deterministic | `FS01` | 0.06900 (0.00%) | 0.07708 (0.00%) | **0.07456 (+0.75%)** | **0.07258 (+0.31%)** | **0.07331** | **2 / 4** |
| **LSTM Residual (Expanding)** | PP1 | Holdout 1-seed (42) | `FS01` | 0.06920 (-0.29%) | **0.07701 (+0.09%)** | 0.07536 (-0.33%) | **0.07273 (+0.09%)** | 0.07358 | **2 / 4** |
| **GRU Residual (Expanding)** | PP1 | Holdout 1-seed (42) | `FS02` | **0.06878 (+0.32%)** | 0.07714 (-0.09%) | 0.07521 (-0.12%) | **0.07277 (+0.04%)** | **0.07348** | **2 / 4** |
| **SA–LSTM Residual (Expanding)** | PP2 | Holdout TB 5-seed | `FS01` | 0.06921 (-0.31%) | 0.07745 (-0.49%) | 0.07520 (-0.12%) | 0.07291 (-0.15%) | 0.07369 | 0 / 4 |
| **SA–LSTM Residual (Expanding)** | PP2 | Holdout TB 5-seed | `FS02` | 0.09176 (-32.99%) | 0.11095 (-43.94%) | 0.08456 (-12.57%) | 0.07858 (-7.94%) | 0.09146 | 0 / 4 |
| **TPE–LSTM Residual (Expanding)** | PP2 | Holdout TB 5-seed | `FS01` | 0.06980 (-1.15%) | 0.07801 (-1.21%) | 0.07567 (-0.73%) | 0.07335 (-0.75%) | 0.07421 | 0 / 4 |
| **TPE–LSTM Residual (Expanding)** | PP2 | Holdout TB 5-seed | `FS02` | 0.08117 (-17.63%) | 0.07880 (-2.23%) | 0.07564 (-0.70%) | 0.07494 (-2.94%) | 0.07764 | 0 / 4 |
| **iTransformer Residual (Expanding)** | PP3 | Holdout TB 5-seed | `FS01` | 0.06982 (-1.18%) | 0.07853 (-1.88%) | 0.07573 (-0.82%) | 0.07372 (-1.26%) | 0.07445 | 0 / 4 |
| **iTransformer Residual (Expanding)** | PP3 | Holdout TB 5-seed | `FS02` | 0.07067 (-2.42%) | 0.09409 (-22.07%) | 0.07777 (-3.53%) | 0.07375 (-1.31%) | 0.07907 | 0 / 4 |
| **iTransformer Residual (Expanding)** | PP3 | Holdout TB 5-seed | `FS06` | 0.07254 (-5.13%) | 0.07738 (-0.39%) | 0.07600 (-1.18%) | 0.07552 (-3.74%) | 0.07536 | 0 / 4 |
| **Hybrid ARIMA–GRU (Expanding)** | PP5 | Holdout TB 5-seed | `FS01` | 0.07275 (-5.43%) | 0.07745 (-0.48%) | **0.07475 (+0.48%)** | 0.07284 (-0.06%) | 0.07445 | 1 / 4 |
| **Static (Train+Val) Residual** | PP4 | Holdout 1-seed (42) | `FS02` | 0.07505 (-8.76%) | 0.07812 (-1.35%) | 0.07684 (-2.30%) | **0.07213 (+0.91%)** | 0.07553 | 1 / 4 |
| **WF Expanding Residual (`W=756, S=40`)** | PP4 | WF 1-seed (42)* | `FS02` | 0.07108 (-3.02%) | 0.07852 (-1.87%) | 0.07714 (-2.70%) | 0.07573 (-4.02%) | 0.07562 | 0 / 4 |

*\*Ghi chú `PP4 FS02`: Bảng Cross-Ratio 4 mốc giữ 1-seed (`42`) để khớp đủ 4 split; khi mở rộng 5 seeds ở `80/20` và `90/10` (`Bảng 2.4`), TB 5-seed `WF Expanding FS02` cải thiện xuống `0.07004` (`80/20`) và `0.07828` (`90/10`).*

> **Nhận xét Bảng 2.3:** Ở chế độ 5-seed, chỉ **`PP5 Hybrid ARIMA–GRU` trên `FS01` tại mốc `92.5/7.5`** (`0.07475`, nhờ thành phần `auto_arima(1,1,0)` đạt `0.07456`) vượt nhẹ được Naive ở một mốc đơn lẻ; tất cả cấu hình 5-seed của `PP2` và `PP3` đều đạt `0/4` mốc vượt Naive trên TB 5 seeds. Trên `FS01`, `PP2 SA–LSTM (TB 5 seeds = 0.07369)` và `PP5 Hybrid (0.07445)` / `PP3 iTransformer (0.07445)` ít lệch Naive hơn hẳn so với khi ghép `FS02` (nơi `PP2 SA–LSTM` nhảy lên `0.09146` và `PP3 iTransformer` lên `0.07907` do nhạy cảm seed).

### 2.4. Đối chuẩn Ổn định Đa Hạt giống (`5 Seeds: 41–45`) & Ensemble 5-Seed trên `90/10` và `80/20`

Bảng dữ liệu chi tiết: [group_multiseed_5seeds_summary.csv](group_multiseed_5seeds_summary.csv)

#### Bảng 2.4a — Tóm tắt 5-seed (`41–45`) & 5-seed Ensemble tại `Split 90/10 Expanding` ($N=271$, Naive = `0.07708`)

| Bộ biến | `PP2 SA-LSTM` (Mean±Std \| Ens) | `PP2 TPE-LSTM` (Mean±Std \| Ens) | `PP3 iTrans` (Mean±Std \| Ens) | `PP4 WF-Exp` (Mean±Std \| Ens) | `PP4 WF-Roll` (Mean±Std \| Ens) | `PP5 Hybrid` (Mean±Std \| Ens) |
|---|---|---|---|---|---|---|
| **`FS01_CU`** | `0.07745±.00061` \| `0.07729` *(p=.248)* | `0.07801±.00082` \| `0.07785` *(p=.077)* | `0.07853±.00268` \| — | `0.07732±.00018` \| `0.07727` *(p=.469)* | **`0.07730±.00013`** \| `0.07719` *(p=.734)* | `0.07745±.00063` \| `0.07733` *(p=.033↓)* |
| **`FS02_PAPER_CU`** | `0.11095±.03938` \| `0.08085` *(p=.001↓)* | `0.07880±.00196` \| **`0.07717`** *(p=.814)* | `0.09409±.03694` \| — | **`0.07828±.00064`** \| `0.07793` *(p=.164)* | `0.07864±.00122` \| `0.07818` *(p=.096)* | `0.08110±.00305` \| `0.07806` *(p=.002↓)* |
| **`FS03_PAPER_EXO`** | `0.07890±.00247` \| `0.07707` *(p=.986)* | `0.08547±.00759` \| `0.07900` *(p=.051)* | `0.08388±.00608` \| — | `0.07790±.00116` \| `0.07717` *(p=.887)* | **`0.07763±.00055`** \| **`0.07702`** *(p=.928)* | `0.08334±.00660` \| `0.07838` *(p=.248)* |
| **`FS04_METALS_CU`** | `0.07960±.00410` \| `0.07786` *(p=.420)* | `0.08326±.01223` \| `0.07826` *(p=.313)* | **`0.07797±.00137`** \| — | `0.07816±.00092` \| **`0.07755`** *(p=.528)* | `0.07833±.00078` \| `0.07783` *(p=.180)* | `0.08185±.00592` \| `0.07922` *(p=.045↓)* |
| **`FS05_METALS_EXO`** | `0.07797±.00078` \| **`0.07691`** *(p=.322)* | `0.08556±.00711` \| `0.07761` *(p=.168)* | `0.07883±.00212` \| — | **`0.07762±.00035`** \| `0.07735` *(p=.681)* | `0.07798±.00046` \| `0.07741` *(p=.547)* | `0.08071±.00468` \| `0.07779` *(p=.460)* |
| **`FS06_WTI_CU`** | `0.07873±.00210` \| `0.07720` *(p=.708)* | `0.07891±.00256` \| `0.07806` *(p=.019↓)* | **`0.07738±.00041`** \| **`0.07684`** *(p=.468)* | `0.07778±.00038` \| `0.07735` *(p=.450)* | `0.07802±.00091` \| `0.07756` *(p=.374)* | `0.07844±.00151` \| `0.07753` *(p=.090)* |

#### Bảng 2.4b — Tóm tắt 5-seed (`41–45`) & 5-seed Ensemble tại `Split 80/20` ($N=542$, Naive = `0.06900`) đủ `5/5 Phương pháp`

- **`PP1` Walk-Forward 5-seed tại `80/20 FS06` (`V23/V25`):** `WF GRU-Res` đạt `0.06910 ± 0.00015` (Ens = **`0.06907`**, `p = 0.3879`, `0/5` thua DM), `WF LSTM-Res` đạt `0.06918 ± 0.00017` (Ens = `0.06913`, `p = 0.2161`), `WF CNN-LSTM-Res` đạt `0.06932 ± 0.00037` (Ens = `0.06925`, `p = 0.1211`), `WF CNN-GRU-Res` đạt `0.06945 ± 0.00051` (Ens = `0.06922`, `p = 0.0620`).
- **Tập đa phương pháp ở `80/20 Expanding` (`FS01–FS06`):**
  - `PP2`: `SA-LSTM FS01 = 0.06921 ± 0.00047` (Ens `0.06910, p=0.445`) và `TPE FS01 = 0.06980 ± 0.00077` (`p=0.038↓` ở Ens); trên các bộ đa biến `FS02–FS04`, Ensemble của TPE thua Naive có ý nghĩa (`p < 0.0001`, lên tới `0.08145` ở `FS04`).
  - `PP3`: `FS05 = 0.06912 ± 0.00025`, `FS04 = 0.06950 ± 0.00018`, `FS01 = 0.06982 ± 0.00121`, `FS06 = 0.07254 ± 0.00760` (Ens = `0.06921, p=0.5487`, `0/5` thắng, `1/5` thua DM).
  - `PP4`: Toàn bộ `6/6` Ensemble **Expanding** đều **hòa Naive** (`MAE ∈ [0.06926, 0.06972]`, `p ∈ [0.1025, 0.5218]`), trong khi `5/6` Ensemble **Rolling** kém Naive có ý nghĩa (`p ≤ 0.0363`, chỉ `FS03` hòa `p = 0.1281`).
  - `PP5`: `Hybrid` có `5/6` bộ biến (`FS01, FS02, FS03, FS04, FS06`) có 5-seed Ensemble kém Naive có ý nghĩa (`MAE = 0.06981–0.07413, p ≤ 0.0284`), chỉ `FS05` hòa Naive (`0.06885, p = 0.3958`).
- **Tổng kết Ensemble 5-seed toàn nhóm (`66` bộ Ensemble có dự báo ngày ở `90/10` + `80/20`):** **`0 / 66 (0.0%)` Ensemble 5-seed vượt được Naive có ý nghĩa thống kê dưới $L_1$** (`(↓)` trong bảng chỉ các trường hợp Ensemble kém Naive ý nghĩa ở mức 5%).

---

## 3. Kiểm định Ý nghĩa Thống kê Toàn Nhóm (Diebold–Mariano Significance Tests)

Biểu đồ kiểm định đối xứng toàn nhóm: [fig18_group_dm_significance_and_pros_cons.png](fig18_group_dm_significance_and_pros_cons.png) · Bảng CSV đầy đủ (`551` dòng, `0` giá trị thiếu): [group_significance_dm_tests_complete.csv](group_significance_dm_tests_complete.csv)

Toàn bộ các phép kiểm định được chuẩn hóa theo **Diebold–Mariano (1995)** với ước lượng phương sai vững **Newey–West HAC** ($\text{lag} = 5$) và hiệu chỉnh mẫu nhỏ **Harvey–Leybourne–Newbold (HLN)** trên cả hàm mất mát $L_1$ ($|e_t|$) và $L_2$ ($e_t^2$).

### 3.1. Kiểm định giả thuyết $H_0^{(1)}$: *"Có mô hình Residual nào vượt Naive Persistence có ý nghĩa thống kê không?"*

Bảng kiểm kê đầy đủ trên **toàn bộ `992` lượt chạy Residual** có dữ liệu kiểm định từng phiên từ **cả `5/5` nhóm phương pháp (`PP1–PP5`)**:

| Nhóm PP | Tập lượt chạy kiểm định với Naive Persistence | Tổng lượt (Expanding chuẩn) | Thắng Naive ý nghĩa 5% ($L_1$) | Thắng Naive ý nghĩa 5% ($L_2$) | Kém Naive ý nghĩa 5% ($L_1$) | Thống kê $\text{DM}_{L1}$ & $p$-value tiêu biểu |
|---|---|---:|---:|---:|---:|---|
| **PP1 (Grid Non-SA)** | `5` mô hình (`lstm, gru, cnn_lstm, cnn_gru, ridge`) × `4` FS × `4` splits × `3` schemes | `240` (`80` Exp) | **0 / 240** (`0 / 80` Exp) | **2 / 240** *(chỉ `gru_res`)* | `102 / 240` (`18 / 80` Exp; `lstm_res` Exp: **`0/16`**) | `lstm_res FS04`: $\text{DM}=-0.28, p=0.780$; `gru_res FS04`: $\text{DM}=-0.49, p=0.622$ |
| **PP1 (Walk-Forward)** | `V20` (`5` seeds `FS02`), `V23` (`5` seeds `gru, lstm` `FS06`), `V25` (`cnn_gru, cnn_lstm, ridge` `FS06`) | `26` (`26` Exp WF) | **0 / 26** | — | `5 / 26` | V20 `gru_res` s41 ($N=2214$): $p=0.704$; V23 `gru_res` s41: $p=0.493$; V25 `cnn_lstm_res` s44: $p=0.727$ |
| **PP2 (`SA–LSTM` 5s)** | `6` FS × `6` bối cảnh (`4` splits Exp + `Rec/Roll` 90/10) × `5` seeds (`41–45`) | `180` (`120` Exp) | **2 / 180** (`2/120` Exp: chỉ `FS01 s44` 80/20 & 90/10) | **2 / 180** | `74 / 180` (`42 / 120` Exp) | 5s-Ens 90/10 `FS05`: $\text{DM}=-0.99, p=0.322$; `FS01`: $\text{DM}=+1.16, p=0.248$ |
| **PP2 (`TPE–LSTM` 5s)** | `6` FS × `6` bối cảnh × `5` seeds (`41–45`) | `180` (`120` Exp) | **2 / 180** (`2/120` Exp: chỉ `FS01 s44` 80/20 & 90/10) | **1 / 180** | `91 / 180` (`45 / 120` Exp) | 5s-Ens 90/10 `FS02`: $\text{DM}=+0.24, p=0.814$; `FS01`: $\text{DM}=+1.77, p=0.077$ |
| **PP3 (`iTransformer FS06`)** | `FS06_WTI_CU` × `4` splits × `3` schemes × `5` seeds (`41–45`) | `60` (`20` Exp) | **0 / 60** (`0 / 20` Exp) | **0 / 60** | `17 / 60` (`3 / 20` Exp) | 5s-Ens 90/10 `FS06`: $\text{DM}=-0.73, p=0.468$; 80/20 `FS06`: $\text{DM}=+0.60, p=0.549$ |
| **PP4 (`Walk-Forward + Static`)** | Phase 2D (`6` FS × `2` ratios `90/10 & 80/20` × Exp/Roll × `5` seeds = `120`) + `Static T+V` (`6`) | `126` (`60` WF Exp) | **0 / 126** (`0 / 60` Exp) | **0 / 126** | `44 / 126` (`90/10`: `5/60`, `80/20`: `37/60`, Static: `2/6`) | 5s-Ens 90/10 `Roll FS03`: $\text{DM}=-0.09, p=0.928$; `Exp FS01`: $\text{DM}=+0.72, p=0.469$ |
| **PP5 (`Hybrid ARIMA–GRU`)** | `Expanding` (`4` splits × `6` FS × `5` seeds = `120`) + `Rec/Roll` 90/10 (`60`) | `180` (`120` Exp) | **1 / 180** (`0/120` Exp; chỉ `Rec FS01 s42`, $p=0.0385$) | **0 / 180** | `109 / 180` (`57 / 120` Exp) | 5s-Ens 90/10 `FS01`: $\text{DM}=+2.14, p=0.033$; `FS05`: $\text{DM}=+0.74, p=0.460$ |

> **Kết luận Kiểm định $H_0^{(1)}$ (Đồng thuận tuyệt đối `5/5` phương pháp):**
> - Trên **toàn bộ `992` lượt chạy Residual** (`PP1–PP5`), chỉ **`5 / 992` (`0.50%`)** lượt đạt `p < 0.05` thắng Naive ở $L_1$; trên **`546` lượt Expanding chuẩn**, tỷ lệ thắng là **`4 / 546` (`0.73%`)** (thực chất chỉ là `Seed 44` trên `FS01` của `PP2`). Cả hai tỷ lệ đều **thấp hơn rõ rệt ngưỡng dương tính giả `2.5%` của kiểm định đuôi trái ở $\alpha = 5\%$** dưới giả thuyết $H_0$ đúng.
> - Mở rộng ra **`66` mô hình Ensemble 5-seed**, tỷ lệ thắng có ý nghĩa thống kê trước Naive là **`0 / 66` (`0.0%`)**. Chuỗi sai phân giá đồng ngày hành xử như quá trình **Martingale Difference Sequence (MDS)** ($|r_{\text{lag-1}}| \le 0.091$), khiến mọi mô hình học dưới MSE hội tụ co phương sai về $\hat{y}_t \approx y_{t-1}$.

### 3.2. Kiểm định đối đầu nội bộ `PP2: TPE–LSTM vs. SA–LSTM` (180 cặp theo Seed & 36 cặp 5-Seed Ensemble)

- **Trên `180` cặp Residual theo từng seed (`6 FS × 6 bối cảnh × 5 seeds`):** `88/180` (`48.9%`) hòa thống kê; **SA–LSTM thắng ý nghĩa `54/180` (`30.0%`)**; **TPE–LSTM thắng ý nghĩa `38/180` (`21.1%`)**.
- **Trên `36` cặp 5-seed Ensemble (`6 FS × 6 bối cảnh`):** `21/36` (`58.3%`) hòa thống kê; **SA–LSTM thắng `10/36` (`27.8%`)**; **TPE–LSTM thắng `5/36` (`13.9%`)**.

| Bộ biến (`Split 90/10 Exp`, $N=271$) | `SA–LSTM` 5s Mean (`Ens`) | `TPE–LSTM` 5s Mean (`Ens`) | $\text{DM}_{L1}$ (`TPE vs SA`, 5s-Ens) | $p_{L1}$ | $\text{DM}_{L2}$ ($p_{L2}$) | Kết luận ở mức ý nghĩa $\alpha = 5\%$ |
|---|---:|---:|---:|---:|---:|---|
| **`FS01` (Copper)** | **0.07745** (`0.07729`) | 0.07801 (`0.07785`) | **`+2.148`** | **`0.0326`** | `+1.348` (`0.1787`) | **SA–LSTM thắng ý nghĩa $L_1$** |
| **`FS02` (Paper3 + Cu)** | *0.11095* (`0.08085`) | **0.07880 (`0.07717`)** | **`-3.279`** | **`0.0012`** | **`-2.741` (`0.0065`)** | **TPE–LSTM thắng ý nghĩa ($p<0.01$)** |
| **`FS03` (Paper3 - Cu)** | **0.07890 (`0.07707`)** | 0.08547 (`0.07900`) | `+1.615` | `0.1075` | `+1.394` (`0.1644`) | Hòa thống kê |
| **`FS04` (5 Kim loại)** | **0.07960 (`0.07786`)** | 0.08326 (`0.07826`) | `+0.903` | `0.3673` | `+1.817` (`0.0704`) | Hòa thống kê |
| **`FS05` (4 Kim loại - Cu)** | **0.07797 (`0.07691`)** | 0.08556 (`0.07761`) | `+1.687` | `0.0928` | `+0.736` (`0.4624`) | Hòa thống kê |
| **`FS06` (WTI + Cu)** | **0.07873 (`0.07720`)** | 0.07891 (`0.07806`) | `+1.821` | `0.0697` | `+1.153` (`0.2499`) | Hòa thống kê ($p<0.10$ nghiêng SA) |

### 3.3. Kiểm định DM chéo đối xứng giữa `5/5 Phương pháp` (`90/10 Expanding`, $N = 271$)

Bảng tổng hợp đối đầu trực tiếp sử dụng **5-seed Ensemble (`41–45`)** cho `PP2, PP3 (FS06), PP4, PP5` và **Holdout 1-seed (`42`)** cho `PP1` (kèm kiểm định nhạy khớp riêng `Seed = 42` trong [group_significance_dm_tests_complete.csv](group_significance_dm_tests_complete.csv)):

| Cặp so sánh (`A vs B`, `90/10 Exp Primary`) | Số FS chung | `A` thắng ý nghĩa | `B` thắng ý nghĩa | Hòa thống kê ($p \ge 0.05$) | $\text{DM}_{L1}$ TB | Chi tiết các bộ biến khác biệt ý nghĩa ($p_{L1} < 0.05$) |
|---|---:|---:|---:|---:|---:|---|
| **`PP1: LSTM-Res (1s)` vs `PP1: GRU-Res (1s)`** | 4 | **1** | 0 | 3 | -1.15 | `PP1 LSTM-Res` thắng `FS01` ($p=0.0071$) |
| **`PP1: LSTM-Res (1s)` vs `PP2: SA-LSTM (5s-Ens)`** | 4 | **1** | 0 | 3 | -1.33 | `PP1 LSTM-Res` thắng `FS02` ($p=0.0067$) |
| **`PP1: LSTM-Res (1s)` vs `PP2: TPE-LSTM (5s-Ens)`** | 4 | **1** | 0 | 3 | -1.37 | `PP1 LSTM-Res` thắng `FS06` ($p=0.0293$) |
| **`PP1: LSTM-Res (1s)` vs `PP3: iTrans (5s-Ens)`** | 1 (`FS06`) | 0 | 0 | **1** | +0.68 | Hòa trên `FS06` ($p=0.4996$) |
| **`PP1: LSTM-Res (1s)` vs `PP4: WF-Exp (5s-Ens)`** | 4 | 0 | 0 | **4** | -0.81 | **Hòa cả `4/4`**, `PP1` có MAE thấp hơn ở `FS01, 02, 04, 06` |
| **`PP1: LSTM-Res (1s)` vs `PP5: Hybrid (5s-Ens)`** | 4 | **2** | 0 | 2 | -2.00 | `PP1 LSTM-Res` thắng `FS04` ($p=0.0055$) & `FS06` ($p=0.0346$) |
| **`PP2: SA-LSTM (5s-Ens)` vs `PP4: WF-Exp (5s-Ens)`** | 6 | 0 | **1** | **5** | +0.30 | `PP4` thắng `FS02` ($p=0.0100$); hòa `5/6` bộ |
| **`PP2: SA-LSTM (5s-Ens)` vs `PP5: Hybrid (5s-Ens)`** | 6 | **1** | **1** | 4 | -0.66 | `PP5` thắng `FS02` ($p=0.0096$); `PP2 SA` thắng `FS04` ($p=0.0016$) |
| **`PP2: TPE-LSTM (5s-Ens)` vs `PP4: WF-Exp (5s-Ens)`** | 6 | 0 | 0 | **6** | +0.64 | **Hòa tuyệt đối `6/6` bộ** |
| **`PP2: TPE-LSTM (5s-Ens)` vs `PP5: Hybrid (5s-Ens)`** | 6 | **1** | 0 | 5 | -0.29 | `PP2 TPE` thắng `FS02` ($p=0.0173$); hòa `5/6` bộ |
| **`PP3: iTrans (5s-Ens)` vs `PP2 SA / PP2 TPE / PP4 Exp / PP5`** | 1 (`FS06`) | 0 | 0 | **1** | `-0.90 / -1.87 / -0.88 / -1.69` | **Hòa thống kê trên `FS06`** (nhưng `PP3` có MAE Ens `0.07684` thấp nhất ở `FS06`) |
| **`PP4: WF-Exp (5s-Ens)` vs `PP5: Hybrid (5s-Ens)`** | 6 | 0 | 0 | **6** | -0.66 | **Hòa tuyệt đối `6/6` bộ** khi cả hai cùng Ensemble 5 seeds (`DM < 0` ở cả 6 bộ) |
| **`PP4: WF-Exp (5s-Ens)` vs `PP4: WF-Roll (5s-Ens)`** | 6 | 0 | 0 | **6** | -0.13 | Hòa `6/6` ở `90/10`; ngược lại ở **`80/20`** `Exp` vượt trội `Roll` (`6/6` hòa Naive vs `5/6` Roll thua Naive) |

- **Kiểm định độ nhạy khi khớp `Seed = 42`:** Ở cấu hình đơn hạt giống (`Seed 42`), `PP5 Hybrid` có phương sai cao hơn nên thua ý nghĩa nhiều hơn (`PP1 LSTM`, `PP2 SA`, `PP2 TPE`, `PP4 WF-Exp` lần lượt thắng `3/4`, `4/6`, `4/6`, `4/6` bộ trước `PP5 s42`); khi lấy **5-seed Ensemble**, đa số các cặp `PP2–PP5` hội tụ về **hòa thống kê** (`5/6` đến `6/6` bộ biến).
- **Kiểm định DM bổ sung của `PP3` trên benchmark `K=5`:** `iTransformer Residual (0.07812)` thắng có ý nghĩa thống kê cả `SA–LSTM Fixed (0.08293)` lẫn `SA–LSTM Tuned (0.08535)` trên cấu hình `90/10 Expanding`.

---

## 4. Nghiên cứu Cắt lớp Toàn Nhóm (Group-Wide Ablation Studies)

Biểu đồ Heatmap & đối chiếu Direct/Residual 6 bộ biến: [fig17_group_ablation_6_feature_sets.png](fig17_group_ablation_6_feature_sets.png)

### 4.1. Ablation 1 — Vai trò 6 Bộ biến (`FS01 → FS06`) trên cả `5/5 Phương pháp` (`Split 90/10 — Expanding`)

*(Các nhóm `PP2, PP3, PP4 (WF), PP5` trình bày **trung bình 5 seeds `41–45`**; `PP1` & `PP4 Direct` trình bày 1 seed `42`.)*

| Mã bộ biến | Các biến trong tensor $X_{t-22:t-1}$ | $K$ | `PP1` (`LSTM-Res 1s`) | `PP1` (`Ridge-Res 1s`) | `PP2` (`SA-LSTM 5s`) | `PP2` (`TPE-LSTM 5s`) | `PP3` (`iTrans-Res 5s`) | `PP4` (`WF-Exp-Res 5s`) | `PP4` (`WF-Roll-Res 5s`) | `PP5` (`Hybrid-Res 5s`) | `PP4` (`WF-Exp-Dir 1s`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`FS01_CU`** | `Copper` | 1 | 0.07701 | **0.07731** | **0.07745** | **0.07801** | 0.07853 | **0.07732** | **0.07730** | **0.07745** | **0.20265** |
| **`FS02_PAPER_CU`** | `Copper, WTI, Gold, Silver` | 4 | 0.07701 | 0.08071 | *0.11095* | 0.07880 | *0.09409* | 0.07828 | 0.07864 | 0.08110 | 0.23696 |
| **`FS03_PAPER_EXO`** | `WTI, Gold, Silver` *(Bỏ Cu)* | 3 | — | — | 0.07890 | *0.08547* | 0.08388 | 0.07790 | 0.07763 | *0.08334* | *0.38606* |
| **`FS04_METALS_CU`** | `Copper, Al, Tin, Silver, Gold` | 5 | **0.07698** | *0.08171* | 0.07960 | 0.08326 | 0.07797 | 0.07816 | 0.07833 | 0.08185 | 0.33844 |
| **`FS05_METALS_EXO`** | `Al, Tin, Silver, Gold` *(Bỏ Cu)* | 4 | — | — | 0.07797 | *0.08556* | 0.07883 | 0.07762 | 0.07798 | 0.08071 | *0.36182* |
| **`FS06_WTI_CU`** | `Copper, Al, Tin, WTI` | 4 | 0.07704 | 0.07774 | 0.07873 | 0.07891 | **0.07738** | 0.07778 | 0.07802 | 0.07844 | 0.21921 |
| **Trung bình** | — | — | **0.07701** | 0.07937 | 0.08393 | 0.08167 | 0.08178 | **0.07784** | 0.07798 | 0.08048 | 0.29086 |

**3 phát hiện chung toàn nhóm từ Ablation 6 bộ biến (đã kiểm chứng 5-seed):**
1. **Nhiễu chiều & bất ổn đa hạt giống ở biến ngoại sinh:** Tương quan trễ bậc 1 ngày trên Train của toàn bộ biến ngoại sinh vs Copper đều rất thấp ($|r| \le 0.064$ Pearson, $\le 0.091$ Spearman). Trên trung bình 5 seeds, bộ đơn biến **`FS01`** dẫn đầu trên `PP1 Ridge (0.07731)`, `PP2 SA (0.07745)`, `PP2 TPE (0.07801)`, `PP4 WF-Exp (0.07732)`, `PP4 WF-Roll (0.07730)` và `PP5 Hybrid (0.07745)`. Ngược lại, đưa bộ biến bài báo `FS02` vào làm bùng phương sai ở các mô hình có dung lượng HPO/Attention lớn (`PP2 SA-LSTM FS02 = 0.11095 ± 0.03938`; `PP3 iTransformer FS02 = 0.09409 ± 0.03694`), chứng tỏ quá khớp cục bộ theo seed trên tập Validate đa biến.
2. **Tác động khi loại bỏ `Copper` (`FS03`, `FS05`):** Trên nhánh Direct, bỏ Copper làm sai số tăng `1.5–2.0` lần. Trên nhánh Residual, `PP2 TPE-LSTM` (`0.08547 / 0.08556`), `PP3 iTransformer` (`FS03 = 0.08388`) và `PP5 Hybrid` (`FS03 = 0.08334`) giảm rõ rệt so với `FS01`; riêng `PP4 Walk-Forward` giữ được MAE thấp và ổn định trên cả `FS03/FS05` (`0.07762–0.07790`).
3. **Bộ đa biến ổn định nhất (`FS06`):** Trong các tổ hợp đa biến, **`FS06 (Copper, Al, Tin, WTI)`** là bộ biến đa chiều ít gây quá khớp nhất trên hầu hết phương pháp (`PP3` đạt đỉnh 5-seed `0.07738 ± 0.00041`, `PP1 LSTM 0.07704`, `PP4 WF-Exp 0.07778`, `PP5 Hybrid 0.07844`, `PP2 SA/TPE 0.07873/0.07891`).

### 4.2. Ablation 2 — Chiến lược Cửa sổ (`Expanding` vs. `Rolling 756` vs. `Recency`) & Tham số Walk-Forward

1. **So sánh 3 chiến lược cửa sổ qua 5 phương pháp:** **`Expanding Window`** vượt trội rõ rệt so với **`Rolling 756`** khi kiểm chứng 5-seed trên cả `80/20` và `90/10`:
   - Trên **`PP4` (`80/20`, 5 seeds)**: toàn bộ **`6/6` Ensemble Expanding hòa Naive** (`p ∈ [0.1025, 0.5218]`), trong khi **`5/6` Ensemble Rolling thua Naive có ý nghĩa** (`p ≤ 0.0363`, `MAE` tăng lên `0.06946–0.07098`).
   - Trên **`PP5` (`90/10`)**: mức phạt trung bình 6 FS của `Expanding` là `+4.42%`, `Recency` là `+6.18%`, còn `Rolling 756` vọt lên **`+32.45%`** (`FS03 Rolling 756 = +101.22%`).
   - Trên **`PP3 FS06` (5 seeds)**: số lượt thua Naive ý nghĩa ở `Expanding` chỉ là `3/20`, nhưng tập trung ở `Rolling_756` (chiếm phần lớn trong `17/60` lượt thua của `PP3 FS06`). Riêng `Recency Weighted` (`FS01`) giúp `PP3` giữ độ phân tán 5-seed thấp (`0.07707 ± 0.00045` ở `90/10`).
2. **Ablation tham số Walk-Forward (`W`, `S`, `Cold/Warm`) của `PP4`:** Trên `Rolling Direct`, $W=756$ (`0.28371`) tốt hơn $W=252$ (`0.29233`) và $W=504$ (`0.29514`); giảm Stride $S$ từ `60 → 10` hạ `Expanding Direct` từ `0.32841 → 0.22756` nhưng thời gian chạy tăng `5.7×` (`64.4s → 364.2s`, chọn $S^*=40$); `Cold-Start` (`0.28371`) tránh tích lũy trôi trọng số tốt hơn `Warm-Start` (`0.37394`).

---

## 5. Đánh giá Ưu / Nhược điểm & Tính Có Ý Nghĩa của `5/5 Phương Pháp` (`PP1–PP5`)

| Mã & Phương pháp | Ưu điểm nổi bật (Strengths) | Nhược điểm & Giới hạn (Weaknesses) | Kết luận thống kê cốt lõi |
|---|---|---|---|
| **PP1 — Non-SA Baselines (`LSTM, GRU, Ridge`) & SA-LSTM** | - Khung kiểm chứng chặt chẽ nhất về mặt kinh tế lượng: `240` lượt Grid + **Walk-Forward 5 seeds (`V20/V23/V25`)** trên nhiều họ mô hình + chuỗi kiểm định chuẩn đoán (ADF, KPSS, Zivot–Andrews, Brown–Forsythe) và chứng minh *Variance Shrinkage*.<br>- `LSTM Residual` giữ sai số TB thấp và ổn định nhất trên 4 bộ có Copper (`0.07701 ± 0.00002`, `0/16` Exp thua Naive), đồng thời `WF GRU/LSTM FS06` ở `80/20` bám sát Naive (`0.06910 / 0.06918`). | - Bộ Holdout Grid chỉ chạy **`1 seed (42)`** và chưa chạy `FS03, FS05` (Non-SA); các mô hình WF `V23/V25` 5-seed chỉ phủ chủ yếu `80/20 FS06` (và `V20 FS02`).<br>- `Ridge Residual` nhạy cảm nhiễu đa biến (`40/48` lượt thua Naive ý nghĩa). | `0/240` (Grid $L_1$) & `0/26` (WF $L_1$) thắng Naive; cung cấp chuẩn baseline đa họ mô hình hoàn chỉnh nhất cho bài toán Random Walk. |
| **PP2 — Tối ưu Siêu tham số SA–LSTM vs. TPE–LSTM (Optuna)** | - Hoàn thiện lưới **full `5 seeds × 6 FS × Direct & Residual`** (`1.920` trials HPO + `720` final fits + `203.400` dòng dự báo), so sánh công bằng cùng ngân sách HPO (`16 trials`).<br>- Ở chế độ đơn biến `FS01`, cả 5-seed Mean (`0.07745`) lẫn Direct TPE (`0.19139`) đều kiểm soát sai số tốt; `TPE–LSTM` thắng `SA–LSTM` có ý nghĩa ($p=0.0012$) ở `FS02`. | - Dễ bị **Validation overfitting & bất ổn theo seed** trên các bộ đa biến có nhiễu (`PP2 SA FS02 = 0.11095 ± 0.03938`; `PP2 TPE 80/20 FS03/FS04` lên tới `0.09862 / 0.08588`, Ensemble thua Naive $p < 0.0001$).<br>- Chỉ đạt `2/180` lượt thắng đơn lẻ trên cả hai nhánh (`Seed 44 FS01`). | `88/180` (`58.3%` ở Ens) hòa nhau giữa SA và TPE; HPO Validation không tạo được lợi thế ngoại mẫu bền vững trước Naive (`4/360` lượt thắng đơn lẻ). |
| **PP3 — iTransformer (Inverted Attention)** | - Kiến trúc Transformer đảo trục biến thành token (`Dense 22→48`) đạt đỉnh 5-seed ở tổ hợp năng lượng-kim loại **`FS06_WTI_CU` (`0.07738 ± 0.00041`, Ens = `0.07684`)** và thắng `SA-LSTM Fixed/Tuned` ($p < 0.025$) ở benchmark `K=5`.<br>- Hoàn thiện full `5 seeds × 6 FS × 4 splits × 3 schemes × Direct/Residual` (`720` lượt) + gói `FS06` đầy đủ DM-HAC & raw predictions. | - Chưa xuất tệp dự báo từng phiên cho `FS01–FS05`, nên chưa tính được 5-seed Ensemble và ma trận DM chéo cho 5 bộ này.<br>- Khi chạy 5-seed, `FS01` từng dẫn đầu ở seed 42 (`0.07681`) tăng lên `0.07853 ± 0.00268` và `FS02` vọt `0.09409 ± 0.03694`; nhánh Direct kém nhất nhóm (`1.08–1.71`). | Đạt `0/60` ($L_1$) thắng Naive và `17/60` thua ý nghĩa ở `FS06`; `68/300` (`FS01–05`) & `16/60` (`FS06`) vượt điểm raw Naive nhưng chưa ổn định thống kê. |
| **PP4 — Rolling & Expanding Window Walk-Forward LSTM** | - Xử lý hiệu quả **Regime Shift** trên nhánh Direct: hạ `63.7%` MAE TB (`0.80154 → 0.29086`, $\text{DM}=-51.36, p<0.0001$).<br>- Hoàn tất full **6/6 FS × Exp/Roll × 5 seeds** ở cả `90/10` & `80/20` (`120` lượt Phase 2D): đạt **MAE TB qua 6 bộ biến thấp nhất trong 4 nhóm 5-seed (`0.07784 ± 0.00035`)** và chứng minh thực nghiệm **`Expanding` thắng `Rolling` ở `80/20`** (`6/6` Exp Ens hòa Naive vs `5/6` Roll Ens thua Naive). | - Tốn tài nguyên huấn luyện gấp nhiều lần mô hình tĩnh (`~110–145s`/run GPU `80/20`, 7–14 lần retrain).<br>- Trên nhánh `1-seed FS02` Cross-Ratio đứng cuối bảng (`0.07562`, `0/4` mốc vượt Naive); trên 5-seed `FS01/02/06` vẫn không vượt được Naive ở điểm trung bình hay Ensemble. | `0/126` lượt Residual thắng Naive (`0/12` Ensemble ở `90/10` và `80/20` thắng Naive); hòa DM chéo tuyệt đối `6/6` bộ với `PP2 TPE` và `PP5` 5-seed Ensemble ở `90/10`. |
| **PP5 — Hybrid ARIMA–GRU (`FS01–FS06`, 5 seeds)** | - Đồng bộ đầy đủ `360` lượt GRU 2 tầng chuẩn nhóm (`6 FS × 5 seeds`) + xuất đủ 36 file dự báo chi tiết.<br>- Thành phần tuyến tính **`auto_arima`** đạt **TB Cross-Ratio 4 mốc thấp nhất toàn nhóm (`0.07331`)** (`92.5/7.5: 0.07456`, `95/5: 0.07258`).<br>- Ở chế độ 5-seed Ensemble trên `FS02`, thắng `PP2 SA` ($p=0.0096$) và hòa `6/6` với `PP4`. | - Xếp tầng GRU phi tuyến lên phần dư ARIMA làm **tăng sai số so với cả ARIMA thuần & Naive** trên cả 6/6 bộ ở `90/10 Expanding` (`FS01 = 0.07745`, TB `0.08048`) và bùng sai số nặng ở `Rolling 756` (`+32.45%` TB).<br>- `auto_arima` chọn `(0,1,0)` trùng Naive ở `80/20` & `90/10`. | `1/180` lượt (`0/120` Exp) thắng Naive $L_1$; `57/120` Exp và `4/6` Ensemble ở `80/20` thua Naive ý nghĩa. Khẳng định ghép phần dư ARIMA không vượt được bước ngẫu nhiên. |

---

## 6. Danh mục Tệp Tổng hợp Chung (`group_benchmark_summary/`)

- Báo cáo tổng hợp: [THONG_KE_CHISO_SIGNIFICANCE_ABLATION.md](THONG_KE_CHISO_SIGNIFICANCE_ABLATION.md)
- Bảng Ablation 6 bộ biến toàn nhóm: [group_ablation_6_feature_sets_all_members.csv](group_ablation_6_feature_sets_all_members.csv) (`23` dòng)
- Bảng Cross-Ratio 4 mốc chia toàn nhóm: [group_cross_ratio_4_splits_all_members.csv](group_cross_ratio_4_splits_all_members.csv) (`64` dòng)
- Bảng tổng hợp 5-seed & Ensemble `90/10` + `80/20`: [group_multiseed_5seeds_summary.csv](group_multiseed_5seeds_summary.csv) (`76` dòng)
- Bảng kiểm định Diebold–Mariano toàn nhóm: [group_significance_dm_tests_complete.csv](group_significance_dm_tests_complete.csv) (`551` dòng, `0` null)
- 3 biểu đồ tổng hợp cân bằng toàn nhóm:
  - [fig16_group_benchmark_all_5_members.png](fig16_group_benchmark_all_5_members.png)
  - [fig17_group_ablation_6_feature_sets.png](fig17_group_ablation_6_feature_sets.png)
  - [fig18_group_dm_significance_and_pros_cons.png](fig18_group_dm_significance_and_pros_cons.png)
