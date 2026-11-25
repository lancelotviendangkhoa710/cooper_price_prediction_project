# Tổng hợp Pipeline Toàn Nhóm (8 Kiến trúc Chính)

Thư mục này tổng hợp đầy đủ **8 sơ đồ pipeline** cho toàn bộ các mô hình Deep Learning / Hybrid chính của 5 thành viên trong dự án **Copper Futures Price Forecasting** (định dạng PNG độ phân giải cao ~5.5K–6.1K, đã đối chiếu trực tiếp từng notebook, đồng bộ chuẩn in đậm/gạch đầu dòng theo mẫu LSTM và kiểm tra khối `(9) Metrics` khớp 100% từng hàm đánh giá trong code).

| # | Tệp sơ đồ | Mô hình / Phương pháp | Phạm vi bộ biến, mục tiêu & giao thức chính xác theo code | `(9) Metrics` khớp đúng notebook | Nguồn mã nguồn |
|---:|---|---|---|---|---|
| 1 | `01_Pineline-LSTM-SA-LSTM.png` | **LSTM & SA-LSTM** | **Non-SA (`64/32, h=16, lr=0.001`)**: `FS01/02/04/06` · `Direct & Residual` (NB 03). **Paper Fixed (`39/111, h=32, lr=5e-5`) & SA-optimized**: `FS02–FS06` · `Direct` (NB 02). | `MAE, RMSE, MAPE, R²` · `DM-HAC vs. Naive` | `Module1/offical-notebooks/` (`02`, `03`, `04`) |
| 2 | `02_Pineline-GRU.png` | **GRU Baseline** | `GRU(64) → Drop(0.2) → GRU(32) → Drop(0.2) → Dense(16, ReLU) → Dense(1)`, `Adam(0.001)`, `Direct & Residual`, chạy `FS01/02/04/06`. | `MAE, RMSE, MAPE, R²` · `DM-HAC vs. Naive` | `Module1/offical-notebooks/03_*` |
| 3 | `03_Pineline-CNN-LSTM.png` | **Causal 1D CNN–LSTM** | `Conv1D(32, k=3, causal, ReLU) → MaxPool1D(2) → LSTM(48→24)`, `Direct & Residual`, chạy `FS01/02/04/06`. | `MAE, RMSE, MAPE, R²` · `DM-HAC vs. Naive` | `Module1/offical-notebooks/03_*` |
| 4 | `04_Pineline-CNN-GRU.png` | **Causal 1D CNN–GRU** | `Conv1D(32, k=3, causal, ReLU) → MaxPool1D(2) → GRU(48→24)`, `Direct & Residual`, chạy `FS01/02/04/06`. | `MAE, RMSE, MAPE, R²` · `DM-HAC vs. Naive` | `Module1/offical-notebooks/03_*` |
| 5 | `05_Pineline-iTransformer.png` | **iTransformer** | `Inverted Embedding (22→48) + 2× Encoder`, `Direct & Residual`. Tách rõ 2 giao thức: **Full Grid (`FS01–FS06`)**: `5 seeds (41–45), max 65 epochs`; **Fair `K=5`**: `seed 42, max 100 epochs`. | `MAE, RMSE, MSE, MAPE, R², DA` · `DM-HAC vs. Naive` | `Vi Task/` |
| 6 | `06_Pineline-TPE-LSTM.png` | **TPE–LSTM (Optuna)** | TPE tìm `(u₁, u₂, lr)` riêng theo từng `(FS, Direct/Residual)` trên `80/20 Expanding val` (`Dense h=32`), rồi **khóa cấu hình tốt nhất** cho toàn bộ split & strategy còn lại *(notebook đối chứng song song SA-LSTM)*. | `MAE, RMSE, MSE, MASE, R², DA` · `Naive comparison` | `TPE+LSTM/final-tpe-lstm.ipynb` |
| 7 | `07_Pineline-WalkForward-LSTM-Quan.png` | **Walk-Forward LSTM** | `LSTM(39→111) → Dense(32, ReLU) → Drop(0.2) → Dense(1)`, `lr=5e-5`, `W=756, S=40`, `cold_start`. `Val` chỉ chọn `(W, S)`, `EarlyStopping` theo **train loss** (`monitor="loss"`); `5 seeds` chỉ cho `FS02 90/10 Expanding`. | `MAE, RMSE, MSE, MAPE, MASE, R², DA` · `DM vs. Naive` | `quan task/copper_rolling_expanding_quan.ipynb` |
| 8 | `08_Pineline-Hybrid-ARIMA-GRU-Huy.png` | **Hybrid ARIMA–GRU** | `Auto-ARIMA` fit trên Copper trước test → phần dư ARIMA `rₜ = yₜ − ŷₜᴬᴿᴵᴹᴬ` (khác sai phân `Δyₜ`) đi qua `Preprocessing → Input Sequence → GRU(64→32) → Inverse Scaler → cộng ŷₜᴬᴿᴵᴹᴬ`. | `MAE, RMSE, MAPE, R², DA` · `DM test vs. Naive & ARIMA` | `huy task/copper_hybrid_arima_gru_huy.ipynb` |

---

## Xem nhanh các sơ đồ

### 1. LSTM & SA-LSTM
![LSTM & SA-LSTM](01_Pineline-LSTM-SA-LSTM.png)

### 2. GRU Baseline
![GRU Baseline](02_Pineline-GRU.png)

### 3. Causal 1D CNN–LSTM
![CNN-LSTM](03_Pineline-CNN-LSTM.png)

### 4. Causal 1D CNN–GRU
![CNN-GRU](04_Pineline-CNN-GRU.png)

### 5. iTransformer
![iTransformer](05_Pineline-iTransformer.png)

### 6. TPE–LSTM (Optuna)
![TPE-LSTM](06_Pineline-TPE-LSTM.png)

### 7. Walk-Forward LSTM (Quân)
![Walk-Forward LSTM](07_Pineline-WalkForward-LSTM-Quan.png)

### 8. Hybrid ARIMA–GRU (Huy)
![Hybrid ARIMA-GRU](08_Pineline-Hybrid-ARIMA-GRU-Huy.png)
