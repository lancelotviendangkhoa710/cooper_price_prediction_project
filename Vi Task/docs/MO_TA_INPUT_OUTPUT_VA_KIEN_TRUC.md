# Mô tả Input, Output và kiến trúc iTransformer

## 1. Bài toán dự báo

Mục tiêu là dự báo giá Copper ở phiên giao dịch kế tiếp. Mỗi mẫu dùng 22 phiên ngay trước phiên cần dự báo:

- X có kích thước (số_mẫu, 22, K).
- 22 là số phiên quá khứ (lookback); K là số biến đầu vào.
- y là một giá Copper tại thời điểm kế tiếp.

Chuỗi được sắp theo thời gian. Tại thời điểm t, code dùng t−22 đến t−1, không đưa Copper(t) vào cửa sổ. Scaler fit trên train trước khi transform validation/test.

## 2. Input trong từng notebook

| Notebook | Input X | Shape mỗi mẫu | Target |
|---|---|---:|---|
| FS01_CU | Copper | 22 × 1 | Copper(t) |
| FS02_PAPER_CU | Copper, WTI, Gold, Silver | 22 × 4 | Copper(t) |
| FS03_PAPER_EXO | WTI, Gold, Silver | 22 × 3 | Copper(t) |
| FS04_METALS_CU | Copper, Aluminium, Tin, Silver, Gold | 22 × 5 | Copper(t) |
| FS05_METALS_EXO | Aluminium, Tin, Silver, Gold | 22 × 4 | Copper(t) |
| FS06_WTI_CU | Copper, Aluminium, Tin, WTI | 22 × 4 | Copper(t) |

FS03/FS05 không có Copper trong X; Residual vẫn dùng Copper(t−1) để cộng phần thay đổi dự báo vào mức giá.

## 3. Inverted Embedding

Input (batch, 22, K) được hoán đổi thành (batch, K, 22). Mỗi biến thành một token chứa lịch sử 22 phiên; Dense dùng chung chiếu vector 22 chiều thành 48 chiều. Attention trộn thông tin giữa các biến trong cửa sổ lịch sử.

Với K=1 như FS01, attention chỉ có một token nên không học được quan hệ liên biến.

## 4. Hai khối Encoder

Mỗi block dùng Multi-Head Attention 4 head, dropout 0,1, residual connection và LayerNormalization; sau đó Feed-Forward Dense(96, GELU) → Dropout(0,1) → Dense(48), thêm residual và LayerNormalization. Output giữ dạng (batch, K, 48).

## 5. Prediction head

Sau encoder thứ hai: Flatten(K × 48) → Dense(32, GELU) → Dropout(0,1) → Dense(1, linear).

| Feature set | K | Encoder output | Flatten | Tổng tham số |
|---|---:|---:|---:|---:|
| FS01_CU | 1 | 1 × 48 | 48 | 40.625 |
| FS02_PAPER_CU | 4 | 4 × 48 | 192 | 45.233 |
| FS03_PAPER_EXO | 3 | 3 × 48 | 144 | 43.697 |
| FS04_METALS_CU | 5 | 5 × 48 | 240 | 46.769 |
| FS05_METALS_EXO | 4 | 4 × 48 | 192 | 45.233 |
| FS06_WTI_CU | 4 | 4 × 48 | 192 | 45.233 |

## 6. Direct, Residual và Naive

- **Direct:** mạng dự báo trực tiếp Copper(t) sau chuẩn hóa; output được inverse-transform về thang giá gốc.
- **Residual:** mạng dự báo Δzₜ = zₜ − zₜ₋₁. Khi suy luận, code cộng phần thay đổi dự báo vào giá phiên trước để khôi phục mức giá. Đây là model riêng với Direct.
- **Naive:** không train mạng; dự báo phiên tiếp theo bằng giá Copper gần nhất, ŷₜ = yₜ₋₁.

## 7. Cấu hình train và đánh giá

Mỗi FS dùng seeds 41–45, bốn tỷ lệ chia thời gian (80%, 90%, 92,5%, 95%), ba chiến lược (`expanding`, `rolling_756`, `recency_weighted`) và hai mục tiêu: 120 neural fits, 12 baseline Naive. Cấu hình train: Adam, learning rate 0,0003, clipnorm 1,0, MSE, batch size 64, tối đa 65 epochs, EarlyStopping patience 9; shuffle tắt.

Metric sau inverse-transform gồm MAE, RMSE, MAPE, R², Directional Accuracy và skill so với Naive. MAE/RMSE/MAPE thấp hơn là tốt hơn; skill dương nghĩa là MAE thấp hơn baseline.

Cả sáu FS đã hoàn tất. FS06 dùng fingerprint CSV khác FS01–FS05 nên được phân tích riêng; xem [báo cáo thực nghiệm](BAO_CAO_THUC_NGHIEM_ITRANSFORMER.md). Raw predictions, histories, metrics full-precision và DM của FS06 nằm trong ZIP. Các bảng/hình hiện hành nằm trong `../outputs/tables/five_seed/` và `../outputs/figures/five_seed/`.

## 8. Giới hạn khi diễn giải

- Đánh giá rolling one-step-ahead, không phải dự báo nhiều bước mà không nhận dữ liệu mới.
- FS03/FS05 không đưa Copper vào X, nhưng Residual cần Copper(t−1) để khôi phục giá.
- Dữ liệu ngoại sinh căn theo ngày trong CSV; thời điểm công bố thực tế chưa được kiểm tra.
- FS06 khác fingerprint nên không so sánh trực tiếp với FS01–FS05.
