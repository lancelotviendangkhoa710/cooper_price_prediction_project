# Mô tả Input, Output và kiến trúc iTransformer

## 1. Bài toán dự báo

Mục tiêu là dự báo một giá Copper tại phiên giao dịch kế tiếp. Mỗi mẫu lấy 22 phiên ngay trước phiên cần dự báo:

- X có kích thước (số_mẫu, 22, K).
- 22 là lookback, tức số phiên quá khứ.
- K là số biến input trong feature set.
- y là một giá Copper tại thời điểm kế tiếp, nên mỗi dự báo có một giá trị đầu ra.

Các chuỗi được sắp theo thời gian. Tại thời điểm t, code dùng dữ liệu từ t−22 đến t−1, không đưa giá Copper(t) vào cửa sổ đầu vào. Scaler được fit trên đoạn train của từng split rồi mới transform validation/test.

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

Input ban đầu có dạng (batch, 22, K): trục thứ hai là thời gian, trục cuối là biến. Code hoán đổi hai trục để nhận (batch, K, 22). Khi đó mỗi biến trở thành một token, và 22 giá trị lịch sử của biến đó là vector token. Dense dùng chung chiếu vector 22 chiều thành 48 chiều. Attention trộn thông tin giữa các biến trong cửa sổ lịch sử.

Khi K=1 như FS01, chỉ có một token nên không có quan hệ liên biến để attention học.

## 4. Hai khối Encoder

Mỗi block dùng Multi-Head Attention 4 head, dropout 0,1, residual connection và LayerNormalization; tiếp theo là Feed-Forward Dense(96, GELU) → Dropout(0,1) → Dense(48), rồi residual và LayerNormalization. Output giữ dạng (batch, K, 48).

## 5. Prediction head

Sau encoder thứ hai, Flatten K token rồi qua Dense(32, GELU) → Dropout(0,1) → Dense(1, linear).

| Feature set | K | Encoder output | Flatten | Tổng tham số |
|---|---:|---:|---:|---:|
| FS01_CU | 1 | 1 × 48 | 48 | 40.625 |
| FS02_PAPER_CU | 4 | 4 × 48 | 192 | 45.233 |
| FS03_PAPER_EXO | 3 | 3 × 48 | 144 | 43.697 |
| FS04_METALS_CU | 5 | 5 × 48 | 240 | 46.769 |
| FS05_METALS_EXO | 4 | 4 × 48 | 192 | 45.233 |
| FS06_WTI_CU | 4 | 4 × 48 | 192 | 45.233 |

## 6. Direct, Residual và Naive

- **Direct:** mạng dự báo trực tiếp giá Copper(t) sau chuẩn hóa; output được inverse-transform về thang giá gốc.
- **Residual:** mạng dự báo thay đổi đã chuẩn hóa Δzₜ = zₜ − zₜ₋₁. Code cộng dự báo thay đổi vào giá phiên trước để khôi phục mức giá. Direct và Residual là hai model train riêng.
- **Naive:** không train mạng; dự báo giá phiên tiếp theo bằng giá gần nhất, ŷₜ = yₜ₋₁.

## 7. Cấu hình train và đánh giá

FS01–FS06 dùng seeds 41–45. Mỗi FS có 4 tỷ lệ chia thời gian (80%, 90%, 92,5%, 95%), 3 chiến lược train (`expanding`, `rolling_756`, `recency_weighted`), 2 mục tiêu và 5 seed: 120 neural fits cùng 12 baseline Naive. Cấu hình train là Adam, learning rate 0,0003, clipnorm 1,0, MSE, batch size 64, tối đa 65 epochs, EarlyStopping patience 9; shuffle tắt.

Các metric được tính sau khi inverse-transform: MAE, RMSE, MAPE, R², Directional Accuracy và skill so với Naive. Với MAE/RMSE/MAPE thấp hơn là tốt hơn; skill dương nghĩa là MAE thấp hơn baseline.

FS01–FS05 đã hoàn tất. FS06 đang dở và dùng fingerprint CSV khác; xem [báo cáo thực nghiệm](BAO_CAO_THUC_NGHIEM_ITRANSFORMER.md) trước khi so sánh kết quả. Bảng MAE, trạng thái và hình được lưu trong `../outputs/tables/five_seed/` và `../outputs/figures/five_seed/`.

## 8. Giới hạn khi diễn giải

- Đánh giá là rolling one-step-ahead: lịch sử thực tế mới nhất có thể vào cửa sổ dự báo kế tiếp; không phải forecast nhiều bước không nhận thêm dữ liệu.
- FS03/FS05 không dùng Copper làm input của mạng, nhưng Residual cần Copper(t−1) để khôi phục mức giá.
- Dữ liệu ngoại sinh được căn theo ngày trong CSV; thời điểm công bố thực tế của từng biến chưa được kiểm tra.
- Các MAE đã trích từ log làm tròn; raw predictions và bảng kiểm định thống kê chưa có trong thư mục kết quả này.
