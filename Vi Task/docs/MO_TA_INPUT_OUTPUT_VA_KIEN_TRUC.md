# Mô tả Input, Output và kiến trúc iTransformer

## 1. Bài toán dự báo

Mục tiêu là dự báo một giá Copper tại phiên giao dịch kế tiếp. Mỗi mẫu lấy 22 phiên ngay trước phiên cần dự báo:

- X có kích thước (số_mẫu, 22, K).
- 22 là lookback, tức số phiên quá khứ.
- K là số biến input trong feature set.
- y là một giá Copper tại thời điểm kế tiếp, nên mỗi dự báo có một giá trị đầu ra.

Các chuỗi được sắp theo thời gian. Tại thời điểm t, code dùng dữ liệu từ t−22 đến t−1, không đưa giá Copper(t) vào cửa sổ đầu vào. Scaler được fit trên đoạn train của từng split rồi mới transform validation/test.

## 2. Input trong từng notebook

Thứ tự cột chính xác như code:

| Notebook | Input X | Shape mỗi mẫu | Target |
|---|---|---:|---|
| FS01_CU | Copper | 22 × 1 | Copper(t) |
| FS02_PAPER_CU | Copper, WTI, Gold, Silver | 22 × 4 | Copper(t) |
| FS03_PAPER_EXO | WTI, Gold, Silver | 22 × 3 | Copper(t) |
| FS04_METALS_CU | Copper, Aluminium, Tin, Silver, Gold | 22 × 5 | Copper(t) |
| FS05_METALS_EXO | Aluminium, Tin, Silver, Gold | 22 × 4 | Copper(t) |
| FS06_WTI_CU | Copper, Aluminium, Tin, WTI | 22 × 4 | Copper(t) |

FS01 là trường hợp chỉ dựa vào Copper quá khứ. FS03 tái hiện nhóm ba biến ngoại sinh WTI, Gold và Silver; FS02 thêm lịch sử Copper. FS04/FS05 kiểm tra nhóm kim loại có hoặc không có Copper. FS06 thử một tổ hợp có WTI, Aluminium, Tin và Copper.

Trong FS03 và FS05, Copper không có mặt trong X. Tuy nhiên, mô hình Residual cần Copper(t−1) để cộng phần thay đổi dự báo vào mức giá trước đó. Điều này là một phép khôi phục đầu ra, không phải một variable-token của iTransformer.

## 3. Inverted Embedding hoạt động ra sao?

Input ban đầu có dạng (batch, 22, K): trục thứ hai là thời gian, trục cuối là biến. Code hoán đổi hai trục để nhận (batch, K, 22). Khi đó mỗi biến trở thành một token, và 22 giá trị lịch sử của biến đó là vector token.

Một lớp Dense dùng chung chiếu vector 22 chiều của mỗi token thành 48 chiều:

**(batch, K, 22) → Dense(48) → (batch, K, 48)**

Khác cách Transformer thông thường vốn thường xem mỗi thời điểm là token, inverted embedding xem mỗi biến là token. Attention do đó trộn thông tin giữa các biến trong cùng cửa sổ lịch sử. Thời gian vẫn có thứ tự cố định trong vector 22 phần tử được chiếu; code này không thêm một positional embedding riêng.

Khi K = 1 như FS01, chỉ có một token nên không tồn tại quan hệ liên biến để attention học. Đây là giới hạn cấu trúc của ablation Copper-only, không phải lỗi chạy.

## 4. Hai khối Encoder

Mỗi block nhận tensor (batch, K, 48) và thực hiện lần lượt:

1. **Multi-Head Attention:** 4 head, key dimension 12 mỗi head; query/key/value đều xuất phát từ các variable-token hiện tại.
2. **Dropout attention:** dropout 0,1 bên trong MultiHeadAttention và một lớp Dropout 0,1 riêng sau đầu ra attention.
3. **Residual + LayerNormalization:** cộng đầu vào block với attention output, sau đó LayerNormalization với epsilon 1e−6.
4. **Feed-Forward:** Dense(96, GELU) → Dropout(0,1) → Dense(48).
5. **Residual + LayerNormalization:** cộng nhánh feed-forward với đầu vào của nhánh này, rồi chuẩn hóa.

Output mỗi encoder vẫn có dạng (batch, K, 48). Output encoder thứ nhất được đưa trực tiếp sang encoder thứ hai.

## 5. Prediction head và số tham số

Sau encoder thứ hai, code Flatten K token, mỗi token 48 chiều; Dense head biến vector này thành dự báo:

**Flatten(K × 48) → Dense(32, GELU) → Dropout(0,1) → Dense(1, linear)**

| Feature set | K | Encoder output | Flatten | Tổng tham số |
|---|---:|---:|---:|---:|
| FS01_CU | 1 | 1 × 48 | 48 | 40.625 |
| FS02_PAPER_CU | 4 | 4 × 48 | 192 | 45.233 |
| FS03_PAPER_EXO | 3 | 3 × 48 | 144 | 43.697 |
| FS04_METALS_CU | 5 | 5 × 48 | 240 | 46.769 |
| FS05_METALS_EXO | 4 | 4 × 48 | 192 | 45.233 |
| FS06_WTI_CU | 4 | 4 × 48 | 192 | 45.233 |

Các block encoder dùng chung cấu trúc; phần số tham số thay đổi theo K chủ yếu vì embedding và prediction head nhận số token khác nhau. Không nên so số tham số giữa các feature set như thể chúng là kiến trúc hoàn toàn khác.

## 6. Direct, Residual và Naive

### Direct

Mạng học ánh xạ từ X sang giá Copper(t) sau chuẩn hóa. Dự báo được inverse-transform về đơn vị giá gốc.

### Residual

Mạng học chênh lệch ở thang đã chuẩn hóa:

**Δzₜ = zₜ − zₜ₋₁**

Trong đó z là giá sau MinMaxScaler. Khi dự báo, code cộng Δz dự báo vào zₜ₋₁, sau đó inverse-transform để thu được giá gốc. Vì vậy, biểu thức trực quan là:

**ŷₜ = yₜ₋₁ + Δŷₜ**

Direct và Residual là hai model được train riêng, mỗi model có loss và trọng số riêng.

### Naive

Naive không có mạng học: dự báo giá phiên tiếp theo bằng giá Copper phiên gần nhất, ŷₜ = yₜ₋₁. Đây là baseline quan trọng vì chuỗi giá có tính liên tục cao.

## 7. Cấu hình train và đánh giá

Sáu notebook FS dùng seed 42, Adam với learning rate 0,0003 và clipnorm 1,0, loss MSE, batch size 64, tối đa 65 epochs, EarlyStopping patience 9 trên val_loss và khôi phục trọng số tốt nhất. Shuffle bị tắt để giữ thứ tự thời gian.

Bốn vị trí bắt đầu test là 80%, 90%, 92,5% và 95% chuỗi; mỗi vị trí chạy Expanding, Rolling-756 và Recency Weighted. Từng test là đoạn cuối theo thời gian. Validation được giữ trước test để early stopping quan sát dữ liệu chưa dùng cập nhật gradient.

Các metric được tính trên giá đã inverse-transform:

- MAE: trung bình trị tuyệt đối sai số.
- RMSE: căn bậc hai của MSE; nhạy hơn với sai số lớn.
- MAPE: sai số tuyệt đối tương đối theo phần trăm.
- R²: mức độ giải thích biến thiên của giá; có thể âm khi dự báo kém hơn dự báo bằng trung bình.
- Directional Accuracy: tỷ lệ phiên mà dấu thay đổi dự báo khớp dấu thay đổi thực tế.
- Skill so với Naive: 100 × (MAE_Naive − MAE_Model) / MAE_Naive; dương nghĩa là MAE thấp hơn Naive.

Directional Accuracy của Naive có thể rất thấp vì Naive dự báo thay đổi bằng 0; chỉ số này không phản ánh đầy đủ chất lượng dự báo mức giá. Nên xem cùng MAE/RMSE và biểu đồ giá.

## 8. Giới hạn khi diễn giải

- Test được đánh giá rolling one-step-ahead: lịch sử thực tế mới nhất có thể đi vào cửa sổ của phiên kế tiếp. Kết quả không phải dự báo nhiều bước mà không nhận thêm dữ liệu.
- FS03/FS05 không dùng Copper trong X, nhưng nhánh Residual vẫn cần Copper(t−1) để tái tạo mức giá.
- FS01 không thể học attention giữa nhiều biến vì K = 1.
- Dữ liệu ngoại sinh được căn theo ngày trong CSV; thời điểm công bố thực tế của từng biến chưa được kiểm tra.
- Một seed và một giai đoạn cuối không đủ để kết luận model hoặc feature set ưu việt tổng quát.
