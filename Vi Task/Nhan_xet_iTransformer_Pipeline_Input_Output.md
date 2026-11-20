# Nhận xét mô hình iTransformer: pipeline, input và output

## 1. Phạm vi nhận xét

Tài liệu này nhận xét lần chạy được lưu trong `Vi Task/iTransformer_vs_SA_LSTM_K5_Fair_FINAL_Colab.ipynb` và các kết quả được trích ra ở `Vi Task/outputs/`. Đây là benchmark iTransformer so với SA-LSTM và mô hình dự báo ngây thơ (Naive), trên cùng bộ dữ liệu và các mốc chia thời gian.

**Kết luận ngắn:** pipeline đã kiểm soát được một số lỗi quan trọng như dùng dữ liệu tương lai và scaler nhìn thấy tập test. Tuy vậy, kết quả hiện tại chưa cho thấy iTransformer dự báo giá đồng tốt hơn baseline. Ở chế độ dự báo trực tiếp giá, sai số rất lớn; ở chế độ dự báo phần thay đổi so với giá phiên trước, iTransformer gần như chỉ ngang dự báo Naive. Vì thế nên xem đây là một lần benchmark chẩn đoán, chưa phải bằng chứng iTransformer có lợi thế.

## 2. Pipeline thực nghiệm

```text
CSV giá hàng hóa
    ↓ kiểm tra ngày, cột, giá thiếu/trùng và sắp xếp thời gian
Tạo cửa sổ 22 phiên quá khứ → dự báo Copper tại phiên kế tiếp
    ↓ chia theo thời gian; scaler chỉ fit trên phần train
Train / validation / test dùng chung cho các mô hình
    ↓
iTransformer ───────── SA-LSTM cố định ───────── SA-LSTM tối ưu bằng SA
    ↓                         ↓                            ↓
Giá trực tiếp hoặc phần thay đổi so với phiên trước; thêm baseline Naive
    ↓
MAE / RMSE / MSE / hướng thay đổi / so sánh với Naive / biểu đồ
```

### 2.1 Dữ liệu và cách chia

- CSV có **2.710 quan sát**, từ **2016-01-13 đến 2026-11-13**; lần chạy ghi nhận không loại dòng nào vì thiếu dữ liệu.
- Mỗi mẫu dùng **22 phiên trước** (lookback = 22) để dự báo **một phiên tiếp theo** (horizon = 1).
- Notebook chạy bốn mốc bắt đầu test: 80%, 90%, 92,5% và 95% của chuỗi. Validation là 10% phần dữ liệu đứng trước mốc test. Ví dụ, mốc “90/10” có nghĩa test bắt đầu ở khoảng 90% chuỗi; validation được lấy ngay trước đó, chứ không phải chia ngẫu nhiên train/validation/test.
- Có ba cách tổ chức phần train: `expanding` dùng toàn bộ phần lịch sử trước validation; `rolling_756` chỉ dùng tối đa 756 phiên ngay trước validation; `recency_weighted` giữ dữ liệu mở rộng nhưng giảm trọng số các quan sát cũ.
- Các scaler MinMax cho input và target được fit trên train của từng cấu hình. Cùng một split, ngày test và năm kênh input được dùng cho các mô hình trong cùng một nhóm.
- Trong mỗi ngày test, mô hình dự báo một bước và có thể dùng 22 quan sát thực tế gần nhất làm lịch sử cho bước kế tiếp. Đây là **rolling one-step-ahead evaluation**. Nó không tương đương với dự báo toàn bộ nhiều tháng từ một ngày xuất phát mà không cập nhật dữ liệu thực tế.

### 2.2 Những mô hình được so sánh

Notebook ghi nhận **84/84 job hoàn thành**: 4 mốc test × 3 cách tổ chức train × 7 cấu hình (Naive, iTransformer direct/residual, SA-LSTM fixed direct/residual, SA-LSTM tuned direct/residual). Có thêm 32 lượt đánh giá SA để tìm siêu tham số cho hai mục tiêu.

Phần iTransformer đổi dạng input từ `(22 phiên, 5 biến)` thành **5 token biến**, mỗi token biểu diễn lịch sử 22 phiên của một biến. Sau đó Dense chiếu mỗi token thành embedding 48 chiều, hai khối Transformer dùng 4 attention heads để trộn thông tin giữa các biến, rồi Flatten và Dense dự báo một giá trị Copper. Đây là **biến thể iTransformer cho bài toán một target scalar**; không phải đầu ra đa biến/horizon dài nguyên bản.

SA-LSTM có hai lớp LSTM với 39 và 111 units trong bản cố định, Dropout 0,2, Dense 32 và đầu ra tuyến tính. Bản tuned cho SA tìm số units và learning rate trên validation. iTransformer dùng Adam với learning rate cố định `3e-4`; do đó mức độ tìm kiếm siêu tham số giữa hai kiến trúc không hoàn toàn ngang nhau.

## 3. Input: mô hình thực sự được xem những gì?

Năm kênh được đưa vào là:

| Kênh | Nội dung của mỗi cửa sổ |
|---|---|
| Aluminium | Giá Aluminium của 22 phiên trước |
| Tin | Giá Tin của 22 phiên trước |
| Silver | Giá Silver của 22 phiên trước |
| Gold | Giá Gold của 22 phiên trước |
| Copper | **Giá Copper quá khứ** trong 22 phiên trước |

Tensor mỗi mẫu có dạng `(22, 5)`, tương ứng 22 hàng thời gian và 5 biến. `Copper` vẫn là target, nhưng các giá trị Copper trước thời điểm dự báo đồng thời được đưa vào làm lịch sử. Không có WTI trong cấu hình này.

Việc có Copper quá khứ **không tự động là leakage**: với target tại thời điểm `t`, cửa sổ chỉ lấy từ `t−22` đến `t−1`; giá Copper tại `t` không được đưa vào input. Notebook có kiểm tra các chỉ số cửa sổ nhỏ hơn thời điểm target và kiểm tra giá Copper cuối cửa sổ khớp với Copper phiên liền trước. Tuy nhiên, cách đánh giá rolling chỉ hợp lệ trong tình huống vận hành có thể cập nhật giá thực tế sau mỗi phiên.

**Khác phạm vi bài báo gốc:** đây là dataset 2016–2026 với Aluminium, Tin, Silver, Gold và Copper quá khứ; notebook ghi rõ không dùng WTI. Vì vậy không nên gọi đây là tái hiện nguyên trạng dữ liệu/thí nghiệm của paper gốc. Đây là một thiết kế benchmark mới trên bộ dữ liệu hiện có của nhóm.

## 4. Output: mô hình dự báo gì và đọc kết quả ra sao?

Đầu ra của mỗi mẫu là **một số duy nhất**, giá Copper của phiên kế tiếp. Notebook thử hai cách đặt mục tiêu:

1. **Direct:** mạng dự báo trực tiếp mức giá Copper đã chuẩn hóa; khi đánh giá, giá được đổi về thang đo gốc.
2. **Residual:** mạng dự báo phần chênh lệch giữa Copper phiên kế tiếp và Copper phiên trước. Dự báo giá cuối cùng bằng giá phiên trước cộng phần chênh lệch dự báo.

Residual là một ablation bổ sung, không nên gộp với direct khi kết luận kiến trúc nào tốt hơn. So sánh hợp lý là direct với direct, residual với residual, đồng thời xem Naive `Copper(t−1)` làm mốc.

Các chỉ số được tính trên giá Copper ở thang đo gốc: MAE, RMSE, MSE; notebook còn ghi directional accuracy, mức cải thiện so với Naive và tương quan của phần thay đổi dự báo. Với MAE/RMSE/MSE, **thấp hơn là tốt hơn**. Biểu đồ mức giá cần được đọc cùng biểu đồ `ΔCopper`: đường dự báo có thể trông gần giá thực tế vì nó bám giá phiên trước, dù chưa dự báo tốt các biến động.

### Ví dụ kết quả: mốc test 90%, `expanding`

MAE dưới đây dùng đơn vị giá gốc của Copper trong CSV:

| Cấu hình | MAE |
|---|---:|
| Naive: dùng giá Copper phiên trước | 0.077077 |
| iTransformer, direct | 1.768602 |
| iTransformer, residual | 0.078119 |
| SA-LSTM fixed, direct | 0.798291 |
| SA-LSTM fixed, residual | 0.082928 |
| SA-LSTM tuned, direct | 0.562087 |
| SA-LSTM tuned, residual | 0.085346 |

Ở ví dụ này, iTransformer direct sai số lớn hơn nhiều so với Naive. iTransformer residual gần baseline nhưng MAE vẫn cao hơn một chút (`0.078119` so với `0.077077`). Kết quả các split khác trong bảng cũng có cùng dấu hiệu tổng quát: direct khá kém; residual thường nằm gần Naive. Điều này cho thấy mô hình chưa chứng minh được khả năng dự báo biến động tốt hơn quy tắc “giữ giá phiên trước”.

Biểu đồ `delta_predictions.png` cần được chú ý: Copper thực tế thay đổi khá thất thường, còn phần thay đổi dự báo mượt và nhỏ. Đây là dấu hiệu mô hình có thể đang co dự báo về gần 0; khi cộng với giá phiên trước, đường dự báo mức giá vẫn có vẻ bám theo xu hướng, nhưng phần biến động ngắn hạn chưa được nắm bắt tốt.

## 5. Nhận xét chất lượng thiết kế

### Điểm làm tốt

- Chia theo thời gian, không xáo trộn thứ tự chuỗi.
- Dùng scaler fit trên train, không fit bằng dữ liệu validation/test.
- Các mô hình nhận cùng cửa sổ, biến đầu vào và ngày test trong mỗi cấu hình.
- Có baseline Naive, lưu kết quả theo từng job, kiểm tra job thiếu và dự báo finite.
- Có tách direct/residual, nhờ vậy tránh nhầm lợi ích của residual với lợi ích riêng của Transformer.

### Hạn chế cần ghi rõ

- Một seed (`42`) chưa cho biết kết quả có ổn định qua các lần khởi tạo khác nhau hay không.
- Chỉ có 16 lượt SA cho mỗi objective; đó là ngân sách tìm kiếm hạn chế, không bảo đảm tìm được siêu tham số tối ưu toàn cục.
- SA được tìm trên split 80/20 expanding rồi cấu hình được dùng ở các split khác; kết quả các split sau do đó không phải là một lần tối ưu SA độc lập cho từng split.
- iTransformer có 46.769 tham số, SA-LSTM cố định có 77.681 tham số. Cùng input và split tạo điều kiện so sánh tốt hơn, nhưng số tham số, optimizer và ngân sách tuning vẫn khác.
- Các khoảng test ở bốn tỷ lệ có phần thời gian chồng lấn; không nên coi các kết quả đó là bốn thí nghiệm độc lập để suy luận thống kê.
- Cần xác minh thời điểm công bố/khớp phiên giao dịch của Aluminium, Tin, Silver, Gold và Copper. Việc mọi giá cùng mang ngày `t−1` trong file chưa tự chứng minh rằng chúng đã sẵn có tại thời điểm dự báo thực tế.
- Biểu đồ mức giá trong test là dự báo cuốn chiếu một bước; không thể dùng nó để tuyên bố mô hình dự báo trước toàn bộ quãng test nhiều tháng.

## 6. Kết luận có thể trình bày

Trong thí nghiệm hiện tại, iTransformer nhận 22 phiên lịch sử của bốn kim loại Aluminium, Tin, Silver và Gold cùng lịch sử Copper để dự báo giá Copper phiên kế tiếp. Pipeline đã có các kiểm tra cơ bản chống leakage do chia sai thời gian và fit scaler trên toàn bộ dữ liệu. Tuy nhiên, iTransformer dự báo trực tiếp mức giá cho sai số lớn; khi dự báo phần thay đổi, kết quả gần với baseline Naive và phần thay đổi dự báo còn quá mượt. Vì vậy, kết quả hiện tại cho thấy iTransformer **chưa cải thiện rõ ràng** so với Naive trên cấu hình này; không nên kết luận mô hình Transformer nói chung không phù hợp.

Trước khi khẳng định hiệu năng, nên kiểm tra thêm nhiều seed, dùng walk-forward validation/test không chồng lấn, báo cáo sai số và directional accuracy của `ΔCopper`, xác minh căn chỉnh thời gian công bố dữ liệu, rồi mới thử thay đổi lookback, cách chuẩn hóa hoặc mục tiêu dự báo. Mọi thay đổi cần chọn bằng validation và chỉ mở test cuối cùng để đánh giá.

## 7. File kết quả liên quan

- Notebook: `iTransformer_vs_SA_LSTM_K5_Fair_FINAL_Colab.ipynb`
- Giá dự báo so với thực tế và Naive: `outputs/figures/actual_pred_vs_naive.png`
- Phần thay đổi thực tế/dự báo: `outputs/figures/delta_predictions.png`
- So sánh MAE các mô hình: `outputs/figures/mae_comparison.png`
- Nhật ký: `outputs/execution_log.txt`
- Các bảng HTML được trích từ output notebook: `outputs/tables/`

Lưu ý: notebook ghi file CSV/JSON chi tiết và ZIP đầy đủ lên Google Drive. Trong thư mục GitHub hiện có hình, các bảng HTML và log, không có các CSV/JSON chi tiết từ Drive.
