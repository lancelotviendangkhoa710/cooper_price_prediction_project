# ĐÁNH GIÁ THỰC NGHIỆM MÔ HÌNH HỌC PHẦN DƯ VÀ BỔ SUNG RESIDUAL-XGBOOST TRÊN HORIZON 1 — 17-11

## Tóm tắt

Báo cáo này đánh giá chuỗi thực nghiệm ngày 17-11 trên 2.710 quan sát giá đồng (13/01/2016 – 13/11/2026), đánh dấu bước chuyển phương pháp luận quyết định: **chuyển dứt khoát từ $HORIZON = 0$ (nowcasting) sang $HORIZON = 1$ (dự báo trước một phiên)**. Thí nghiệm kế thừa toàn bộ khung so sánh Non-SA ngày 16-11 (LSTM, GRU, CNN1D-LSTM, CNN-GRU, Ridge Delta, Residual-GRU) và bổ sung thuật toán cây tăng cường độ dốc **Residual-XGBoost** với kỹ thuật trích xuất đặc trưng trễ (lags 1, 2, 3, 5, 10, 21), lợi suất lịch sử và độ biến động ngắn hạn.

Kết quả thực nghiệm trên $HORIZON = 1$ cho thấy:
1. **Các mô hình nơ-ron dự báo mức giá trực tiếp (Direct Level)**: Tiếp tục thất bại trước Naive Persistence. Mặc dù GRU và CNN-GRU đạt kết quả tốt nhất nhóm direct (RMSE khoảng 0,49–0,51 ở 90/10), sai số của chúng vẫn cao gấp 5 đến 12 lần Naive Persistence (RMSE 0,1010).
2. **Sự thật khi chuyển sang $HORIZON = 1$ đối với Ridge Delta**: Khi không còn được tiếp cận giá đóng cửa cùng ngày của các biến ngoại sinh, ưu thế của Ridge Delta biến mất hoàn toàn. Trên phân chia expanding 90/10, Ridge Delta có MAE 0,08067 USD (thua Naive 4,67%); trên rolling 756 phiên, Ridge Delta thua Naive tới 37,70%.
3. **Hiệu năng của Residual-GRU và Residual-XGBoost**:
   - `Residual-GRU` (expanding) bám sát Naive Persistence: RMSE đạt 0,10100 USD (cải thiện danh nghĩa 0,03% so với Naive 0,10103 USD), nhưng MAE là 0,07709 USD (thua Naive 0,018%). Độ chính xác định hướng (*Directional Accuracy - DA*) chỉ đạt **50,56%**, tương đương tung đồng xu ngẫu nhiên.
   - `Residual-XGBoost` (expanding) đạt RMSE 0,10642 USD, MAE 0,08383 USD (thua Naive 8,76%), DA đạt **47,21%**. Ở chiến lược rolling 756 phiên, Residual-XGBoost đạt RMSE 0,10275 USD, MAE 0,07990 USD (thua Naive 3,66%), DA đạt **47,58%**.
4. **Kết luận khoa học**: Khi tuân thủ nghiêm ngặt ranh giới thông tin trước một ngày ($HORIZON=1$), không có bất kỳ mô hình nơ-ron hay mô hình học máy nào đánh bại được Naive Persistence một cách có ý nghĩa thống kê.

**Từ khóa:** giá đồng, $HORIZON=1$, dự báo chuỗi thời gian, Residual Learning, Residual-XGBoost, Residual-GRU, Directional Accuracy, Naive Persistence.

---

## I. Mục tiêu và phạm vi

Thí nghiệm ngày 17-11 (thực hiện qua notebook `17-11-NonSA-Residual-GRU-XGBoost.ipynb`) tập trung vào ba câu hỏi cốt lõi:
1. **Thẩm định tính thực chất của dự báo ($HORIZON=1$)**: Khi loại bỏ hoàn toàn thông tin cùng ngày $t$ và chỉ cho phép mô hình nhìn thấy thông tin tới $t-1$, liệu các mô hình từng thắng Naive ở ngày 15-11 và 16-11 (như Ridge Delta) có còn duy trì được lợi thế?
2. **Mở rộng thuật toán cây với Residual-XGBoost**: XGBoost là thuật toán cực kỳ hiệu quả trên dữ liệu bảng. Bằng cách thiết kế các đặc trưng trễ (lags), sai phân lợi suất và độ biến động lăn ngắn hạn từ chuỗi $Copper$ lịch sử kết hợp với thông tin ngoại sinh, liệu Residual-XGBoost có vượt được mạng nơ-ron hồi quy chuỗi (GRU/LSTM)?
3. **Đo lường Directional Accuracy (Độ chính xác dự báo chiều biến động)**: Bổ sung chỉ số xác định tỷ lệ dự đoán đúng chiều tăng/giảm so với giá hôm trước:
   $$DA = \frac{1}{N_{nonflat}} \sum_{t} \mathbb{I}\left( \text{sgn}(\widehat{y}_t - y_{t-1}) == \text{sgn}(y_t - y_{t-1}) \right)$$
   nhằm đánh giá giá trị kinh tế/thực tế của tín hiệu giao dịch.

*Phạm vi giới hạn:* Thí nghiệm chạy trên môi trường Kaggle GPU (`FULL_MODE`, tổng thời gian chạy 6,3 phút). Chế độ thích nghi chế độ (B_adaptation) và phần dư (B_residual_extension) được thực hiện trên phân chia chính 90/10; nhánh mô hình cố định (A_fixed_models) được chạy qua cả ba mốc 90/10, 92,5/7,5 và 95/5.

---

## II. Thiết kế thực nghiệm và phương pháp

### 2.1. Căn chỉnh thời gian và bảo vệ ranh giới thông tin (No-leakage Horizon 1)

Với $HORIZON = 1$, cửa sổ đầu vào $X[t-22 : t-1]$ (kéo dài 22 phiên) chỉ chứa các giá trị ghi nhận từ ngày $t-1$ trở về trước. Nhãn mục tiêu là $Copper(t)$.
- Mẫu train, val, test được kiểm tra độc lập bằng assert: không có chỉ số thời gian nào của test hoặc nhãn $y(t)$ xuất hiện trong ma trận đầu vào.
- MinMaxScaler cho cả $X$ và $y$ chỉ được khớp (*fit*) trên tập train từ `fit_start` đến `val_start`.

| Tỷ lệ train/test | Train | Validation | Test | Ngày bắt đầu test |
|---|---:|---:|---:|---|
| **90/10** | 2.173 | 244 | 271 | 29/10/2025 |
| **92,5/7,5** | 2.233 | 251 | 204 | 30/01/2026 |
| **95/5** | 2.294 | 258 | 136 | 07/05/2026 |

### 2.2. Đặc trưng và siêu tham số cho Residual-XGBoost

Khác với mạng nơ-ron tiếp nhận chuỗi 3D $(N, 22, D)$, XGBoost yêu cầu định dạng bảng phẳng 2D. Các đặc trưng được tạo ra hoàn toàn từ thông tin $\le t-1$:
1. Ma trận ngoại sinh trải phẳng $X[t-22 : t-1]$ (kích thước $22 \times 3 = 66$ đặc trưng).
2. Các mốc giá trễ của Copper: $k \in \{1, 2, 3, 5, 10, 21\}$ phiên trước.
3. Lợi suất phiên gần nhất: $\Delta Copper_{t-1} = Copper_{t-1} - Copper_{t-2}$.
4. Giá trị trung bình và độ lệch chuẩn của chuỗi biến động 10 phiên gần nhất: $\text{mean}(\Delta Copper_{t-k})$, $\text{std}(\Delta Copper_{t-k})$ với $k \in [1, 10]$.

**Tối ưu siêu tham số XGBoost (XGB_TUNING)**: Chỉ chọn cấu hình có $val\_mse\_scaled$ thấp nhất trên tập validation, không dùng test:
- `expanding`: 350 cây (`n_estimators=350`), độ sâu tối đa 2 (`max_depth=2`), tốc độ học 0,015 (`learning_rate=0.015`). Validation MSE đạt 0,0014243.
- `rolling_756`: 150 cây, `max_depth=2`, `learning_rate=0.03`. Validation MSE đạt 0,0039061.
- `recency_weighted`: 150 cây, `max_depth=2`, `learning_rate=0.03`. Validation MSE đạt 0,0014967.

---

## III. Kết quả thực nghiệm

### 3.1. Nhánh A: Các mô hình Direct Level trên Horizon 1

Bảng kết quả kiểm tra trên thang giá gốc qua 3 mốc phân chia:

| Tập biến | Tỷ lệ | Mô hình | MSE_scaled | RMSE_raw | MAE_raw | Relative MAE vs Naive |
|---|---|---|---:|---:|---:|---:|
| — | **90/10** | **Naive Persistence** | **0,001021** | **0,10103** | **0,07708** | **1,0000** |
| `paper3` | 90/10 | gru | 0,024441 | 0,49441 | 0,42647 | 5,5330 |
| `legacy14` | 90/10 | cnn_gru | 0,025863 | 0,50859 | 0,44726 | 5,8028 |
| `legacy14` | 90/10 | gru | 0,033663 | 0,58024 | 0,51469 | 6,6777 |
| `paper3` | 90/10 | cnn_gru | 0,038906 | 0,62379 | 0,54010 | 7,0073 |
| `paper3` | 90/10 | cnn1d_lstm | 0,067688 | 0,82278 | 0,68826 | 8,9295 |
| `legacy14` | 90/10 | lstm | 0,090500 | 0,95138 | 0,85413 | 11,0815 |
| `legacy14` | 90/10 | cnn1d_lstm | 0,104981 | 1,02468 | 0,92628 | 12,0176 |
| `paper3` | 90/10 | lstm | 0,153661 | 1,23969 | 1,14599 | 14,8681 |
| — | **92,5/7,5** | **Naive Persistence** | **0,000968** | **0,09840** | **0,07512** | **1,0000** |
| `legacy14` | 92,5/7,5 | cnn_gru | 0,024964 | 0,49967 | 0,44702 | 5,9510 |
| `legacy14` | 92,5/7,5 | gru | 0,027054 | 0,52017 | 0,47058 | 6,2647 |
| `paper3` | 92,5/7,5 | gru | 0,032286 | 0,56825 | 0,51213 | 6,8179 |
| `paper3` | 92,5/7,5 | cnn_gru | 0,037391 | 0,61152 | 0,55331 | 7,3660 |
| `paper3` | 92,5/7,5 | cnn1d_lstm | 0,103789 | 1,01884 | 0,93039 | 12,3860 |
| `legacy14` | 92,5/7,5 | cnn1d_lstm | 0,135308 | 1,16330 | 1,10056 | 14,6515 |
| `legacy14` | 92,5/7,5 | lstm | 0,153304 | 1,23825 | 1,18745 | 15,8081 |
| `paper3` | 92,5/7,5 | lstm | 0,227004 | 1,50677 | 1,46309 | 19,4777 |
| — | **95/5** | **Naive Persistence** | **0,000870** | **0,09661** | **0,07280** | **1,0000** |
| `legacy14` | 95/5 | gru | 0,024566 | 0,51331 | 0,45230 | 6,2129 |
| `paper3` | 95/5 | cnn_gru | 0,036366 | 0,62454 | 0,55742 | 7,6569 |
| `paper3` | 95/5 | gru | 0,054187 | 0,76236 | 0,70459 | 9,6785 |
| `legacy14` | 95/5 | cnn_gru | 0,062863 | 0,82113 | 0,78198 | 10,7417 |
| `legacy14` | 95/5 | cnn1d_lstm | 0,104557 | 1,05898 | 1,03512 | 14,2188 |
| `paper3` | 95/5 | cnn1d_lstm | 0,112152 | 1,09677 | 1,07602 | 14,7806 |
| `legacy14` | 95/5 | lstm | 0,143917 | 1,24242 | 1,22702 | 16,8549 |
| `paper3` | 95/5 | lstm | 0,224139 | 1,55049 | 1,53932 | 21,1447 |

---

### 3.2. So sánh Nhánh B: Residual-GRU vs Residual-XGBoost vs Ridge Delta ($HORIZON=1$)

Bảng đối chiếu tổng hợp từ `residual_gru_xgboost_vs_naive.csv` trên 271 điểm test của tỷ lệ 90/10 ($HORIZON=1$):

| Mô hình | Chiến lược | RMSE_raw | MAE_raw | MAE Gain vs Naive (%) | Directional Accuracy (%) | Thắng Naive MAE? | Thắng Naive RMSE? |
|---|---|---:|---:|---:|---:|:---:|:---:|
| **Naive Persistence** | — | **0,10103** | **0,07708** | **0,00%** | — | — | — |
| `residual_gru` | `expanding` | **0,10100** | 0,07709 | -0,02% | 50,56% | ❌ False | ✅ True |
| `residual_gru` | `recency_weighted` | 0,10102 | 0,07716 | -0,11% | 48,70% | ❌ False | ✅ True |
| `residual_xgboost` | `rolling_756` | 0,10275 | 0,07990 | -3,66% | 47,58% | ❌ False | ❌ False |
| `ridge_delta` | `expanding` | 0,10524 | 0,08067 | -4,67% | 50,56% | ❌ False | ❌ False |
| `residual_gru` | `rolling_756` | 0,10602 | 0,08208 | -6,49% | 46,47% | ❌ False | ❌ False |
| `residual_xgboost` | `expanding` | 0,10642 | 0,08383 | -8,76% | 47,21% | ❌ False | ❌ False |
| `ridge_delta` | `recency_weighted` | 0,11820 | 0,09155 | -18,78% | 46,84% | ❌ False | ❌ False |
| `residual_xgboost` | `recency_weighted` | 0,11762 | 0,09513 | -23,43% | 46,84% | ❌ False | ❌ False |
| `ridge_delta` | `rolling_756` | 0,13214 | 0,10613 | -37,70% | 46,47% | ❌ False | ❌ False |

---

## IV. Phân tích chuyên sâu các phát hiện chính

### 4.1. Sự sụp đổ của ưu thế Ridge Delta khi chuyển từ $HORIZON=0$ sang $HORIZON=1$
- Ở ngày 15-11 và 16-11 ($HORIZON=0$), Ridge Delta đạt RMSE 0,09420 (giảm MAE hơn 6,2% so với Naive). Khi đó mô hình dùng biến động cùng ngày của vàng, bạc, dầu để ước lượng biến động của đồng.
- Khi chuyển sang $HORIZON=1$, mô hình chỉ biết biến động của các biến ngoại sinh đến ngày $t-1$. Kết quả: RMSE tăng từ 0,09420 lên 0,10524; MAE tăng từ 0,07227 lên 0,08067. Ridge Delta lập tức bị Naive đánh bại (thua 4,67% ở expanding và thua 37,70% ở rolling 756).
- **Kết luận:** Ưu thế của Ridge Delta ở các ngày trước hoàn toàn là hiện tượng tương quan đồng thời giữa các tài sản tài chính (*simultaneous co-movement*), không phải là khả năng dự báo trước giá tương lai.

### 4.2. Bản chất "tiệm cận Naive" của Residual-GRU
- Residual-GRU expanding đạt RMSE 0,10100 (thấp hơn Naive 0,10103 một lượng siêu nhỏ: 0,00003 USD). Tuy nhiên, trên MAE, Residual-GRU lại kém hơn Naive (0,07709 vs 0,07708).
- Directional Accuracy của Residual-GRU chỉ đạt 50,56%.
- Điều này chứng minh rằng khi huấn luyện tối ưu hóa hàm loss MSE trên phần dư $\Delta y(t)$ của chuỗi bước đi ngẫu nhiên gần đúng (*martingale difference*), mạng GRU chỉ học được cách đưa đầu ra dự đoán $\widehat{\Delta y}(t) \approx 0$. Nói cách khác, Residual-GRU chỉ đang bắt chước Naive Persistence chứ không trích xuất được tín hiệu định hướng thực chất nào từ dữ liệu.

### 4.3. Đánh giá thuật toán cây: Residual-XGBoost
- Khi dùng cửa sổ `expanding`, Residual-XGBoost có xu hướng khớp quá mức (*overfitting*) trên nhiễu lịch sử của phần dư, dẫn đến MAE kém Naive 8,76%.
- Khi chuyển sang `rolling_756`, XGBoost kiểm soát tốt hơn và thu hẹp khoảng cách thua Naive xuống còn 3,66% (RMSE 0,10275), tốt hơn cả Residual-GRU rolling (RMSE 0,10602).
- Tuy nhiên, Directional Accuracy của XGBoost trên mọi chiến lược chỉ dao động từ 46,8% đến 47,6% (dưới 50%), cho thấy việc cố gắng dự đoán chiều biến động hàng ngày của giá đồng dựa trên các đặc trưng kỹ thuật trễ không mang lại lợi thế thống kê.

---

## V. Kết luận và định hướng tiếp theo

1. **Khẳng định nguyên tắc phương pháp luận**: Toàn bộ các nghiên cứu tiếp theo bắt buộc phải cố định $HORIZON=1$. Mọi kết quả dựa trên $HORIZON=0$ chỉ được xem là bài toán nowcasting, không được sử dụng để tuyên bố tính khả thi của chiến lược giao dịch hay dự báo vận hành.
2. **Naive Persistence là cột mốc chuẩn mực tối thượng**: Mọi mô hình phức tạp (LSTM, GRU, XGBoost, Ridge) đều không chứng minh được sự vượt trội có ý nghĩa thực tế trước Naive Persistence trên một bước dự báo giá hàng ngày.
3. **Đòi hỏi kiểm định walk-forward nhiều năm**: Cần triển khai kiểm định Walk-Forward liên tục qua nhiều năm với kiểm định thống kê chuẩn mực (Diebold-Mariano) để trả lời dứt khoát liệu có bất kỳ giai đoạn thị trường nào mô hình học máy thực sự tạo ra giá trị gia tăng so với bước đi ngẫu nhiên hay không.

---

## Hiện vật và tệp kết quả

- [Notebook thực nghiệm](../../17-11-NonSA-Residual-GRU-XGBoost.ipynb)
- [Bảng tổng hợp summary_v17.csv](summary_v17.csv)
- [Bảng so sánh chi tiết residual_gru_xgboost_vs_naive.csv](residual_gru_xgboost_vs_naive.csv)
- [Chi tiết từng lần chạy results_v17.csv](results_v17.csv)
- [Cấu hình tốt nhất của XGBoost expanding](xgb_best_expanding_paper3_0.9.json)
- [Biểu đồ Actual vs Naive vs Residual-GRU vs XGBoost](actual_naive_residual_gru_xgboost.png)
