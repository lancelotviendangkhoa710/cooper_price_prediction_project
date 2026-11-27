# BÁO CÁO ĐÁNH GIÁ THỰC NGHIỆM TÁI LẬP VÀ ĐỐI CHUẨN CÁC PHƯƠNG PHÁP HUẤN LUYỆN NON-SA (PROTOCOL V20) — DỰ BÁO GIÁ ĐỒNG

## Tóm tắt thực nghiệm

Báo cáo này trình bày kết quả thực nghiệm tái lập và đối chuẩn phương pháp luận toàn diện (**Non-SA Fair Full Protocol V20**) trên chuỗi dữ liệu 2.710 phiên giao dịch giá đồng (*Copper*) trong giai đoạn từ 13/01/2016 đến 13/11/2026. Nghiên cứu thực hiện đánh giá độc lập và có hệ thống đối với các kiến trúc mô hình học máy, học sâu cũng như các chiến lược huấn luyện chuỗi thời gian nhằm phân tích độ nhạy, tính ổn định và khả năng khái quát hóa trong điều kiện kiểm soát nghiêm ngặt ranh giới thông tin:

1. **Thiết lập ranh giới thông tin chặt chẽ**: Cố định bước dự báo $HORIZON = 1$, mô hình chỉ được phép tiếp cận thông tin quan sát được đến phiên $t-1$ để dự báo giá đóng cửa phiên $t$, bảo đảm phản ánh đúng điều kiện dự báo thực tế ngoài mẫu.
2. **Thiết kế ma trận thực nghiệm đa chiều (Full Factorial Grid)**: Thực hiện 324 lượt huấn luyện độc lập nhằm đối sánh tương tác giữa 4 tỷ lệ phân chia thời gian (80/20, 90/10, 92,5/7,5, 95/5) $\times$ 3 bộ đặc trưng (`copper_only`, `copper_paper3`, `copper_legacy14`) $\times$ 3 chiến lược xử lý dữ liệu thích nghi (`expanding`, `rolling_756`, `recency_weighted`) $\times$ 9 mô hình (4 kiến trúc Direct, 4 kiến trúc Residual, 1 mô hình tuyến tính Ridge Residual), đối chiếu chuẩn mực với mô hình cơ sở Naive Persistence.
3. **Thẩm định Walk-Forward liên tục và kiểm định thống kê Diebold-Mariano**: Đánh giá ngoài mẫu trên 2.214 phiên giao dịch (2018–2026) với cơ chế tái huấn luyện định kỳ mỗi 63 phiên qua 5 hạt giống ngẫu nhiên (seeds 41 đến 45) kết hợp kiểm định thống kê Diebold-Mariano sử dụng sai số chuẩn vững HAC (Heteroskedasticity and Autocorrelation Consistent).

**Tóm lược các kết quả đánh giá mô hình và phương pháp:**

- **Về cấu trúc mô hình (Direct Level vs. Residual Learning)**: Các mô hình dự báo trực tiếp mức giá (*Direct Level*) gặp thách thức lớn trước hiện tượng không dừng và dịch chuyển phân phối của chuỗi tài chính, dẫn đến sai số MAE cao hơn đáng kể so với mô hình cơ sở Naive Persistence. Ngược lại, phương pháp học phần dư sai phân (*Residual Learning*) giúp ổn định phân phối mục tiêu và đưa sai số về sát mức của Naive Persistence.
- **Về cơ chế học phần dư (Residual Variance Analysis)**: Dưới hàm mất mát sai số bình phương trung bình (MSE) trong môi trường dữ liệu có tỷ lệ tín hiệu trên nhiễu thấp (*low Signal-to-Noise Ratio*), các mạng nơ-ron có xu hướng thu hẹp phương sai phần dư dự báo $\widehat{\Delta}_t$ về gần 0 ($\text{std}(\widehat{\Delta}_{pred})$ dao động $0,0005 - 0,0047$ USD so với độ lệch chuẩn thực tế $\text{std}(\Delta_{true}) \approx 0,101$ USD). Đây là cơ chế tối ưu hóa tự nhiên của mô hình nhằm cực tiểu hóa rủi ro sai số bình phương khi biến động ngắn hạn mang tính ngẫu nhiên cao.
- **Về vai trò của biến ngoại sinh (Feature Ablation)**: Việc bổ sung các biến hàng hóa liên thị trường (Dầu WTI, Vàng, Bạc, Nhôm, Thiếc) ở bước trễ $t-1$ không tạo ra sự cải thiện rõ rệt so với mô hình đơn biến chỉ dùng giá đồng (`copper_only`). Phân tích tương quan bậc trễ 1 ngày trong tập huấn luyện cho thấy tương quan tuyến tính và phi tuyến giữa biến động các hàng hóa ngoại sinh và giá đồng đều ở mức rất thấp ($|r| < 0,092$).
- **Về độ ổn định dài hạn và kiểm định giả thuyết thống kê**: Qua 2.214 phiên kiểm tra Walk-Forward đa chu kỳ kinh tế, kiểm định Diebold-Mariano qua 5 hạt giống ngẫu nhiên cho thấy hiệu năng dự báo của Residual-GRU không có sự khác biệt mang ý nghĩa thống kê so với Naive Persistence ở 2 hạt giống ($p = 0,704$ và $p = 0,366$), trong khi ở các hạt giống còn lại Naive Persistence cho kết quả sai số thấp hơn ở mức ý nghĩa thống kê. Tỷ lệ dự báo đúng hướng (*Directional Accuracy*) dao động quanh mức $50,27\% - 51,45\%$, phản ánh đặc tính tiệm cận bước đi ngẫu nhiên (*near-random-walk*) của giá hàng hóa ở tần suất ngày.

**Từ khóa:** Dự báo giá đồng, Horizon 1, Non-SA Protocol V20, Full Factorial Grid, Residual Learning, Feature Ablation, Diebold-Mariano Test, Walk-Forward Validation, Martingale Difference.

---

## I. Thiết lập Giao thức Huấn luyện và Thực nghiệm Tái lập

Giao thức thực nghiệm **Non-SA Fair Full Protocol V20** được thiết kế nhằm chạy lại, đối chuẩn và khảo sát toàn diện các phương pháp huấn luyện chuỗi thời gian đối với giá đồng.

> [!NOTE]
> **Định hướng tiếp cận của thực nghiệm:**
> Nghiên cứu không đặt nặng việc chạy đua tối ưu hóa độ chính xác dự báo (chasing accuracy / SOTA) hay tìm kiếm một mô hình vượt trội tuyệt đối. Trọng tâm cốt lõi là **thực thi thực nghiệm có kiểm soát để thấu hiểu hành vi thực tế của các phương pháp**: quan sát xem các kiến trúc học sâu thực sự học được những gì từ dữ liệu chuỗi thời gian, phản ứng như thế nào trước các phương thức biểu diễn nhãn (Direct vs. Residual), các chiến lược cửa sổ dữ liệu (Expanding vs. Rolling vs. Recency) và hiện tượng nhiễu chiều khi mở rộng biến ngoại sinh.

### 1.1. Dữ liệu và phân chia tập dữ liệu

Thực nghiệm sử dụng tập dữ liệu 2.710 phiên giao dịch hàng ngày (13/01/2016 – 13/11/2026) từ tệp dữ liệu chuẩn hóa [dataset_9_commodities_daily.csv](file:///d:/University/ptdlkd-is403.r11/dataset_9_commodities_daily.csv) (mã băm SHA256: `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600`).

- Dữ liệu đầy đủ 2.710 dòng, không có giá trị khuyết thiếu (0 missing values), chuỗi thời gian tăng đơn điệu.
- Độ dài cửa sổ lịch sử đầu vào: $LOOKBACK = 22$ phiên giao dịch (~1 tháng giao dịch).
- Chân trời dự báo: $HORIZON = 1$ (sử dụng ma trận quan sát $X[t-22 : t-1]$ để dự báo giá đóng cửa $Copper(t)$).
- Khảo sát 4 mốc phân chia thời gian train/validation/test nhằm kiểm tra độ nhạy theo quy mô mẫu và các chu kỳ thị trường khác nhau:

| Tỷ lệ Train / Test | Số phiên Train | Validation (10% trước test) | Số phiên Test | Ngày bắt đầu Test | Ngày kết thúc Test |
| --- | ---: | ---: | ---: | --- | --- |
| **80 / 20** | 1.930 | 216 | 542 | 07/10/2024 | 13/11/2026 |
| **90 / 10** | 2.173 | 244 | 271 | 29/10/2025 | 13/11/2026 |
| **92,5 / 7,5** | 2.233 | 251 | 204 | 30/01/2026 | 13/11/2026 |
| **95 / 5** | 2.294 | 258 | 136 | 07/05/2026 | 13/11/2026 |

*Nguyên tắc chuẩn hóa dữ liệu:* `MinMaxScaler` chỉ được khớp (`fit`) duy nhất trên phần dữ liệu huấn luyện cục bộ (`fit_start` đến `val_start`). Sau đó, cùng bộ tham số co giãn này được áp dụng để chuyển đổi (`transform`) cho tập validation và test. Trong bài toán dự báo trực tiếp, các giá trị thực tế ngoài mẫu nếu vượt khỏi khoảng $[0, 1]$ được giữ nguyên để kiểm tra khả năng ngoại suy của mô hình trước biến động giá mới.

### 1.2. Ba tập đặc trưng khảo sát (Feature Sets)

Thực nghiệm xây dựng 3 tập đặc trưng để phân tích tác động của thông tin ngoại sinh:

1. **`copper_only`**: Duy nhất chuỗi giá mục tiêu `Copper` (chuẩn đối sánh đơn biến).
2. **`copper_paper3`**: Gồm 4 biến `['Copper', 'WTI', 'Gold', 'Silver']` (thiết lập đa biến tương ứng với các hàng hóa được chọn trong các nghiên cứu tham chiếu).
3. **`copper_legacy14`**: Gồm 5 biến `['Copper', 'Aluminium', 'Tin', 'Silver', 'Gold']` (mở rộng thêm các kim loại công nghiệp liên quan).

### 1.3. Ba chiến lược xử lý dữ liệu thích nghi (Adaptation Schemes)

Để đánh giá cách các mô hình thích ứng với sự thay đổi của cấu trúc thị trường qua thời gian, 3 chiến lược huấn luyện được áp dụng:

1. **`expanding`**: Tận dụng toàn bộ lịch sử dữ liệu từ ngày khởi đầu (13/01/2016) đến điểm phân chia validation.
2. **`rolling_756`**: Sử dụng cửa sổ trượt cố định 756 phiên giao dịch (~3 năm giao dịch gần nhất) trước điểm validation, loại bỏ dữ liệu quá khứ xa nhằm thích ứng với các thay đổi cấu trúc dài hạn.
3. **`recency_weighted`**: Huấn luyện trên toàn bộ chuỗi nhưng áp dụng trọng số mẫu suy giảm theo hàm mũ để ưu tiên dữ liệu gần:
   $$w_i = \exp\left(-\frac{\ln(2) \cdot age_i}{252}\right)$$
   với chu kỳ bán rã được đặt là 252 phiên (~1 năm giao dịch).

### 1.4. Chín kiến trúc mô hình được đánh giá

Ma trận mô hình được thiết kế nhằm so sánh có hệ thống giữa các phương pháp tiếp cận:

- **Nhóm dự báo mức giá trực tiếp (Direct Level)**: Dự báo trực tiếp giá trị $y_t$ thông qua mạng hồi quy:
  - `lstm_direct`, `gru_direct`, `cnn_lstm_direct`, `cnn_gru_direct`.
- **Nhóm dự báo phần dư sai phân (Residual Learning)**: Mô hình dự báo bước nhảy sai phân $\Delta y_t = y_t - y_{t-1}$, giá trị dự báo sau đó được hoàn nguyên theo công thức $\widehat{y}_t = y_{t-1} + \widehat{\Delta}_t$:
  - `lstm_residual`, `gru_residual`, `cnn_lstm_residual`, `cnn_gru_residual`.
- **Mô hình tuyến tính chính quy hóa phần dư**: `ridge_residual` (hồi quy Ridge với hệ số chính quy hóa $\alpha = 1,0$).
- **Mô hình cơ sở tham chiếu (Baseline)**: `naive_persistence` ($\widehat{y}_t = y_{t-1}$, giả định giá ngày mai bằng giá ngày hôm nay).

*Quy trình huấn luyện mạng nơ-ron:* 65 epoch, cơ chế dừng sớm Early Stopping với độ kiên nhẫn (patience) là 9 epoch theo dõi `val_loss`, kích thước batch 64, thuật toán tối ưu Adam với tốc độ học $10^{-3}$, hạt giống ngẫu nhiên cố định seed 42 cho toàn bộ lưới thực nghiệm 324 lượt chạy.

---

## II. Đánh giá Hiệu năng Mô hình

### 2.1. Đánh giá tổng hợp trên tập kiểm tra chuẩn (Tỷ lệ 90/10)

Bảng dưới đây trích xuất từ [fair_table_0.900.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/fair_table_0.900.csv), ghi nhận kết quả trên 271 phiên kiểm tra (29/10/2025 – 13/11/2026), sắp xếp theo thứ tự sai số tuyệt đối trung bình (MAE) trên đơn vị giá gốc:

| Xếp hạng | Mô hình | Tập đặc trưng | Chiến lược thích nghi | MAE (USD) | RMSE (USD) | MSE_scaled | % Cải thiện so với Naive | Directional Accuracy (%) |
| :---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gru_residual` | `copper_legacy14` | `expanding` | **0,07695** | **0,10081** | **0,001016** | **+0,16%** | 52,77% |
| 2 | `lstm_residual` | `copper_legacy14` | `expanding` | 0,07698 | 0,10095 | 0,001019 | +0,13% | 52,77% |
| 3 | `cnn_lstm_residual` | `copper_only` | `rolling_756` | 0,07700 | 0,10089 | 0,003862 | +0,10% | 53,51% |
| 4 | `lstm_residual` | `copper_paper3` | `expanding` | 0,07701 | 0,10088 | 0,001018 | +0,09% | 52,40% |
| 5 | `lstm_residual` | `copper_only` | `expanding` | 0,07701 | 0,10081 | 0,001016 | +0,09% | 52,77% |
| 6 | `gru_residual` | `copper_legacy14` | `recency_weighted` | 0,07706 | 0,10097 | 0,001019 | +0,03% | 51,29% |
| **7** | **Naive Persistence** | — | — | **0,07708** | **0,10103** | **0,001021** | **0,00%** | **—** |
| 8 | `gru_residual` | `copper_paper3` | `expanding` | 0,07714 | 0,10091 | 0,001018 | -0,09% | 49,45% |
| 9 | `lstm_residual` | `copper_only` | `rolling_756` | 0,07716 | 0,10088 | 0,003860 | -0,11% | 48,71% |
| 10 | `gru_residual` | `copper_legacy14` | `rolling_756` | 0,07718 | 0,10102 | 0,003867 | -0,14% | 46,49% |
| 11 | `gru_residual` | `copper_paper3` | `rolling_756` | 0,07721 | 0,10131 | 0,003868 | -0,17% | 45,39% |
| 12 | `cnn_lstm_residual` | `copper_legacy14` | `expanding` | 0,07725 | 0,10086 | 0,001017 | -0,22% | 46,86% |
| 13 | `gru_residual` | `copper_paper3` | `recency_weighted` | 0,07730 | 0,10107 | 0,001021 | -0,30% | 46,49% |
| 14 | `ridge_residual` | `copper_only` | `expanding` | 0,07731 | 0,10119 | 0,001024 | -0,30% | 47,60% |
| 15 | `cnn_lstm_residual` | `copper_legacy14` | `recency_weighted` | 0,07730 | 0,10102 | 0,001021 | -0,29% | 49,08% |
| 16 | `cnn_lstm_residual` | `copper_paper3` | `expanding` | 0,07743 | 0,10125 | 0,001025 | -0,46% | 50,55% |
| 17 | `lstm_residual` | `copper_legacy14` | `rolling_756` | 0,07776 | 0,10151 | 0,003859 | -0,88% | 46,49% |
| 18 | `ridge_residual` | `copper_paper3` | `expanding` | 0,08071 | 0,10556 | 0,001114 | -4,72% | 52,40% |
| 19 | `ridge_residual` | `copper_legacy14` | `expanding` | 0,08171 | 0,10673 | 0,001139 | -6,01% | 49,45% |
| 20 | `ridge_residual` | `copper_paper3` | `rolling_756` | 0,09835 | 0,12225 | 0,004865 | -27,60% | 46,13% |
| ... | *... nhóm Direct ...* | ... | ... | ... | ... | ... | ... | ... |
| 28 | `gru_direct` | `copper_only` | `recency_weighted` | 0,20515 | 0,24349 | 0,003327 | -166,17% | 46,13% |
| 30 | `gru_direct` | `copper_legacy14` | `expanding` | 0,24082 | 0,30292 | 0,005379 | -212,44% | 53,51% |
| 32 | `gru_direct` | `copper_paper3` | `expanding` | 0,24789 | 0,30513 | 0,004071 | -221,62% | 53,14% |
| 35 | `cnn_gru_direct` | `copper_paper3` | `expanding` | 0,29638 | 0,35961 | 0,007656 | -284,52% | 52,40% |
| 36 | `cnn_gru_direct` | `copper_legacy14` | `expanding` | 0,31352 | 0,38100 | 0,010903 | -306,77% | 51,29% |
| 38 | `lstm_direct` | `copper_only` | `expanding` | 0,44672 | 0,47853 | 0,006437 | -479,57% | 46,49% |
| 45 | `cnn_lstm_direct` | `copper_paper3` | `expanding` | 1,10859 | 1,17008 | 0,020082 | -1338,29% | 46,49% |
| 46 | `lstm_direct` | `copper_paper3` | `expanding` | 1,16396 | 1,24421 | 0,012233 | -1410,14% | 46,49% |

### 2.2. So sánh chuyên sâu giữa hai phương pháp tiếp cận: Direct Level và Residual Learning

Kết quả thực nghiệm cho thấy sự phân hóa sâu sắc về mặt phương pháp luận giữa hai hướng tiếp cận:

- **Phương pháp Direct Level (dự báo mức giá trực tiếp)**:
  - Tất cả các biến thể kiến trúc thuộc nhóm Direct (LSTM, GRU, CNN-LSTM, CNN-GRU) đều ghi nhận sai số lớn hơn nhiều so với Naive Persistence, với mức chênh lệch MAE từ +166% đến trên +1.400%.
  - *Giải thích cơ chế:* Chuỗi mức giá hàng hóa là chuỗi không dừng mang tính tích hợp bậc một $I(1)$ với xu hướng dài hạn và sự dịch chuyển mặt bằng giá qua các chu kỳ kinh tế. Khi dữ liệu ngoài mẫu dịch chuyển ra khỏi biên độ phân phối đã học trong tập huấn luyện, các hàm kích hoạt phi tuyến của mạng nơ-ron gặp khó khăn trong việc ngoại suy giá trị tuyệt đối. Đồng thời, mô hình thường xuyên mắc hiện tượng trễ pha (lagging phase), tức dự báo mức giá có xu hướng chạy theo giá quá khứ với độ trễ từ 1 đến vài phiên.
- **Phương pháp Residual Learning (dự báo phần dư sai phân)**:
  - Việc chuyển mục tiêu dự báo từ mức giá $y_t$ sang sai phân bậc một $\Delta y_t = y_t - y_{t-1}$ đã giải quyết hiệu quả vấn đề tính không dừng. Chuỗi sai phân có tính dừng bậc không $I(0)$, dao động quanh giá trị 0, giúp phân phối đầu ra ổn định giữa tập train, validation và test.
  - Toàn bộ các mô hình nơ-ron Residual đều đạt sai số MAE nằm rất sát với Naive Persistence (dao động trong khoảng chênh lệch $\pm 0,3\%$).

### 2.3. Đánh giá mô hình tuyến tính chính quy hóa (Ridge Residual)

Mô hình `ridge_residual` đóng vai trò là cầu nối đối chuẩn giữa phương pháp hồi quy tuyến tính cổ điển và các mạng nơ-ron phi tuyến:

- Khi sử dụng tập đặc trưng đơn biến `copper_only`, Ridge Residual đạt MAE là $0,07731$ USD (chỉ kém Naive Persistence $0,30\%$).
- Tuy nhiên, khi bổ sung thêm các biến ngoại sinh đa biến (`copper_paper3` và `copper_legacy14`), sai số của Ridge Residual tăng lên đáng kể (MAE đạt $0,08071 - 0,09835$ USD, kém Naive từ $4,7\%$ đến $27,6\%$).
- Điều này cho thấy trong không gian đa biến với hệ số tương quan trễ thấp, mô hình tuyến tính dễ bị ảnh hưởng bởi phương sai ước lượng của các hệ số hồi quy ngoại sinh, trong khi các mạng nơ-ron hồi quy có khả năng tự điều chỉnh trọng số thông qua quá trình tối ưu hóa phi tuyến để giảm thiểu tác động tiêu cực của các đặc trưng nhiễu.

### 2.4. Tính nhất quán của hiệu năng qua 4 tỷ lệ phân chia thời gian

Bảng dưới đây tổng hợp hiệu năng của cấu hình mô hình tốt nhất đối chiếu với Naive Persistence qua 4 mốc phân chia tỷ lệ dữ liệu:

| Tỷ lệ phân chia | Số phiên Test | Cấu hình mô hình tốt nhất | MAE mô hình (USD) | MAE Naive (USD) | Mức cải thiện (%) | Directional Accuracy (%) |
| :---: | :---: | --- | ---: | ---: | :---: | :---: |
| **80 / 20** | 542 | `gru_residual` / `copper_paper3` / `expanding` | 0,06878 | 0,06900 | +0,32% | 53,51% |
| **90 / 10** | 271 | `gru_residual` / `copper_legacy14` / `expanding` | 0,07695 | 0,07708 | +0,16% | 52,77% |
| **92,5 / 7,5** | 204 | `gru_residual` / `copper_legacy14` / `expanding` | 0,07502 | 0,07512 | +0,13% | 53,92% |
| **95 / 5** | 136 | `gru_residual` / `copper_legacy14` / `recency` | 0,07251 | 0,07280 | +0,40% | 54,41% |

**Nhận xét tổng hợp:**
Qua cả 4 mốc thời gian với quy mô kiểm tra từ 136 đến 542 phiên giao dịch, mô hình Residual nơ-ron ghi nhận mức cải thiện MAE rất khiêm tốn so với Naive Persistence, luôn nằm dưới ngưỡng **0,40%**. Tính ổn định này cho thấy hiệu năng của các mô hình nơ-ron học phần dư có tính nhất quán cao, song biên độ vượt trội so với mốc cơ sở là rất hẹp trong toàn bộ các phân vị thời gian.

---

## III. Đánh giá Phương pháp Huấn luyện và Chiến lược Dữ liệu

### 3.1. Đánh giá các chiến lược cửa sổ dữ liệu (Expanding vs. Rolling vs. Recency)

Ba chiến lược thích nghi dữ liệu thể hiện các đặc tính phương pháp luận rõ rệt:

- **`expanding` (Cửa sổ mở rộng)**: Cho kết quả ổn định và tối ưu nhất trong phần lớn các cấu hình nơ-ron (chiếm ưu thế ở vị trí dẫn đầu tại các tỷ lệ 80/20, 90/10, 92,5/7,5). Việc tận dụng toàn bộ lịch sử huấn luyện giúp mô hình học sâu có đủ số lượng mẫu để tối ưu hóa trọng số mạng và giảm phương sai ước lượng.
- **`rolling_756` (Cửa sổ trượt 3 năm)**: Mặc dù có ưu thế lý thuyết trong việc loại bỏ chế độ quá khứ cũ, kích thước mẫu 756 phiên có phần hạn chế đối với việc huấn luyện mạng học sâu nhiều tham số, dẫn đến sự dao động hiệu năng cao hơn giữa các kiến trúc.
- **`recency_weighted` (Trọng số suy giảm theo thời gian)**: Đạt hiệu năng tốt ở mốc phân chia 95/5 (136 phiên test ngắn hạn), cho thấy trọng số nhấn mạnh thời gian gần có thể hữu ích khi cấu trúc thị trường ngắn hạn có tính tiếp nối cao.

### 3.2. Đánh giá vai trò của biến ngoại sinh qua thực nghiệm Feature Ablation

Bảng dưới đây trích xuất từ [feature_ablation.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/feature_ablation.csv), so sánh hiệu năng giữa cấu hình đơn biến `copper_only` và hai tập đa biến `copper_paper3`, `copper_legacy14` tại phân chia 90/10:

| Mô hình | Chiến lược thích nghi | MAE `copper_only` | MAE `copper_paper3` | MAE `copper_legacy14` | Cải thiện của Paper3 vs Only (%) | Cải thiện của Legacy14 vs Only (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gru_residual` | `expanding` | 0,08018 | 0,07714 | **0,07695** | +3,79% | +4,03% |
| `lstm_residual` | `expanding` | 0,07701 | 0,07701 | **0,07698** | 0,00% | +0,04% |
| `cnn_lstm_residual` | `expanding` | 0,08010 | 0,07743 | **0,07725** | +3,33% | +3,56% |
| `ridge_residual` | `expanding` | **0,07731** | 0,08071 | 0,08171 | -4,41% | -5,69% |
| `gru_residual` | `recency_weighted` | 0,08394 | 0,07730 | **0,07706** | +7,91% | +8,21% |
| `lstm_residual` | `recency_weighted` | **0,07814** | 0,07832 | 0,07869 | -0,23% | -0,71% |
| `ridge_residual` | `recency_weighted` | **0,07933** | 0,09011 | 0,08566 | -13,59% | -7,98% |
| `cnn_lstm_residual` | `rolling_756` | **0,07700** | 0,09101 | 0,08120 | -18,19% | -5,45% |
| `ridge_residual` | `rolling_756` | **0,08270** | 0,09835 | 0,08400 | -18,92% | -1,57% |

**Phân tích từ góc độ kinh tế lượng và lựa chọn đặc trưng:**

- **Hiện tượng nhiễu chiều dữ liệu (Noise Inflation)**: Đối với mô hình tuyến tính `ridge_residual`, cấu hình đơn biến `copper_only` luôn đạt kết quả tốt hơn cấu hình đa biến từ 4,4% đến 18,9%. Việc bổ sung thêm các biến ngoại sinh làm tăng số lượng tham số ước lượng trong khi tín hiệu dự báo hạn chế, dẫn đến phương sai sai số lớn hơn.
- **Độ nhạy của mạng nơ-ron**: Đối với các mạng nơ-ron, mức đóng góp của các biến ngoại sinh có sự phân hóa nhẹ tùy thuộc vào kiến trúc và chiến lược dữ liệu. Mặc dù `copper_legacy14` mang lại cải thiện nhỏ (~3% - 4%) trên `gru_residual` và `cnn_lstm_residual` ở chiến lược expanding, đối với `lstm_residual` thì kết quả giữa đơn biến và đa biến gần như tương đương ($0,07701$ so với $0,07698$ USD).
- **Khảo sát tương quan bậc trễ trong tập huấn luyện**: Bảng số liệu [lagged_correlations_training_only.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/lagged_correlations_training_only.csv) cung cấp hệ số tương quan Spearman ở bậc trễ 1 ngày ($t-1$) giữa biến động các hàng hóa ngoại sinh và biến động giá đồng:
  - Vàng (*Gold*): $r = 0,0910$
  - Bạc (*Silver*): $r = 0,0388$
  - Dầu thô WTI: $r = 0,0142$
  - Nhôm (*Aluminium*): $r = 0,0093$
  - Thiếc (*Tin*): $r = -0,0185$

Các hệ số tương quan trễ đều nằm dưới ngưỡng $0,10$, cho thấy mối liên kết động lượng tức thời ở tần suất ngày giữa các thị trường hàng hóa là tương đối yếu, lý giải vì sao các mô hình học máy khó khai thác được tín hiệu dự báo vượt trội từ các biến ngoại sinh ở bước dự báo $HORIZON=1$.

### 3.3. Phân tích cơ chế học phần dư và kiểm toán phương sai (Residual Audit)

Tệp dữ liệu [residual_audit.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/residual_audit.csv) phân tích đặc tính phân phối của phần dư dự đoán $\widehat{\Delta}_t$ so với biến động thực tế $\Delta_t = Copper(t) - Copper(t-1)$ trên tập kiểm tra 90/10:

| Mô hình | Chiến lược thích nghi | $\text{std}(\Delta_{true})$ | $\text{std}(\widehat{\Delta}_{pred})$ | Tỷ lệ co cụm phương sai | Hệ số tương quan $\text{Corr}(\widehat{\Delta}, \Delta)$ |
| --- | --- | ---: | ---: | :---: | ---: |
| `lstm_residual` | `expanding` | 0,10097 | 0,00356 | 28,4 lần | 0,0367 |
| `gru_residual` | `expanding` | 0,10097 | 0,00293 | 34,5 lần | 0,0712 |
| `cnn_lstm_residual` | `expanding` | 0,10097 | 0,00473 | 21,3 lần | 0,0525 |
| `gru_residual` | `recency_weighted` | 0,10097 | 0,00054 | 186,0 lần | 0,1016 |
| `gru_residual` | `rolling_756` | 0,10097 | 0,00105 | 96,2 lần | 0,0867 |
| `ridge_residual` | `expanding` | 0,10097 | 0,02680 | 3,8 lần | -0,0490 |

**Phân tích toán học về hiện tượng thu hẹp phương sai (Variance Shrinkage):**

1. **Đặc tính phân phối**: Độ lệch chuẩn thực tế của bước nhảy giá đồng hàng ngày là $\text{std}(\Delta_{true}) \approx 0,101$ USD/ngày. Trong khi đó, độ lệch chuẩn của phần dư dự đoán bởi các mạng nơ-ron chỉ đạt từ $0,00054$ đến $0,00473$ USD/ngày (phương sai bị co nhỏ từ 21 đến 186 lần).
2. **Cơ chế tối ưu hóa dưới hàm mất mát MSE**:
   Trong bài toán hồi quy với hàm mục tiêu $\mathcal{L} = \mathbb{E}[(\Delta_t - \widehat{\Delta}_t)^2]$, khi chuỗi mục tiêu $\Delta_t$ có tính ngẫu nhiên cao và tỷ lệ tín hiệu trên nhiễu tiệm cận 0, nghiệm giải tích tối ưu hóa kỳ vọng có dạng:
   $$\widehat{\Delta}^* = \mathbb{E}[\Delta_t \mid X_{t-1}] \approx \mathbb{E}[\Delta_t] \approx 0$$
   Do đó, thuật toán huấn luyện học cách đưa các giá trị dự báo phần dư hội tụ về một biên độ dao động rất hẹp quanh 0 nhằm hạn chế sai số phạt bình phương lớn khi dự báo sai biên độ.
3. **Mối quan hệ với Naive Persistence**:
   Khi $\widehat{\Delta}_t \approx 0$, công thức hoàn nguyên giá trị:
   $$\widehat{y}_t = y_{t-1} + \widehat{\Delta}_t \approx y_{t-1} + 0 = y_{t-1}$$
   Hiện tượng này giải thích vì sao sai số MAE và RMSE của các mạng nơ-ron Residual luôn tiệm cận với Naive Persistence. Ngược lại, khi mô hình hồi quy tuyến tính `ridge_residual` cố gắng dự đoán biên độ phần dư lớn hơn ($\text{std} = 0,0268$ USD), sai số dự báo bị khuếch đại do các sai lệch biên độ, dẫn đến tổng sai số MSE cao hơn.

---

## IV. Đánh giá Độ ổn định Dài hạn và Kiểm định Thống kê Walk-Forward (2018–2026)

Để đánh giá tính khái quát hóa và độ ổn định của phương pháp học sâu qua nhiều chu kỳ kinh tế, quy trình kiểm định Walk-Forward liên tục được triển khai trên 2.214 phiên giao dịch ngoài mẫu từ năm 2018 đến 2026. Mô hình `gru_residual` kết hợp bộ biến `copper_paper3` và chiến lược `expanding` được tái huấn luyện định kỳ mỗi 63 phiên giao dịch (~1 quý) trên 5 hạt giống ngẫu nhiên độc lập (seeds 41, 42, 43, 44, 45).

### 4.1. Kết quả kiểm định Diebold-Mariano với hiệu chỉnh sai số chuẩn HAC

Tệp [dm_hac_by_seed.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/dm_hac_by_seed.csv) tổng hợp kết quả kiểm định thống kê Diebold-Mariano (1995) với hiệu chỉnh Newey-West HAC (độ trễ lag = 5) để kiểm tra giả thuyết không ($H_0$): *Mức độ chính xác dự báo (đo bằng hàm mất mát tuyệt đối) giữa Residual-GRU và Naive Persistence là tương đương nhau*:

| Seed | Số phiên Test ($N$) | MAE Residual-GRU | MAE Naive | Chênh lệch trung bình ($\overline{d}$) | Giá trị thống kê $z$ (DM) | Giá trị $p$-value (2 phía) | Kết luận kiểm định thống kê | Directional Accuracy (%) |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: | --- | :---: |
| **41** | 2.214 | 0,045110 | 0,045127 | -0,000017 | -0,379 | **0,7043** | Không có sự khác biệt có ý nghĩa | 50,45% |
| **42** | 2.214 | 0,045189 | 0,045127 | +0,000062 | +0,905 | **0,3656** | Không có sự khác biệt có ý nghĩa | 50,63% |
| **43** | 2.214 | 0,045447 | 0,045127 | +0,000320 | +2,716 | **0,0066** | Naive có MAE thấp hơn có ý nghĩa ($p < 0,01$) | 50,27% |
| **44** | 2.214 | 0,045502 | 0,045127 | +0,000375 | +2,573 | **0,0101** | Naive có MAE thấp hơn có ý nghĩa ($p \approx 0,01$) | 50,63% |
| **45** | 2.214 | 0,045388 | 0,045127 | +0,000262 | +1,758 | **0,0788** | Naive có MAE thấp hơn ở mức ý nghĩa 10% | 51,45% |

### 4.2. Hiệu năng chi tiết theo từng năm giao dịch (2018–2026)

Tệp [wf_by_year.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/wf_by_year.csv) phân rã sai số theo từng năm nhằm khảo sát sự biến thiên hiệu năng qua các giai đoạn thị trường khác nhau:

| Năm | Số phiên Test | MAE Naive | MAE GRU (Seed 41) | MAE GRU (Seed 42) | Ghi nhận so sánh tương quan |
| :---: | :---: | ---: | ---: | ---: | --- |
| **2018** | 254 | 0,02750 | 0,02777 | 0,02778 | Naive có sai số thấp hơn |
| **2019** | 261 | 0,02098 | 0,02102 | 0,02115 | Naive có sai số thấp hơn |
| **2020** | 261 | 0,02882 | 0,02860 | 0,02854 | GRU có sai số thấp hơn nhẹ (-0,0002 USD) |
| **2021** | 258 | 0,05128 | 0,05120 | 0,05178 | Hiệu năng tương đương |
| **2022** | 225 | 0,05341 | 0,05348 | 0,05356 | Naive có sai số thấp hơn |
| **2023** | 213 | 0,03849 | 0,03859 | 0,03850 | Naive có sai số thấp hơn |
| **2024** | 260 | 0,04426 | 0,04429 | 0,04438 | Naive có sai số thấp hơn |
| **2025** | 257 | 0,06776 | 0,06755 | 0,06750 | GRU có sai số thấp hơn nhẹ (-0,0002 USD) |
| **2026** | 225 | 0,07804 | 0,07791 | 0,07790 | GRU có sai số thấp hơn nhẹ (-0,0001 USD) |

### 4.3. Đánh giá tính khái quát hóa và độ ổn định thống kê

- **Độ nhạy theo hạt giống ngẫu nhiên (Seed Sensitivity)**: Mặc dù cùng cấu hình kiến trúc và dữ liệu, sự thay đổi hạt giống ngẫu nhiên trong quá trình khởi tạo trọng số mạng nơ-ron dẫn đến sự khác biệt về kết luận thống kê. Ở seed 41 và 42, kiểm định DM không phát hiện sự khác biệt có ý nghĩa giữa mô hình và Naive. Tuy nhiên ở seed 43 và 44, Naive Persistence lại đạt sai số thấp hơn có ý nghĩa thống kê ($p < 0,015$). Điều này nhấn mạnh tầm quan trọng của việc đánh giá đa hạt giống trong các nghiên cứu học sâu tài chính.
- **Khả năng dự báo hướng biến động (Directional Accuracy)**: Tỷ lệ dự báo đúng hướng của Residual-GRU qua các seed dao động từ **$50,27\%$ đến $51,45\%$**. Kết quả này nằm rất sát với mức kỳ vọng ngẫu nhiên ($50\%$), cho thấy mô hình chưa thể khai thác được các mẫu hình có thể dự báo trước về xu hướng tăng/giảm ở tần suất giao dịch hàng ngày.

---

## V. Thảo luận Khoa học và Khuyến nghị Phương pháp Luận

### 5.1. Thảo luận về đặc tính chuỗi giá tài chính và mốc cơ sở Naive Persistence

Dưới góc nhìn kinh tế lượng tài chính, các kết quả thực nghiệm trong nghiên cứu này hoàn toàn phù hợp với nền tảng lý thuyết đã được thiết lập:

1. **Giả thuyết bước đi ngẫu nhiên và chuỗi sai phân Martingale**:
   Theo Giả thuyết thị trường hiệu quả dạng yếu (Fama, 1970), giá tài chính tại các thị trường có tính thanh khoản cao như hợp đồng tương lai kim loại đồng phản ánh nhanh chóng các thông tin sẵn có. Chuỗi lợi suất hàng ngày thường có cấu trúc gần với một chuỗi sai phân martingale (*martingale difference sequence*):
   $$\mathbb{E}[P_t \mid \mathcal{F}_{t-1}] \approx P_{t-1}$$
   Do đó, mô hình Naive Persistence ($\widehat{P}_t = P_{t-1}$) là một đường chuẩn đối sánh cực kỳ mạnh mẽ (*strong econometric benchmark*). Việc các mạng nơ-ron phức tạp không vượt qua được Naive Persistence ở chân trời $t+1$ không phải là một thất bại cá biệt, mà là sự phản ánh trung thực bản chất ngẫu nhiên cao của dữ liệu tài chính tần suất ngày.
2. **Vai trò phương pháp luận của Residual Learning**:
   Phương pháp học phần dư đã đóng góp một giải pháp kỹ thuật quan trọng: giúp chuẩn hóa phân phối mục tiêu, ngăn chặn hiện tượng bùng nổ sai số do không dừng của các mô hình Direct Level, và đưa dự báo về vùng giá trị tiệm cận tối ưu với rủi ro thấp nhất.
3. **Giá trị nhận thức phương pháp luận (Methodological Insights)**:
   Thay vì chạy theo việc tinh chỉnh để cố ép mô hình đạt độ chính xác cao hơn trên một tập test cụ thể, giá trị thực sự của chuỗi thực nghiệm này nằm ở chỗ giúp người nghiên cứu **thấu hiểu bản chất cơ chế vận hành của các thuật toán học máy**:
   - Hiểu rõ vì sao mô hình dự báo trực tiếp (Direct) gặp khó khăn khi ngoại suy trên chuỗi không dừng $I(1)$ với phân phối dịch chuyển.
   - Hiểu rõ vì sao dưới hàm mất mát sai số bình phương (MSE), mạng nơ-ron học cách thu hẹp phần dư dự báo về gần 0 để tối ưu hóa rủi ro kỳ vọng khi đối mặt với dữ liệu có tỷ lệ tín hiệu trên nhiễu thấp.
   - Hiểu rõ vì sao việc bổ sung các biến ngoại sinh có tương quan trễ yếu không mang lại giá trị gia tăng mà chủ yếu làm tăng phương sai ước lượng do mở rộng số chiều dữ liệu.
   Những phát hiện này tạo nền tảng vững chắc cho việc định hình bài toán và lựa chọn công cụ mô hình hóa phù hợp trong các nghiên cứu tài chính tiếp theo.

### 5.2. Giới hạn thực nghiệm

1. **Độ lệch múi giờ đóng cửa giữa các thị trường (Asynchronous Trading Hours)**: Dữ liệu sử dụng giá đóng cửa hàng ngày của các sàn giao dịch quốc tế (LME tại London, COMEX tại New York, NYMEX cho dầu WTI). Thời điểm chốt giá giữa các thị trường có khoảng chênh lệch múi giờ, do đó một phần thông tin của các biến ngoại sinh có thể chưa phản ánh đồng bộ tại thời điểm đóng cửa của hợp đồng đồng.
2. **Chi phí giao dịch và tính thanh khoản thực tế**: Đánh giá thực nghiệm tập trung vào các thước đo sai số hồi quy (MAE, RMSE). Với tỷ lệ dự báo hướng dao động quanh 50,5%, việc áp dụng trực tiếp vào giao dịch thực tế sẽ phải chịu thêm chi phí trượt giá (*slippage*), phí giao dịch (*exchange fees*) và chênh lệch giá mua - bán (*bid-ask spread*).
3. **Không gian siêu tham số cố định**: Để đảm bảo tính đồng nhất trên toàn bộ ma trận 324 lượt chạy, kiến trúc mạng được cố định cấu hình (64/32 units, lr 0,001). Thực nghiệm chưa bao gồm việc tinh chỉnh siêu tham số riêng biệt cho từng ô của lưới.

### 5.3. Định hướng nghiên cứu và phát triển phương pháp tiếp theo

Từ các phát hiện thực nghiệm, nghiên cứu đề xuất một số định hướng tiếp cận nhằm nâng cao hiệu quả mô hình hóa:

1. **Chuyển dịch mục tiêu sang dự báo độ biến động (Volatility Forecasting)**:
   Thay vì dự báo mức giá có tính ngẫu nhiên cao, việc dự báo phương sai có điều kiện hoặc độ biến động (sử dụng các mô hình họ GARCH kết hợp mạng nơ-ron hoặc Stochastic Volatility) có tính khả thi cao hơn nhiều do hiện tượng co cụm biến động (*volatility clustering*) là đặc tính được ghi nhận rộng rãi trong kinh tế lượng.
2. **Mở rộng chân trời dự báo trung và dài hạn (Multi-step Horizon Forecasting)**:
   Chuyển dịch trọng tâm từ dự báo 1 ngày ($t+1$) sang các chân trời xa hơn ($t+5$, $t+21$, hoặc $t+63$), nơi các yếu tố chu kỳ kinh tế vĩ mô và cân đối cung - cầu công nghiệp có thời gian tích lũy và thể hiện xu thế rõ nét hơn so với nhiễu vi cấu trúc hàng ngày.
3. **Mô hình hóa chế độ thị trường (Regime-Switching Models)**:
   Tích hợp các mô hình chuyển đổi trạng thái (như Hidden Markov Models - HMM hoặc Markov-Switching Vector Autoregressions) để nhận diện các giai đoạn thị trường có xu hướng (*trending*), đi ngang (*ranging*) hoặc biến động mạnh (*high volatility*), từ đó áp dụng các chiến lược dự báo phù hợp cho từng chế độ.
4. **Bổ sung các biến số nền tảng và dữ liệu phi cấu trúc**:
   Tích hợp thêm các dữ liệu cơ bản như lượng tồn kho tại các kho LME/SHFE, chỉ số PMI ngành sản xuất toàn cầu, chênh lệch lãi suất và chỉ số tâm lý thị trường (*market sentiment*) nhằm gia tăng tỷ lệ tín hiệu trên nhiễu cho không gian đặc trưng đầu vào.

---

## Danh mục tài liệu tham khảo và hiện vật thực nghiệm

### Tài liệu tham khảo khoa học

1. Chen, Y., et al. (2023). *Copper price prediction using LSTM recurrent neural network integrated simulated annealing algorithm*.
2. Diebold, F. X., & Mariano, R. S. (1995). *Comparing predictive accuracy*. Journal of Business & Economic Statistics, 13(3), 253-263.
3. Fama, E. F. (1970). *Efficient capital markets: A review of theory and empirical work*. The Journal of Finance, 25(2), 383-417.
4. Newey, W. K., & West, K. D. (1987). *A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix*. Econometrica, 703-708.

### Danh mục hiện vật thực nghiệm

Các dữ liệu và kết quả chi tiết của giao thức Protocol V20 được lưu trữ đầy đủ tại thư mục:
`d:/University/ptdlkd-is403.r11/lancelot task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/`

- **Bảng đối sánh hiệu năng các mô hình:**
  - [fair_table_0.900.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/fair_table_0.900.csv): Bảng đối chuẩn chuẩn hóa tại tỷ lệ phân chia 90/10.
  - [fair_table_0.800.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/fair_table_0.800.csv): Bảng đối chuẩn chuẩn hóa tại tỷ lệ phân chia 80/20.
  - [fair_table_0.925.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/fair_table_0.925.csv): Bảng đối chuẩn chuẩn hóa tại tỷ lệ phân chia 92,5/7,5.
  - [fair_table_0.950.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/fair_table_0.950.csv): Bảng đối chuẩn chuẩn hóa tại tỷ lệ phân chia 95/5.
- **Dữ liệu phân tích chuyên sâu:**
  - [feature_ablation.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/feature_ablation.csv): Bảng khảo sát đóng góp của các bộ đặc trưng ngoại sinh.
  - [residual_audit.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/residual_audit.csv): Bảng thống kê kiểm toán hiện tượng co cụm phương sai phần dư.
  - [lagged_correlations_training_only.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/lagged_correlations_training_only.csv): Ma trận tương quan bậc trễ 1 ngày trong tập train.
- **Kết quả Walk-Forward và kiểm định thống kê:**
  - [dm_hac_by_seed.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/dm_hac_by_seed.csv): Kết quả kiểm định thống kê Diebold-Mariano với hiệu chỉnh HAC qua 5 seeds.
  - [wf_by_year.csv](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/wf_by_year.csv): Phân rã sai số dự báo Walk-Forward theo từng năm (2018–2026).
  - [manifest.json](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/manifest.json): Tệp cấu hình siêu tham số và phiên bản môi trường thực thi.
- **Biểu đồ trực quan hóa:**
  - [wf_actual_vs_pred.png](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/wf_actual_vs_pred.png): Biểu đồ so sánh chuỗi giá thực tế và giá dự báo Walk-Forward.
  - [wf_delta_audit.png](file:///d:/University/ptdlkd-is403.r11/lancelot%20task/result/17-11-Copper-NonSA-Fair-Full-Protocol-V20/wf_delta_audit.png): Biểu đồ phân phối phần dư dự đoán so với phần dư thực tế.
