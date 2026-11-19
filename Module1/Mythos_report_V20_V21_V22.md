# TÁI HIỆN VÀ ĐỐI CHIẾU CÁC PHƯƠNG PHÁP HỌC MÁY, HỌC SÂU TRONG DỰ BÁO GIÁ ĐỒNG MỘT PHIÊN TRƯỚC

**Tác giả thực nghiệm:** Viên Đăng Khoa  
**Học phần:** Phân tích dữ liệu kinh doanh (IS403)  
**Phiên bản báo cáo:** Hợp nhất V20–V21 và đối chiếu V22 WTI; cập nhật 19/11/2026  
**Tình trạng:** Báo cáo kết quả thực nghiệm, không phải bài báo đã bình duyệt.

## Tóm tắt (Abstract)

Báo cáo tái hiện có kiểm soát cấu hình SA-LSTM theo thông số được trình bày trong Chen và cộng sự (2023), đồng thời đối chiếu với LSTM, GRU, SimpleRNN, CNN-LSTM, CNN-GRU, Ridge Residual và Naive Persistence. Trên tập dữ liệu 2.710 phiên giao dịch từ 13/01/2016 đến 13/11/2026, toàn bộ thí nghiệm sử dụng `HORIZON=1`, `LOOKBACK=22`; 3 chiến lược là Expanding, Rolling 756 và Recency Weighting. V20 có 360 lượt đánh giá; V21 gốc có 100 lượt; phần bổ sung 80/20 thêm 60 lượt (tổng 520 lượt đánh giá, trong đó 452 lượt fitted và 68 lượt Naive). Kết quả 90/10 cho thấy Residual-GRU/V20 với Copper + Legacy14 có MAE 0,076951 so với Naive 0,077077 (+0,163% về MAE), nhưng walk-forward nhiều seeds không xác nhận cải thiện ổn định có ý nghĩa thống kê. Trong V21, các mạng dự báo mức giá trực tiếp đều kém Naive trên tập holdout được thử; thí nghiệm bỏ/thêm lịch sử Copper có tác động không nhất quán. Giá trị chính là hiểu rõ thiết kế đầu vào, target, học tham số, SA hyperparameter search, các lỗi dự báo và giới hạn suy luận, thay vì chỉ xếp hạng độ chính xác.

**Bổ sung V22 (19/11/2026):** Kiểm tra bộ biến Copper/Aluminium/Tin/WTI trên 120 đánh giá Non-SA và 30 đánh giá SA-LSTM, với 16 trial SA tìm tham số mới. Báo cáo bổ sung bảng xếp hạng fitted models (không tính Naive) theo tỷ lệ train/test, so sánh cùng giao thức V20/V21 và đánh giá walk-forward Non-SA; các kết quả mới trình bày chi tiết ở §V-A. Không có bằng chứng để khẳng định biến WTI luôn cải thiện dự báo.

**Từ khóa:** Copper forecasting; LSTM; Simulated Annealing; GRU; Residual Learning; Spearman; feature ablation; walk-forward; Diebold–Mariano.

## I. Giới thiệu (Introduction)

Đồng là vật liệu quan trọng trong xây dựng, điện khí hóa và công nghiệp chế tạo. Dự báo mức giá ngắn hạn giúp đặt câu hỏi về giá trị của lịch sử hàng hóa và tính ứng dụng của phương pháp học sâu, nhưng dao động từng phiên có nhiều nhiễu. Đề tài lấy công trình *Copper price prediction using LSTM recurrent neural network integrated simulated annealing algorithm* (Chen et al., 2023) làm nghiên cứu nền: lựa chọn biến ngoại sinh bằng Spearman, sử dụng LSTM và tìm siêu tham số bằng Simulated Annealing. Do bài báo gốc không công bố đầy đủ code/định nghĩa lookback, Dense head và forecast origin, đây là **tái hiện theo thông số có công bố, có ghi nhận giả định**, không được xem là reproduction tuyệt đối.

Mục tiêu không phải chứng minh mạng phức tạp luôn tốt hơn. Bốn câu hỏi chính: (RQ1) cấu hình LSTM/SA-LSTM tái hiện có hành vi dự báo như thế nào; (RQ2) biểu diễn target theo mức giá hay chênh lệch giá ảnh hưởng đến quá trình học ra sao; (RQ3) lịch sử Copper và các biến ngoại sinh bổ sung tín hiệu gì; (RQ4) Expanding/Rolling/Recency và đánh giá nhiều thời điểm/seeds dẫn đến kết luận đáng tin cậy đến đâu.

**Phạm vi thời gian cần làm rõ:** Người thực hiện đề xuất phạm vi từ 01/01/2016 đến cuối tháng 10/2026. Tuy nhiên **các file ZIP thực nghiệm thực tế** sử dụng 2.710 quan sát **13/01/2016–13/11/2026**. Báo cáo này giữ đúng thời gian gốc từ kết quả đã chạy; nếu muốn báo cáo đúng phạm vi đề xuất, cần cắt dữ liệu, chạy lại và cập nhật toàn bộ bảng. Không được sửa ngày bằng lời văn mà giữ nguyên số liệu.

## II. Tổng quan nghiên cứu và khoảng trống tái hiện (Related Work & Reproduction Gap)

Chen et al. (2023) sử dụng WTI, Gold, Silver làm ba biến ngoại sinh công bố trong mô hình SA-LSTM. Tài liệu nhóm cung cấp ghi nhận 2 tầng LSTM, dropout 0,2, batch size 64, 100 epochs, LR 0,00005 và số units 39/111; tác giả không mô tả đủ để suy ra chính xác toàn bộ preprocessing/lookback/Dense head. Trong nghiên cứu này, `paper_sa_lstm` là mô hình dựng theo các HP đã công bố; `reopt_sa_lstm` là LSTM có HP từ lần dò SA trước đó, **không phải đã chạy lại quá trình SA search trong V21**.

**Không nên suy luận:** (i) Chen đã thử Copper lag rồi loại bỏ vì không cần; (ii) ngưỡng Spearman cao tự động tạo dự báo tốt; hoặc (iii) SA-LSTM V21 yếu nghĩa là thuật toán SA không có giá trị. Những tuyên bố này không được kiểm chứng từ nguồn đang có.

## III. Dữ liệu, EDA và lựa chọn biến (Dataset & Exploratory Data Analysis)

### 3.1. Dữ liệu và tính toàn vẹn

- File nguồn theo manifest: `dataset_9_commodities_daily.csv`, SHA256 `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600`; hai bộ V20/V21 có fingerprint giống nhau.
- 2.710 hàng, Date tăng dần; kiểm tra chất lượng V20 không có missing trong 10 cột: Date, Copper, Aluminium, Nickel, Tin, Iron_Ore_62, Silver, Gold, WTI, Brent.
- `LOOKBACK=22` (một tháng giao dịch gần đúng); `HORIZON=1`, mẫu `X[t−22 : t−1]` dự báo `Copper(t)`; không có feature ngày `t` trong tensor. Các thị trường quốc tế có thể chênh lệch múi giờ, vì vậy kiểm toán `as-of availability` vẫn cần thực hiện.
- Các split V20: 80/20 (542 test, bắt đầu 07/10/2024), 90/10 (271 test, bắt đầu 29/10/2025), 92,5/7,5 (204 test, bắt đầu 30/01/2026), 95/5 (136 test, bắt đầu 07/05/2026); kết thúc 13/11/2026.

### 3.2. Cấu hình bộ biến — tên Legacy14 không có nghĩa là 14 biến

| Bộ đặc trưng | Input đầy đủ | Số biến | Dùng ở đâu |
|---|---|---:|---|
| Copper-only | Copper | 1 | V20 |
| Paper3 | WTI, Gold, Silver | 3 | V21 |
| Paper3 + Copper | Copper, WTI, Gold, Silver | 4 | V20/V21 |
| Legacy14 | Aluminium, Tin, Silver, Gold | 4 | V21 |
| Legacy14 + Copper | Copper, Aluminium, Tin, Silver, Gold | 5 | V20/V21 |

### 3.3. EDA: Spearman trên **mức giá**

| Biến      |   Spearman với Copper (giá mức) |
|:----------|--------------------------------:|
| WTI       |                          0.7711 |
| Gold      |                          0.7425 |
| Silver    |                          0.7424 |
| Aluminium |                          0.9279 |
| Tin       |                          0.8829 |

Spearman là tương quan **thứ hạng**, đánh giá quan hệ đơn điệu, không phải xác suất dự báo đúng và không suy ra quan hệ nhân quả. Mức tương quan giá cao có thể bị tác động bởi xu thế dài hạn chung; nếu lựa chọn features để dự báo Copper(t) từ dữ liệu đến t−1, cần khảo sát riêng returns/differences có độ trễ và không được lựa chọn features theo test.

### 3.4. EDA: Spearman giữa biến động hàng hóa trước một phiên và biến động Copper

| Biến      |   Pearson lag 1 |   Spearman lag 1 |   Số cặp |
|:----------|----------------:|-----------------:|---------:|
| WTI       |          0.0079 |           0.0142 |     2193 |
| Gold      |          0.064  |           0.091  |     2193 |
| Silver    |          0.0573 |           0.0388 |     2193 |
| Aluminium |          0.0202 |           0.0093 |     2193 |
| Tin       |         -0.0115 |          -0.0185 |     2193 |

Các hệ số tương quan trễ 1 nhỏ hơn 0,10 theo trị tuyệt đối. Điều này làm giảm kỳ vọng về **quan hệ đơn biến tức thời**, nhưng chưa loại trừ tương tác đa biến, quan hệ phi tuyến và độ trễ 2/5/10/22/50 phiên (đã được khảo sát trong `lagged_correlations_training_only.csv`).

### 3.5. Có nên bỏ một biến yếu nhất khỏi Legacy14?

**Chưa nên tự ý xóa.** “Paper3” có 3 ngoại sinh vì đây là **cấu hình được bài báo lựa chọn**, không phải quy tắc rằng mọi nghiên cứu chỉ được dùng 3 biến. “Legacy14” có 4 ngoại sinh vì nhóm muốn khảo sát tập kim loại khác. Nếu buộc thử một biến thể 3/4, cần chọn tiêu chí **trên train/validation** và ghi trước quy trình: (a) chọn feature theo Spearman trên biến đổi phù hợp và lag được phép; (b) so sánh `Legacy14-4` với 4 phép leave-one-feature-out; (c) giữ cố định kiến trúc, seed/epochs, split và cùng test timestamps; (d) ưu tiên nhiều folds hoặc validation chọn feature, báo cáo test cuối duy nhất. Không dùng quy tắc “Spearman > một ngưỡng cố định” nếu không chứng minh ngưỡng đó được định nghĩa bởi công trình gốc.

**Đối chứng đặc biệt:** Trong EDA mức giá, Silver (0,7424) hơi thấp hơn Gold (0,7425) và thấp hơn các biến khác, nhưng kết quả này **không đủ** để gọi Silver là biến kém hữu ích nhất trong dự báo. Dữ liệu lagged returns còn thể hiện quan hệ khác; khi cần làm ablation chọn biến, phải dùng hiệu năng validation chứ không nhìn riêng correlation level.

### 3.6. Đồ thị EDA đã có và cần nhúng vào bản PDF/Word cuối

ZIP V20 lưu `eda_levels_returns_volatility.png`, `eda_lagged_pearson.png`, `eda_lagged_spearman.png`; có thể trích trực tiếp làm Hình 1–3 khi xuất báo cáo. Việc nội suy/điền dữ liệu, mô tả nguồn giá và lịch đóng cửa các thị trường phải được dẫn ra từ pipeline nếu muốn kết luận dữ liệu khả dụng vào lúc dự báo.

## IV. Phương pháp và giải pháp thực nghiệm (Methodology / Proposed Solution)

### 4.1. Định nghĩa bài toán và tiền xử lý

Đầu vào `X_t ∈ R^(22×F)` gồm 22 phiên lịch sử; nếu batch size 64 thì tensor `64×22×F`. Với Direct, `ŷ_t=fθ(X_t)`; với Residual, nhãn `Δ_t=P_t−P_{t−1}`, mô hình học `Δ̂_t=fθ(X_t)`, sau đó `ŷ_t=P_{t−1}+Δ̂_t`. Naive `ŷ_t=P_{t−1}` tương đương dự đoán `Δ̂=0` và không cần fit. Các scaler V20/V21 được fit trên train trước validation, sau đó transform các tập còn lại; MAE/RMSE được tính trên giá gốc. `mse_scaled` không dùng xếp hạng giữa những scheme có scaler khác nhau.

### 4.2. Kiến trúc các mô hình — trình tự từng layer

Các layer dưới đây được dựng lại từ `model_layer_parameter_details_v20_v21.csv`, trong đó Dropout không có tham số học. Direct và Residual V20 **cùng kiến trúc và số tham số**; khác target và bước cộng lại mức giá.

| Mô hình | Kiến trúc theo thứ tự (minh họa F biến) | Output |
|---|---|---|
| LSTM V20 / LSTM baseline V21 | LSTM(64, return_sequences=True) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(1) | Direct hoặc Δ (V20) |
| GRU V20 | GRU(64) → Dropout → GRU(32) → Dropout → Dense(16) → Dense(1) | Direct hoặc Δ |
| CNN-LSTM V20 | Causal Conv1D(32, kernel=3) → MaxPooling1D(2) → LSTM(48) → Dropout → LSTM(24) → Dropout → Dense(16) → Dense(1) | Direct hoặc Δ |
| CNN-GRU V20 | Causal Conv1D(32, kernel=3) → MaxPooling1D(2) → GRU(48) → Dropout → GRU(24) → Dropout → Dense(16) → Dense(1) | Direct hoặc Δ |
| SimpleRNN V21 | SimpleRNN(64) → Dropout → SimpleRNN(32) → Dropout → Dense(16) → Dense(1) | Copper(t) |
| SA-LSTM Paper HP V21 | LSTM(39) → Dropout(0.2) → LSTM(111) → Dropout(0.2) → Dense(32, ReLU) → Dense(1, linear) | Copper(t) |
| SA-LSTM SA cache V21 (Paper3) | LSTM(22) → Dropout(0.2) → LSTM(91) → Dropout(0.2) → Dense(32, ReLU) → Dense(1) | Copper(t) |
| SA-LSTM SA cache V21 (Legacy14) | LSTM(22) → Dropout(0.2) → LSTM(83) → Dropout(0.2) → Dense(32, ReLU) → Dense(1) | Copper(t) |
| Ridge Residual V20 | Flatten(22×F) → Ridge linear(α=1) | Δ |
| Naive | Không có mạng; lấy Copper(t−1) | Copper(t) |

**Lưu ý:** `Dense(32)` là giả định triển khai trong SA-LSTM; không phải cấu hình Dense được Chen et al. xác nhận. Dropout của V20 cần lấy chính xác từ notebook tương ứng khi mô tả tỷ lệ (bảng layer-audit chỉ cho biết layer loại Dropout, không cho biết rate).

### 4.3. Số tham số học được của mô hình (audit thực tế)

**V20** — Tổng tham số mô hình theo bộ biến (các nhánh Direct tương ứng với Residual cùng số tham số):

| Model             |   Copper+Legacy14 (F=5) |   Copper-only (F=1) |   Copper+Paper3 (F=4) |
|:------------------|------------------------:|--------------------:|----------------------:|
| cnn_gru_residual  |                   18065 |               17681 |                 17969 |
| cnn_lstm_residual |                   23489 |               23105 |                 23393 |
| gru_residual      |                   23585 |               22817 |                 23393 |
| lstm_residual     |                   30881 |               29857 |                 30625 |
| naive_persistence |                       0 |                   0 |                     0 |
| ridge_residual    |                     111 |                  23 |                    89 |

**V21** — Tổng tham số theo biến thể feature set:

| Model         |   Paper3 (F=3) |   Paper3+Cu (F=4) |   Legacy14 (F=4) |   Legacy14+Cu (F=5) |
|:--------------|---------------:|------------------:|-----------------:|--------------------:|
| baseline_lstm |          30369 |             30625 |            30625 |               30881 |
| baseline_rnn  |           8001 |              8065 |             8065 |                8129 |
| naive         |              0 |                 0 |                0 |                   0 |
| paper_sa_lstm |          77369 |             77525 |            77525 |               77681 |
| reopt_sa_lstm |          46761 |             46849 |            40289 |               40377 |

Tất cả model neural trong CSV audit có `nontrainable_params=0`; Naive có 0 tham số; Ridge có `22×F+1` hệ số gồm intercept (không phải neural network weights). **Số tham số không đổi giữa Expanding/Rolling/Recency, và không phụ thuộc train ratio; chỉ phụ thuộc kiến trúc và input dimensionality.** Dữ liệu layer-by-layer đầy đủ 288 dòng nằm ở CSV phụ lục; không được nhầm số tham số với số lượt train.

### 4.4. Mô hình tính toán như thế nào?

- **SimpleRNN:** ở mỗi bước k, cập nhật hidden state từ input xₖ và hidden trước đó, `hₖ=tanh(Wₓxₖ+Wₕhₖ₋₁+b)`.
- **LSTM:** các cổng quên, ghi và xuất học các phép chiếu tuyến tính có sigmoid/tanh, điều tiết cell state để lưu/loại thông tin; hidden cuối được đưa qua Dense.
- **GRU:** các cổng update/reset điều khiển trộn hidden cũ và trạng thái ứng viên; không có cell state riêng như LSTM.
- **CNN-hybrid:** Conv1D học bộ lọc cục bộ, MaxPool giảm chiều thời gian (22→11), mạng recurrent học chuỗi đặc trưng mới; Dense đưa về scalar dự báo.
- **Ridge:** ước lượng tuyến tính có điều chuẩn L2; không dùng gradient backprop theo các epoch như neural network.
- **SA:** Simulated Annealing dò hyperparameters (số units/learning rate); **không** thêm neuron SA hay tham số SA vào forward pass. Sau khi chốt HP, LSTM vẫn tối ưu weights bằng Adam/MSE. Có thể có cùng số tham số nhưng khác kỹ thuật học target → hiệu năng khác nhau.

### 4.5. Huấn luyện và chiến lược thời gian

V20: 65 epochs tối đa, EarlyStopping patience 9, batch 64, Adam LR 0,001 và seed 42 ở lưới; V21 holdout: tối đa 100 epochs, patience 9, batch 64, seed 42. Reoptimized SA: Paper3 `(22, 91, lr 0.000922)`; Legacy14 `(22, 83, lr 0.000655)`; Paper HP `(39, 111, lr 0.00005)`. V21 **không thực hiện SA search lại**.
- **Expanding:** cố định mọi quan sát train đến ranh giới validation/test.
- **Rolling 756:** chỉ dùng 756 phiên gần nhất trước ranh giới fit.
- **Recency weighted:** dùng toàn bộ mẫu nhưng trọng số `wᵢ=exp(−ln(2)·ageᵢ/252)`.
Ba strategy ở các lưới holdout **không có nghĩa mô hình retrain mỗi ngày**. Walk-forward V20 retrain theo lịch riêng mỗi 63 phiên; kết quả holdout không được gọi walk-forward.

### 4.6. Độ đo và kiểm định

MAE = mean(|P−P̂|), RMSE = sqrt(mean((P−P̂)²)), MSE = mean((P−P̂)²) theo giá gốc; MAE là metric chính, RMSE bổ sung kiểm soát lỗi lớn. Skill MAE so với Naive `100×(1−MAE_model/MAE_naive)%`. Directional Accuracy (DA) đo tính đúng dấu của `P̂_t−P_{t−1}` so với `P_t−P_{t−1}`. Naive dự đoán delta 0, vì vậy **không báo DA tăng/giảm như một tín hiệu phân lớp**. Kiểm định Diebold–Mariano–HAC kiểm tra chênh lệch expected loss; p-value không nhỏ hơn 0,05 không chứng minh mô hình tương đương tuyệt đối.

## V. Kết quả (Experimental Results)

### 5.1. Đối chứng Non-SA V20 ở train/test 90/10, Expanding

Tổng hợp đã đối chiếu theo `grid_results.csv` trong báo cáo gốc:

| Model | Input (lựa chọn hậu nghiệm trong Expanding) | MAE | RMSE | DA (%) |
|---|---|---:|---:|---:|
| Naive | — | **0.077077** | 0.101033 | — |
| Residual-GRU | Copper+Legacy14 | **0.076951** | 0.100810 | 52.77 |
| Residual-LSTM | Copper+Legacy14 | 0.076978 | 0.100948 | 52.77 |
| Residual-CNN-LSTM | Copper+Legacy14 | 0.077245 | 0.100856 | 46.86 |
| Residual-CNN-GRU | Copper-only | 0.078067 | 0.101486 | 45.76 |
| Residual Ridge | Copper-only | 0.077306 | 0.101190 | 47.60 |
| GRU Direct | Copper+Legacy14 | 0.240815 | 0.302923 | 53.51 |
| LSTM Direct | Copper+Legacy14 | 0.247410 | 0.304620 | 52.77 |
| CNN-GRU Direct | Copper+Paper3 | 0.296379 | 0.359605 | 52.40 |
| CNN-LSTM Direct | Copper+Legacy14 | 0.393469 | 0.492822 | 49.82 |

**Vì sao nhấn mạnh Residual:** Naive = Copper(t−1) là đường chuẩn rất mạnh cho mức giá; residual mô hình hóa phần điều chỉnh quanh baseline thay vì cố tái lập toàn bộ mặt bằng giá. Tuy vậy chênh lệch 0,000126 đơn vị giá (+0,163%) ở split 90/10 là nhỏ và được chọn sau khi nhìn test; không phải bằng chứng xác nhận cải tiến.

### 5.2. SA-LSTM V21 gốc, cùng split 90/10 + Expanding

| Model         |   Paper3+Cu |    Paper3 |   Legacy14+Cu |   Legacy14 |
|:--------------|------------:|----------:|--------------:|-----------:|
| naive         |   0.0770767 | 0.0770767 |     0.0770767 |  0.0770767 |
| baseline_lstm |   1.16396   | 1.14599   |     0.247409  |  0.854128  |
| baseline_rnn  |   1.38573   | 1.00123   |     0.831186  |  0.962108  |
| paper_sa_lstm |   0.978075  | 0.54072   |     0.470929  |  0.464991  |
| reopt_sa_lstm |   0.87742   | 0.595605  |     0.510467  |  0.466595  |

V21 90/10 cho thấy LSTM baseline + Legacy14+Cu MAE 0,247409 là tốt nhất trong nhóm học sâu Direct, nhưng vẫn cao hơn Naive 0,077077. Không thể nói SA-LSTM thất bại về nguyên lý; ở đây chỉ quan sát các HP và giao thức train hiện tại không vượt được Naive.

### 5.3. BỔ SUNG MỚI — V21 train/test 80/20, 542 phiên test

**MAE, Expanding** (kết quả mới từ ZIP 80/20):

| Model         |   Paper3+Cu |    Paper3 |   Legacy14+Cu |   Legacy14 |
|:--------------|------------:|----------:|--------------:|-----------:|
| naive         |   0.0690015 | 0.0690015 |     0.0690015 |  0.0690015 |
| baseline_lstm |   0.432026  | 0.626871  |     0.713145  |  0.550635  |
| baseline_rnn  |   0.913549  | 0.843475  |     0.795791  |  0.888142  |
| paper_sa_lstm |   0.848981  | 0.506237  |     0.760563  |  0.669122  |
| reopt_sa_lstm |   0.616924  | 0.460221  |     1.00042   |  0.361893  |

**RMSE, Expanding:**

| Model         |   Paper3+Cu |   Paper3 |   Legacy14+Cu |   Legacy14 |
|:--------------|------------:|---------:|--------------:|-----------:|
| naive         |    0.108    | 0.108    |      0.108    |   0.108    |
| baseline_lstm |    0.526865 | 0.812337 |      0.878396 |   0.700947 |
| baseline_rnn  |    1.11899  | 1.11998  |      0.916974 |   1.04504  |
| paper_sa_lstm |    1.03974  | 0.605086 |      1.00231  |   0.797336 |
| reopt_sa_lstm |    0.728432 | 0.544544 |      1.18919  |   0.464548 |

V21 80/20 có **đủ 60/60** tổ hợp 4 feature sets × 3 strategies × 5 models (48 lượt fitted và 12 lượt Naive). ZIP đã lưu `holdout_80_results.csv`, `holdout_80_predictions.csv` và 10 ảnh. Không có `wf_fold_metrics.csv` hoặc `wf_predictions.csv`.

**Các kết quả 80/20 tốt nhất trong 48 lượt fitted (chọn hậu nghiệm):**

| model         | feature_set             | scheme           |      mae |     rmse |   directional_accuracy_pct |
|:--------------|:------------------------|:-----------------|---------:|---------:|---------------------------:|
| paper_sa_lstm | paper3_without_copper   | recency_weighted | 0.349026 | 0.450344 |                      51.11 |
| reopt_sa_lstm | legacy14_without_copper | expanding        | 0.361893 | 0.464548 |                      52.95 |
| baseline_lstm | paper3_with_copper      | expanding        | 0.432026 | 0.526865 |                      46.86 |
| reopt_sa_lstm | paper3_without_copper   | expanding        | 0.460221 | 0.544544 |                      51.48 |
| paper_sa_lstm | paper3_without_copper   | expanding        | 0.506237 | 0.605086 |                      51.48 |
| baseline_lstm | legacy14_without_copper | expanding        | 0.550635 | 0.700947 |                      48.52 |
| baseline_lstm | legacy14_without_copper | recency_weighted | 0.565972 | 0.700684 |                      46.86 |
| reopt_sa_lstm | legacy14_without_copper | recency_weighted | 0.57386  | 0.691455 |                      46.49 |

Dù một số phương pháp hưởng lợi từ thêm hoặc bỏ Copper history, tất cả mạng V21 80/20 vẫn kém Naive (MAE 0,069001). Việc so sánh V20/V21 phải cùng ratio và test timestamps; ngân sách huấn luyện/HP không hoàn toàn bằng nhau.

### 5.4. So sánh chiến lược 80/20 cho đủ Paper3/Legacy14 ± Copper

**paper3_with_copper:**

| Model         |   Expanding |   Rolling 756 |   Recency |
|:--------------|------------:|--------------:|----------:|
| naive         |   0.0690015 |     0.0690015 | 0.0690015 |
| baseline_lstm |   0.432026  |     1.36682   | 0.950984  |
| baseline_rnn  |   0.913549  |     1.44868   | 1.07948   |
| paper_sa_lstm |   0.848981  |     1.04105   | 0.811499  |
| reopt_sa_lstm |   0.616924  |     1.07819   | 0.759957  |

**paper3_without_copper:**

| Model         |   Expanding |   Rolling 756 |   Recency |
|:--------------|------------:|--------------:|----------:|
| naive         |   0.0690015 |     0.0690015 | 0.0690015 |
| baseline_lstm |   0.626871  |     0.986493  | 0.985067  |
| baseline_rnn  |   0.843475  |     1.82844   | 1.04629   |
| paper_sa_lstm |   0.506237  |     0.703631  | 0.349026  |
| reopt_sa_lstm |   0.460221  |     0.638912  | 0.638208  |

**legacy14_with_copper:**

| Model         |   Expanding |   Rolling 756 |   Recency |
|:--------------|------------:|--------------:|----------:|
| naive         |   0.0690015 |     0.0690015 | 0.0690015 |
| baseline_lstm |   0.713145  |     1.09639   | 0.574219  |
| baseline_rnn  |   0.795791  |     1.25843   | 0.833112  |
| paper_sa_lstm |   0.760563  |     0.786877  | 0.586941  |
| reopt_sa_lstm |   1.00042   |     1.24165   | 0.919743  |

**legacy14_without_copper:**

| Model         |   Expanding |   Rolling 756 |   Recency |
|:--------------|------------:|--------------:|----------:|
| naive         |   0.0690015 |     0.0690015 | 0.0690015 |
| baseline_lstm |   0.550635  |     1.08387   | 0.565972  |
| baseline_rnn  |   0.888142  |     1.32301   | 0.909091  |
| paper_sa_lstm |   0.669122  |     1.42327   | 0.664396  |
| reopt_sa_lstm |   0.361893  |     0.833539  | 0.57386   |

Đây là dữ liệu đầy đủ cho các biểu đồ `plots/80_strategy_mae_all_features.png` và `plots/80_strategy_rmse_all_features.png` của ZIP add-on. Cần đặt các trục y thống nhất trong báo cáo cuối để tránh trực quan hóa phóng đại chênh lệch.

### 5.5. V21 90/10 Legacy14 + Copper: so sánh chiến lược (giữ từ báo cáo cũ)

| Model         |   Expanding |   Rolling 756 |   Recency |
|:--------------|------------:|--------------:|----------:|
| naive         |   0.0770767 |     0.0770767 | 0.0770767 |
| baseline_lstm |   0.247409  |     0.958466  | 0.65906   |
| baseline_rnn  |   0.831186  |     1.11515   | 1.04567   |
| paper_sa_lstm |   0.470929  |     0.77331   | 0.70535   |
| reopt_sa_lstm |   0.510467  |     0.977821  | 0.735621  |

Ở cấu hình này Expanding có MAE nhỏ hơn Rolling và Recency cho bốn mạng Direct, song chỉ phản ánh một tập test. Rolling có ít mẫu train hơn, Recency hạ trọng số dữ liệu cũ; không chiến lược nào đảm bảo tốt hơn ở mọi market regime.

### 5.6. V21 feature ablation: Copper history

V21 bỏ Copper từ **cả chuỗi 22 phiên đầu vào**, không chỉ bỏ scalar `Copper(t−1)`. Lợi ích thêm Copper trong MAE ở 90/10 + Expanding: LSTM Paper3 −1,57%; Legacy14 +71,03%; SimpleRNN Paper3 −38,40%, Legacy14 +13,61%; SA-LSTM paper HP Paper3 −80,88%, Legacy14 −1,28%; SA-LSTM cached HP Paper3 −47,32%, Legacy14 −9,40%. Dấu dương nghĩa là MAE giảm khi thêm Copper. Các số liệu là kết quả khám phá, chưa kiểm định ý nghĩa; Copper history không có lợi phổ quát.

### 5.7. Kết quả V20 xuyên 4 tỷ lệ split — đã sửa bảng xếp hạng

| Train ratio | Mô hình fitted có MAE thấp nhất | Feature | Scheme | MAE | Skill vs Naive |
|---|---|---|---|---:|---:|
| 80% | GRU Residual | Copper+Paper3 | Expanding | 0.068780 | +0.321% |
| 90% | GRU Residual | Copper+Legacy14 | Expanding | 0.076951 | +0.163% |
| 92.5% | CNN-LSTM Residual | Copper-only | Recency | 0.075050 | +0.089% |
| 95% | GRU Residual | Copper+Paper3 | Recency | 0.072357 | +0.608% |

**Sửa sai bản V20 cũ:** mô hình dẫn đầu 92,5% và 95% trước đó bị ghi nhầm; cải thiện holdout lớn nhất là +0,608%, không phải tất cả dưới 0,40%. Các tỷ lệ khác nhau có các giai đoạn test khác nhau; đây không phải phép kiểm soát nhân quả tác động của train ratio.

### 5.8. Walk-forward V20 và Diebold–Mariano–HAC

Chỉ **Residual-GRU + Copper/Paper3 + Expanding** đã có walk-forward V20 trên 2.214 phiên 2018–2026, retrain mỗi 63 phiên, seeds 41–45; kết quả:

| Seed | MAE GRU | MAE Naive | p-value 2 phía | Diễn giải tại α=5% |
|---|---:|---:|---:|---|
| 41 | 0.045110 | 0.045127 | 0.7043 | Chưa thấy khác biệt |
| 42 | 0.045189 | 0.045127 | 0.3656 | Chưa thấy khác biệt |
| 43 | 0.045447 | 0.045127 | 0.0066 | Naive tốt hơn có ý nghĩa |
| 44 | 0.045502 | 0.045127 | 0.0101 | Naive tốt hơn có ý nghĩa |
| 45 | 0.045388 | 0.045127 | 0.0788 | Chưa có ý nghĩa ở 5% |

DA Residual-GRU dao động khoảng 50,27–51,45%. Walk-forward V20 không chứng minh Residual-GRU thắng Naive ổn định. **Walk-forward SA-LSTM V21 vẫn thiếu trong ZIP mới**, dù file ZIP đặt tên “80_wf_plots”. Không được áp kết quả DM của V20 cho SA-LSTM V21.

### 5.9. Residual audit và giải thích sai số

| Model V20 (Paper3+Cu, Expanding, 90/10) | std(Δ thực) | std(Δ dự báo) | Corr(Δ̂,Δ) | MAE |
|---|---:|---:|---:|---:|
| LSTM Residual | 0.10097 | 0.00580 | 0.0517 | 0.077006 |
| GRU Residual | 0.10097 | 0.00567 | 0.0401 | 0.077143 |
| CNN-LSTM Residual | 0.10097 | 0.01026 | 0.0314 | 0.077432 |
| Ridge Residual | 0.10097 | 0.02365 | −0.0593 | 0.080712 |

Mô hình Residual dự báo Δ với biên độ nhỏ so với biến động thật; vì vậy `Copper(t−1)+Δ̂` tiệm cận Naive. Dưới MSE, nghiệm tối ưu theo kỳ vọng là `E[Δ|X]`; nếu không có nhiều thông tin, ước lượng này có thể gần 0. **Đây là giải thích cơ chế khả dĩ**, không chứng minh chuỗi hoàn toàn random walk hay Δ có tính dừng nếu chưa kiểm định ADF/KPSS và các điều kiện cần thiết.

## V-A. MỞ RỘNG THỰC NGHIỆM V22 — THAY GOLD/SILVER BẰNG WTI (cập nhật 19/11/2026)

### 5A.1. Xuất xứ, phạm vi và nguyên tắc đối chiếu

Phần này bổ sung trực tiếp từ hai ZIP thực nghiệm ngày 18/11/2026: `[]18-11- Copper_NonSA_V22_WTI_Only_Kaggle.zip` và `[]18-11copper_sa_lstm_wti_only_v22_results.zip`; số liệu gốc V20/V21 lấy từ ZIP gốc tương ứng, riêng V21 80/20 từ `holdout_80_results.csv`. Các bảng so sánh MAE và RMSE theo **giá gốc**, ưu tiên MAE nhỏ nhất. Naive chỉ là **baseline**, không nằm trong xếp hạng fitted models. Mọi thứ hạng là **mô tả hậu nghiệm trên holdout**, không phải lựa chọn mô hình đã được xác nhận ngoài mẫu.

- V22 dùng chính xác `Copper, Aluminium, Tin, WTI` (4 biến) với cửa sổ 22 phiên để dự báo Copper ở phiên tiếp theo. Legacy14 cũ gồm `Copper, Aluminium, Tin, Silver, Gold` (5 biến); Paper3 + Copper gồm `Copper, WTI, Gold, Silver` (4 biến). Vì vậy so sánh WTI mới với Legacy14 là **thay đổi đồng thời thành phần và số biến**; không thể quy mọi thay đổi hiệu năng chỉ riêng cho WTI.
- Dataset SHA256 V22 SA theo manifest: `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600`, trùng hash trong báo cáo V20/V21; Non-SA V22 có fingerprint để đối chiếu. Các tỷ lệ split là 80%, 90%, 92,5%, 95% train.
- Non-SA V22: **120/120 hàng holdout** = 4 tỷ lệ × 3 chiến lược × (9 fitted + Naive); có walk-forward Residual-GRU 5 seeds, 36 folds/seed. SA-LSTM V22: **30/30 hàng holdout** = 80/20 Expanding (5), 90/10 đủ ba chiến lược (15), 92,5/7,5 Expanding (5), 95/5 Expanding (5). **Không có** SA-LSTM Rolling/Recency cho ba tỷ lệ 80%, 92,5%, 95%; không được gán số liệu cho các ô này.
- V22 SA chạy **16 trials SA mới** với seed SA 2026, ở mốc tham chiếu 80/20, mỗi trial tối đa 15 epochs; mô hình cuối tối đa 100 epochs. Các siêu tham số tối ưu chỉ theo validation ở mốc tham chiếu. Không thể gọi đây là đánh giá kiểm soát riêng tác động của thuật toán SA khi ngân sách/tìm kiếm của V21 khác V22.

### 5A.2. Xếp hạng số 1 ở mỗi tỷ lệ — chỉ tính các mô hình được huấn luyện

| Train/test (%) | Test (phiên) | Non-SA V22 #1 / chiến lược | MAE | RMSE | SA-V22 #1 / chiến lược | MAE | RMSE | Naive MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80/20 | 542 | Residual-LSTM / Expanding | 0.068840 | 0.107996 | SA-LSTM (SA tìm mới) / Expanding | 0.223809 | 0.297888 | 0.069001 |
| 90/10 | 271 | Residual-LSTM / Expanding | 0.077037 | 0.100980 | SA-LSTM (paper HP) / Expanding | 0.184634 | 0.225723 | 0.077077 |
| 92.5/7.5 | 204 | Residual-GRU / Recency | 0.075028 | 0.098307 | SA-LSTM (paper HP) / Expanding | 0.194868 | 0.236334 | 0.075116 |
| 95/5 | 136 | Residual-GRU / Recency | 0.072619 | 0.096531 | SA-LSTM (paper HP) / Expanding | 0.200803 | 0.235800 | 0.072799 |

Hai nhánh xếp hạng riêng vì Non-SA V22 có nhóm mô hình học trực tiếp và residual, còn SA-V22 chỉ dự báo trực tiếp. Đối với quyết định *mô hình fitted tốt nhất xét tất cả mô hình V22 theo MAE*, cả bốn tỷ lệ đều thuộc nhánh Non-SA residual; cần xem từng dòng trong bảng, không suy diễn độ ổn định từ các split có tập test lồng nhau.

### 5A.3. Toàn bộ mô hình Non-SA V22 — MAE 90/10 theo chiến lược

| Mô hình | MAE Expanding | MAE Rolling | MAE Recency | Tốt nhất trong 3 | RMSE tương ứng |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual-LSTM | 0.077037 | 0.080271 | 0.078856 | Expanding | 0.100980 |
| Residual-CNN-GRU | 0.077229 | 0.085898 | 0.077639 | Expanding | 0.101285 |
| Residual-GRU | 0.077463 | 0.081939 | 0.077851 | Expanding | 0.101052 |
| Residual-CNN-LSTM | 0.077638 | 0.085373 | 0.078023 | Expanding | 0.101076 |
| Residual-Ridge | 0.077737 | 0.094420 | 0.081044 | Expanding | 0.101669 |
| GRU Direct | 0.243208 | 0.256292 | 0.235421 | Recency | 0.275645 |
| LSTM Direct | 0.324937 | 0.346713 | 1.236604 | Expanding | 0.381590 |
| CNN-GRU Direct | 0.619798 | 0.427192 | 0.643456 | Rolling 756 | 0.470401 |
| CNN-LSTM Direct | 0.888314 | 0.873802 | 0.886170 | Rolling 756 | 0.945043 |

Bảng trên giữ đầy đủ 9 mô hình fitted. So sánh chiến lược trong cùng mô hình; không dùng `val_mse_scaled` để so sánh chéo scaler.

### 5A.4. Đối chiếu thay đổi MAE khi thay input — Non-SA V20 ↔ V22 (90/10 + Expanding)

| Mô hình | MAE Legacy14+Cu V20 | MAE Paper3+Cu V20 | MAE WTI V22 | Δ MAE vs Legacy14 | Chiều thay đổi | RMSE WTI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM Direct | 0.247410 | 1.163964 | 0.324937 | +31.34% | tăng | 0.381590 |
| GRU Direct | 0.240815 | 0.247893 | 0.243208 | +0.99% | tăng | 0.287070 |
| CNN-LSTM Direct | 0.393469 | 1.108590 | 0.888314 | +125.76% | tăng | 0.952557 |
| CNN-GRU Direct | 0.313523 | 0.296379 | 0.619798 | +97.69% | tăng | 0.664925 |
| Residual-LSTM | 0.076978 | 0.077006 | 0.077037 | +0.08% | tăng | 0.100980 |
| Residual-GRU | 0.076951 | 0.077143 | 0.077463 | +0.67% | tăng | 0.101052 |
| Residual-CNN-LSTM | 0.077245 | 0.077432 | 0.077638 | +0.51% | tăng | 0.101076 |
| Residual-CNN-GRU | 0.090563 | 0.078132 | 0.077229 | -14.72% | giảm | 0.101285 |
| Residual-Ridge | 0.081707 | 0.080712 | 0.077737 | -4.86% | giảm | 0.101669 |

Quy ước `Δ MAE = 100 × (MAE_WTI / MAE_Legacy14 − 1)`: **âm = WTI giảm lỗi (tốt hơn)**; dương = WTI tăng lỗi. Đây là **một cấu hình split/strategy cố định**, không kết luận WTI hữu ích hay kém hữu ích trong mọi điều kiện.

### 5A.5. So sánh fitted model đứng đầu theo split — giữa bộ WTI và hai bộ cũ Non-SA

| Train % | WTI fitted #1 | MAE | Legacy14 fitted #1 | MAE | Paper3 fitted #1 | MAE | V20 tất cả feature sets #1 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80% | Residual-LSTM / Expanding | 0.068840 | Residual-GRU / Expanding | 0.068857 | Residual-GRU / Expanding | 0.068780 | Residual-GRU (copper_paper3) | 0.068780 |
| 90% | Residual-LSTM / Expanding | 0.077037 | Residual-GRU / Expanding | 0.076951 | Residual-LSTM / Expanding | 0.077006 | Residual-GRU (copper_legacy14) | 0.076951 |
| 92.5% | Residual-GRU / Recency | 0.075028 | Residual-GRU / Expanding | 0.075205 | Residual-GRU / Expanding | 0.075206 | Residual-CNN-LSTM (copper_only) | 0.075050 |
| 95% | Residual-GRU / Recency | 0.072619 | Residual-GRU / Recency | 0.072507 | Residual-GRU / Recency | 0.072357 | Residual-GRU (copper_paper3) | 0.072357 |

Mỗi ô trên là **min test MAE sau khi xem kết quả**, có selection bias. Cột V20 tất cả feature sets còn gồm Copper-only nên không đồng nhất phạm vi candidate với hai cột input cố định. Không diễn giải sự khác biệt giữa min các tập candidate là hiệu quả nhân quả của một feature.

### 5A.6. SA-LSTM V22: tham số SA mới và bảng chiến lược 90/10

SA search chọn `units_1=47`, `units_2=127`, `learning_rate=5.04788067547e-05` từ **16 trial** theo validation (mốc 80/20). Đây là HP của bản V22, không phải HP công bố bởi Chen et al.; `paper_sa_lstm` tiếp tục mang cấu hình mô phỏng thông số paper.

| Mô hình | MAE Expanding | MAE Rolling | MAE Recency | Tốt nhất | RMSE tương ứng |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM baseline | 0.324940 | 0.321192 | 1.236605 | Rolling 756 | 0.350564 |
| SimpleRNN | 0.818738 | 0.859431 | 1.009153 | Expanding | 0.900169 |
| SA-LSTM (paper HP) | 0.184634 | 0.575763 | 0.294741 | Expanding | 0.225723 |
| SA-LSTM (SA tìm mới) | 0.358921 | 0.496288 | 0.326683 | Recency | 0.376388 |

### 5A.7. SA-LSTM: so sánh 90/10 + Expanding với V21 có Copper history

| Mô hình | MAE V21 Legacy14+Cu | MAE V21 Paper3+Cu | MAE V22 WTI | Δ MAE vs Legacy14 | RMSE V22 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM baseline | 0.247409 | 1.163964 | 0.324940 | +31.34% | 0.381594 |
| SimpleRNN | 0.831186 | 1.385730 | 0.818738 | -1.50% | 0.900169 |
| SA-LSTM (paper HP) | 0.470929 | 0.978075 | 0.184634 | -60.79% | 0.225723 |
| SA-LSTM (SA tìm mới) | 0.510467 | 0.877420 | 0.358921 | -29.69% | 0.415277 |

**Quan trọng:** Với `reopt_sa_lstm`, thay input đồng thời **tìm lại siêu tham số SA**, nên không phải phép ablation thuần về feature; mức thay đổi MAE là hiệu ứng tổng hợp của feature, HP, và dao động huấn luyện. Với `paper_sa_lstm`, HP được giữ theo cấu hình paper, song diễn giải vẫn giới hạn trong protocol đang dùng.

### 5A.8. SA-LSTM fitted #1 theo split; đối chứng V21 và Naive

| Train % | V22 WTI fitted #1 | MAE | V21 Legacy+Cu fitted #1 | MAE | V21 Paper+Cu fitted #1 | MAE | Naive MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80% | SA-LSTM (SA tìm mới) / Expanding | 0.223809 | LSTM baseline / Recency | 0.574219 | LSTM baseline / Expanding | 0.432026 | 0.069001 |
| 90% | SA-LSTM (paper HP) / Expanding | 0.184634 | LSTM baseline / Expanding | 0.247409 | SA-LSTM (SA tìm mới) / Recency | 0.511784 | 0.077077 |
| 92.5% | SA-LSTM (paper HP) / Expanding | 0.194868 | LSTM baseline / Expanding | 0.292571 | LSTM baseline / Expanding | 0.402367 | 0.075116 |
| 95% | SA-LSTM (paper HP) / Expanding | 0.200803 | SA-LSTM (SA tìm mới) / Expanding | 0.541005 | LSTM baseline / Expanding | 0.452409 | 0.072799 |

V21 80/20 lấy từ ZIP bổ sung, chỉ được xác nhận đối với kết quả có trong CSV. Không có walk-forward SA-V22 trong ZIP đã cung cấp.

### 5A.9. Non-SA V22 walk-forward — so sánh cùng phiên, cùng seed với V20

| Seed | Phiên | V20 Paper3+Cu GRU MAE | V22 WTI GRU MAE | Δ MAE vs V20 | Naive MAE | V22 Skill vs Naive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 2214 | 0.045110 | 0.045158 | +0.108% | 0.045127 | -0.069% |
| 42 | 2214 | 0.045189 | 0.045340 | +0.333% | 0.045127 | -0.471% |
| 43 | 2214 | 0.045447 | 0.045169 | -0.611% | 0.045127 | -0.093% |
| 44 | 2214 | 0.045502 | 0.045373 | -0.283% | 0.045127 | -0.546% |
| 45 | 2214 | 0.045388 | 0.045201 | -0.413% | 0.045127 | -0.165% |

Các MAE walk-forward ở bảng này được **tính lại từ file predictions** theo `mean(abs(actual − pred))`, không lấy trung bình đơn giản MAE của folds. Hai ZIP chứa **36 folds × 5 seeds** và 2.214 phiên/seed. V22 không kèm file kiểm định DM-HAC theo seed, nên **chưa có p-value DM mới**, không thể tuyên bố có ý nghĩa thống kê cho chênh lệch V22. Kết quả DM-HAC cũ chỉ áp dụng V20 như báo cáo ở §5.8.

### 5A.10. Diễn giải khoa học, giới hạn và kết luận sau V22

**(1) Feature selection.** Quyết định thử WTI có cơ sở khám phá từ Spearman trên *mức giá*, nhưng tương quan mức giá không chứng minh khả năng dự báo thay đổi giá ngày tiếp theo. Do WTI mới có bốn biến còn Legacy14 có năm, thí nghiệm hiện tại là **feature-set comparison**, chưa phải phép leave-one-feature-out kiểm soát duy nhất một biến.

**(2) Xếp hạng.** Để trả lời “mô hình tốt nhất”, báo cáo xếp hạng **chỉ các fitted models theo MAE trên từng split**; đồng thời vẫn bắt buộc báo Naive MAE tách riêng để biết cải thiện có vượt được tham chiếu đơn giản không. Mô hình tốt nhất *trong số fitted* không đồng nghĩa mô hình có hiệu năng sử dụng thực tế tốt nhất nếu MAE vẫn cao hơn Naive.

**(3) Direct so với residual.** Mọi kết quả cho thấy cần phân biệt thang target. Một mô hình residual có lợi thế neo ở Copper(t−1), vì vậy MAE gần Naive chưa đủ chứng minh nó học được tín hiệu dự báo biến động. Không dùng mức MAE của Direct để khẳng định SA nói chung yếu hơn một thuật toán khác; hai cách biểu diễn target khác nhau.

**(4) Chiến lược.** Expanding/Rolling/Recency ở các holdout là cách chọn hoặc gán trọng số dữ liệu train, không phải tự retrain trên từng ngày test. Chỉ bảng có đầy đủ ba chiến lược mới được đưa ra kết luận nội bộ ba-way; ở SA-V22 các ratio ngoài 90/10 **chỉ có Expanding**.

**(5) Phạm vi bằng chứng.** Các kết quả xếp hạng chủ yếu lấy seed 42 và các test lồng nhau theo thời gian; nhìn rồi chọn cấu hình sẽ dẫn đến selection bias. SA-V22 chưa có walk-forward, và walk-forward V22 Non-SA chưa có kiểm định significance mới. Vì vậy chỉ kết luận **mô tả thực nghiệm trong dataset và giao thức đã chạy**, không khẳng định mô hình chiến thắng phổ quát.

### 5A.11. Kiểm toán tự động trước khi đưa vào báo cáo

| Nguồn | Số hàng | MAE không null | Số hàng RMSE ≥ MAE | Khóa ratio/scheme/model bị trùng |
| --- | ---: | ---: | ---: | ---: |
| Non-SA V22 | 120 | 120 | 120 | 0 |
| SA-LSTM V22 | 30 | 30 | 30 | 0 |

Số liệu trong các bảng được sinh trực tiếp từ CSV có sẵn, làm tròn **6 chữ số thập phân** khi hiển thị; phép tính tăng/giảm dùng dữ liệu chưa làm tròn. Tính toàn vẹn dữ liệu ngoài các file đầu ra (múi giờ công bố giá, dữ liệu thô, thời điểm thực sự dự báo) vẫn cần kiểm toán riêng.

## VI. Thảo luận (Discussion)

### 6.1. Vì sao độ chính xác không phải đóng góp duy nhất?

Thử nghiệm cho thấy cùng cấu trúc mạng, chỉ thay target từ mức giá sang sai phân cộng lại Copper trước đó đã đưa MAE sát Naive. Điều đó phản ánh vai trò của **inductive bias / target representation** chứ không nhất thiết là việc tăng năng lực biểu diễn bằng nhiều layers. Thực nghiệm SA-LSTM giúp hiểu SA là công cụ tìm HP; HP cached tốt trên một cấu hình không mặc nhiên tốt khi số features hoặc chiến lược khác đi. Đây là bài học phương pháp học máy có giá trị ngay khi không thắng baseline.

### 6.2. Các lý do có thể khiến Direct kém hơn Residual

Mô hình Direct phải học quy mô mức giá, chịu áp lực ngoại suy khi test có mặt bằng giá khác train, và nhạy với lựa chọn scaling/output activation/early stopping. Residual có điểm neo Copper(t−1) và chỉ cần ước lượng Δ. Tuy nhiên, các khẳng định “giá đồng là I(1)”, “chắc chắn concept drift” hoặc “thiếu feature là nguyên nhân duy nhất” **chưa được chứng minh** chỉ nhờ MAE cao. Cần kiểm tra histogram y_pred, systematic bias, phân phối train/test, thời điểm cập nhật thị trường và learning curves.

### 6.3. Đánh giá vai trò tương quan và số lượng biến

Không nên bớt một biến Legacy14 theo tương quan Spearman của **mức giá** chỉ để làm mô hình chính xác hơn. Mức tương quan theo giá và theo delta lag 1 là hai vấn đề khác nhau. Bài báo gốc chọn WTI, Gold, Silver; bản mở rộng Legacy14 dùng Aluminium, Tin, Silver, Gold vì mục tiêu so sánh **thông tin kinh tế khác**. Nếu muốn một Legacy3 công bằng, chọn bằng validation hoặc leave-one-out, giải thích trước, giữ nguyên Paper3 làm reproduction baseline.

### 6.4. Khi nào cần xem xét concept drift/regime shift?

Sai số thay đổi theo năm và các mô hình Direct dự báo lệch là tín hiệu **nên điều tra**, không phải phép kiểm định drift. Muốn kiểm chứng: vẽ rolling MAE và delta-volatility, kiểm tra thay đổi phân phối có kiểm soát, thử refit định kỳ cùng chính sách và giữ một test period chưa dùng để chọn strategy. Nên tránh overclaim khi chi phí compute giới hạn.

## VII. Giới hạn nghiên cứu và nguy cơ sai lệch (Limitations)

1. **Khác phạm vi ngày:** khoảng 01/01/2016–31/10/2026 theo đề xuất không trùng dữ liệu thực nghiệm 13/01/2016–13/11/2026.
2. **Khác training budget:** V20 65 epochs tối đa, V21 100 epochs; không kết luận “SA tốt/yếu hơn kiến trúc” theo nhân quả.
3. **SA search:** V21 chỉ train bằng HP công bố hoặc cached; không đánh giá chi phí/tính tối ưu toàn cục của SA so với tuning có ngân sách bằng nhau.
4. **Thiếu V21 SA walk-forward:** ZIP add-on chỉ có 80/20, chưa có CSV folds, predictions, DM theo seeds.
5. **Selection bias:** các cấu hình dẫn đầu được tìm khi nhìn vào holdout; cần khóa phương pháp và đánh giá lần cuối trên khoảng chưa tối ưu.
6. **Biến ngoại sinh:** lịch đóng cửa WTI, Gold, Silver, Copper giữa thị trường khác nhau có thể không đồng bộ theo ngày.
7. **EDA Spearman:** tương quan levels cao không chứng minh predictability of returns; correlated features có thể bổ sung thông tin trùng lặp.
8. **Kết luận thống kê:** DM p>0.05 không chứng minh tương đương; kiểm định 5 seeds trên cùng lịch test không tạo 5 datasets độc lập.
9. **MSE scaled:** scale thay đổi theo scheme, không so sánh trực tiếp qua schemes.
10. **DA:** 50–54% chưa đủ chứng cứ để tuyên bố khai thác xu hướng; Naive không có hướng.
11. **Cơ chế drift/I(1):** chưa có chứng cứ kiểm định riêng, không nên ghi như sự thật đã xác nhận.
12. **Reproduction:** Chen et al. không cung cấp đủ thông tin để xác nhận Dense(32), lookback=22, horizon=1, early stopping; cần tách giả định.
13. **Tham số:** kiểm toán 50 tổ hợp là đếm cấu trúc (không train lại); số layer-audit không đồng nghĩa với chứng thực checkpoint weights V20/V21.
14. **Dữ liệu gốc / nguồn sàn:** dataset được chuẩn hóa từ nhiều hàng hóa; báo cáo cần bổ sung tài liệu lịch sử nguồn/API và quy tắc xử lý ngày nghỉ để đảm bảo replication.

## VIII. Ma trận đầy đủ/chưa đầy đủ của V20 và V21

| Hạng mục | V20 | V21 sau add-on | Đánh giá |
|---|---|---|---|
| 80/20 (3 schemes) | Có Copper-only, Copper+Paper3, Copper+Legacy14 | **Đã có đủ 4 biến thể ±Copper**, 60/60 | Có thể ghép các cấu hình chung để mô tả |
| 90/10 (3 schemes) | Có cho 3 input sets | Có đủ 4 input variants | Có thể ghép có Copper, không mặc định cùng budget |
| 92,5/7,5 và 95/5 | Đủ 3 schemes | Chỉ Expanding | Thiếu V21 Rolling/Recency ở hai tỷ lệ |
| Copper-only | Có | Không | Thiếu V21 Copper-only |
| Không có Copper history | Không | Có | V20 thiếu exogenous-only; có thể phân tích ablation nội bộ V21 |
| Residual architectures | Có đủ LSTM/GRU/CNN/Ridge | Không | V21 tập trung Direct, SA-LSTM |
| SimpleRNN và SA-LSTM | Không | Có | V21 mở rộng danh mục kiến trúc |
| Walk-forward, 5 seeds, DM-HAC | Residual-GRU | **Chưa có file WF mới** | Không thể so sánh khả năng tổng quát hóa SA bằng thống kê |
| SA search tại từng variant | Không | Không (dùng SA cache) | Chưa có thực nghiệm search đầy đủ, ghi rõ giới hạn |
| Số tham số từng layer | Đã có CSV audit | Đã có CSV audit | 50 tổ hợp, 288 hàng layers |

## IX. Kết luận (Conclusion)

Nhóm đã tái hiện có giả định một cấu hình SA-LSTM theo bài báo và so sánh hệ thống với nhiều mô hình ML/DL trong V20/V21. Đóng góp quan trọng là **thiết kế giao thức H=1**, phân biệt thông tin ngoại sinh và Copper history, phân biệt forward-pass Direct với target Residual, kiểm tra Naive làm baseline và công khai những kết quả không vượt được baseline. Residual-GRU V20 ở một split cải thiện MAE 0,163% nhưng walk-forward chưa cho thấy lợi thế ổn định; SA-LSTM V21 chưa vượt Naive trên holdout. Việc thêm Copper history có lợi hay hại tùy tập biến, kiến trúc và điều kiện huấn luyện. Việc bổ sung 80/20 đã làm đầy ma trận holdout V21 ở ratio còn thiếu; riêng walk-forward SA cần tiếp tục nếu mục tiêu là kết luận về tổng quát hóa theo thời gian. Báo cáo không cần khẳng định phát minh thuật toán mới mà nên chứng minh hiểu đúng phương pháp, đối chứng có kiểm soát, lý giải cơ chế và trình bày giới hạn thẳng thắn.

**Cập nhật kết luận V22:** Bảng §5A.2 xác định fitted model Non-SA/SA-LSTM đạt MAE thấp nhất ở từng split (Naive được đặt thành baseline riêng). V22 không mang lại cải thiện đồng đều so với Legacy14/Paper3; các hiệu ứng phụ thuộc kiến trúc, tỷ lệ train/test và chiến lược. Việc SA search lại đồng thời với thay input khiến riêng so sánh reoptimized SA không thể quy kết tác động chỉ do feature. Bản báo cáo này không gọi các mô hình đứng đầu holdout là nghiệm đã được xác nhận thống kê.

## X. Tài liệu tham khảo (References)

1. Chen, Y. và cộng sự (2023), *Copper price prediction using LSTM recurrent neural network integrated simulated annealing algorithm*, PLOS ONE. (Cần đối chiếu metadata tác giả đầy đủ khi nộp.)
2. Diebold, F. X., & Mariano, R. S. (1995). *Comparing predictive accuracy*. Journal of Business & Economic Statistics, 13(3), 253–263.
3. Newey, W. K., & West, K. D. (1987). *A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix*. Econometrica.
4. Fama, E. F. (1970). *Efficient capital markets: A review of theory and empirical work*. Journal of Finance, 25(2), 383–417.
5. Dữ liệu nguồn nội bộ và artifacts kiểm toán: xem Phụ lục A.

## Phụ lục A. Danh mục dữ liệu nguồn, CSV, ảnh cần nhúng và khả năng tái lập

- `17-11-Copper-NonSA-Fair-Full-Protocol-V20(1).zip`: `grid_results.csv`, `grid_predictions.csv`, `quality.csv`, `spearman_training_only.csv`, `lagged_correlations_training_only.csv`, `feature_ablation.csv`, `residual_audit.csv`, `walk_forward_predictions.csv`, `dm_hac_by_seed.csv`, các ảnh EDA, predictions, walk-forward và `manifest.json`.
- `copper_sa_lstm_h1_ablation_v21_results.zip`: `results_h1_ablation.csv`, `predictions_h1_ablation.csv`, `ratio_summary.csv`, `strategy_summary.csv`, `copper_lag_ablation.csv`, ảnh và file HP cache.
- `copper_v21_addon_80_wf_plots(1).zip`: `holdout_80_results.csv` (60 hàng), `holdout_80_predictions.csv`, 10 ảnh PNG và `visualization_manifest.json`; manifest xác nhận **chưa có** walk-forward metrics/predictions.
- `model_parameter_counts_v20_v21.csv` (50 hàng) và `model_layer_parameter_details_v20_v21.csv` (288 hàng) là nguồn kiểm tra tham số.
- `BAO_CAO_TONG_HOP_COPPER_V20_V21_DA_DOI_CHUNG(2).md` là bản báo cáo tiền nhiệm, được giữ **nguyên văn trong Phụ lục B** để không bỏ mất nội dung/nhận định lịch sử.

### Hình ảnh sử dụng trong bản cuối

| Hình | File | Nội dung |
|---|---|---|
| 1 | `eda_levels_returns_volatility.png` | Giá, returns, rolling volatility |
| 2 | `eda_lagged_spearman.png` | Tương quan trễ Spearman |
| 3 | `eda_lagged_pearson.png` | Tương quan trễ Pearson |
| 4 | `actual_pred_0.9_copper_legacy14.png` | V20 actual vs predicted |
| 5 | `wf_actual_vs_pred.png` | V20 walk-forward actual vs predicted |
| 6 | `wf_delta_audit.png` | So sánh delta thực/dự báo |
| 7 | `plots/80_strategy_mae_all_features.png` | V21 80/20 strategy MAE (4 bộ) |
| 8 | `plots/80_strategy_rmse_all_features.png` | V21 80/20 strategy RMSE |
| 9 | `plots/80_ablation_paper3_mae.png` | Copper history Paper3 |
| 10 | `plots/80_ablation_legacy14_mae.png` | Copper history Legacy14 |

### Phụ lục A bổ sung: V22

- `[]18-11- Copper_NonSA_V22_WTI_Only_Kaggle.zip`: `grid_results.csv` (120), `grid_predictions.csv`, `walk_forward_predictions.csv` (11.070), `walk_forward_fold_metrics.csv` (180), EDA, bảng MAE theo split, fingerprint.
- `[]18-11copper_sa_lstm_wti_only_v22_results.zip`: `results_h1_ablation.csv` (30), `sa_search_trials.csv` (16), `fresh_sa_best_hp.json`, `experiment_manifest.json`, predictions, biểu đồ.
- Bảng chính ở §V-A lấy từ các CSV này; **phụ lục lịch sử V20/V21 dưới đây không cập nhật theo V22**.

## Phụ lục B. BẢN BÁO CÁO V20–V21 TRƯỚC KHI TÁI CẤU TRÚC (LƯU NGUYÊN VĂN)

> **Ghi chú bảo toàn:** Phụ lục này sao chép nguyên văn tài liệu người dùng cung cấp, **không áp dụng hiệu chỉnh tự động**. Nếu có bảng/khẳng định khác nội dung chính của báo cáo mới, sử dụng số liệu đối chiếu và phần hiệu chỉnh ở nội dung chính; bản cũ chỉ giữ để truy xuất nguồn và bảo đảm không lược mất thông tin.

# BÁO CÁO TỔNG HỢP THỰC NGHIỆM DỰ BÁO GIÁ ĐỒNG — V20 (HỢP NHẤT NON-SA V20 VÀ SA-LSTM V21)

**Ngày đối chiếu:** 18/11/2026  
**Định hướng:** Tái hiện có kiểm soát, đối chuẩn phương pháp, giải thích cách tính toán và giới hạn suy luận.  
**Tư cách nguồn:** Các số liệu trong báo cáo này được kiểm tra trên CSV/ZIP gốc và đối chiếu mã notebook; **không huấn luyện lại** mô hình trong quá trình biên soạn.  
**Lưu ý cách gọi:** V20 là giao thức Non-SA; V21 là đợt thực nghiệm SA-LSTM độc lập được **tích hợp vào bản báo cáo V20**, không ngụ ý cả hai chạy cùng một pipeline hoàn toàn đồng nhất.

## Tóm tắt điều hành

Hai bộ kết quả sử dụng **cùng dataset 2.710 hàng** (13/01/2016–13/11/2026), chung SHA-256 `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600`, cùng `LOOKBACK=22`, `HORIZON=1`, và seed 42 trong các phép đánh giá holdout. V20 gồm **360 lượt đánh giá (324 lượt fitted + 36 Naive)**; V21 gồm **100 lượt (80 fitted + 20 Naive)**. Tổng số là **460 lượt đánh giá, 404 lượt huấn luyện mô hình**; các lượt Naive lặp lại theo feature set/strategy không đại diện cho các mô hình khác nhau.

V20 cho thấy nhóm học sai phân/residual có MAE sát Naive; tốt nhất trên test 90/10 là Residual-GRU với Copper + Legacy14, MAE **0.076951** so với Naive **0.077077**, cải thiện **0.163%** (chưa phải bằng chứng thống kê). V21 cho thấy các mạng dự báo *mức giá trực tiếp* dùng cấu hình SA-LSTM vẫn chưa vượt Naive; tốt nhất ở 90/10 trong nhóm V21 là LSTM baseline với Copper + Legacy14 (MAE **0.247409**). Việc thêm Copper history có lợi rõ rệt trong một cấu hình LSTM–Legacy14 nhưng **không cải thiện nhất quán** ở các cấu hình SA-LSTM.

Giá trị chính của đề tài là phân biệt **cơ chế học mức giá**, **cơ chế học biến động**, **ảnh hưởng tập thông tin**, **cách lấy mẫu lịch sử** và **độ tin cậy của kết luận ngoài mẫu** — không phải tạo ra tuyên bố vượt baseline bằng mọi giá.

---

## 1. Mục tiêu và câu hỏi nghiên cứu

1. **Reproduction:** Thực thi lại SA-LSTM dùng siêu tham số báo cáo trong Chen et al. (2023), ghi rõ những phần phải tự giả định do bài báo không mô tả đủ (lookback, Dense head, exact preprocessing/horizon). Đây là **tái hiện theo thông số công bố**, không phải bản sao xác minh được mã gốc.
2. **Architecture comparison:** So sánh LSTM, GRU, CNN-LSTM, CNN-GRU, SimpleRNN và hai cấu hình SA-LSTM trong giới hạn các điều kiện kiểm soát được.
3. **Representation comparison:** So sánh dự đoán mức giá trực tiếp với dự đoán sai phân/residual (V20).
4. **Feature ablation:** Kiểm tra giá trị của Copper history và biến ngoại sinh.
5. **Training strategy:** Khảo sát Expanding, Rolling 756 và Recency Weighting.
6. **Robustness:** Sử dụng walk-forward nhiều seeds và DM-HAC cho **Residual-GRU**, không suy diễn kết quả này sang SA-LSTM.

## 2. Nguồn số liệu, giao thức và kiểm tra đối chứng

### 2.1. Nguồn dữ liệu và ranh giới dự báo

- Dataset: `dataset_9_commodities_daily.csv`, **2.710 quan sát**, 13/01/2016–13/11/2026; ZIP báo không có missing ở 10 cột được kiểm tra.
- Dự báo **Copper(t)** chỉ từ cửa sổ các phiên **t−22 … t−1** (`HORIZON=1`, `LOOKBACK=22`). Các quan sát cùng ngày *t* không được đưa vào tensor input.
- Scaler X/y được fit trên vùng *train trước validation* rồi transform validation/test. Các mô hình trong hai notebook đều đánh giá MAE, RMSE, MSE trên **thang giá gốc** sau inverse transform.
- Dataset có cùng hash không tự chứng minh dữ liệu giá từ các thị trường khác nhau đã được căn chỉnh theo **giờ thực sự khả dụng**. Việc ghép ngày theo timestamp không thay thế audit múi giờ/giờ đóng cửa.

### 2.2. Bộ đặc trưng

| Bộ thông tin | V20 | V21 | Tính chất |
|---|---|---|---|
| Copper only | `copper_only`: Copper | Không có | Chỉ thông tin tự hồi quy |
| Paper3 + Copper | `copper_paper3`: Copper, WTI, Gold, Silver | `paper3_with_copper` | Ghép được để so sánh mô tả |
| Paper3 không Copper | Không có | `paper3_without_copper` | Gần đặc tả ba ngoại sinh của paper |
| Legacy14 + Copper | `copper_legacy14`: Copper, Aluminium, Tin, Silver, Gold | `legacy14_with_copper` | Ghép được để so sánh mô tả |
| Legacy14 không Copper | Không có | `legacy14_without_copper` | Kiểm tra giá trị Copper lag |

**Quy ước:** “Legacy14” là **tên kế thừa của nhóm đặc trưng**, không phải 14 features: V21 có 4 ngoại sinh hoặc 5 biến khi thêm Copper.

### 2.3. Ma trận thực nghiệm và mức độ so sánh được

- V20: 4 tỷ lệ train/test × 3 bộ đầu vào × 3 scheme × (9 fitted models + Naive) = **360**; 324 fitted.
- V21: ở **90/10** chạy 4 bộ đầu vào × 3 scheme × (4 fitted models + Naive) = **60**; ở **92.5/7.5** và **95/5**, mỗi tỷ lệ chỉ chạy 4 đầu vào × Expanding × 5 mô hình = **20**; tổng **100**.
- Cùng khoảng test ở một ratio, cùng nhãn target, cùng lookback/horizon, song **khác kiến trúc, epochs, early stopping và phân bổ các cấu hình**. Vì vậy chỉ có thể nói đây là **so sánh holdout cùng đích dự báo**, chưa phải đánh giá nguyên nhân nhân quả của *thuật toán SA*.
- V20: neural network tối đa **65 epochs**, early stopping patience 9; V21 `FULL_MODE`: tối đa **100 epochs**, early stopping patience 9. Hai bên cùng batch size 64 cho fitted neural networks. V21 **không chạy SA search trong đợt này**, mà dùng bộ tham số SA đã lưu.
- Trong cả hai notebook, `rolling_756` giới hạn dữ liệu fit **một lần theo mỗi split**, không phải rolling refit ở từng ngày test; V20 chỉ triển khai **walk-forward định kỳ** độc lập cho Residual-GRU.

### 2.4. Kiểm tra tính khớp độc lập giữa V20 và V21

**LSTM baseline** của V21 có cấu trúc (64, 32, Dense16), tương ứng LSTM Direct của V20. So sánh 90/10 + Expanding từ hai tệp CSV:

| Bộ đầu vào | MAE V20 | MAE V21 | |Chênh lệch| |
| --- | --- | --- | --- |
| Paper3 + Copper | 1.1639640 | 1.1639639 | 0.00000012 |
| Legacy14 + Copper | 0.2474096 | 0.2474090 | 0.00000057 | |

Chênh lệch dưới 10⁻⁶ là phép kiểm tra đối chứng hữu ích về cùng dữ liệu/triển khai LSTM, **không chứng minh tất cả kiến trúc và training budget hoàn toàn tương đương**.

---

## 3. Cách mô hình tính toán: từ input đến dự báo

### 3.1. Trượt cửa sổ dữ liệu

Với `LOOKBACK=22`, một mẫu đầu vào X có kích thước `(22, F)` và nhãn là `y_t = Copper(t)`. Nếu Paper3 không Copper, `F=3`; nếu Paper3 + Copper, `F=4`; nếu Legacy14 + Copper, `F=5`. Một batch size 64 có tensor `(64, 22, F)`.

### 3.2. Các kiến trúc dự báo Direct

- **SimpleRNN:** Trạng thái ẩn cập nhật theo `h_k = tanh(W_x x_k + W_h h_{k−1} + b)`. Hai tầng recurrent tạo biểu diễn chuỗi; lớp Dense dự báo giá chuẩn hóa. Dễ triển khai nhưng có thể khó giữ phụ thuộc dài do gradient suy giảm.
- **LSTM:** Tính các cổng quên `f`, ghi `i`, xuất `o` và ô nhớ `c` để điều khiển lượng thông tin giữ lại; tầng Dense biến hidden state cuối thành **mức Copper(t)**. LSTM được thiết kế để học phụ thuộc theo thời gian, **không tự đảm bảo ngoại suy giá tốt**.
- **GRU:** Dùng cổng cập nhật `z` và cổng reset `r` để tạo `h_k = (1−z_k)·h_{k−1} + z_k·	ilde{h}_k` (quy ước cổng có thể đảo tùy tài liệu). Ít cơ chế cổng hơn LSTM, chi phí thường thấp hơn trong cấu hình tương đương.
- **CNN-LSTM / CNN-GRU:** Conv1D causal kernel 3 trích xuất mẫu cục bộ → MaxPool → các tầng recurrent học phụ thuộc trình tự → Dense(16) → Dense(1). Tầng convolution **không tự động tạo thông tin dự báo mới**.
- **Paper SA-LSTM V21:** LSTM(39) → Dropout(0.2) → LSTM(111) → Dropout(0.2) → Dense(32, ReLU) → Dense(1, linear), Adam LR `0.00005`, MSE. `Dense(32)` là **giả định triển khai**, không được gán là cấu hình được paper xác nhận.
- **Reoptimized SA-LSTM V21:** Cùng công thức hai tầng LSTM/Dense; số units và LR lấy từ cache SA trước đó: Paper3 `(22, 91, 0.000922)`, Legacy14 `(22, 83, 0.000655)`. **Không tìm tham số mới trong V21**.

`SA` (Simulated Annealing) **không nằm trong forward pass**: SA là thuật toán tìm *hyperparameters* như units/LR, sau đó LSTM được train bằng gradient descent/Adam. Một khi cố định hyperparameters, dự đoán của “SA-LSTM” chính là forward pass của LSTM đã huấn luyện.

### 3.3. Residual Learning — khác Direct ở nhãn và cách hoàn nguyên

- **Direct:** `ŷ_t = f_θ(X_{t−22:t−1})`; network tối ưu MSE giữa dự báo và `Copper(t)`.
- **Residual:** Học `Δ_t = Copper(t) − Copper(t−1)` với `Δ̂_t = f_θ(X_{t−22:t−1})`; đầu ra `ŷ_t = Copper(t−1) + Δ̂_t`.
- **Naive:** `ŷ_t = Copper(t−1)`; đây là trường hợp residual đặc biệt `Δ̂_t = 0`.
- **Ridge Residual:** Tuyến tính trên vector hóa input 22×F và tối ưu MSE + `α||β||²` (α=1 trong V20).

**Giải thích lý thuyết, không phải kết luận đo được:** Với loss MSE, ước lượng Bayes của Δ là `E[Δ_t | X]`; nếu tín hiệu có điều kiện yếu và trung bình Δ gần 0, mạng có thể học dự báo rất nhỏ. Để khẳng định điều đó xảy ra cần nhìn **độ lệch chuẩn/ tương quan của Δ dự báo với Δ thực**, được đo ở §5.

### 3.4. Ba chiến lược train

- **Expanding:** Sử dụng mọi mẫu từ đầu chuỗi đến ranh giới fit; lợi về số mẫu nhưng có thể giữ quan hệ lỗi thời.
- **Rolling 756:** Dùng 756 quan sát gần vùng train, thải quá khứ xa; có thể thích nghi với thay đổi nhưng làm giảm số mẫu.
- **Recency Weighting:** Giữ mọi mẫu nhưng giảm trọng số của mẫu cũ, `w_i ∝ exp(−ln(2)·age_i/252)`; half-life = 252 phiên. Chỉ thay loss có trọng số, **không phải thay kiến trúc**.

---

## 4. Kết quả holdout — đối chiếu mô hình trên cùng 90/10

**Protocol cố định:** 271 phiên test từ 29/10/2025 đến 13/11/2026. MAE/RMSE đều trên giá Copper gốc. Các hàng V20 dưới đây chỉ lấy `expanding`, còn bảng chiến lược riêng trình bày ở §6. 

### 4.1. Non-SA V20: mô hình tốt nhất theo từng kiến trúc (90/10 + Expanding)

| Mô hình | Bộ input tốt nhất | MAE | RMSE | DA (%) | Skill MAE vs Naive |
| --- | --- | --- | --- | --- | --- |
| Naive | Không cần features | 0.077077 | 0.101033 | — | 0.000% |
| Residual-GRU | copper_legacy14 | 0.076951 | 0.100810 | 52.77 | +0.163% |
| Residual-LSTM | copper_legacy14 | 0.076978 | 0.100948 | 52.77 | +0.128% |
| Residual-CNN-LSTM | copper_legacy14 | 0.077245 | 0.100856 | 46.86 | -0.219% |
| Residual Ridge | copper_only | 0.077306 | 0.101190 | 47.60 | -0.297% |
| Residual-CNN-GRU | copper_only | 0.078067 | 0.101486 | 45.76 | -1.284% |
| GRU (Direct) | copper_legacy14 | 0.240815 | 0.302923 | 53.51 | -212.436% |
| LSTM (Direct) | copper_legacy14 | 0.247410 | 0.304620 | 52.77 | -220.991% |
| CNN-GRU (Direct) | copper_paper3 | 0.296379 | 0.359605 | 52.40 | -284.524% |
| CNN-LSTM (Direct) | copper_legacy14 | 0.393469 | 0.492822 | 49.82 | -410.490% | |

**Diễn giải:** Cách ghi “bộ input tốt nhất” là *xếp sau khi nhìn test*, chỉ có giá trị **khám phá/hậu nghiệm**; không được dùng để tuyên bố mô hình đó đã được chọn trước và tổng quát hóa ngoài mẫu. Ở cùng 90/10, Residual-GRU + Legacy14 đạt MAE 0.076951 so với Naive 0.077077, chênh lệch chỉ **0.000126** đơn vị giá (0.163%). Phải đánh giá trên thời đoạn độc lập trước khi tuyên bố cải thiện.

### 4.2. SA-LSTM V21: MAE theo cấu hình đầu vào (90/10 + Expanding)

| Mô hình | Paper3 + Cu | Paper3 − Cu | Legacy14 + Cu | Legacy14 − Cu |
| --- | --- | --- | --- | --- |
| Naive | 0.077077 | 0.077077 | 0.077077 | 0.077077 |
| LSTM | 1.163964 | 1.145988 | 0.247409 | 0.854128 |
| SimpleRNN | 1.385730 | 1.001232 | 0.831186 | 0.962108 |
| SA-LSTM (paper HP) | 0.978075 | 0.540720 | 0.470929 | 0.464991 |
| SA-LSTM (SA cache) | 0.877420 | 0.595605 | 0.510467 | 0.466595 | |

Trong nhóm V21, MAE tốt nhất ngoài Naive ở 90/10 thuộc **LSTM baseline + Legacy14 + Copper** (**0.247409**), tương ứng RMSE **0.304619**; vẫn lớn gấp khoảng **3.21 lần** MAE của Naive. SA-LSTM với thông số công bố của paper **không thắng** cấu hình SA đã lưu ở mọi trường hợp, và ngược lại. Không tồn tại kết luận “SA luôn tốt hơn LSTM” từ kết quả V21.

### 4.3. Kết quả Naive theo các kỳ test

| Train/Test | Phiên test | MAE Naive | RMSE Naive | V21 có đánh giá? |
| --- | --- | --- | --- | --- |
| 80/20 | 542 | 0.069001 | 0.108000 | Không |
| 90/10 | 271 | 0.077077 | 0.101033 | Có |
| 92.5/7.5 | 204 | 0.075116 | 0.098399 | Có |
| 95/5 | 136 | 0.072799 | 0.096612 | Có | |

Các tỷ lệ 80/20, 90/10, 92.5/7.5, 95/5 có **những giai đoạn test khác nhau**; so sánh MAE chéo ratio cần thận trọng. Đặc biệt, V21 **không có 80/20**.

### 4.4. Sửa những chỗ không chính xác trong báo cáo V20 ban đầu

Dựa trên **toàn bộ `grid_results.csv`**, cấu hình có MAE nhỏ nhất (không tính Naive) tại từng split được xác định lại:

| Train ratio | Mô hình | Features | Strategy | MAE | Skill vs Naive |
| --- | --- | --- | --- | --- | --- |
| 0.800 | gru_residual | copper_paper3 | expanding | 0.068780 | +0.321% |
| 0.900 | gru_residual | copper_legacy14 | expanding | 0.076951 | +0.163% |
| 0.925 | cnn_lstm_residual | copper_only | recency_weighted | 0.075050 | +0.089% |
| 0.950 | gru_residual | copper_paper3 | recency_weighted | 0.072357 | +0.608% | |

**Hiệu chỉnh:** Bản báo cáo V20 cũ ghi sai cấu hình dẫn đầu tại **92.5%** và **95%**. Ở 92.5%, thực tế `cnn_lstm_residual/copper_only/recency_weighted` dẫn đầu; ở 95%, `gru_residual/copper_paper3/recency_weighted` dẫn đầu. Mức skill cao nhất của *full grid* là **+0.608%**, nên câu “tất cả cải thiện đều dưới 0.40%” trong báo cáo cũ là không chính xác. Tất cả các kết quả dẫn đầu này đều được chọn khi nhìn test, do đó **không được diễn giải như chứng cứ đã xác nhận về tối ưu mô hình**.

---

## 5. Cơ chế và nguyên nhân khả dĩ của kết quả

### 5.1. Vì sao nhóm Direct có thể sai số lớn?

Các mô hình Direct dự báo **mức giá chuẩn hóa**, rồi inverse-transform về đơn vị gốc. Khi giá test vượt vùng train, MSE cao có thể đến từ khả năng ngoại suy hạn chế, bias của đầu ra, dynamic range và cấu hình head. Đây là **những cơ chế cần kiểm tra**, không đủ căn cứ kết luận chuỗi chắc chắn `I(1)` hoặc xảy ra *concept drift* chỉ nhờ MAE cao. Muốn khẳng định cần kiểm định tính dừng, so sánh phân phối, lỗi theo thời kỳ và kiểm tra giới hạn output activation.

### 5.2. Vì sao Residual gần Naive?

Residual cộng `Copper(t−1)` trước khi dự báo; mạng chỉ phải điều chỉnh phần thay đổi. `residual_audit.csv` của V20 ở 90/10 cung cấp các chỉ số trên giá gốc. Dưới đây lấy **cùng bộ input `copper_paper3` và scheme `expanding`**:

| Mô hình | std(Δ thực) | std(Δ dự báo) | Corr(Δ̂, Δ) | MAE |
| --- | --- | --- | --- | --- |
| lstm_residual | 0.10097 | 0.00580 | 0.0517 | 0.077006 |
| gru_residual | 0.10097 | 0.00567 | 0.0401 | 0.077143 |
| cnn_lstm_residual | 0.10097 | 0.01026 | 0.0314 | 0.077432 |
| ridge_residual | 0.10097 | 0.02365 | -0.0593 | 0.080712 | |

Mạng residual dự báo Δ có độ dao động nhỏ hơn đáng kể Δ thực. Do vậy, `ŷ_t = Copper(t−1)+Δ̂_t` sát với `Copper(t−1)` và MAE gần Naive. **Đây là quan sát trực tiếp**. Giả thuyết nguyên nhân là loss MSE đẩy mô hình về dự báo trung bình có điều kiện gần 0 khi phần biến động khó dự báo; muốn kết luận đây là cơ chế duy nhất cần thêm đối chứng loss/learning curves và phân tích return predictability.

### 5.3. Feature ablation: có/không có Copper(t−1)

V21 đánh giá trực tiếp việc bỏ chuỗi Copper khỏi **toàn bộ input 22 phiên**; đây không phải chỉ bỏ riêng một scalar `Copper(t−1)`. Lấy 90/10 + Expanding, mức thay đổi phần trăm MAE khi **thêm** Copper history (dương = giảm MAE):

| Mô hình | Lợi ích thêm Copper — Paper3 | Lợi ích thêm Copper — Legacy14 |
| --- | --- | --- |
| LSTM đối chứng (V21) | -1.57% | +71.03% |
| SimpleRNN (V21) | -38.40% | +13.61% |
| SA-LSTM cấu hình paper (V21) | -80.88% | -1.28% |
| SA-LSTM tham số SA đã lưu (V21) | -47.32% | -9.40% | |

Kết quả cho thấy Copper history **hữu ích cho LSTM baseline + Legacy14**, nhưng ở Paper3 lại gây hại cho cả bốn fitted models (trên split này). Do đó **không được tuyên bố Copper lag luôn tăng độ chính xác**. Trong V20, toàn bộ cấu hình đều chứa Copper, nên ablation của V20 trả lời câu hỏi khác: *ngoại sinh có bổ sung thông tin so với Copper-only hay không?*

### 5.4. Vai trò ngoại sinh và lagged correlation

Ở V20, Spearman giữa **biến động hàng hóa lag-1** và biến động Copper trên tập train thấp (Gold ~0.0910, Silver ~0.0388, WTI ~0.0142, Aluminium ~0.0093, Tin ~−0.0185 theo báo cáo/CSV). Đây chỉ là liên hệ **đơn biến tại độ trễ cụ thể**, không chứng minh biến ngoại sinh vô ích dưới quan hệ phi tuyến, lag khác hoặc tương tác với Copper. Feature ablation cho thấy hiệu quả của các ngoại sinh còn tùy mô hình/strategy.

### 5.5. Directional Accuracy (DA) và sự khác biệt giữa giá và biến động

`DA` được tính dựa trên dấu của `(ŷ_t−Copper(t−1))` so với `(Copper(t)−Copper(t−1))`. MAE thấp không đảm bảo đoán chiều tốt. Với Naive, Δ̂ = 0 cho mọi ngày nên **DA kiểu hai lớp tăng/giảm là không áp dụng**. Trong V20, công thức `sign(0)==sign(Δ)` có thể tạo ra một giá trị DA rất nhỏ ở ngày Δ thực = 0, nên khi trình bày nên ghi `—`, không diễn giải như tín hiệu giao dịch.

---

## 6. Expanding, Rolling 756, Recency — kết quả và diễn giải

### 6.1. V21: Legacy14 + Copper, 90/10 (MAE)

| Mô hình | Expanding | Rolling 756 | Recency |
| --- | --- | --- | --- |
| Naive | 0.077077 | 0.077077 | 0.077077 |
| LSTM | 0.247409 | 0.958466 | 0.659060 |
| SimpleRNN | 0.831186 | 1.115149 | 1.045665 |
| SA-LSTM paper HP | 0.470929 | 0.773310 | 0.705350 |
| SA-LSTM SA cache | 0.510467 | 0.977821 | 0.735621 | |

Ở bộ Legacy14 + Copper này, Expanding thắng về MAE cho cả bốn mạng; nhưng đây là **một cấu hình và một split**, không phải định luật chung. Rolling có ít mẫu fit hơn; Recency giảm trọng số mẫu cũ; hai cách không nhất thiết cải thiện nếu cấu trúc dữ liệu còn ổn định hoặc dung lượng mẫu không đủ.

### 6.2. Tại sao không thể coi đây là walk-forward?

Ba scheme ở lưới chính **chọn tập fit tại đầu test** và dự đoán cả giai đoạn test sau đó. Với `rolling_756`, cửa sổ không trượt và retrain mỗi ngày test. Muốn nhận xét “model thích nghi với thị trường đang thay đổi” một cách đáng tin cần **walk-forward refit** theo lịch xác định trước.

---

## 7. Walk-forward V20: mức độ tin cậy của Residual-GRU

V20 chạy Residual-GRU + Copper/Paper3 + Expanding, 2.214 phiên out-of-sample (2018–2026), retrain mỗi 63 phiên, 5 seed (41–45). Loss differential sử dụng sai số tuyệt đối, sai số chuẩn Newey–West HAC lag 5. Kết quả:

| Seed | N test | MAE GRU | MAE Naive | GRU − Naive | p (2 phía) | Ý nghĩa 5% |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | 2214 | 0.045110 | 0.045127 | -0.000017 | 0.7043 | Không ở 5% |
| 42 | 2214 | 0.045189 | 0.045127 | 0.000062 | 0.3656 | Không ở 5% |
| 43 | 2214 | 0.045447 | 0.045127 | 0.000320 | 0.0066 | Có; Naive tốt hơn |
| 44 | 2214 | 0.045502 | 0.045127 | 0.000375 | 0.0101 | Có; Naive tốt hơn |
| 45 | 2214 | 0.045388 | 0.045127 | 0.000262 | 0.0788 | Không ở 5% | |

**Diễn giải đúng:** Seed 41 và 42 không bác bỏ giả thuyết bằng nhau về *expected loss*; seed 43, 44 có bằng chứng Naive tốt hơn ở mức 5%; **seed 45 p=0.0788, không có ý nghĩa tại 5%** (chỉ ở mức 10% nếu áp dụng). **Không đủ chứng cứ Residual-GRU cải thiện có ý nghĩa thống kê so với Naive**. Kiểm định nhiều seeds trên những dự báo có cùng giai đoạn không tự tạo năm tập test độc lập.

**Giới hạn suy rộng:** V21 SA-LSTM chưa có walk-forward hay DM-HAC riêng. Không được viết “SA-LSTM không tổng quát hóa” từ riêng V20; hiện chỉ được nói SA-LSTM có MAE kém hơn Naive trên các holdout V21.

---

## 8. Ma trận thiếu kết quả khi hợp nhất hai bộ

| Hạng mục / Lát cắt | Non-SA V20 | SA-LSTM V21 | Có thể ghép so sánh trực tiếp? |
|---|---|---|---|
| 80/20 × Paper3 + Copper, Legacy14 + Copper | Đủ 3 scheme | **Thiếu toàn bộ** | Không |
| 90/10 × Paper3 + Copper, Legacy14 + Copper | Đủ 3 scheme | Đủ 3 scheme | **Có, mô tả holdout**, khác training budget |
| 92.5/7.5 và 95/5 × có Copper | Đủ 3 scheme | **Chỉ Expanding** | Chỉ hàng Expanding |
| Paper3 / Legacy14 **không Copper** | **Thiếu** | Có (3 scheme ở 90/10, Expanding ở tỷ lệ khác) | So sánh trong V21; không ghép V20 |
| Copper only | Có | **Thiếu** | Đánh giá Copper-only trong V20 |
| Residual-LSTM/GRU/CNN và Ridge | Có | Không | Chỉ V20 |
| SimpleRNN và hai cấu hình SA-LSTM | Không | Có | Chỉ V21 |
| SA-LSTM **re-search** tại từng input/scheme/split | Không | **Không**; sử dụng SA cache cũ | Không thể kết luận SA tìm được tối ưu toàn cục |
| Walk-forward 2.214 phiên, 5 seeds, DM-HAC | Chỉ Residual-GRU | **Thiếu** | Không thể so sánh thống kê SA vs Residual-GRU |
| Đánh giá DA & biến thiên Δ | Có | Có | Có với chú ý DA Naive |
| Các model trên **cùng** ngân sách epochs/HP tuning | Chưa chuẩn hóa hoàn toàn | Chưa chuẩn hóa hoàn toàn | Chưa phải controlled algorithm-level test |

**Ưu tiên bổ sung nếu chỉ còn thời gian cho môn học:** (1) báo cáo giới hạn reproduction chính xác; (2) bảng 90/10 + Expanding trên cùng dữ liệu; (3) diễn giải input/target/optimization; (4) thí nghiệm thêm Copper history đã có từ V21; (5) cân nhắc chạy SA-LSTM ở 80/20 + Expanding nếu cần bảng 4 ratio, **không cần thiết phải chạy lại toàn bộ 460 jobs**.

---

## 9. Giới hạn khoa học và các khẳng định cần tránh

1. **Tối ưu tham số SA:** `paper_sa_lstm` là **LSTM huấn luyện với siêu tham số Chen et al. công bố**, không phải SA search được chạy trong V21. `reopt_sa_lstm` dùng cấu hình SA được tìm ở đợt trước. Kết quả chỉ phản ánh **lợi/hại của hyperparameter choices và training protocol hiện tại**, chưa đo giá trị đầy đủ của tối ưu SA.
2. **Paper reproduction:** `HORIZON=1`, `LOOKBACK=22`, `Dense(32)`, EarlyStopping trong V21 là lựa chọn/giả định thực nghiệm; không được gán cho Chen et al. nếu bài báo không công bố. Đặc biệt **bản paper 3 ngoại sinh không Copper** khác biệt có chủ ý với phần extension thêm lịch sử Copper.
3. **Thống kê:** Nhiều phép xếp hạng trên cùng test tạo nguy cơ selection bias. Kết luận “model tốt nhất” chỉ là *trong các cấu hình được đánh giá*; nếu muốn tuyên bố cải tiến, cần test mới hoặc nested/rolling validation.
4. **Concept drift/random walk/I(1):** MAE cao, ngoại suy kém và Δ̂ gần 0 có thể phù hợp với các giả thuyết này, nhưng **không phải bằng chứng xác nhận** concept drift, martingale difference hay bậc tích hợp I(1).
5. **Độ đo thống nhất:** `mse_scaled`, `val_mse_scaled` chịu ảnh hưởng scaler fit của từng scheme nên **không dùng làm thước đo xếp hạng xuyên scheme**; ưu tiên MAE/RMSE theo giá gốc và test timestamps giống nhau.
6. **DA:** Đừng xem 50–54% DA là “dự báo hướng đáng tin” nếu chưa so với base-rate, uncertainty và kiểm định phù hợp.
7. **Nguồn code:** Chỉ kiểm toán cấu trúc và số liệu đầu ra đã gửi, **không thực hiện rerun**; các file ZIP không thay thế xác minh phiên bản môi trường, dữ liệu phát hành tại thời điểm thực tế và lịch sử tuning.

## 10. Kết luận tổng hợp theo mục tiêu môn học

- **Về tái hiện:** Nhóm đã chạy cấu hình LSTM hai tầng với các HP công bố trong bài báo và một tập SA HP được tìm trước đó. Nhóm cần gọi đây là **reproduction có giả định** vì thiếu mô tả đầy đủ từ tác giả gốc.
- **Về hiểu thuật toán:** Việc phân tách Direct/Residual cho thấy hình thức biểu diễn target quyết định cách mạng học: Direct học toàn bộ mức giá, Residual học sai phân và cộng lại giá phiên trước. Các cổng LSTM/GRU xử lý thông tin dọc 22 phiên; SA tìm hyperparameters chứ không dự báo trực tiếp.
- **Về độ chính xác:** Trên 90/10, Residual-GRU V20 có MAE tốt nhất trong các mô hình được đánh giá (0.076951), nhưng lợi thế trước Naive chỉ 0.163%; V21 SA-LSTM chưa vượt Naive. Walk-forward V20 không hỗ trợ tuyên bố Residual-GRU thắng Naive một cách ổn định.
- **Về yếu tố dữ liệu:** Copper history có lúc giúp giảm MAE đáng kể (LSTM/Legacy14 V21), nhưng không tạo lợi ích phổ quát; chỉ số tương quan có độ trễ nhỏ không đủ để kết luận ngoại sinh vô ích.
- **Về giá trị học thuật:** Không cần cố tăng số mô hình. Giá trị lớn nhất của đồ án là giải thích rõ cách tính, đối chứng giả thuyết, nêu các trường hợp không cải thiện và xác định ranh giới những gì có/không có đủ bằng chứng.

---

## 11. Tài liệu nguồn và khả năng truy xuất

**Nguồn dữ liệu đã đối chiếu (tên file do nhóm cung cấp):**

- `17-11-Copper-NonSA-Fair-Full-Protocol-V20.zip`: `grid_results.csv`, `grid_predictions.csv`, `fair_table_0.900.csv`, `feature_ablation.csv`, `residual_audit.csv`, `dm_hac_by_seed.csv`, `walk_forward_predictions.csv`, `manifest.json`, ...
- `copper_sa_lstm_h1_ablation_v21_results.zip`: `results_h1_ablation.csv`, `predictions_h1_ablation.csv`, `main_comparison_90_expanding.csv`, `strategy_summary.csv`, `fixed_sa_hyperparameters.json`, ...
- `17-11-Copper-NonSA-Fair-Full-Protocol-V20(1).ipynb` và `Copper_SA_LSTM_H1_Full_Comparison_Kaggle.ipynb`: dùng để kiểm tra tạo cửa sổ, scaling, cơ chế loss, kiến trúc và chương trình train.
- `BAO_CAO_DANH_GIA_17-11-COPPER_NON_SA_FAIR_FULL_PROTOCOL_V20.md`: tài liệu V20 gốc, được đối chiếu và hiệu chỉnh các nhận định chưa khớp CSV.

**Tài liệu khoa học nền tảng:** Chen et al. (2023), *Copper price prediction using LSTM recurrent neural network integrated simulated annealing algorithm*, PLOS ONE; Diebold & Mariano (1995), *Comparing Predictive Accuracy*; Newey & West (1987), HAC covariance. Các thông tin paper không được xác minh trực tiếp trong lần biên soạn này được ghi là thông số do nhóm cung cấp và giả định triển khai.


