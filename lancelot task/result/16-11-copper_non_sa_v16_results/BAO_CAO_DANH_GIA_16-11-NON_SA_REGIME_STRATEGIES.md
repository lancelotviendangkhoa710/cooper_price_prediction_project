# ĐÁNH GIÁ THỰC NGHIỆM CÁC MÔ HÌNH NON-SA VÀ CHIẾN LƯỢC THÍCH NGHI CHẾ ĐỘ — 16-11

## Tóm tắt

Báo cáo này đánh giá hệ thống thực nghiệm ngày 16-11 trên chuỗi giá đồng (*Copper*) gồm 2.710 quan sát (13/01/2016 – 13/11/2026), tập trung vào hai hướng chính: (1) Khảo sát các kiến trúc mạng học sâu không qua tối ưu Simulated Annealing (Non-SA: LSTM, GRU, CNN1D-LSTM, CNN-GRU) đối chiếu giữa hai bộ biến `legacy14` và `paper3`; (2) Thử nghiệm các chiến lược thích nghi chế độ thị trường (*Regime Adaptation*: Expanding, Rolling 756 phiên, Recency Weighting) kết hợp phương pháp học phần dư (*Residual Learning* với Residual-GRU và Ridge Delta). Toàn bộ thí nghiệm ngày 16-11 duy trì thiết lập $HORIZON = 0$ (nowcasting cùng ngày).

Kết quả thực nghiệm cho thấy:
1. **Mô hình dự báo mức giá trực tiếp (Direct Level)**: Naive Persistence vượt trội toàn bộ các mạng học sâu ở cả ba tỷ lệ chia 90/10, 92,5/7,5 và 95/5. RMSE giá gốc của các mạng nơ-ron cao hơn Naive từ 4,5 đến 16,7 lần. GRU và CNN-GRU là hai kiến trúc học sâu ổn định nhất trong nhóm direct.
2. **Chiến lược cửa sổ huấn luyện**: Khi dự báo mức giá trực tiếp, Rolling 756 phiên cải thiện nhẹ RMSE giá gốc cho GRU (từ 0,5572 xuống 0,5002) và CNN-GRU (từ 0,6284 xuống 0,5485) trên test 90/10, nhưng MSE chuẩn hóa lại cao hơn do miền co giãn của scaler rolling hẹp hơn.
3. **Mô hình học phần dư (Residual Learning)**: Ridge Delta đạt MSE_scaled 0,000887 và RMSE 0,09420 (giảm MAE 6,23% so với Naive Persistence trên phân chia expanding 90/10). Tuy nhiên, đây là ưu thế xuất phát từ việc sử dụng giá đóng cửa các hàng hóa ngoại sinh cùng ngày $t$ ($HORIZON=0$). Trong khi đó, Residual-GRU rolling 756 có MAE thua Naive 4,40% và chỉ thắng Naive ở 41,33% số ngày giao dịch trong phép kiểm định từng cặp ngày.

**Từ khóa:** giá đồng, chuỗi thời gian, Non-SA, GRU, CNN-GRU, rolling window, recency weighting, residual learning, nowcasting.

---

## I. Mục tiêu và phạm vi

Thí nghiệm ngày 16-11 được thiết kế nhằm giải quyết các câu hỏi mở từ báo cáo ngày 14-11 và 15-11:
1. **Định chuẩn mô hình Non-SA**: Đánh giá hiệu năng thực tế của các kiến trúc học sâu chuẩn (LSTM, GRU, CNN1D-LSTM, CNN-GRU) khi không dùng thuật toán tìm kiếm Simulated Annealing (SA), nhằm phân tách rõ ràng xem ưu thế/hạn chế nằm ở bản thân kiến trúc nơ-ron hay ở quá trình tối ưu SA.
2. **Ablation tập biến (Paper3 vs. Legacy14)**: So sánh khách quan giữa bộ biến gốc của Chen và cộng sự (2023) (`paper3`: WTI, Gold, Silver) và bộ biến 4 hàng hóa từ ngày 14-11 (`legacy14`: Aluminium, Tin, Silver, Gold) trên cùng điều kiện chuẩn hóa và chia tập.
3. **Chiến lược ứng phó dịch chuyển chế độ (Regime Shift)**: Kiểm tra giả thuyết liệu việc giới hạn cửa sổ lịch sử (`rolling_756` ~ 3 năm giao dịch) hoặc chiết khấu quá khứ xa bằng trọng số suy giảm hàm mũ (`recency_weighted`, chu kỳ bán rã 252 phiên) có giúp mô hình thích nghi tốt hơn so với toàn bộ dữ liệu lịch sử (`expanding`).
4. **Học phần dư (Residual Learning)**: Tái cấu trúc mục tiêu học từ việc dự đoán trực tiếp mức giá $Copper(t)$ sang dự đoán biến động phần dư $\Delta Copper(t) = Copper(t) - Copper(t-1)$, sau đó cộng lại giá $Copper(t-1)$ (Naive Persistence) để triệt tiêu sai số trôi dạt do phân phối giá dịch chuyển.

*Phạm vi giới hạn:* Thí nghiệm ngày 16-11 chạy trên notebook `16-11-NonSA-Regime-standalone.ipynb` với ngân sách tính toán `FULL_MODE` (100 epoch, early stopping patience 7). Mặc định $HORIZON = 0$, nghĩa là dự đoán giá đồng ngày $t$ có sử dụng giá các biến ngoại sinh tại ngày $t$ (nowcast), chưa phải là dự báo tương lai trước ngày $t$ ($HORIZON = 1$).

---

## II. Thiết kế thực nghiệm và phương pháp

### 2.1. Dữ liệu và phân chia tập

Dữ liệu gồm 2.710 phiên giao dịch từ ngày 13/01/2016 đến ngày 13/11/2026 lấy từ `dataset_9_commodities_daily.csv`. Dữ liệu sạch, không khuyết thiếu, không trùng lặp ngày. Lookback cố định 22 phiên.

Dữ liệu được chia tuần tự theo thời gian thành ba mốc kiểm tra lồng nhau. Tập validation chiếm 10% vùng dữ liệu trước mốc test:

| Tỷ lệ train/test | Train | Validation | Test | Khoảng thời gian test |
|---|---:|---:|---:|---|
| **90/10** | 2.174 | 244 | 271 | 29/10/2025 – 13/11/2026 |
| **92,5/7,5** | 2.234 | 251 | 204 | 30/01/2026 – 13/11/2026 |
| **95/5** | 2.295 | 258 | 136 | 07/05/2026 – 13/11/2026 |

MinMaxScaler được fit duy nhất trên tập train (từ `fit_start` đến `val_start`), sau đó áp dụng biến đổi cho validation và test nhằm loại trừ rò rỉ dữ liệu.

### 2.2. Hai tập biến đầu vào (Feature Sets)

- **`paper3`**: WTI, Gold, Silver (theo lựa chọn dựa trên tương quan Spearman trong nghiên cứu Chen và cộng sự, 2023).
- **`legacy14`**: Aluminium, Tin, Silver, Gold (bộ 4 kim loại cố định từ thử nghiệm ngày 14-11).

### 2.3. Ba chiến lược thích nghi chế độ (Regime Schemes)

Khảo sát trên phân chia 90/10 với tập biến `paper3`:
- **`expanding`**: Huấn luyện từ đầu chuỗi (index 0) đến trước validation (`fit_start = 0`).
- **`rolling_756`**: Chỉ sử dụng 756 phiên giao dịch gần nhất trước validation (`fit_start = val_start - 756`), mô phỏng cửa sổ động 3 năm.
- **`recency_weighted`**: Huấn luyện trên toàn bộ lịch sử nhưng áp dụng trọng số mẫu hàm mũ:
  $$w_i = \exp\left(-\frac{\ln(2) \cdot age_i}{252}\right)$$
  với $age_i$ là khoảng cách phiên so với điểm cuối train, chuẩn hóa sao cho trung bình trọng số bằng 1.

### 2.4. Nhánh mô hình và kiến trúc

1. **Nhánh A (Fixed Models — Direct Level)**:
   - `naive_persistence`: $\widehat{y}(t) = y(t-1)$
   - `lstm`: 2 lớp LSTM (64, 32 units), Dropout 0.20, Dense(16, ReLU), Dense(1, Linear). Adam lr $10^{-3}$.
   - `gru`: 2 lớp GRU (64, 32 units), Dropout 0.20, Dense(16, ReLU), Dense(1, Linear). Adam lr $10^{-3}$.
   - `cnn1d_lstm`: Conv1D (32 filters, kernel 3, causal padding), MaxPool(2), LSTM (48, 24 units), Dropout 0.20, Dense(16, ReLU), Dense(1).
   - `cnn_gru`: Conv1D (32 filters, kernel 3, causal padding), MaxPool(2), GRU (48, 24 units), Dense(16, ReLU), Dropout 0.20, Dense(1).
2. **Nhánh B (Adaptation & Residual Extension)**:
   - `residual_gru`: Mạng GRU học phần dư $\Delta y(t) = y(t) - y(t-1)$. Dự đoán cuối cùng: $\widehat{y}(t) = y(t-1) + \widehat{\Delta y}(t)$.
   - `ridge_delta`: Mô hình hồi quy tuyến tính Ridge ($\alpha = 1.0$) hồi quy $\Delta y(t)$ theo đặc trưng trải phẳng, sau đó cộng lại $y(t-1)$.

---

## III. Kết quả thực nghiệm

### 3.1. So sánh Nhánh A: Các mô hình Direct Level trên hai tập biến

Bảng dưới tổng hợp kết quả trên thang đo giá gốc (RMSE_raw, MAE_raw) và sai số tương đối so với Naive Persistence:

| Tập biến | Tỷ lệ | Mô hình | MSE_scaled | RMSE_raw | MAE_raw | Relative MAE vs Naive |
|---|---|---|---:|---:|---:|---:|
| — | **90/10** | **Naive Persistence** | **0,001021** | **0,10103** | **0,07708** | **1,0000** |
| `legacy14` | 90/10 | cnn_gru | 0,026616 | 0,51594 | 0,45025 | 5,8416 |
| `legacy14` | 90/10 | gru | 0,037890 | 0,61559 | 0,54425 | 7,0611 |
| `legacy14` | 90/10 | lstm | 0,089674 | 0,94703 | 0,84944 | 11,0207 |
| `legacy14` | 90/10 | cnn1d_lstm | 0,104528 | 1,02246 | 0,92334 | 11,9795 |
| `paper3` | 90/10 | gru | 0,031038 | 0,55716 | 0,49293 | 6,3953 |
| `paper3` | 90/10 | cnn_gru | 0,039478 | 0,62836 | 0,54687 | 7,0952 |
| `paper3` | 90/10 | cnn1d_lstm | 0,067767 | 0,82326 | 0,68747 | 8,9193 |
| `paper3` | 90/10 | lstm | 0,156046 | 1,24927 | 1,15628 | 15,0016 |
| — | **92,5/7,5** | **Naive Persistence** | **0,000968** | **0,09840** | **0,07512** | **1,0000** |
| `legacy14` | 92,5/7,5 | gru | 0,020287 | 0,45045 | 0,40018 | 5,3274 |
| `legacy14` | 92,5/7,5 | cnn_gru | 0,035185 | 0,59321 | 0,51355 | 6,8368 |
| `paper3` | 92,5/7,5 | gru | 0,030935 | 0,55624 | 0,50861 | 6,7710 |
| `paper3` | 92,5/7,5 | cnn_gru | 0,037120 | 0,60931 | 0,55287 | 7,3602 |
| `paper3` | 92,5/7,5 | cnn1d_lstm | 0,104806 | 1,02382 | 0,93601 | 12,4608 |
| `legacy14` | 92,5/7,5 | cnn1d_lstm | 0,140721 | 1,18634 | 1,12575 | 14,9867 |
| `legacy14` | 92,5/7,5 | lstm | 0,155070 | 1,24536 | 1,19486 | 15,9069 |
| `paper3` | 92,5/7,5 | lstm | 0,229302 | 1,51438 | 1,47105 | 19,5836 |
| — | **95/5** | **Naive Persistence** | **0,000870** | **0,09661** | **0,07280** | **1,0000** |
| `paper3` | 95/5 | cnn_gru | 0,036407 | 0,62489 | 0,55934 | 7,6833 |
| `paper3` | 95/5 | gru | 0,040163 | 0,65633 | 0,59996 | 8,2413 |
| `legacy14` | 95/5 | gru | 0,049889 | 0,73150 | 0,66405 | 9,1217 |
| `legacy14` | 95/5 | cnn_gru | 0,062458 | 0,81848 | 0,78018 | 10,7169 |
| `legacy14` | 95/5 | cnn1d_lstm | 0,104731 | 1,05986 | 1,03627 | 14,2346 |
| `paper3` | 95/5 | cnn1d_lstm | 0,111527 | 1,09371 | 1,07318 | 14,7416 |
| `legacy14` | 95/5 | lstm | 0,142015 | 1,23418 | 1,21886 | 16,7427 |
| `paper3` | 95/5 | lstm | 0,225365 | 1,55473 | 1,54365 | 21,2042 |

**Nhận xét chính:**
- **Vị thế thống trị của Naive Persistence**: Không có bất kỳ mạng nơ-ron direct nào tiệm cận được Naive Persistence. MAE của mô hình nơ-ron tốt nhất vẫn gấp từ 5,3 đến 5,8 lần Naive.
- **GRU vs LSTM**: Mạng GRU và CNN-GRU luôn xếp trên LSTM thuần túy và CNN1D-LSTM. LSTM thuần túy có sai số nghiêm trọng nhất (MAE gấp 11 đến 21 lần Naive), phản ánh tình trạng trôi dự báo (*lagged smoothing*) khi gặp vùng giá test cao hơn train.
- **Paper3 vs. Legacy14**: Sự khác biệt giữa hai bộ biến là không đáng kể đối với bài toán direct level; việc bổ sung Aluminium và Tin (`legacy14`) giúp GRU cải thiện nhẹ tại 90/10 và 92,5/7,5 nhưng không làm thay đổi bản chất thất bại của việc dự báo mức giá trực tiếp.

---

### 3.2. So sánh Nhánh B: Các chiến lược thích nghi chế độ (Expanding vs Rolling vs Recency)

Thực hiện tại mốc 90/10, bộ biến `paper3`:

| Mô hình | Chiến lược (Scheme) | MSE_scaled | RMSE_raw | MAE_raw | Relative MAE vs Naive |
|---|---|---:|---:|---:|---:|
| **Naive Persistence** | — | **0,001021** | **0,10103** | **0,07708** | **1,0000** |
| `gru` | `rolling_756` | 0,069717 | 0,50022 | 0,45362 | 5,8853 |
| `gru` | `expanding` | 0,031038 | 0,55716 | 0,49293 | 6,3953 |
| `gru` | `recency_weighted` | 0,050898 | 0,71348 | 0,60080 | 7,7949 |
| `cnn_gru` | `rolling_756` | 0,083834 | 0,54854 | 0,45157 | 5,8587 |
| `cnn_gru` | `recency_weighted` | 0,031774 | 0,56372 | 0,47506 | 6,1634 |
| `cnn_gru` | `expanding` | 0,039478 | 0,62836 | 0,54687 | 7,0952 |

**Phân tích nghịch lý thang đo Scaled vs Raw**:
- Trên thang đo giá gốc, `rolling_756` giúp giảm RMSE của GRU từ 0,55716 xuống 0,50022 (giảm 10,2%) và CNN-GRU từ 0,62836 xuống 0,54854 (giảm 12,7%). Điều này xác nhận rằng việc loại bỏ dữ liệu quá khứ xa (2016–2022) giúp mô hình co cụm dự đoán gần hơn với mặt bằng giá cao của giai đoạn test (2025–2026).
- Tuy nhiên, MSE_scaled của `rolling_756` lại cao hơn gấp đôi so với `expanding` (0,0697 vs 0,0310). Nguyên nhân là vì scaler của `rolling_756` chỉ fit trên 756 hàng có khoảng biến thiên hẹp hơn, khiến cùng một sai số giá gốc bị phóng đại khi quy đổi ra khoảng $[0, 1]$. Do đó, **tuyệt đối không so sánh các chiến lược huấn luyện bằng MSE_scaled**.

---

### 3.3. So sánh Nhánh B: Học phần dư (Residual Learning) và Kiểm định từng cặp ngày

Khi chuyển sang học phần dư $\Delta y(t)$, sai số giảm mạnh về sát mức Naive Persistence:

| Mô hình | Chiến lược | MSE_scaled | RMSE_raw | MAE_raw | Relative MAE vs Naive |
|---|---|---:|---:|---:|---:|
| **Ridge Delta** | `expanding` | **0,000887** | **0,09420** | **0,07227** | **0,9377** |
| **Ridge Delta** | `recency_weighted` | **0,000887** | **0,09420** | **0,07227** | **0,9377** |
| **Residual-GRU** | `recency_weighted` | 0,001019 | 0,10097 | 0,07697 | 0,9987 |
| **Naive Persistence** | `expanding` | 0,001021 | 0,10103 | 0,07708 | 1,0000 |
| **Residual-GRU** | `expanding` | 0,001021 | 0,10107 | 0,07712 | 1,0006 |
| **Residual-GRU** | `rolling_756` | 0,003028 | 0,10425 | 0,08047 | 1,0440 |
| **Ridge Delta** | `rolling_756` | 0,003620 | 0,11398 | 0,09062 | 1,1758 |

#### Đánh giá chi tiết cặp kiểm định ngày (Paired Daily Test): Residual-GRU (Rolling 756) vs Naive
Từ tệp kiểm định chi tiết `residual_rolling756_vs_naive.json` trên 271 ngày test (29/10/2025 – 13/11/2026):
- **Naive MAE**: 0,077077 USD | **Residual-GRU MAE**: 0,080466 USD (kém hơn 4,40%).
- **Naive RMSE**: 0,101033 USD | **Residual-GRU RMSE**: 0,104253 USD (kém hơn 3,19%).
- **Tỷ lệ số ngày Residual-GRU thắng Naive**: **41,33%** (112 / 271 ngày).
- **Kết luận kiểm định**: Mô hình phần dư GRU rolling thua Naive Persistence trên cả MAE lẫn RMSE, và thua ở đa số các phiên giao dịch.

---

## IV. Giới hạn phương pháp và rủi ro diễn giải

1. **Hiệu ứng Nowcasting ($HORIZON=0$)**: Ưu thế của Ridge Delta (-6,23% MAE) tại ngày 16-11 đạt được nhờ sử dụng giá đóng cửa ngày $t$ của WTI, Gold, Silver để ước lượng giá đóng cửa ngày $t$ của Copper. Trong bối cảnh giao dịch thực tế, giá các hàng hóa này thường đóng cửa đồng thời hoặc chênh lệch múi giờ không đáng kể; mô hình chưa chứng minh được khả năng dự báo giá ngày hôm sau ($t+1$).
2. **Sai số chuẩn hóa không đồng nhất**: MinMaxScaler fit riêng trên từng cửa sổ rolling khiến việc so sánh `MSE_scaled` giữa các chiến lược bị sai lệch; bắt buộc phải quy đổi về `RMSE_raw` và `MAE_raw`.
3. **Số lượng seed**: Các mô hình mạng nơ-ron chỉ chạy 1 seed (seed 42) trong `FULL_MODE` nhằm đảm bảo giới hạn thời gian chạy trên GPU (10,2 phút).

---

## V. Kết luận ngày 16-11

- Loại bỏ hoàn toàn phương pháp dự báo mức giá trực tiếp (Direct Level) bằng mạng học sâu cho bài toán giá đồng: kể cả khi dùng GRU hay CNN-GRU kết hợp rolling window, mô hình vẫn kém Naive Persistence từ 5 đến 7 lần.
- Tiếp cận học phần dư (Residual Learning) là hướng đi đúng đắn duy nhất để đưa sai số mạng nơ-ron về cùng bậc độ lớn với thị trường.
- Bước chuyển tiếp bắt buộc cho ngày 17-11: Chuyển toàn bộ hệ thống sang $HORIZON = 1$ (chỉ dùng dữ liệu $\le t-1$ để dự báo ngày $t$), tích hợp thêm XGBoost cho phần dư và triển khai Walk-Forward kiểm định thống kê nghiêm ngặt.

---

## Hiện vật và tệp kết quả

- [Notebook chạy độc lập](../../16-11-NonSA-Regime-standalone.ipynb)
- [Tóm tắt kết quả summary_v16.csv](summary_v16.csv)
- [Chi tiết từng lần chạy results_v16.csv](results_v16.csv)
- [Kiểm định cặp ngày Residual vs Naive](residual_rolling756_vs_naive.json)
- [Bảng xếp hạng theo validation](validation_ranked_candidates.csv)
- [Biểu đồ so sánh Rolling vs Expanding](rolling756_vs_expanding_gru.png)
