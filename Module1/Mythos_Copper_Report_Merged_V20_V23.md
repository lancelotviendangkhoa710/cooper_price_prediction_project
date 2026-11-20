# BÁO CÁO KHOA HỌC HỢP NHẤT — DỰ BÁO GIÁ ĐỒNG MỘT PHIÊN TRƯỚC

**Bản tích hợp Mythos V20–V23 (20/11/2026)** · **Tác giả thực nghiệm:** Viên Đăng Khoa · **Học phần:** IS403 — Phân tích dữ liệu kinh doanh

**Phạm vi:** Các mạng học sai phân (Residual), Ridge Residual, hai cấu hình SA-LSTM và Naive Persistence. Các kết quả LSTM/GRU/CNN Direct thông thường và SimpleRNN Direct được loại hoàn toàn khỏi nội dung xếp hạng theo yêu cầu. **SA-LSTM vẫn dự báo mức giá trực tiếp**, nên so sánh với Residual phản ánh cả khác biệt biểu diễn target.

**Quy ước nguồn:** `(V20)`, `(V21)`, `(V21-80)`, `(V22-N)`, `(V22-S)`, `(V23)` xuất hiện ngay cạnh giá trị đo, tương ứng các ZIP ở phụ lục. Không tự suy ra số liệu cho cấu hình chưa thực nghiệm.

## Tóm tắt (Abstract)

Nghiên cứu dự báo mức giá Copper ở phiên t từ 22 phiên quá khứ bằng các kiến trúc học sai phân Residual-LSTM/GRU/CNN-hybrid, Residual-Ridge, hai cấu hình SA-LSTM và baseline Naive. Bộ dữ liệu gồm 2.710 phiên (13/01/2016–13/11/2026); các tập đầu vào gồm Copper-only, Paper3, Legacy14 và WTI mới. Cùng định nghĩa MAE/RMSE trên thang giá gốc, thí nghiệm khảo sát bốn tỷ lệ train/test cùng ba chiến lược Expanding, Rolling 756, Recency; V23 bổ sung bốn ô split–strategy của nhánh SA và đánh giá walk-forward 542 phiên/5 seeds. Báo cáo ưu tiên xếp hạng các **mô hình fitted** theo từng điều kiện và dùng Naive làm mốc kiểm tra. Việc đứng đầu một bảng hậu nghiệm không xác lập ưu thế thống kê hoặc nhân quả.

**Từ khóa:** Copper forecasting, Residual Learning, LSTM, GRU, Simulated Annealing, Walk-forward, Diebold–Mariano, Spearman.

## 1. Bài toán, dữ liệu và kiểm toán tính hợp lệ

### 1.1. Dự báo một bước

Với cửa sổ đầu vào `X_t ∈ ℝ^(22×F)` bao gồm các ngày `t−22,...,t−1`, mục tiêu là `P_t = Copper(t)`, horizon 1. Không đưa biến tại ngày `t` vào đầu vào. Dữ liệu `dataset_9_commodities_daily.csv` được manifest báo 2.710 dòng, 13/01/2016–13/11/2026, SHA-256 `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600`. Chưa xác nhận giờ phát hành/đóng cửa của từng thị trường; cần phân biệt chia theo ngày với kiểm soát availability theo thời điểm thực.

### 1.2. Các split kiểm tra

| Train/test | Số phiên test | Test bắt đầu | Kết thúc |
|---|---:|---|---|
| 80/20 | 542 | 07/10/2024 | 13/11/2026 |
| 90/10 | 271 | 29/10/2025 | 13/11/2026 |
| 92.5/7.5 | 204 | 30/01/2026 | 13/11/2026 |
| 95/5 | 136 | 07/05/2026 | 13/11/2026 |

Các khoảng test lồng nhau, không độc lập; vì vậy không xem MAE giữa hai split khác thời gian như tác động riêng của tỷ lệ train.

### 1.3. Bộ đặc trưng và giải thích EDA

| Bộ đầu vào | Biến | Nguồn |
|---|---|---|
| Copper-only | Copper | V20 |
| Paper3+Cu | Copper, WTI, Gold, Silver | V20 |
| Paper3−Cu | WTI, Gold, Silver | V21 SA |
| Legacy14+Cu | Copper, Aluminium, Tin, Silver, Gold | V20, V21 SA |
| Legacy14−Cu | Aluminium, Tin, Silver, Gold | V21 SA |
| WTI+Cu | Copper, Aluminium, Tin, WTI | V22, V23 |

Legacy14 là nhãn lịch sử, không phải 14 features. Việc chuyển từ Legacy14+Cu sang WTI+Cu **đổi cả thành phần (Gold, Silver → WTI) và số chiều (5 → 4)**, không phải ablation một biến có kiểm soát.

| Hàng hóa | Spearman mức giá với Copper | Spearman biến động lag 1 |
|---|---:|---:|
| Aluminium | 0.9279 | 0.0093 |
| Tin | 0.8829 | −0.0185 |
| WTI | 0.7711 | 0.0142 |
| Gold | 0.7425 | 0.0910 |
| Silver | 0.7424 | 0.0388 |

Số tương quan level cao không bảo đảm dự báo delta tốt; Spearman lag-1 yếu không loại trừ tương tác phi tuyến và nhiều độ trễ trong cửa sổ 22 phiên. Chọn feature trên train/validation, không dựa vào test. Số liệu EDA từ CSV V20/V22.

## 2. Phương pháp dùng chung và kiến trúc

### 2.1. Residual, SA-LSTM và Naive

- **Residual:** `ΔP_t = P_t − P_(t−1)`; mạng hoặc Ridge học `Δ̂P_t=f(X_t)` và hoàn nguyên `P̂_t = P_(t−1) + Δ̂P_t`.
- **Naive Persistence:** `P̂_t=P_(t−1)`; không có tham số fit, không tính vào xếp hạng mô hình fitted nhưng luôn dùng đối chứng.
- **SA-LSTM:** mạng học mức giá `P_t` trực tiếp; Simulated Annealing chỉ dò hyperparameters, không phải tầng neural hoặc thuật toán huấn luyện weight. Giữ cấu hình này vì tái hiện bài Chen et al. (2023); không đánh đồng target với Residual.

### 2.2. Kiến trúc thực hiện

| Phương pháp | Kiến trúc triển khai | Target |
|---|---|---|
| Residual-LSTM | LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(1) | ΔP |
| Residual-GRU | GRU(64) → Dropout → GRU(32) → Dropout → Dense(16) → Dense(1) | ΔP |
| Residual-CNN-LSTM | Conv1D(32, causal, k=3) → MaxPool(2) → LSTM(48) → Dropout → LSTM(24) → Dropout → Dense(16) → Dense(1) | ΔP |
| Residual-CNN-GRU | Conv1D(32, causal, k=3) → MaxPool(2) → GRU(48) → Dropout → GRU(24) → Dropout → Dense(16) → Dense(1) | ΔP |
| Residual-Ridge | Flatten(22×F) → Ridge(α=1) | ΔP |
| Paper SA-LSTM | LSTM(39) → Dropout(0.2) → LSTM(111) → Dropout(0.2) → Dense(32, ReLU) → Dense(1) | P |
| Reoptimized SA-LSTM V21 | LSTM(22) → LSTM(91) (Paper3), hoặc LSTM(22) → LSTM(83) (Legacy14); cùng Dense head | P |
| Reoptimized SA-LSTM V22/V23 | LSTM(47) → LSTM(127) → Dense(32) → Dense(1) | P |

Tham số Paper SA-LSTM là cấu hình dựng theo những chi tiết đã công bố; Dense(32), lookback 22 và head là lựa chọn triển khai, **không gán là thông số paper xác nhận**. Cần đọc notebook để xác định dropout exact rate của nhóm Residual nếu trình bày chi tiết từng layer.

### 2.3. Chiến lược

- **Expanding:** tại mốc fit dùng lịch sử tích lũy.
- **Rolling 756:** dùng 756 phiên gần thời điểm fit.
- **Recency Weighting:** trọng số `w_i=exp[−ln(2)×age_i/252]` (half-life 252 phiên).
- Holdout train một lần tại ranh giới mỗi split; **không gọi đó là walk-forward**. Walk-forward refit định kỳ 63 phiên.

### 2.4. Tối ưu, định nghĩa độ đo và phạm vi công bằng

`MAE=mean(|P_t−P̂_t|)`, `RMSE=sqrt(mean((P_t−P̂_t)^2))`, `Skill_MAE(%)=100×(1−MAE_model/MAE_Naive)`. DA chỉ đo hướng và không dùng để xếp hạng chính; Naive có delta dự báo 0 nên không gọi DA là độ chính xác dự báo hướng. Scalers fit trên train, không fit test; tránh so `mse_scaled` khi scaler khác nhau.

So sánh trong cùng split/strategy/test timestamps có tính đối chiếu mô tả; **không đồng nghĩa matched target representation hoặc matched hyperparameter-search budget**. V21 reoptimized dùng HP cache, V22 chạy SA search 16 trials trên reference split 80/20, V23 dùng lại HP từ V22; tuyệt đối không diễn giải cải thiện của V22 như tác động riêng của SA hoặc WTI.

## 3. Xếp hạng tổng hợp mọi mô hình retained (không tính Naive)

Tiêu chí xếp hạng: MAE gốc tăng dần; bảng Top 5 theo từng split trên tất cả cấu hình Residual và SA-LSTM đã thực nghiệm. Các mô hình Direct thông thường không đưa vào tập ứng viên. Ties xếp theo RMSE rồi tên; kết quả xếp hạng **hậu nghiệm**.

| Train | Hạng | Mô hình | Features | Strategy | MAE (nguồn) | RMSE (nguồn) |
|---|---|---|---|---|---|---|
| 80% | 1 | Residual-GRU | Paper3+Cu | Expanding | 0.068780 (V20) | 0.107853 (V20) |
| 80% | 2 | Residual-LSTM | WTI+Cu | Expanding | 0.068840 (V22-N) | 0.107996 (V22-N) |
| 80% | 3 | Residual-CNN-LSTM | Copper-only | Recency | 0.068841 (V20) | 0.108019 (V20) |
| 80% | 4 | Residual-CNN-LSTM | Paper3+Cu | Recency | 0.068843 (V20) | 0.107912 (V20) |
| 80% | 5 | Residual-LSTM | Paper3+Cu | Recency | 0.068850 (V20) | 0.107992 (V20) |
| 90% | 1 | Residual-GRU | Legacy14+Cu | Expanding | 0.076951 (V20) | 0.100810 (V20) |
| 90% | 2 | Residual-LSTM | Legacy14+Cu | Expanding | 0.076978 (V20) | 0.100948 (V20) |
| 90% | 3 | Residual-CNN-LSTM | Copper-only | Rolling 756 | 0.077000 (V20) | 0.100895 (V20) |
| 90% | 4 | Residual-LSTM | Paper3+Cu | Expanding | 0.077006 (V20) | 0.100878 (V20) |
| 90% | 5 | Residual-LSTM | Copper-only | Expanding | 0.077008 (V20) | 0.100810 (V20) |
| 92.5% | 1 | Residual-GRU | WTI+Cu | Recency | 0.075028 (V22-N) | 0.098307 (V22-N) |
| 92.5% | 2 | Residual-CNN-LSTM | Copper-only | Recency | 0.075050 (V20) | 0.098426 (V20) |
| 92.5% | 3 | Residual-CNN-LSTM | WTI+Cu | Expanding | 0.075054 (V22-N) | 0.098459 (V22-N) |
| 92.5% | 4 | Residual-CNN-LSTM | Copper-only | Expanding | 0.075057 (V20) | 0.098416 (V20) |
| 92.5% | 5 | Residual-CNN-LSTM | Copper-only | Rolling 756 | 0.075059 (V20) | 0.098468 (V20) |
| 95% | 1 | Residual-GRU | Paper3+Cu | Recency | 0.072357 (V20) | 0.096414 (V20) |
| 95% | 2 | Residual-GRU | Legacy14+Cu | Recency | 0.072507 (V20) | 0.096463 (V20) |
| 95% | 3 | Residual-LSTM | Paper3+Cu | Expanding | 0.072517 (V20) | 0.096496 (V20) |
| 95% | 4 | Residual-LSTM | Paper3+Cu | Rolling 756 | 0.072554 (V20) | 0.096450 (V20) |
| 95% | 5 | Residual-CNN-GRU | Paper3+Cu | Expanding | 0.072575 (V20) | 0.096666 (V20) |



### Naive baseline, không tham gia xếp hạng

| Train | n_test | MAE | RMSE | Nguồn |
|---|---|---|---|---|
| 80% | 542 | 0.069001 | 0.108000 | Các ZIP cùng split |
| 90% | 271 | 0.077077 | 0.101033 | Các ZIP cùng split |
| 92.5% | 204 | 0.075116 | 0.098399 | Các ZIP cùng split |
| 95% | 136 | 0.072799 | 0.096612 | Các ZIP cùng split |



## 4. Mô hình tốt nhất theo từng họ kiến trúc và split

Chọn cấu hình tối thiểu MAE đã quan sát trong mỗi họ mô hình; do số cấu hình mỗi họ khác nhau, bảng này chỉ mang tính mô tả.

| Split | Mô hình fitted | Features | Chiến lược | MAE | RMSE |
|---|---|---|---|---|---|
| 80% | Paper SA-LSTM | WTI+Cu | Expanding | 0.298955 (V22-S) | 0.358768 |
| 80% | Reoptimized SA-LSTM | WTI+Cu | Expanding | 0.223809 (V22-S) | 0.297888 |
| 80% | Residual-CNN-GRU | WTI+Cu | Recency | 0.068947 (V22-N) | 0.107682 |
| 80% | Residual-CNN-LSTM | Copper-only | Recency | 0.068841 (V20) | 0.108019 |
| 80% | Residual-GRU | Paper3+Cu | Expanding | 0.068780 (V20) | 0.107853 |
| 80% | Residual-LSTM | WTI+Cu | Expanding | 0.068840 (V22-N) | 0.107996 |
| 80% | Residual-Ridge | Copper-only | Expanding | 0.069408 (V20) | 0.108075 |
| 90% | Paper SA-LSTM | WTI+Cu | Expanding | 0.184634 (V22-S) | 0.225723 |
| 90% | Reoptimized SA-LSTM | WTI+Cu | Recency | 0.326683 (V22-S) | 0.376388 |
| 90% | Residual-CNN-GRU | WTI+Cu | Expanding | 0.077229 (V22-N) | 0.101285 |
| 90% | Residual-CNN-LSTM | Copper-only | Rolling 756 | 0.077000 (V20) | 0.100895 |
| 90% | Residual-GRU | Legacy14+Cu | Expanding | 0.076951 (V20) | 0.100810 |
| 90% | Residual-LSTM | Legacy14+Cu | Expanding | 0.076978 (V20) | 0.100948 |
| 90% | Residual-Ridge | Copper-only | Expanding | 0.077306 (V20) | 0.101190 |
| 92.5% | Paper SA-LSTM | WTI+Cu | Expanding | 0.194868 (V22-S) | 0.236334 |
| 92.5% | Reoptimized SA-LSTM | Legacy14+Cu | Expanding | 0.352332 (V21) | 0.409124 |
| 92.5% | Residual-CNN-GRU | Copper-only | Recency | 0.075340 (V20) | 0.098272 |
| 92.5% | Residual-CNN-LSTM | Copper-only | Recency | 0.075050 (V20) | 0.098426 |
| 92.5% | Residual-GRU | WTI+Cu | Recency | 0.075028 (V22-N) | 0.098307 |
| 92.5% | Residual-LSTM | WTI+Cu | Expanding | 0.075094 (V22-N) | 0.098551 |
| 92.5% | Residual-Ridge | Copper-only | Expanding | 0.075285 (V20) | 0.098422 |
| 95% | Paper SA-LSTM | WTI+Cu | Expanding | 0.200803 (V22-S) | 0.235800 |
| 95% | Reoptimized SA-LSTM | WTI+Cu | Expanding | 0.325427 (V22-S) | 0.360107 |
| 95% | Residual-CNN-GRU | Paper3+Cu | Expanding | 0.072575 (V20) | 0.096666 |
| 95% | Residual-CNN-LSTM | Copper-only | Expanding | 0.072625 (V20) | 0.096669 |
| 95% | Residual-GRU | Paper3+Cu | Recency | 0.072357 (V20) | 0.096414 |
| 95% | Residual-LSTM | Paper3+Cu | Expanding | 0.072517 (V20) | 0.096496 |
| 95% | Residual-Ridge | Copper-only | Expanding | 0.073684 (V20) | 0.096928 |



## 5. Ma trận chiến lược chung tại 90/10

Mỗi hàng cố định mô hình + features + nguồn. `—` = chưa thử hoặc không có output; không tự điền. Đây là phương pháp đọc trực tiếp ảnh hưởng strategy trong một cấu hình giữ nguyên input.

| Mô hình | Feature set | Expanding MAE | Rolling 756 MAE | Recency MAE |
|---|---|---|---|---|
| Paper SA-LSTM | Legacy14+Cu | 0.470929 (V21) | 0.773310 (V21) | 0.705350 (V21) |
| Paper SA-LSTM | Legacy14−Cu | 0.464991 (V21) | 0.869224 (V21) | 0.524705 (V21) |
| Paper SA-LSTM | Paper3+Cu | 0.978075 (V21) | 0.897077 (V21) | 0.917524 (V21) |
| Paper SA-LSTM | Paper3−Cu | 0.540720 (V21) | 0.604594 (V21) | 0.585950 (V21) |
| Paper SA-LSTM | WTI+Cu | 0.184634 (V22-S) | 0.575763 (V22-S) | 0.294741 (V22-S) |
| Reoptimized SA-LSTM | Legacy14+Cu | 0.510467 (V21) | 0.977821 (V21) | 0.735621 (V21) |
| Reoptimized SA-LSTM | Legacy14−Cu | 0.466595 (V21) | 0.624100 (V21) | 0.460227 (V21) |
| Reoptimized SA-LSTM | Paper3+Cu | 0.877420 (V21) | 0.974593 (V21) | 0.511784 (V21) |
| Reoptimized SA-LSTM | Paper3−Cu | 0.595605 (V21) | 0.644180 (V21) | 0.482221 (V21) |
| Reoptimized SA-LSTM | WTI+Cu | 0.358921 (V22-S) | 0.496288 (V22-S) | 0.326683 (V22-S) |
| Residual-CNN-GRU | Copper-only | 0.078067 (V20) | 0.088180 (V20) | 0.080173 (V20) |
| Residual-CNN-GRU | Legacy14+Cu | 0.090563 (V20) | 0.098870 (V20) | 0.077522 (V20) |
| Residual-CNN-GRU | Paper3+Cu | 0.078132 (V20) | 0.097694 (V20) | 0.083405 (V20) |
| Residual-CNN-GRU | WTI+Cu | 0.077229 (V22-N) | 0.085898 (V22-N) | 0.077639 (V22-N) |
| Residual-CNN-LSTM | Copper-only | 0.080098 (V20) | 0.077000 (V20) | 0.078311 (V20) |
| Residual-CNN-LSTM | Legacy14+Cu | 0.077245 (V20) | 0.081200 (V20) | 0.077304 (V20) |
| Residual-CNN-LSTM | Paper3+Cu | 0.077432 (V20) | 0.091009 (V20) | 0.079093 (V20) |
| Residual-CNN-LSTM | WTI+Cu | 0.077638 (V22-N) | 0.085373 (V22-N) | 0.078023 (V22-N) |
| Residual-GRU | Copper-only | 0.080182 (V20) | 0.078851 (V20) | 0.083943 (V20) |
| Residual-GRU | Legacy14+Cu | 0.076951 (V20) | 0.077182 (V20) | 0.077055 (V20) |
| Residual-GRU | Paper3+Cu | 0.077143 (V20) | 0.077209 (V20) | 0.077304 (V20) |
| Residual-GRU | WTI+Cu | 0.077463 (V22-N) | 0.081939 (V22-N) | 0.077851 (V22-N) |
| Residual-LSTM | Copper-only | 0.077008 (V20) | 0.077159 (V20) | 0.078135 (V20) |
| Residual-LSTM | Legacy14+Cu | 0.076978 (V20) | 0.077756 (V20) | 0.078692 (V20) |
| Residual-LSTM | Paper3+Cu | 0.077006 (V20) | 0.077230 (V20) | 0.078318 (V20) |
| Residual-LSTM | WTI+Cu | 0.077037 (V22-N) | 0.080271 (V22-N) | 0.078856 (V22-N) |
| Residual-Ridge | Copper-only | 0.077306 (V20) | 0.082705 (V20) | 0.079330 (V20) |
| Residual-Ridge | Legacy14+Cu | 0.081707 (V20) | 0.084002 (V20) | 0.085664 (V20) |
| Residual-Ridge | Paper3+Cu | 0.080712 (V20) | 0.098350 (V20) | 0.090112 (V20) |
| Residual-Ridge | WTI+Cu | 0.077737 (V22-N) | 0.094420 (V22-N) | 0.081044 (V22-N) |



## 6. Thay đổi feature Legacy14+Cu → WTI+Cu (cặp đúng mô hình, split, strategy)

`ΔMAE% = 100×(MAE_WTI/MAE_Legacy14 − 1)`; **âm = giảm lỗi**, dương = tăng lỗi. So sánh đúng nhánh Residual ở V20 và V22-N nên tránh bị nhiễu bởi thay đổi SA HP. Tuy nhiên số chiều và thành phần cùng thay đổi, do đó đây không phải kiểm định tác động riêng của WTI.

### Train 80%

| Mô hình | Strategy | MAE Legacy14 | MAE WTI | ΔMAE | RMSE Legacy14 | RMSE WTI |
|---|---|---|---|---|---|---|
| Residual-LSTM | Expanding | 0.068912 (V20) | 0.068840 (V22-N) | -0.10% | 0.108011 | 0.107996 |
| Residual-GRU | Expanding | 0.068857 (V20) | 0.068966 (V22-N) | +0.16% | 0.107932 | 0.107862 |
| Residual-CNN-LSTM | Expanding | 0.068916 (V20) | 0.068972 (V22-N) | +0.08% | 0.108116 | 0.107827 |
| Residual-CNN-GRU | Expanding | 0.074771 (V20) | 0.069742 (V22-N) | -6.73% | 0.114417 | 0.108045 |
| Residual-Ridge | Expanding | 0.072588 (V20) | 0.069603 (V22-N) | -4.11% | 0.112238 | 0.108010 |
| Residual-LSTM | Rolling 756 | 0.076509 (V20) | 0.069943 (V22-N) | -8.58% | 0.116666 | 0.108243 |
| Residual-GRU | Rolling 756 | 0.133698 (V20) | 0.069751 (V22-N) | -47.83% | 0.175810 | 0.108199 |
| Residual-CNN-LSTM | Rolling 756 | 0.069998 (V20) | 0.069118 (V22-N) | -1.26% | 0.108669 | 0.107833 |
| Residual-CNN-GRU | Rolling 756 | 0.095740 (V20) | 0.069534 (V22-N) | -27.37% | 0.135472 | 0.107919 |
| Residual-Ridge | Rolling 756 | 0.163842 (V20) | 0.076286 (V22-N) | -53.44% | 0.192049 | 0.112686 |
| Residual-LSTM | Recency | 0.068907 (V20) | 0.069815 (V22-N) | +1.32% | 0.108036 | 0.108448 |
| Residual-GRU | Recency | 0.068933 (V20) | 0.068914 (V22-N) | -0.03% | 0.107941 | 0.107935 |
| Residual-CNN-LSTM | Recency | 0.068914 (V20) | 0.068931 (V22-N) | +0.02% | 0.107779 | 0.107935 |
| Residual-CNN-GRU | Recency | 0.111344 (V20) | 0.068947 (V22-N) | -38.08% | 0.151494 | 0.107682 |
| Residual-Ridge | Recency | 0.077906 (V20) | 0.071724 (V22-N) | -7.94% | 0.119160 | 0.109358 |



### Train 90%

| Mô hình | Strategy | MAE Legacy14 | MAE WTI | ΔMAE | RMSE Legacy14 | RMSE WTI |
|---|---|---|---|---|---|---|
| Residual-LSTM | Expanding | 0.076978 (V20) | 0.077037 (V22-N) | +0.08% | 0.100948 | 0.100980 |
| Residual-GRU | Expanding | 0.076951 (V20) | 0.077463 (V22-N) | +0.67% | 0.100810 | 0.101052 |
| Residual-CNN-LSTM | Expanding | 0.077245 (V20) | 0.077638 (V22-N) | +0.51% | 0.100856 | 0.101076 |
| Residual-CNN-GRU | Expanding | 0.090563 (V20) | 0.077229 (V22-N) | -14.72% | 0.117479 | 0.101285 |
| Residual-Ridge | Expanding | 0.081707 (V20) | 0.077737 (V22-N) | -4.86% | 0.106728 | 0.101669 |
| Residual-LSTM | Rolling 756 | 0.077756 (V20) | 0.080271 (V22-N) | +3.23% | 0.101509 | 0.103472 |
| Residual-GRU | Rolling 756 | 0.077182 (V20) | 0.081939 (V22-N) | +6.16% | 0.101018 | 0.104597 |
| Residual-CNN-LSTM | Rolling 756 | 0.081200 (V20) | 0.085373 (V22-N) | +5.14% | 0.103617 | 0.107714 |
| Residual-CNN-GRU | Rolling 756 | 0.098870 (V20) | 0.085898 (V22-N) | -13.12% | 0.122720 | 0.108409 |
| Residual-Ridge | Rolling 756 | 0.084002 (V20) | 0.094420 (V22-N) | +12.40% | 0.108523 | 0.116679 |
| Residual-LSTM | Recency | 0.078692 (V20) | 0.078856 (V22-N) | +0.21% | 0.103507 | 0.102206 |
| Residual-GRU | Recency | 0.077055 (V20) | 0.077851 (V22-N) | +1.03% | 0.100969 | 0.101376 |
| Residual-CNN-LSTM | Recency | 0.077304 (V20) | 0.078023 (V22-N) | +0.93% | 0.101023 | 0.101646 |
| Residual-CNN-GRU | Recency | 0.077522 (V20) | 0.077639 (V22-N) | +0.15% | 0.100867 | 0.101300 |
| Residual-Ridge | Recency | 0.085664 (V20) | 0.081044 (V22-N) | -5.39% | 0.112437 | 0.104682 |



### Train 92.5%

| Mô hình | Strategy | MAE Legacy14 | MAE WTI | ΔMAE | RMSE Legacy14 | RMSE WTI |
|---|---|---|---|---|---|---|
| Residual-LSTM | Expanding | 0.076541 (V20) | 0.075094 (V22-N) | -1.89% | 0.100764 | 0.098551 |
| Residual-GRU | Expanding | 0.075205 (V20) | 0.075198 (V22-N) | -0.01% | 0.098704 | 0.098460 |
| Residual-CNN-LSTM | Expanding | 0.076109 (V20) | 0.075054 (V22-N) | -1.39% | 0.099989 | 0.098459 |
| Residual-CNN-GRU | Expanding | 0.080471 (V20) | 0.076010 (V22-N) | -5.54% | 0.105340 | 0.099649 |
| Residual-Ridge | Expanding | 0.082249 (V20) | 0.076510 (V22-N) | -6.98% | 0.107058 | 0.099999 |
| Residual-LSTM | Rolling 756 | 0.075741 (V20) | 0.075551 (V22-N) | -0.25% | 0.098783 | 0.098938 |
| Residual-GRU | Rolling 756 | 0.075867 (V20) | 0.082713 (V22-N) | +9.02% | 0.100204 | 0.103863 |
| Residual-CNN-LSTM | Rolling 756 | 0.076066 (V20) | 0.077000 (V22-N) | +1.23% | 0.098508 | 0.099843 |
| Residual-CNN-GRU | Rolling 756 | 0.076120 (V20) | 0.079791 (V22-N) | +4.82% | 0.100282 | 0.102662 |
| Residual-Ridge | Rolling 756 | 0.085453 (V20) | 0.092756 (V22-N) | +8.55% | 0.108047 | 0.114818 |
| Residual-LSTM | Recency | 0.076629 (V20) | 0.075120 (V22-N) | -1.97% | 0.100968 | 0.098612 |
| Residual-GRU | Recency | 0.075708 (V20) | 0.075028 (V22-N) | -0.90% | 0.099996 | 0.098307 |
| Residual-CNN-LSTM | Recency | 0.076140 (V20) | 0.075275 (V22-N) | -1.14% | 0.099782 | 0.098951 |
| Residual-CNN-GRU | Recency | 0.079462 (V20) | 0.081707 (V22-N) | +2.82% | 0.103679 | 0.106210 |
| Residual-Ridge | Recency | 0.090000 (V20) | 0.080297 (V22-N) | -10.78% | 0.115616 | 0.103246 |



### Train 95%

| Mô hình | Strategy | MAE Legacy14 | MAE WTI | ΔMAE | RMSE Legacy14 | RMSE WTI |
|---|---|---|---|---|---|---|
| Residual-LSTM | Expanding | 0.072619 (V20) | 0.073211 (V22-N) | +0.81% | 0.096367 | 0.096610 |
| Residual-GRU | Expanding | 0.072622 (V20) | 0.072882 (V22-N) | +0.36% | 0.096631 | 0.096631 |
| Residual-CNN-LSTM | Expanding | 0.072917 (V20) | 0.073456 (V22-N) | +0.74% | 0.096419 | 0.096595 |
| Residual-CNN-GRU | Expanding | 0.072810 (V20) | 0.073054 (V22-N) | +0.34% | 0.097681 | 0.096493 |
| Residual-Ridge | Expanding | 0.074357 (V20) | 0.074315 (V22-N) | -0.06% | 0.098943 | 0.097327 |
| Residual-LSTM | Rolling 756 | 0.074773 (V20) | 0.075774 (V22-N) | +1.34% | 0.096963 | 0.097801 |
| Residual-GRU | Rolling 756 | 0.072720 (V20) | 0.078619 (V22-N) | +8.11% | 0.097324 | 0.099497 |
| Residual-CNN-LSTM | Rolling 756 | 0.073683 (V20) | 0.074180 (V22-N) | +0.67% | 0.096634 | 0.096942 |
| Residual-CNN-GRU | Rolling 756 | 0.072839 (V20) | 0.074883 (V22-N) | +2.81% | 0.097354 | 0.096766 |
| Residual-Ridge | Rolling 756 | 0.078320 (V20) | 0.095167 (V22-N) | +21.51% | 0.100768 | 0.115561 |
| Residual-LSTM | Recency | 0.073271 (V20) | 0.073393 (V22-N) | +0.17% | 0.096492 | 0.096674 |
| Residual-GRU | Recency | 0.072507 (V20) | 0.072619 (V22-N) | +0.16% | 0.096463 | 0.096531 |
| Residual-CNN-LSTM | Recency | 0.074450 (V20) | 0.074737 (V22-N) | +0.38% | 0.097025 | 0.097229 |
| Residual-CNN-GRU | Recency | 0.075763 (V20) | 0.072810 (V22-N) | -3.90% | 0.097734 | 0.096709 |
| Residual-Ridge | Recency | 0.085730 (V20) | 0.082883 (V22-N) | -3.32% | 0.106248 | 0.104497 |



## 7. SA-LSTM trên Legacy14 và WTI — đầy đủ các strategy đã chạy

V21 dùng HP cache, V22 tìm mới qua 16 lần thử SA; V23 chỉ fit với HP V22. Do đó bảng mô tả thay đổi **đồng thời feature và cấu hình học**. Việc bổ sung V23 giúp khép kín các ô còn thiếu của WTI tại 92.5/95%, nhưng không tạo số liệu tương ứng cho Legacy14 Rolling/Recency ở hai split này.

| Split | Phương pháp SA-LSTM | Strategy | MAE Legacy14 | MAE WTI | ΔMAE |
|---|---|---|---|---|---|
| 80% | Paper SA-LSTM | Expanding | 0.760563 (V21-80) | 0.298955 (V22-S) | -60.69% |
| 80% | Paper SA-LSTM | Rolling 756 | 0.786877 (V21-80) | — | — |
| 80% | Paper SA-LSTM | Recency | 0.586941 (V21-80) | — | — |
| 80% | Reoptimized SA-LSTM | Expanding | 1.000423 (V21-80) | 0.223809 (V22-S) | -77.63% |
| 80% | Reoptimized SA-LSTM | Rolling 756 | 1.241647 (V21-80) | — | — |
| 80% | Reoptimized SA-LSTM | Recency | 0.919743 (V21-80) | — | — |
| 90% | Paper SA-LSTM | Expanding | 0.470929 (V21) | 0.184634 (V22-S) | -60.79% |
| 90% | Paper SA-LSTM | Rolling 756 | 0.773310 (V21) | 0.575763 (V22-S) | -25.55% |
| 90% | Paper SA-LSTM | Recency | 0.705350 (V21) | 0.294741 (V22-S) | -58.21% |
| 90% | Reoptimized SA-LSTM | Expanding | 0.510467 (V21) | 0.358921 (V22-S) | -29.69% |
| 90% | Reoptimized SA-LSTM | Rolling 756 | 0.977821 (V21) | 0.496288 (V22-S) | -49.25% |
| 90% | Reoptimized SA-LSTM | Recency | 0.735621 (V21) | 0.326683 (V22-S) | -55.59% |
| 92.5% | Paper SA-LSTM | Expanding | 0.541731 (V21) | 0.194868 (V22-S) | -64.03% |
| 92.5% | Paper SA-LSTM | Rolling 756 | — | 0.617974 (V23) | — |
| 92.5% | Paper SA-LSTM | Recency | — | 0.372534 (V23) | — |
| 92.5% | Reoptimized SA-LSTM | Expanding | 0.352332 (V21) | 0.366077 (V22-S) | +3.90% |
| 92.5% | Reoptimized SA-LSTM | Rolling 756 | — | 0.533281 (V23) | — |
| 92.5% | Reoptimized SA-LSTM | Recency | — | 0.384123 (V23) | — |
| 95% | Paper SA-LSTM | Expanding | 0.600953 (V21) | 0.200803 (V22-S) | -66.59% |
| 95% | Paper SA-LSTM | Rolling 756 | — | 0.556494 (V23) | — |
| 95% | Paper SA-LSTM | Recency | — | 0.383168 (V23) | — |
| 95% | Reoptimized SA-LSTM | Expanding | 0.541005 (V21) | 0.325427 (V22-S) | -39.85% |
| 95% | Reoptimized SA-LSTM | Rolling 756 | — | 0.475157 (V23) | — |
| 95% | Reoptimized SA-LSTM | Recency | — | 0.325904 (V23) | — |



## 8. Walk-forward đồng bộ: kết quả V23

Thực nghiệm chỉ với 3 mô hình WTI: Residual-GRU, Residual-LSTM, Reoptimized SA-LSTM. Dùng **cùng 542 phiên** (07/10/2024–13/11/2026), seeds 41–45, `RETRAIN_EVERY=63`, 9 folds (8×63 phiên và 1×38). Tổng 135 lượt fit và 8.130 dòng dự báo. Mỗi fold fit từ đầu trong training history; một phần lịch sử gần origin dùng validation, không fit dữ liệu trong block test. SA HP giữ cố định từ V22, không chạy search ở mỗi fold. Phạm vi này **không trùng** walk-forward V20 từ 2018–2026 (2.214 phiên); tuyệt đối không so trực tiếp MAE giữa hai giai đoạn.

### 8.1. Từng seed và MAE/RMSE (V23)

| Model | Seed | MAE | RMSE | Naive MAE | Skill vs Naive |
|---|---|---|---|---|---|
| Residual-GRU | 41 | 0.068950 | 0.108020 | 0.069001 | +0.074% |
| Residual-GRU | 42 | 0.069089 | 0.108033 | 0.069001 | -0.127% |
| Residual-GRU | 43 | 0.069346 | 0.108287 | 0.069001 | -0.499% |
| Residual-GRU | 44 | 0.069117 | 0.108236 | 0.069001 | -0.168% |
| Residual-GRU | 45 | 0.068995 | 0.108007 | 0.069001 | +0.010% |
| Residual-LSTM | 41 | 0.069010 | 0.108008 | 0.069001 | -0.013% |
| Residual-LSTM | 42 | 0.069053 | 0.108068 | 0.069001 | -0.075% |
| Residual-LSTM | 43 | 0.069139 | 0.108061 | 0.069001 | -0.199% |
| Residual-LSTM | 44 | 0.069280 | 0.108522 | 0.069001 | -0.403% |
| Residual-LSTM | 45 | 0.069414 | 0.108386 | 0.069001 | -0.598% |
| Reoptimized SA-LSTM | 41 | 0.230471 | 0.298832 | 0.069001 | -234.008% |
| Reoptimized SA-LSTM | 42 | 0.202721 | 0.273849 | 0.069001 | -193.792% |
| Reoptimized SA-LSTM | 43 | 0.209556 | 0.274835 | 0.069001 | -203.698% |
| Reoptimized SA-LSTM | 44 | 0.178116 | 0.254883 | 0.069001 | -158.133% |
| Reoptimized SA-LSTM | 45 | 0.218425 | 0.287161 | 0.069001 | -216.551% |



### 8.2. Kết quả trung bình nhiều seeds (không thay thế kiểm định)

| Mô hình | MAE trung bình | SD MAE giữa seeds | RMSE trung bình | Seeds thắng Naive |
|---|---|---|---|---|
| Residual-GRU | 0.069100 | 0.000154 | 0.108117 | 2/5 |
| Residual-LSTM | 0.069179 | 0.000167 | 0.108209 | 0/5 |
| Reoptimized SA-LSTM | 0.207858 | 0.019601 | 0.277912 | 0/5 |



### 8.3. Kiểm định DM-HAC vs Naive (absolute loss; Newey–West lag 5)

`d_t=|y−ŷ_model|−|y−ŷ_naive|`, `d̄<0` nghiêng về mô hình, p-value hai phía. Nhiều seeds cùng chia sẻ test dates, không phải 5 tập test độc lập; kết quả chưa chỉnh multiple testing.

| Model | Seed | Mean loss diff | p-value (2 phía) |
|---|---|---|---|
| Residual-GRU | 41 | -0.000051 | 0.493343 |
| Residual-GRU | 42 | +0.000088 | 0.189622 |
| Residual-GRU | 43 | +0.000344 | 0.0567291 |
| Residual-GRU | 44 | +0.000116 | 0.397776 |
| Residual-GRU | 45 | -0.000007 | 0.931544 |
| Residual-LSTM | 41 | +0.000009 | 0.874817 |
| Residual-LSTM | 42 | +0.000052 | 0.598004 |
| Residual-LSTM | 43 | +0.000137 | 0.207789 |
| Residual-LSTM | 44 | +0.000278 | 0.190898 |
| Residual-LSTM | 45 | +0.000412 | 0.122574 |
| Reoptimized SA-LSTM | 41 | +0.161469 | 1.16255e-22 |
| Reoptimized SA-LSTM | 42 | +0.133719 | 8.06097e-18 |
| Reoptimized SA-LSTM | 43 | +0.140555 | 4.22525e-20 |
| Reoptimized SA-LSTM | 44 | +0.109114 | 1.7364e-12 |
| Reoptimized SA-LSTM | 45 | +0.149423 | 2.37241e-20 |



### 8.4. Theo từng fold (trung bình 5 seeds; không gộp đều 9 fold để tính MAE tổng)

| Fold | Bắt đầu | Kết thúc | n phiên | Residual-GRU MAE | Residual-LSTM MAE | SA-LSTM MAE |
|---|---|---|---|---|---|---|
| 1 | 2024-10-07 | 2025-01-06 | 63 | 0.041246 | 0.041045 | 0.130602 |
| 2 | 2025-01-07 | 2025-04-07 | 63 | 0.066384 | 0.066539 | 0.209103 |
| 3 | 2025-04-08 | 2025-07-04 | 63 | 0.060036 | 0.060706 | 0.148476 |
| 4 | 2025-07-07 | 2025-10-01 | 63 | 0.073553 | 0.073552 | 0.335969 |
| 5 | 2025-10-02 | 2025-12-29 | 63 | 0.068968 | 0.068937 | 0.150386 |
| 6 | 2025-12-30 | 2026-03-27 | 63 | 0.097913 | 0.097901 | 0.169261 |
| 7 | 2026-03-30 | 2026-06-25 | 63 | 0.081988 | 0.081840 | 0.283881 |
| 8 | 2026-06-26 | 2026-09-22 | 63 | 0.062681 | 0.062522 | 0.238213 |
| 9 | 2026-09-23 | 2026-11-13 | 38 | 0.069145 | 0.069826 | 0.202836 |



### 8.5. Walk-forward V20, giữ riêng theo giai đoạn

V20 đánh giá Residual-GRU + Paper3+Cu trên **2.214 phiên 2018–2026**, 5 seeds, retrain 63 phiên, sử dụng DM-HAC. Không dùng phép so sánh chênh MAE V20 và V23 để khẳng định WTI tốt hơn, vì khác đoạn thời gian và có thể khác giao thức học. Các kết luận V20 trước đây: seeds 43–44 Naive tốt hơn ở mức 5%; các seeds khác chưa thấy khác biệt ở mức 5%. Nguồn: `dm_hac_by_seed.csv` (V20).

## 9. Phân tích diễn giải và độ tin cậy

### 9.1. Vì sao Residual gần Naive?

Trong V23, theo bảng audit trên dự báo: độ lệch chuẩn delta thực khoảng 0.107963, trong khi độ lệch chuẩn delta dự báo của Residual-GRU khoảng 0.003004 và Residual-LSTM khoảng 0.004220, tương quan delta dự báo với delta thực lần lượt −0.0345 và −0.0361 (V23, gộp seeds). Do delta dự báo nhỏ, `P_(t−1)+Δ̂` gần Naive. Đây là **hành vi quan sát được**, không chứng minh chuỗi là random walk hay I(1), cũng không xác lập cơ chế duy nhất của mạng.

### 9.2. Khi đọc kết quả MAE

Các mô hình Residual có thể dẫn đầu *fitted* trong một số split, trong khi Naive vẫn có thể mạnh hơn trên walk-forward. SA-LSTM có thể giảm MAE so với chính cấu hình cũ nhưng vẫn cao hơn Naive và Residual. Khác biệt tương đối rất lớn tính từ MAE Direct/SA lớn không tự chứng minh tác động nhân quả của feature/tuning.

### 9.3. Kết quả đánh giá không phải xếp hạng năng lực vĩnh viễn

Top theo test là lựa chọn hậu nghiệm; để khẳng định vượt trội cần chọn mô hình trên validation rồi đánh giá untouched test, preferably walk-forward nhiều mốc và kiểm định trên cùng ngày. Rủi ro selection bias và khác ngân sách hyperparameter search phải được công khai.

## 10. Ma trận công bằng và các giới hạn còn tồn tại

| Hạng mục | Đã kiểm soát / có dữ liệu | Giới hạn cần bảo lưu |
|---|---|---|
| Dữ liệu | Cùng hash, 2.710 hàng, lookback 22, horizon 1 | Chưa audit giờ thực sự khả dụng của ngoại sinh |
| Holdout theo split | So sánh các kết quả trong cùng khoảng test | Các split có test lồng nhau |
| 3 chiến lược | Residual V20/V22 đủ 3; SA WTI 90% và V23 92.5/95% được bổ sung | SA Legacy14 92.5/95% Rolling/Recency chưa có; riêng V21 bổ sung 80/20 đã có đủ ba chiến lược |
| Target | Residual học ΔP; SA-LSTM học mức P | Không thể quy riêng hiệu ứng kiến trúc |
| SA search | V22 có 16 trials; V23 dùng HP đã tìm | V21 dùng cache; search budget không matched trên mọi mô hình |
| Walk-forward V23 | Ba fitted models, 542 phiên, 5 seeds, retrain 63 | Chưa có WF đầy đủ cho mọi kiến trúc; V20 WF khác giai đoạn |
| Thống kê | DM-HAC lag 5 theo seed | Multiple testing, model selection, seeds chồng lặp |
| Ý nghĩa thực tiễn | Có MAE/RMSE trên đơn vị giá gốc | Chưa kiểm định kinh tế/backtest hoặc cost/slippage |

**Lưu ý quan trọng:** Không được viết “mọi phương pháp hoàn toàn công bằng” khi chưa thiết kế đối chứng match feature, target representation, tuning và walk-forward ở cùng forecast timestamps. Bảng xếp hạng chỉ phản ánh các cấu hình đã chạy, và những ô khuyết được giữ khuyết.

## 11. Các câu hỏi phản biện nên chuẩn bị

**Vì sao WTI thay Gold+Silver?** Vì Spearman mức giá WTI cao hơn hai biến đó trong EDA; đây là giả thuyết, không phải quy tắc tối ưu dự báo. Có thể có tương quan chung do xu hướng giá.

**Ablation chưa đầy đủ ở đâu?** Chưa thực hiện leave-one-out riêng Gold/Silver/WTI với số lượng feature tương thích và validation lựa chọn.

**63 có phải số epochs?** Không, là số phiên cách nhau giữa lần refit; tương đương gần 3 tháng giao dịch.

**Vì sao Naive không xếp hạng nhưng vẫn báo?** Để xác định mô hình fitted tốt nhất trong phạm vi nghiên cứu, đồng thời giữ benchmark xác minh có giá trị tăng thêm hay không.

**Vì sao không tuyên bố SA tối ưu luôn thắng?** SA chỉ tìm HP trong một không gian/ngân sách; kết quả SA sử dụng HP V22 ở V23, chưa xác nhận tối ưu toàn cục và chưa đối chứng tuning budget ngang nhau.

**Vì sao không ghép MAE walk-forward 2018–2026 với 2024–2026?** Vì khoảng thời gian, độ khó và số quan sát khác nhau; chỉ nên so trên cùng timestamps.

**Không có p<0.05 có nghĩa mô hình bằng Naive?** Không; chỉ chưa phát hiện khác biệt thống kê theo phép kiểm định đang dùng.

## 12. Kết luận

Sau khi gộp các nguồn V20–V23, báo cáo thống nhất cách trình bày theo mô hình Residual/SA-LSTM, tập biến, chiến lược và split, đồng thời cung cấp bảng xếp hạng fitted và baseline Naive riêng. V23 hoàn tất các ô holdout WTI còn thiếu của nhánh SA-series và bổ sung walk-forward 5 seeds trên cùng 542 mốc dự báo cho Residual-LSTM, Residual-GRU, Reoptimized SA-LSTM. Kết quả phản ánh độ nhạy theo biểu diễn target, feature set, chiến lược và lịch đánh giá; chưa có bằng chứng đủ để xác lập một thuật toán luôn tối ưu hay quy phần thay đổi do WTI/SA riêng rẽ. Các giới hạn được nêu minh bạch để giữ tính khách quan và khả năng truy xuất.

## Tài liệu tham khảo và phụ lục truy xuất nguồn

1. Chen et al. (2023), *Copper price prediction using LSTM recurrent neural network integrated simulated annealing algorithm*, PLOS ONE. Metadata thư mục cần đối chiếu bản paper khi nộp.
2. Diebold, F. X., & Mariano, R. S. (1995), *Comparing predictive accuracy*, Journal of Business & Economic Statistics.
3. Newey, W. K., & West, K. D. (1987), *A simple, positive semi-definite heteroskedasticity and autocorrelation consistent covariance matrix*, Econometrica.

**Nguồn dữ liệu, mã và bảng:**

- `(V20)` `17-11-Copper-NonSA-Fair-Full-Protocol-V20(2).zip`: `grid_results.csv`, `grid_predictions.csv`, `walk_forward_predictions.csv`, `dm_hac_by_seed.csv`, EDA và residual audit.
- `(V21)` `copper_sa_lstm_h1_ablation_v21_results(1).zip`: `results_h1_ablation.csv`, cache SA và predictions.
- `(V21-80)` `copper_v21_addon_80_wf_plots(2).zip`: `holdout_80_results.csv`, `holdout_80_predictions.csv`.
- `(V22-N)` `[]18-11- Copper_NonSA_V22_WTI_Only_Kaggle.zip`: `grid_results.csv`, `walk_forward_predictions.csv`, EDA.
- `(V22-S)` `[]18-11copper_sa_lstm_wti_only_v22_results.zip`: `results_h1_ablation.csv`, `fresh_sa_best_hp.json`, `sa_search_trials.csv`.
- `(V23)` `20 -11 - copper_v23_gapfill_results.zip`: `holdout_missing_results.csv`, `holdout_missing_predictions.csv`, `wf_seed_summary.csv`, `wf_fold_metrics.csv`, `wf_predictions.csv`, `wf_dm_hac_by_seed.csv`.
- Bản Mythos nền: `Mythos_report_formated_V20_V21_V22(1).md`; kiểm toán bổ sung: `BAO_CAO_BO_SUNG_MODULE_COPPER_V23_KIEM_TOAN_CHI_TIET_20_11_2026.md`.

**Chú ý:** Không khẳng định có kết quả ở bất kỳ ô nào nếu nguồn CSV không có bản ghi; số trong bảng được làm tròn 6 chữ số, sử dụng CSV gốc khi đối chiếu chi tiết.

