# BÁO CÁO THỰC NGHIỆM TỔNG HỢP: DỰ BÁO GIÁ ĐỒNG MỘT PHIÊN TRƯỚC

**Ngày cập nhật:** 19/11/2026  |  **Tác giả thực nghiệm:** Viên Đăng Khoa  |  **Học phần:** IS403

**Quy ước trình bày:** Không chia nội dung kết quả theo phiên bản. Kết quả được hợp nhất theo **mô hình – bộ đặc trưng – chiến lược – split**, với phiên bản ghi ngay sau mỗi số liệu để truy xuất. Mọi bảng được tính trực tiếp từ CSV của các ZIP; không train lại trong lúc biên soạn.

## Tóm tắt

Nghiên cứu đối chiếu các mô hình LSTM, GRU, CNN-LSTM, CNN-GRU, SimpleRNN, Ridge, và SA-LSTM bằng hai cách biểu diễn dự báo giá trực tiếp (Direct) hoặc sai phân giá (Residual). Thí nghiệm mở rộng bộ đầu vào `Copper, Aluminium, Tin, WTI` nhằm kiểm tra liệu thay Gold/Silver bằng WTI có cải thiện sai số ngoài mẫu. **Xếp hạng mô hình tốt nhất chỉ tính các mô hình fitted**, trong khi Naive Persistence luôn được báo riêng như baseline định lượng. Các thứ hạng dựa trên MAE ở từng holdout, chưa đồng nghĩa chứng minh ưu thế tổng quát hóa.

## 1. Mục tiêu và phương pháp thống nhất

### 1.1. Bài toán và nguồn dữ liệu

- Target: `Copper(t)`; đầu vào dùng 22 phiên `t−22 ... t−1` (`LOOKBACK=22`, `HORIZON=1`). Dữ liệu gốc 2.710 quan sát, từ **13/01/2016 đến 13/11/2026** theo fingerprint/manifest cũ. Phải kiểm tra đồng bộ thời điểm đóng cửa của các thị trường nếu ứng dụng dự báo thật.

- Holdout theo thời gian gồm 80/20 (542 test), 90/10 (271), 92.5/7.5 (204), 95/5 (136); bảng cùng split chỉ hợp lệ nếu ngày test tương đồng. MAE/RMSE tính trên đơn vị giá gốc. Không chọn hyperparameters bằng test.

- **Direct:** dự đoán `Copper(t)` trực tiếp. **Residual:** dự đoán `ΔCopper(t) = Copper(t) − Copper(t−1)` rồi cộng lại Copper(t−1). **Naive:** dùng Copper(t−1) làm dự báo. Đây là baseline không huấn luyện, không tham gia xếp hạng fitted.

- **Expanding:** dùng lịch sử đến ranh giới train. **Rolling 756:** giữ 756 phiên gần nhất trước fit. **Recency:** toàn lịch sử nhưng weighted half-life 252 phiên. Các chiến lược holdout không phải mỗi phiên lại retrain; chỉ walk-forward mới refit theo lịch.

### 1.2. Bảng bộ đặc trưng và nhãn nguồn

| Bộ đặc trưng | Danh sách biến | Nguồn |
|---|---|---|
| Copper-only | Copper | V20 |
| Paper3+Cu | Copper, WTI, Gold, Silver | V20/V21 |
| Paper3−Cu | WTI, Gold, Silver | V21 |
| Legacy14+Cu | Copper, Aluminium, Tin, Silver, Gold | V20/V21 |
| Legacy14−Cu | Aluminium, Tin, Silver, Gold | V21 |
| WTI mới+Cu | Copper, Aluminium, Tin, WTI | V22 Non-SA / V22 SA |



> **Nguồn ở mỗi số liệu:** `V20` = Non-SA cũ; `V21` và `V21 bổ sung 80/20` = SA-series cũ; `V22 Non-SA` và `V22 SA` = đầu vào WTI mới. Hai lần triển khai LSTM Direct tương tự ở các pipeline có thể tạo bản ghi trùng kiến trúc; được giữ để truy vết, không xem là hai thuật toán khác nhau.

### 1.3. EDA làm cơ sở đặt giả thuyết

| Ngoại sinh | Spearman mức giá với Copper | Spearman lag-1 của biến động |
|---|---|---|
| Aluminium | 0.9279 | 0.0093 |
| Tin | 0.8829 | −0.0185 |
| WTI | 0.7711 | 0.0142 |
| Gold | 0.7425 | 0.0910 |
| Silver | 0.7424 | 0.0388 |



WTI có Spearman ở **mức giá** cao hơn Gold và Silver nhưng tương quan lag-1 của biến động WTI rất yếu. Việc đổi biến là một **giả thuyết thử nghiệm**, không phải bằng chứng WTI chắc chắn hữu ích hơn.

### 1.4. Mô hình, hyperparameters và tính công bằng

- Nhóm Non-SA: LSTM/GRU/CNN-LSTM/CNN-GRU Direct và Residual, Ridge Residual, Naive; cấu hình cũ 65 epochs tối đa, EarlyStopping patience 9, batch 64, Adam LR 0.001. Các so sánh WTI với Legacy14 nội bộ nhóm Non-SA gần nhất với phép thay feature có kiểm soát.

- Nhóm SA-series: LSTM Direct, SimpleRNN Direct, LSTM dùng Paper HP, LSTM dùng SA tìm/tối ưu HP; tối đa 100 epochs ở holdout. Paper HP của Chen et al. (2023) được tái hiện theo những chi tiết đã công bố, không phải code gốc xác minh được.

- **V21 Reoptimized** dùng hyperparameters đã lưu, còn **V22 Reoptimized** tìm mới bằng SA (16 trials, reference split 80/20, 15 epochs/trial). Do vậy, MAE V21–V22 của nhánh này đồng thời chịu tác động đổi feature **và** đổi hyperparameters, không phải ablation thuần feature.

## 2. Xếp hạng tổng hợp theo 4 split (không tính Naive)

**Top 5 mỗi split** được xếp theo MAE tăng dần trên tất cả kết quả fitted có sẵn, bất kể phiên bản hay họ mô hình. Đây là top cấu hình **hậu nghiệm**, chưa phải xác nhận từ một test hoàn toàn mới.

| Split | Hạng | Mô hình | Features | Chiến lược | MAE (nguồn) | RMSE (nguồn) |
|---|---|---|---|---|---|---|
| 80/20 | 1 | GRU Residual | Paper3+Cu | Expanding | 0.068780 (V20) | 0.107853 (V20) |
| 80/20 | 2 | LSTM Residual | WTI mới+Cu | Expanding | 0.068840 (V22-N) | 0.107996 (V22-N) |
| 80/20 | 3 | CNN-LSTM Residual | Copper-only | Recency | 0.068841 (V20) | 0.108019 (V20) |
| 80/20 | 4 | CNN-LSTM Residual | Paper3+Cu | Recency | 0.068843 (V20) | 0.107912 (V20) |
| 80/20 | 5 | LSTM Residual | Paper3+Cu | Recency | 0.068850 (V20) | 0.107992 (V20) |
| 90/10 | 1 | GRU Residual | Legacy14+Cu | Expanding | 0.076951 (V20) | 0.100810 (V20) |
| 90/10 | 2 | LSTM Residual | Legacy14+Cu | Expanding | 0.076978 (V20) | 0.100948 (V20) |
| 90/10 | 3 | CNN-LSTM Residual | Copper-only | Rolling 756 | 0.077000 (V20) | 0.100895 (V20) |
| 90/10 | 4 | LSTM Residual | Paper3+Cu | Expanding | 0.077006 (V20) | 0.100878 (V20) |
| 90/10 | 5 | LSTM Residual | Copper-only | Expanding | 0.077008 (V20) | 0.100810 (V20) |
| 92.5/7.5 | 1 | GRU Residual | WTI mới+Cu | Recency | 0.075028 (V22-N) | 0.098307 (V22-N) |
| 92.5/7.5 | 2 | CNN-LSTM Residual | Copper-only | Recency | 0.075050 (V20) | 0.098426 (V20) |
| 92.5/7.5 | 3 | CNN-LSTM Residual | WTI mới+Cu | Expanding | 0.075054 (V22-N) | 0.098459 (V22-N) |
| 92.5/7.5 | 4 | CNN-LSTM Residual | Copper-only | Expanding | 0.075057 (V20) | 0.098416 (V20) |
| 92.5/7.5 | 5 | CNN-LSTM Residual | Copper-only | Rolling 756 | 0.075059 (V20) | 0.098468 (V20) |
| 95/5 | 1 | GRU Residual | Paper3+Cu | Recency | 0.072357 (V20) | 0.096414 (V20) |
| 95/5 | 2 | GRU Residual | Legacy14+Cu | Recency | 0.072507 (V20) | 0.096463 (V20) |
| 95/5 | 3 | LSTM Residual | Paper3+Cu | Expanding | 0.072517 (V20) | 0.096496 (V20) |
| 95/5 | 4 | LSTM Residual | Paper3+Cu | Rolling 756 | 0.072554 (V20) | 0.096450 (V20) |
| 95/5 | 5 | CNN-GRU Residual | Paper3+Cu | Expanding | 0.072575 (V20) | 0.096666 (V20) |



**Naive baseline (không tham gia xếp hạng):**

| Split | MAE | RMSE | Nguồn kiểm tra |
|---|---|---|---|
| 80/20 | 0.069001 | 0.108000 | V20 / V21-80 / V22-N / V22-S |
| 90/10 | 0.077077 | 0.101033 | V20 / V21 / V22-N / V22-S |
| 92.5/7.5 | 0.075116 | 0.098399 | V20 / V21 / V22-N / V22-S |
| 95/5 | 0.072799 | 0.096612 | V20 / V21 / V22-N / V22-S |



## 3. So sánh toàn bộ mô hình theo chiến lược tại split 90/10

Bảng này giải quyết trực tiếp việc đọc rời rạc giữa các phiên bản: mỗi hàng là **một mô hình + một bộ input + một nguồn**, ba cột hiển thị Expanding/Rolling/Recency đặt cạnh nhau. `—` = chưa thực nghiệm theo nguồn đó. Đơn vị MAE trên giá gốc.

| Mô hình | Bộ biến | Expanding MAE | Rolling MAE | Recency MAE |
|---|---|---|---|---|
| CNN-GRU Direct | Copper-only | 0.633998 (V20) | 0.432151 (V20) | 0.508912 (V20) |
| CNN-GRU Direct | Legacy14+Cu | 0.313523 (V20) | 0.483141 (V20) | 0.298726 (V20) |
| CNN-GRU Direct | Paper3+Cu | 0.296379 (V20) | 0.311519 (V20) | 0.457908 (V20) |
| CNN-GRU Direct | WTI mới+Cu | 0.619798 (V22-N) | 0.427192 (V22-N) | 0.643456 (V22-N) |
| CNN-GRU Residual | Copper-only | 0.078067 (V20) | 0.088180 (V20) | 0.080173 (V20) |
| CNN-GRU Residual | Legacy14+Cu | 0.090563 (V20) | 0.098870 (V20) | 0.077522 (V20) |
| CNN-GRU Residual | Paper3+Cu | 0.078132 (V20) | 0.097694 (V20) | 0.083405 (V20) |
| CNN-GRU Residual | WTI mới+Cu | 0.077229 (V22-N) | 0.085898 (V22-N) | 0.077639 (V22-N) |
| CNN-LSTM Direct | Copper-only | 0.702867 (V20) | 0.686135 (V20) | 0.829593 (V20) |
| CNN-LSTM Direct | Legacy14+Cu | 0.393469 (V20) | 0.687898 (V20) | 0.991953 (V20) |
| CNN-LSTM Direct | Paper3+Cu | 1.108590 (V20) | 0.790701 (V20) | 0.716936 (V20) |
| CNN-LSTM Direct | WTI mới+Cu | 0.888314 (V22-N) | 0.873802 (V22-N) | 0.886170 (V22-N) |
| CNN-LSTM Residual | Copper-only | 0.080098 (V20) | 0.077000 (V20) | 0.078311 (V20) |
| CNN-LSTM Residual | Legacy14+Cu | 0.077245 (V20) | 0.081200 (V20) | 0.077304 (V20) |
| CNN-LSTM Residual | Paper3+Cu | 0.077432 (V20) | 0.091009 (V20) | 0.079093 (V20) |
| CNN-LSTM Residual | WTI mới+Cu | 0.077638 (V22-N) | 0.085373 (V22-N) | 0.078023 (V22-N) |
| GRU Direct | Copper-only | 0.437113 (V20) | 0.206817 (V20) | 0.205153 (V20) |
| GRU Direct | Legacy14+Cu | 0.240815 (V20) | 0.357864 (V20) | 0.273153 (V20) |
| GRU Direct | Paper3+Cu | 0.247893 (V20) | 0.492330 (V20) | 0.254494 (V20) |
| GRU Direct | WTI mới+Cu | 0.243208 (V22-N) | 0.256292 (V22-N) | 0.235421 (V22-N) |
| GRU Residual | Copper-only | 0.080182 (V20) | 0.078851 (V20) | 0.083943 (V20) |
| GRU Residual | Legacy14+Cu | 0.076951 (V20) | 0.077182 (V20) | 0.077055 (V20) |
| GRU Residual | Paper3+Cu | 0.077143 (V20) | 0.077209 (V20) | 0.077304 (V20) |
| GRU Residual | WTI mới+Cu | 0.077463 (V22-N) | 0.081939 (V22-N) | 0.077851 (V22-N) |
| LSTM Direct | Copper-only | 0.446716 (V20) | 0.277933 (V20) | 0.690705 (V20) |
| LSTM Direct | Legacy14+Cu | 0.247410 (V20) | 0.958465 (V20) | 0.659060 (V20) |
| LSTM Direct | Legacy14+Cu | 0.247409 (V21) | 0.958466 (V21) | 0.659060 (V21) |
| LSTM Direct | Legacy14−Cu | 0.854128 (V21) | 0.909432 (V21) | 0.790417 (V21) |
| LSTM Direct | Paper3+Cu | 1.163964 (V20) | 0.918796 (V20) | 0.564050 (V20) |
| LSTM Direct | Paper3+Cu | 1.163964 (V21) | 0.918796 (V21) | 0.564049 (V21) |
| LSTM Direct | Paper3−Cu | 1.145988 (V21) | 1.062130 (V21) | 0.686208 (V21) |
| LSTM Direct | WTI mới+Cu | 0.324937 (V22-N) | 0.346713 (V22-N) | 1.236604 (V22-N) |
| LSTM Direct | WTI mới+Cu | 0.324940 (V22-S) | 0.321192 (V22-S) | 1.236605 (V22-S) |
| LSTM Residual | Copper-only | 0.077008 (V20) | 0.077159 (V20) | 0.078135 (V20) |
| LSTM Residual | Legacy14+Cu | 0.076978 (V20) | 0.077756 (V20) | 0.078692 (V20) |
| LSTM Residual | Paper3+Cu | 0.077006 (V20) | 0.077230 (V20) | 0.078318 (V20) |
| LSTM Residual | WTI mới+Cu | 0.077037 (V22-N) | 0.080271 (V22-N) | 0.078856 (V22-N) |
| Ridge Residual | Copper-only | 0.077306 (V20) | 0.082705 (V20) | 0.079330 (V20) |
| Ridge Residual | Legacy14+Cu | 0.081707 (V20) | 0.084002 (V20) | 0.085664 (V20) |
| Ridge Residual | Paper3+Cu | 0.080712 (V20) | 0.098350 (V20) | 0.090112 (V20) |
| Ridge Residual | WTI mới+Cu | 0.077737 (V22-N) | 0.094420 (V22-N) | 0.081044 (V22-N) |
| SA-LSTM Paper HP | Legacy14+Cu | 0.470929 (V21) | 0.773310 (V21) | 0.705350 (V21) |
| SA-LSTM Paper HP | Legacy14−Cu | 0.464991 (V21) | 0.869224 (V21) | 0.524705 (V21) |
| SA-LSTM Paper HP | Paper3+Cu | 0.978075 (V21) | 0.897077 (V21) | 0.917524 (V21) |
| SA-LSTM Paper HP | Paper3−Cu | 0.540720 (V21) | 0.604594 (V21) | 0.585950 (V21) |
| SA-LSTM Paper HP | WTI mới+Cu | 0.184634 (V22-S) | 0.575763 (V22-S) | 0.294741 (V22-S) |
| SA-LSTM Reoptimized | Legacy14+Cu | 0.510467 (V21) | 0.977821 (V21) | 0.735621 (V21) |
| SA-LSTM Reoptimized | Legacy14−Cu | 0.466595 (V21) | 0.624100 (V21) | 0.460227 (V21) |
| SA-LSTM Reoptimized | Paper3+Cu | 0.877420 (V21) | 0.974593 (V21) | 0.511784 (V21) |
| SA-LSTM Reoptimized | Paper3−Cu | 0.595605 (V21) | 0.644180 (V21) | 0.482221 (V21) |
| SA-LSTM Reoptimized | WTI mới+Cu | 0.358921 (V22-S) | 0.496288 (V22-S) | 0.326683 (V22-S) |
| SimpleRNN Direct | Legacy14+Cu | 0.831186 (V21) | 1.115149 (V21) | 1.045665 (V21) |
| SimpleRNN Direct | Legacy14−Cu | 0.962108 (V21) | 1.128623 (V21) | 0.608387 (V21) |
| SimpleRNN Direct | Paper3+Cu | 1.385730 (V21) | 1.316286 (V21) | 0.973459 (V21) |
| SimpleRNN Direct | Paper3−Cu | 1.001232 (V21) | 1.321809 (V21) | 1.568182 (V21) |
| SimpleRNN Direct | WTI mới+Cu | 0.818738 (V22-S) | 0.859431 (V22-S) | 1.009153 (V22-S) |



**Cách đọc:** Trước hết so sánh ba cột trong cùng hàng (đổi strategy), rồi so sánh giữa các hàng có cùng mô hình và cùng cột (đổi features). Không kết luận chiến lược tốt hơn toàn cục từ một hàng.

## 4. Tác động thay Legacy14+Cu bằng WTI mới+Cu — ghép cặp đúng mô hình và chiến lược

Định nghĩa `ΔMAE (%) = 100 × (MAE_WTI / MAE_Legacy14 − 1)`. **Âm = WTI giảm sai số, tốt hơn; dương = WTI tăng sai số, kém hơn**. Chỉ ghép các trường hợp trùng mô hình, split và strategy; bảng dưới còn ghi rõ nguồn. Với SA Reoptimized, chênh lệch bao gồm cả SA search mới.

### 4.1. Split 80/20

| Mô hình | Chiến lược | MAE Legacy14 | MAE WTI | ΔMAE | RMSE cũ | RMSE mới |
|---|---|---|---|---|---|---|
| CNN-GRU Direct | Expanding | 0.214287 (V20) | 0.301326 (V22-N) | +40.62% | 0.273292 | 0.379359 |
| CNN-GRU Direct | Recency | 0.313767 (V20) | 0.321193 (V22-N) | +2.37% | 0.385948 | 0.400114 |
| CNN-GRU Direct | Rolling 756 | 0.594207 (V20) | 0.315794 (V22-N) | -46.85% | 0.717580 | 0.404259 |
| CNN-GRU Residual | Expanding | 0.074771 (V20) | 0.069742 (V22-N) | -6.73% | 0.114417 | 0.108045 |
| CNN-GRU Residual | Recency | 0.111344 (V20) | 0.068947 (V22-N) | -38.08% | 0.151494 | 0.107682 |
| CNN-GRU Residual | Rolling 756 | 0.095740 (V20) | 0.069534 (V22-N) | -27.37% | 0.135472 | 0.107919 |
| CNN-LSTM Direct | Expanding | 0.718548 (V20) | 0.438657 (V22-N) | -38.95% | 0.844535 | 0.556630 |
| CNN-LSTM Direct | Recency | 0.558670 (V20) | 0.435179 (V22-N) | -22.10% | 0.702984 | 0.550786 |
| CNN-LSTM Direct | Rolling 756 | 0.842083 (V20) | 0.636210 (V22-N) | -24.45% | 1.052927 | 0.781975 |
| CNN-LSTM Residual | Expanding | 0.068916 (V20) | 0.068972 (V22-N) | +0.08% | 0.108116 | 0.107827 |
| CNN-LSTM Residual | Recency | 0.068914 (V20) | 0.068931 (V22-N) | +0.02% | 0.107779 | 0.107935 |
| CNN-LSTM Residual | Rolling 756 | 0.069998 (V20) | 0.069118 (V22-N) | -1.26% | 0.108669 | 0.107833 |
| GRU Direct | Expanding | 0.196520 (V20) | 0.164666 (V22-N) | -16.21% | 0.258860 | 0.222705 |
| GRU Direct | Recency | 0.246072 (V20) | 0.156398 (V22-N) | -36.44% | 0.324673 | 0.209462 |
| GRU Direct | Rolling 756 | 0.687417 (V20) | 0.212751 (V22-N) | -69.05% | 0.842202 | 0.282181 |
| GRU Residual | Expanding | 0.068857 (V20) | 0.068966 (V22-N) | +0.16% | 0.107932 | 0.107862 |
| GRU Residual | Recency | 0.068933 (V20) | 0.068914 (V22-N) | -0.03% | 0.107941 | 0.107935 |
| GRU Residual | Rolling 756 | 0.133698 (V20) | 0.069751 (V22-N) | -47.83% | 0.175810 | 0.108199 |
| LSTM Direct | Expanding | 0.713145 (V20) | 0.531803 (V22-N) | -25.43% | 0.878396 | 0.662974 |
| LSTM Direct | Recency | 0.574219 (V20) | 0.528022 (V22-N) | -8.05% | 0.722827 | 0.673269 |
| LSTM Direct | Rolling 756 | 1.096387 (V20) | 0.424343 (V22-N) | -61.30% | 1.355998 | 0.550282 |
| LSTM Residual | Expanding | 0.068912 (V20) | 0.068840 (V22-N) | -0.10% | 0.108011 | 0.107996 |
| LSTM Residual | Recency | 0.068907 (V20) | 0.069815 (V22-N) | +1.32% | 0.108036 | 0.108448 |
| LSTM Residual | Rolling 756 | 0.076509 (V20) | 0.069943 (V22-N) | -8.58% | 0.116666 | 0.108243 |
| Ridge Residual | Expanding | 0.072588 (V20) | 0.069603 (V22-N) | -4.11% | 0.112238 | 0.108010 |
| Ridge Residual | Recency | 0.077906 (V20) | 0.071724 (V22-N) | -7.94% | 0.119160 | 0.109358 |
| Ridge Residual | Rolling 756 | 0.163842 (V20) | 0.076286 (V22-N) | -53.44% | 0.192049 | 0.112686 |
| LSTM Direct | Expanding | 0.713145 (V21-80) | 0.531802 (V22-S) | -25.43% | 0.878396 | 0.662974 |
| SA-LSTM Paper HP | Expanding | 0.760563 (V21-80) | 0.298955 (V22-S) | -60.69% | 1.002313 | 0.358768 |
| SA-LSTM Reoptimized | Expanding | 1.000423 (V21-80) | 0.223809 (V22-S) | -77.63% | 1.189193 | 0.297888 |
| SimpleRNN Direct | Expanding | 0.795791 (V21-80) | 0.837846 (V22-S) | +5.28% | 0.916974 | 0.994480 |



Có 31 cặp được ghép có số liệu tại split 80/20. Các cặp không có trong CSV không được nội suy.

### 4.2. Split 90/10

| Mô hình | Chiến lược | MAE Legacy14 | MAE WTI | ΔMAE | RMSE cũ | RMSE mới |
|---|---|---|---|---|---|---|
| CNN-GRU Direct | Expanding | 0.313523 (V20) | 0.619798 (V22-N) | +97.69% | 0.381001 | 0.664925 |
| CNN-GRU Direct | Recency | 0.298726 (V20) | 0.643456 (V22-N) | +115.40% | 0.372347 | 0.689708 |
| CNN-GRU Direct | Rolling 756 | 0.483141 (V20) | 0.427192 (V22-N) | -11.58% | 0.527339 | 0.470401 |
| CNN-GRU Residual | Expanding | 0.090563 (V20) | 0.077229 (V22-N) | -14.72% | 0.117479 | 0.101285 |
| CNN-GRU Residual | Recency | 0.077522 (V20) | 0.077639 (V22-N) | +0.15% | 0.100867 | 0.101300 |
| CNN-GRU Residual | Rolling 756 | 0.098870 (V20) | 0.085898 (V22-N) | -13.12% | 0.122720 | 0.108409 |
| CNN-LSTM Direct | Expanding | 0.393469 (V20) | 0.888314 (V22-N) | +125.76% | 0.492822 | 0.952557 |
| CNN-LSTM Direct | Recency | 0.991953 (V20) | 0.886170 (V22-N) | -10.66% | 1.048918 | 0.953553 |
| CNN-LSTM Direct | Rolling 756 | 0.687898 (V20) | 0.873802 (V22-N) | +27.02% | 0.810581 | 0.945043 |
| CNN-LSTM Residual | Expanding | 0.077245 (V20) | 0.077638 (V22-N) | +0.51% | 0.100856 | 0.101076 |
| CNN-LSTM Residual | Recency | 0.077304 (V20) | 0.078023 (V22-N) | +0.93% | 0.101023 | 0.101646 |
| CNN-LSTM Residual | Rolling 756 | 0.081200 (V20) | 0.085373 (V22-N) | +5.14% | 0.103617 | 0.107714 |
| GRU Direct | Expanding | 0.240815 (V20) | 0.243208 (V22-N) | +0.99% | 0.302923 | 0.287070 |
| GRU Direct | Recency | 0.273153 (V20) | 0.235421 (V22-N) | -13.81% | 0.325727 | 0.275645 |
| GRU Direct | Rolling 756 | 0.357864 (V20) | 0.256292 (V22-N) | -28.38% | 0.398964 | 0.292692 |
| GRU Residual | Expanding | 0.076951 (V20) | 0.077463 (V22-N) | +0.67% | 0.100810 | 0.101052 |
| GRU Residual | Recency | 0.077055 (V20) | 0.077851 (V22-N) | +1.03% | 0.100969 | 0.101376 |
| GRU Residual | Rolling 756 | 0.077182 (V20) | 0.081939 (V22-N) | +6.16% | 0.101018 | 0.104597 |
| LSTM Direct | Expanding | 0.247410 (V20) | 0.324937 (V22-N) | +31.34% | 0.304620 | 0.381590 |
| LSTM Direct | Recency | 0.659060 (V20) | 1.236604 (V22-N) | +87.63% | 0.772684 | 1.296919 |
| LSTM Direct | Rolling 756 | 0.958465 (V20) | 0.346713 (V22-N) | -63.83% | 1.000966 | 0.382468 |
| LSTM Residual | Expanding | 0.076978 (V20) | 0.077037 (V22-N) | +0.08% | 0.100948 | 0.100980 |
| LSTM Residual | Recency | 0.078692 (V20) | 0.078856 (V22-N) | +0.21% | 0.103507 | 0.102206 |
| LSTM Residual | Rolling 756 | 0.077756 (V20) | 0.080271 (V22-N) | +3.23% | 0.101509 | 0.103472 |
| Ridge Residual | Expanding | 0.081707 (V20) | 0.077737 (V22-N) | -4.86% | 0.106728 | 0.101669 |
| Ridge Residual | Recency | 0.085664 (V20) | 0.081044 (V22-N) | -5.39% | 0.112437 | 0.104682 |
| Ridge Residual | Rolling 756 | 0.084002 (V20) | 0.094420 (V22-N) | +12.40% | 0.108523 | 0.116679 |
| LSTM Direct | Expanding | 0.247409 (V21) | 0.324940 (V22-S) | +31.34% | 0.304619 | 0.381594 |
| LSTM Direct | Recency | 0.659060 (V21) | 1.236605 (V22-S) | +87.63% | 0.772684 | 1.296920 |
| LSTM Direct | Rolling 756 | 0.958466 (V21) | 0.321192 (V22-S) | -66.49% | 1.000967 | 0.350564 |
| SA-LSTM Paper HP | Expanding | 0.470929 (V21) | 0.184634 (V22-S) | -60.79% | 0.527551 | 0.225723 |
| SA-LSTM Paper HP | Recency | 0.705350 (V21) | 0.294741 (V22-S) | -58.21% | 0.769216 | 0.346242 |
| SA-LSTM Paper HP | Rolling 756 | 0.773310 (V21) | 0.575763 (V22-S) | -25.55% | 0.842719 | 0.634142 |
| SA-LSTM Reoptimized | Expanding | 0.510467 (V21) | 0.358921 (V22-S) | -29.69% | 0.591630 | 0.415277 |
| SA-LSTM Reoptimized | Recency | 0.735621 (V21) | 0.326683 (V22-S) | -55.59% | 0.825336 | 0.376388 |
| SA-LSTM Reoptimized | Rolling 756 | 0.977821 (V21) | 0.496288 (V22-S) | -49.25% | 1.034595 | 0.552808 |
| SimpleRNN Direct | Expanding | 0.831186 (V21) | 0.818738 (V22-S) | -1.50% | 0.886049 | 0.900169 |
| SimpleRNN Direct | Recency | 1.045665 (V21) | 1.009153 (V22-S) | -3.49% | 1.113621 | 1.089029 |
| SimpleRNN Direct | Rolling 756 | 1.115149 (V21) | 0.859431 (V22-S) | -22.93% | 1.185744 | 0.937179 |



Có 39 cặp được ghép có số liệu tại split 90/10. Các cặp không có trong CSV không được nội suy.

### 4.3. Split 92.5/7.5

| Mô hình | Chiến lược | MAE Legacy14 | MAE WTI | ΔMAE | RMSE cũ | RMSE mới |
|---|---|---|---|---|---|---|
| CNN-GRU Direct | Expanding | 0.443855 (V20) | 0.719336 (V22-N) | +62.07% | 0.504454 | 0.750956 |
| CNN-GRU Direct | Recency | 0.351300 (V20) | 0.633058 (V22-N) | +80.20% | 0.400391 | 0.672985 |
| CNN-GRU Direct | Rolling 756 | 0.470120 (V20) | 0.571354 (V22-N) | +21.53% | 0.529660 | 0.608944 |
| CNN-GRU Residual | Expanding | 0.080471 (V20) | 0.076010 (V22-N) | -5.54% | 0.105340 | 0.099649 |
| CNN-GRU Residual | Recency | 0.079462 (V20) | 0.081707 (V22-N) | +2.82% | 0.103679 | 0.106210 |
| CNN-GRU Residual | Rolling 756 | 0.076120 (V20) | 0.079791 (V22-N) | +4.82% | 0.100282 | 0.102662 |
| CNN-LSTM Direct | Expanding | 0.618529 (V20) | 1.030310 (V22-N) | +66.57% | 0.711877 | 1.071730 |
| CNN-LSTM Direct | Recency | 0.798469 (V20) | 1.025076 (V22-N) | +28.38% | 0.893468 | 1.065589 |
| CNN-LSTM Direct | Rolling 756 | 0.849618 (V20) | 0.522778 (V22-N) | -38.47% | 0.939924 | 0.560252 |
| CNN-LSTM Residual | Expanding | 0.076109 (V20) | 0.075054 (V22-N) | -1.39% | 0.099989 | 0.098459 |
| CNN-LSTM Residual | Recency | 0.076140 (V20) | 0.075275 (V22-N) | -1.14% | 0.099782 | 0.098951 |
| CNN-LSTM Residual | Rolling 756 | 0.076066 (V20) | 0.077000 (V22-N) | +1.23% | 0.098508 | 0.099843 |
| GRU Direct | Expanding | 0.408172 (V20) | 0.268112 (V22-N) | -34.31% | 0.463556 | 0.303433 |
| GRU Direct | Recency | 0.381128 (V20) | 0.182461 (V22-N) | -52.13% | 0.425980 | 0.215549 |
| GRU Direct | Rolling 756 | 0.379187 (V20) | 0.369189 (V22-N) | -2.64% | 0.429467 | 0.397711 |
| GRU Residual | Expanding | 0.075205 (V20) | 0.075198 (V22-N) | -0.01% | 0.098704 | 0.098460 |
| GRU Residual | Recency | 0.075708 (V20) | 0.075028 (V22-N) | -0.90% | 0.099996 | 0.098307 |
| GRU Residual | Rolling 756 | 0.075867 (V20) | 0.082713 (V22-N) | +9.02% | 0.100204 | 0.103863 |
| LSTM Direct | Expanding | 0.292570 (V20) | 0.318708 (V22-N) | +8.93% | 0.339832 | 0.370594 |
| LSTM Direct | Recency | 0.959042 (V20) | 1.120221 (V22-N) | +16.81% | 1.027179 | 1.164206 |
| LSTM Direct | Rolling 756 | 1.147674 (V20) | 0.351619 (V22-N) | -69.36% | 1.170163 | 0.379217 |
| LSTM Residual | Expanding | 0.076541 (V20) | 0.075094 (V22-N) | -1.89% | 0.100764 | 0.098551 |
| LSTM Residual | Recency | 0.076629 (V20) | 0.075120 (V22-N) | -1.97% | 0.100968 | 0.098612 |
| LSTM Residual | Rolling 756 | 0.075741 (V20) | 0.075551 (V22-N) | -0.25% | 0.098783 | 0.098938 |
| Ridge Residual | Expanding | 0.082249 (V20) | 0.076510 (V22-N) | -6.98% | 0.107058 | 0.099999 |
| Ridge Residual | Recency | 0.090000 (V20) | 0.080297 (V22-N) | -10.78% | 0.115616 | 0.103246 |
| Ridge Residual | Rolling 756 | 0.085453 (V20) | 0.092756 (V22-N) | +8.55% | 0.108047 | 0.114818 |
| LSTM Direct | Expanding | 0.292571 (V21) | 0.318782 (V22-S) | +8.96% | 0.339834 | 0.370703 |
| SA-LSTM Paper HP | Expanding | 0.541731 (V21) | 0.194868 (V22-S) | -64.03% | 0.585612 | 0.236334 |
| SA-LSTM Reoptimized | Expanding | 0.352332 (V21) | 0.366077 (V22-S) | +3.90% | 0.409124 | 0.411546 |
| SimpleRNN Direct | Expanding | 0.724220 (V21) | 0.975897 (V22-S) | +34.75% | 0.772906 | 1.013056 |



Có 31 cặp được ghép có số liệu tại split 92.5/7.5. Các cặp không có trong CSV không được nội suy.

### 4.4. Split 95/5

| Mô hình | Chiến lược | MAE Legacy14 | MAE WTI | ΔMAE | RMSE cũ | RMSE mới |
|---|---|---|---|---|---|---|
| CNN-GRU Direct | Expanding | 0.419085 (V20) | 0.558294 (V22-N) | +33.22% | 0.446224 | 0.580681 |
| CNN-GRU Direct | Recency | 0.460504 (V20) | 0.408640 (V22-N) | -11.26% | 0.496450 | 0.438568 |
| CNN-GRU Direct | Rolling 756 | 0.526928 (V20) | 0.622279 (V22-N) | +18.10% | 0.551093 | 0.645350 |
| CNN-GRU Residual | Expanding | 0.072810 (V20) | 0.073054 (V22-N) | +0.34% | 0.097681 | 0.096493 |
| CNN-GRU Residual | Recency | 0.075763 (V20) | 0.072810 (V22-N) | -3.90% | 0.097734 | 0.096709 |
| CNN-GRU Residual | Rolling 756 | 0.072839 (V20) | 0.074883 (V22-N) | +2.81% | 0.097354 | 0.096766 |
| CNN-LSTM Direct | Expanding | 0.884860 (V20) | 0.950946 (V22-N) | +7.47% | 0.913996 | 0.964782 |
| CNN-LSTM Direct | Recency | 0.697727 (V20) | 0.923828 (V22-N) | +32.41% | 0.734079 | 0.939854 |
| CNN-LSTM Direct | Rolling 756 | 0.910641 (V20) | 0.959533 (V22-N) | +5.37% | 0.934973 | 0.974685 |
| CNN-LSTM Residual | Expanding | 0.072917 (V20) | 0.073456 (V22-N) | +0.74% | 0.096419 | 0.096595 |
| CNN-LSTM Residual | Recency | 0.074450 (V20) | 0.074737 (V22-N) | +0.38% | 0.097025 | 0.097229 |
| CNN-LSTM Residual | Rolling 756 | 0.073683 (V20) | 0.074180 (V22-N) | +0.67% | 0.096634 | 0.096942 |
| GRU Direct | Expanding | 0.370773 (V20) | 0.285936 (V22-N) | -22.88% | 0.390730 | 0.311670 |
| GRU Direct | Recency | 0.384643 (V20) | 0.249627 (V22-N) | -35.10% | 0.413438 | 0.274234 |
| GRU Direct | Rolling 756 | 0.352270 (V20) | 0.521993 (V22-N) | +48.18% | 0.371524 | 0.545712 |
| GRU Residual | Expanding | 0.072622 (V20) | 0.072882 (V22-N) | +0.36% | 0.096631 | 0.096631 |
| GRU Residual | Recency | 0.072507 (V20) | 0.072619 (V22-N) | +0.16% | 0.096463 | 0.096531 |
| GRU Residual | Rolling 756 | 0.072720 (V20) | 0.078619 (V22-N) | +8.11% | 0.097324 | 0.099497 |
| LSTM Direct | Expanding | 0.662176 (V20) | 0.551285 (V22-N) | -16.75% | 0.684761 | 0.571203 |
| LSTM Direct | Recency | 0.854903 (V20) | 1.166861 (V22-N) | +36.49% | 0.869833 | 1.180903 |
| LSTM Direct | Rolling 756 | 1.130368 (V20) | 0.598023 (V22-N) | -47.09% | 1.142445 | 0.615654 |
| LSTM Residual | Expanding | 0.072619 (V20) | 0.073211 (V22-N) | +0.81% | 0.096367 | 0.096610 |
| LSTM Residual | Recency | 0.073271 (V20) | 0.073393 (V22-N) | +0.17% | 0.096492 | 0.096674 |
| LSTM Residual | Rolling 756 | 0.074773 (V20) | 0.075774 (V22-N) | +1.34% | 0.096963 | 0.097801 |
| Ridge Residual | Expanding | 0.074357 (V20) | 0.074315 (V22-N) | -0.06% | 0.098943 | 0.097327 |
| Ridge Residual | Recency | 0.085730 (V20) | 0.082883 (V22-N) | -3.32% | 0.106248 | 0.104497 |
| Ridge Residual | Rolling 756 | 0.078320 (V20) | 0.095167 (V22-N) | +21.51% | 0.100768 | 0.115561 |
| LSTM Direct | Expanding | 0.662176 (V21) | 0.551285 (V22-S) | -16.75% | 0.684761 | 0.571203 |
| SA-LSTM Paper HP | Expanding | 0.600953 (V21) | 0.200803 (V22-S) | -66.59% | 0.625768 | 0.235800 |
| SA-LSTM Reoptimized | Expanding | 0.541005 (V21) | 0.325427 (V22-S) | -39.85% | 0.569432 | 0.360107 |
| SimpleRNN Direct | Expanding | 0.902960 (V21) | 0.941332 (V22-S) | +4.25% | 0.916224 | 0.952494 |



Có 31 cặp được ghép có số liệu tại split 95/5. Các cặp không có trong CSV không được nội suy.

## 5. Xếp hạng chiến lược theo mô hình và theo split

Để tránh kết luận từ một lần thắng, đếm số lần mỗi chiến lược đạt **MAE nhỏ nhất trong ba chiến lược trên cùng một mô hình + features + split + nguồn**. Chỉ tính nhóm có đủ cả 3 chiến lược (không tính Naive). Đây là **thống kê mô tả**, không phải kiểm định độc lập.

| Nguồn | Số bộ so sánh đủ 3 strategy | Expanding thắng | Rolling thắng | Recency thắng |
|---|---|---|---|---|
| V20 | 108 | 61 | 19 | 28 |
| V21 | 16 | 7 | 1 | 8 |
| V21 bổ sung 80/20 | 16 | 10 | 0 | 6 |
| V22 Non-SA | 36 | 18 | 5 | 13 |
| V22 SA | 4 | 2 | 1 | 1 |



**Lưu ý:** Ba chiến lược thay đổi lượng/trọng số dữ liệu train; chiến lược có số lần thắng nhiều nhất chưa chắc tối ưu cho mọi giai đoạn thị trường. Các split holdout 80–95% có giai đoạn test lồng nhau, không độc lập.

## 6. So sánh theo từng kiến trúc: MAE tốt nhất giữa các bộ input

Ở mỗi split, mỗi nhóm mô hình được gom về cùng tên kiến trúc. Chọn kết quả fitted có MAE nhỏ nhất trong các feature sets và strategies **đã thực nghiệm**, ghi cụ thể feature và version (hậu nghiệm).

| Split | Mô hình | MAE tốt nhất (nguồn) | RMSE | Features | Chiến lược |
|---|---|---|---|---|---|
| 80/20 | CNN-GRU Direct | 0.214287 (V20) | 0.273292 | Legacy14+Cu | Expanding |
| 80/20 | CNN-GRU Residual | 0.068947 (V22-N) | 0.107682 | WTI mới+Cu | Recency |
| 80/20 | CNN-LSTM Direct | 0.384733 (V20) | 0.473176 | Copper-only | Recency |
| 80/20 | CNN-LSTM Residual | 0.068841 (V20) | 0.108019 | Copper-only | Recency |
| 80/20 | GRU Direct | 0.141899 (V20) | 0.191724 | Copper-only | Recency |
| 80/20 | GRU Residual | 0.068780 (V20) | 0.107853 | Paper3+Cu | Expanding |
| 80/20 | LSTM Direct | 0.110100 (V20) | 0.161307 | Copper-only | Expanding |
| 80/20 | LSTM Residual | 0.068840 (V22-N) | 0.107996 | WTI mới+Cu | Expanding |
| 80/20 | Ridge Residual | 0.069408 (V20) | 0.108075 | Copper-only | Expanding |
| 80/20 | SA-LSTM Paper HP | 0.298955 (V22-S) | 0.358768 | WTI mới+Cu | Expanding |
| 80/20 | SA-LSTM Reoptimized | 0.223809 (V22-S) | 0.297888 | WTI mới+Cu | Expanding |
| 80/20 | SimpleRNN Direct | 0.795791 (V21-80) | 0.916974 | Legacy14+Cu | Expanding |
| 90/10 | CNN-GRU Direct | 0.296379 (V20) | 0.359605 | Paper3+Cu | Expanding |
| 90/10 | CNN-GRU Residual | 0.077229 (V22-N) | 0.101285 | WTI mới+Cu | Expanding |
| 90/10 | CNN-LSTM Direct | 0.393469 (V20) | 0.492822 | Legacy14+Cu | Expanding |
| 90/10 | CNN-LSTM Residual | 0.077000 (V20) | 0.100895 | Copper-only | Rolling 756 |
| 90/10 | GRU Direct | 0.205153 (V20) | 0.243488 | Copper-only | Recency |
| 90/10 | GRU Residual | 0.076951 (V20) | 0.100810 | Legacy14+Cu | Expanding |
| 90/10 | LSTM Direct | 0.247409 (V21) | 0.304619 | Legacy14+Cu | Expanding |
| 90/10 | LSTM Residual | 0.076978 (V20) | 0.100948 | Legacy14+Cu | Expanding |
| 90/10 | Ridge Residual | 0.077306 (V20) | 0.101190 | Copper-only | Expanding |
| 90/10 | SA-LSTM Paper HP | 0.184634 (V22-S) | 0.225723 | WTI mới+Cu | Expanding |
| 90/10 | SA-LSTM Reoptimized | 0.326683 (V22-S) | 0.376388 | WTI mới+Cu | Recency |
| 90/10 | SimpleRNN Direct | 0.608387 (V21) | 0.710083 | Legacy14−Cu | Recency |
| 92.5/7.5 | CNN-GRU Direct | 0.351300 (V20) | 0.400391 | Legacy14+Cu | Recency |
| 92.5/7.5 | CNN-GRU Residual | 0.075340 (V20) | 0.098272 | Copper-only | Recency |
| 92.5/7.5 | CNN-LSTM Direct | 0.522778 (V22-N) | 0.560252 | WTI mới+Cu | Rolling 756 |
| 92.5/7.5 | CNN-LSTM Residual | 0.075050 (V20) | 0.098426 | Copper-only | Recency |
| 92.5/7.5 | GRU Direct | 0.182461 (V22-N) | 0.215549 | WTI mới+Cu | Recency |
| 92.5/7.5 | GRU Residual | 0.075028 (V22-N) | 0.098307 | WTI mới+Cu | Recency |
| 92.5/7.5 | LSTM Direct | 0.247864 (V20) | 0.280534 | Copper-only | Rolling 756 |
| 92.5/7.5 | LSTM Residual | 0.075094 (V22-N) | 0.098551 | WTI mới+Cu | Expanding |
| 92.5/7.5 | Ridge Residual | 0.075285 (V20) | 0.098422 | Copper-only | Expanding |
| 92.5/7.5 | SA-LSTM Paper HP | 0.194868 (V22-S) | 0.236334 | WTI mới+Cu | Expanding |
| 92.5/7.5 | SA-LSTM Reoptimized | 0.352332 (V21) | 0.409124 | Legacy14+Cu | Expanding |
| 92.5/7.5 | SimpleRNN Direct | 0.724220 (V21) | 0.772906 | Legacy14+Cu | Expanding |
| 95/5 | CNN-GRU Direct | 0.408640 (V22-N) | 0.438568 | WTI mới+Cu | Recency |
| 95/5 | CNN-GRU Residual | 0.072575 (V20) | 0.096666 | Paper3+Cu | Expanding |
| 95/5 | CNN-LSTM Direct | 0.659669 (V20) | 0.679916 | Copper-only | Recency |
| 95/5 | CNN-LSTM Residual | 0.072625 (V20) | 0.096669 | Copper-only | Expanding |
| 95/5 | GRU Direct | 0.167929 (V20) | 0.194203 | Copper-only | Recency |
| 95/5 | GRU Residual | 0.072357 (V20) | 0.096414 | Paper3+Cu | Recency |
| 95/5 | LSTM Direct | 0.404888 (V20) | 0.428586 | Copper-only | Rolling 756 |
| 95/5 | LSTM Residual | 0.072517 (V20) | 0.096496 | Paper3+Cu | Expanding |
| 95/5 | Ridge Residual | 0.073684 (V20) | 0.096928 | Copper-only | Expanding |
| 95/5 | SA-LSTM Paper HP | 0.200803 (V22-S) | 0.235800 | WTI mới+Cu | Expanding |
| 95/5 | SA-LSTM Reoptimized | 0.325427 (V22-S) | 0.360107 | WTI mới+Cu | Expanding |
| 95/5 | SimpleRNN Direct | 0.902960 (V21) | 0.916224 | Legacy14+Cu | Expanding |



**Cảnh báo:** Bảng này không so sánh công bằng độ ổn định giữa các kiến trúc vì số lượng cấu hình được thử khác nhau và lựa chọn tốt nhất đã quan sát test.

## 7. SA-LSTM: kết quả tìm hyperparameters và so sánh phương pháp

V22 dùng SA search mới cho bộ WTI (16 evaluations, seed SA 2026, search trên validation của reference split 80/20). Cấu hình được manifest xác nhận: **LSTM units 47/127, LR 0.000050478806754686805** (V22 SA). Trong V21, Reoptimized dùng cache trước đó; không nên gọi đó là SA search mới ở V21.

### 7.1. Đối chiếu cùng bộ WTI: LSTM thường, SimpleRNN, Paper HP và SA tối ưu mới

**Split 80/20:**

| Mô hình | Strategy | MAE | RMSE | DA (%) |
|---|---|---|---|---|
| SA-LSTM Reoptimized | Expanding | 0.223809 (V22-S) | 0.297888 | 51.29 |
| SA-LSTM Paper HP | Expanding | 0.298955 (V22-S) | 0.358768 | 48.89 |
| LSTM Direct | Expanding | 0.531802 (V22-S) | 0.662974 | 47.60 |
| SimpleRNN Direct | Expanding | 0.837846 (V22-S) | 0.994480 | 46.49 |



**Split 90/10:**

| Mô hình | Strategy | MAE | RMSE | DA (%) |
|---|---|---|---|---|
| SA-LSTM Paper HP | Expanding | 0.184634 (V22-S) | 0.225723 | 51.66 |
| SA-LSTM Paper HP | Recency | 0.294741 (V22-S) | 0.346242 | 49.08 |
| LSTM Direct | Rolling 756 | 0.321192 (V22-S) | 0.350564 | 46.86 |
| LSTM Direct | Expanding | 0.324940 (V22-S) | 0.381594 | 48.34 |
| SA-LSTM Reoptimized | Recency | 0.326683 (V22-S) | 0.376388 | 46.86 |
| SA-LSTM Reoptimized | Expanding | 0.358921 (V22-S) | 0.415277 | 48.34 |
| SA-LSTM Reoptimized | Rolling 756 | 0.496288 (V22-S) | 0.552808 | 46.86 |
| SA-LSTM Paper HP | Rolling 756 | 0.575763 (V22-S) | 0.634142 | 46.86 |
| SimpleRNN Direct | Expanding | 0.818738 (V22-S) | 0.900169 | 46.49 |
| SimpleRNN Direct | Rolling 756 | 0.859431 (V22-S) | 0.937179 | 46.49 |
| SimpleRNN Direct | Recency | 1.009153 (V22-S) | 1.089029 | 46.49 |
| LSTM Direct | Recency | 1.236605 (V22-S) | 1.296920 | 46.49 |



**Split 92.5/7.5:**

| Mô hình | Strategy | MAE | RMSE | DA (%) |
|---|---|---|---|---|
| SA-LSTM Paper HP | Expanding | 0.194868 (V22-S) | 0.236334 | 53.43 |
| LSTM Direct | Expanding | 0.318782 (V22-S) | 0.370703 | 50.00 |
| SA-LSTM Reoptimized | Expanding | 0.366077 (V22-S) | 0.411546 | 48.04 |
| SimpleRNN Direct | Expanding | 0.975897 (V22-S) | 1.013056 | 46.57 |



**Split 95/5:**

| Mô hình | Strategy | MAE | RMSE | DA (%) |
|---|---|---|---|---|
| SA-LSTM Paper HP | Expanding | 0.200803 (V22-S) | 0.235800 | 47.79 |
| SA-LSTM Reoptimized | Expanding | 0.325427 (V22-S) | 0.360107 | 46.32 |
| LSTM Direct | Expanding | 0.551285 (V22-S) | 0.571203 | 44.85 |
| SimpleRNN Direct | Expanding | 0.941332 (V22-S) | 0.952494 | 44.85 |



### 7.2. Diễn giải

Kết quả SA-LSTM được diễn giải theo MAE và RMSE thực đo. Khác biệt giữa Paper HP và SA-optimized cho biết hành vi của **hai bộ hyperparameters**, không chứng minh thuật toán SA luôn tốt hơn hoặc tệ hơn; giới hạn 16 trials không chứng minh tối ưu toàn cục.

## 8. Walk-forward và độ ổn định

Walk-forward là đánh giá ngoài mẫu theo nhiều giai đoạn, có retrain định kỳ; **không trộn** với bảng holdout. V20 và V22 Non-SA có kết quả walk-forward Residual-GRU, còn **SA-series V21/V22 chưa có kết quả walk-forward được xác minh** trong các ZIP dùng ở đây.

### 8.1. Kết quả được báo cáo từ walk-forward V20

| Seed | MAE Residual-GRU | MAE Naive | DM-HAC p-value hai phía |
|---|---|---|---|
| 41 | 0.045110 (V20) | 0.045127 | 0.7043 |
| 42 | 0.045189 (V20) | 0.045127 | 0.3656 |
| 43 | 0.045447 (V20) | 0.045127 | 0.0066 |
| 44 | 0.045502 (V20) | 0.045127 | 0.0101 |
| 45 | 0.045388 (V20) | 0.045127 | 0.0788 |



Kết quả V20 không xác nhận ưu thế ổn định của GRU trước Naive. Không sử dụng p-value này để mô tả V22.

### 8.2. Phần đối chiếu V22 Walk-forward

Để tránh nhầm kết quả **theo fold** với kết quả **toàn bộ walk-forward**, bảng so sánh sau được tổng hợp trực tiếp từ file predictions, theo seed. Số liệu MAE là trung bình |actual − predicted| trên toàn bộ các dự báo có trong file; cần xem xét trùng timestamp nếu kết quả tích hợp bị lặp.

| Seed | Số phiên khớp | MAE Paper3+Cu | MAE WTI mới+Cu | ΔMAE WTI | Naive MAE |
|---|---|---|---|---|---|
| 41 | 2214 | 0.045110 (V20) | 0.045158 (V22-N) | +0.108% | 0.045127 |
| 42 | 2214 | 0.045189 (V20) | 0.045340 (V22-N) | +0.333% | 0.045127 |
| 43 | 2214 | 0.045447 (V20) | 0.045169 (V22-N) | -0.611% | 0.045127 |
| 44 | 2214 | 0.045502 (V20) | 0.045373 (V22-N) | -0.283% | 0.045127 |
| 45 | 2214 | 0.045388 (V20) | 0.045201 (V22-N) | -0.413% | 0.045127 |



Hai phiên bản walk-forward có cùng các timestamp ở từng seed theo phép ghép trực tiếp. Sai khác MAE phản ánh hiệu năng của **hai bộ input** trong giao thức walk-forward quan sát được, không suy diễn p-value từ V20 sang V22.

## 9. Thảo luận chung

### 9.1. Điều gì thay đổi khi dùng WTI?

Sự thay đổi theo MAE không có dấu nhất quán giữa các mô hình: có mô hình Direct/SA giảm rõ rệt, trong khi một số CNN/Direct hoặc Residual kém hơn. Điều này phù hợp với việc tương quan mức giá đơn biến không định lượng được thông tin dự báo có điều kiện của mạng trên 22 phiên. **Không suy ra quan hệ nhân quả** chỉ bằng bảng trước/sau.

### 9.2. Vì sao Residual thường nằm gần Naive?

Residual học phần chênh lệch rồi cộng lại mức giá phiên trước, nên nếu `Δhat` nhỏ thì giá dự báo gần Naive. Các giá trị MAE gần nhau chỉ là hiện tượng đo được; để kết luận nguyên nhân cần audit phân phối delta dự báo, tương quan với delta thực và thử nghiệm trên nhiều giai đoạn.

### 9.3. Ý nghĩa của xếp hạng

“Tốt nhất” trong báo cáo luôn có nghĩa **MAE thấp nhất trong tập cấu hình đã chạy tại một split/điều kiện cụ thể, loại Naive khỏi cuộc xếp hạng fitted**. Không có tuyên bố “mô hình X thắng ổn định” nếu chưa có kiểm định phù hợp với chuỗi thời gian và một holdout chưa từng dùng để chọn cấu hình.

## 10. Giới hạn, kiểm tra hợp lệ và khả năng tái lập

1. V20/V22 Non-SA có giao thức gần nhau để quan sát thay feature; V21/V22 SA Reoptimized **không phải ablation thuần feature** vì SA được dò lại. Nhóm Non-SA và SA-series khác ngân sách epochs/chi tiết training nên so sánh chéo là mô tả, không nhân quả.

2. Các split 80/20, 90/10, 92.5/7.5 và 95/5 có test periods khác nhau; không so sánh MAE qua splits như thí nghiệm chỉ thay tỷ lệ train.

3. Một số tổ hợp SA-series ở 92.5% và 95% chỉ chạy Expanding; ô khác đánh dấu “—”, không được suy đoán. SA walk-forward chưa thực nghiệm/không có output hợp lệ.

4. EDA Spearman cao trên mức giá có thể chứa cùng xu hướng dài hạn; lagged returns mới gần câu hỏi dự báo biến động tiếp theo hơn. Tuy vậy correlations lag-1 yếu không loại trừ tương tác phi tuyến và nhiều lag.

5. Kết quả trên holdout đã được quan sát để lựa chọn best model/feature: có selection bias. Cần validation hoặc nested walk-forward và test chưa dùng để chọn mô hình để xác nhận.

6. Phải audit as-of market close/nguồn dữ liệu để tránh dùng giá ngoại sinh chưa thực sự khả dụng tại forecast origin; hiện chưa thể khẳng định loại bỏ rủi ro này chỉ từ ngày trong CSV.

7. Số liệu MAE/RMSE gốc không tự chứng minh sự khác biệt có ý nghĩa thống kê; không gắn các DM p-values của V20 sang V22 hoặc SA-series.

## 11. Kết luận tổng hợp

- **80/20:** fitted tốt nhất là **GRU Residual** với **Paper3+Cu**, **Expanding**, MAE **0.068780** và RMSE **0.107853** (**V20**).

- **90/10:** fitted tốt nhất là **GRU Residual** với **Legacy14+Cu**, **Expanding**, MAE **0.076951** và RMSE **0.100810** (**V20**).

- **92.5/7.5:** fitted tốt nhất là **GRU Residual** với **WTI mới+Cu**, **Recency**, MAE **0.075028** và RMSE **0.098307** (**V22-N**).

- **95/5:** fitted tốt nhất là **GRU Residual** với **Paper3+Cu**, **Recency**, MAE **0.072357** và RMSE **0.096414** (**V20**).

Kết luận trọng tâm: Việc thay Gold/Silver bằng WTI tạo kết quả **phụ thuộc mô hình, chiến lược và giai đoạn đánh giá**, không tạo thứ hạng thống nhất trên toàn bộ thực nghiệm. Nhóm Residual cho sai số rất sát baseline; nhóm SA-series cho phép khảo sát ảnh hưởng đặc trưng và lựa chọn hyperparameters, nhưng không thay thế đối chứng Naive. Đây là báo cáo mô tả bằng chứng thực nghiệm, không khẳng định phương pháp nào đúng/sai về nguyên lý.

## Phụ lục A. Truy xuất dữ liệu theo phiên bản

| Nhãn nguồn | Artifact gốc | CSV dùng cho số liệu | Số hàng |
|---|---|---|---|
| V20 | 17-11-Copper-NonSA-Fair-Full-Protocol-V20(2).zip | grid_results.csv | 360 |
| V21 | copper_sa_lstm_h1_ablation_v21_results(1).zip | results_h1_ablation.csv | 100 |
| V21 bổ sung 80/20 | copper_v21_addon_80_wf_plots(2).zip | holdout_80_results.csv | 60 |
| V22 Non-SA | []18-11- Copper_NonSA_V22_WTI_Only_Kaggle.zip | grid_results.csv | 120 |
| V22 SA | []18-11copper_sa_lstm_wti_only_v22_results.zip | results_h1_ablation.csv | 30 |



**Tổng dữ liệu bảng holdout gốc:** 670 hàng (bao gồm nhiều lần lặp Naive theo feature/strategy để kiểm tra consistency, không phải 670 mô hình huấn luyện độc lập).

**Tài liệu tham khảo:** Chen et al. (2023), *Copper price prediction using LSTM recurrent neural network integrated simulated annealing algorithm*, PLOS ONE; Diebold & Mariano (1995), *Comparing Predictive Accuracy*; Newey & West (1987), HAC covariance. Thông tin xuất bản của Chen cần đối chiếu metadata trước khi nộp.
