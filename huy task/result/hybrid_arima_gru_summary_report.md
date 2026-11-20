# Báo cáo thực nghiệm Phương pháp 5: Hybrid ARIMA–GRU

- **Người làm:** Phan Ngọc Đức Huy · **Ngày:** 20/11/2026
- **Notebook:** `huy task/copper_hybrid_arima_gru_huy.ipynb` (đã chạy trên Kaggle, giữ nguyên output)
- **Dữ liệu:** `data/raw/dataset_9_commodities_daily.csv` (bộ của Khoa), chỉ dùng giá đồng ($/lb); 2.710 phiên, 13/01/2016 – 13/11/2026
- **Bài toán:** dự báo giá đồng 1 phiên tới từ 22 phiên trước (giống Khoa)

## 1. Ý tưởng
Mô hình lai theo Zhang (2003): ARIMA bắt phần tuyến tính của chuỗi giá, GRU học phần dư (sai số) mà ARIMA để lại.
**Dự báo cuối = dự báo ARIMA + phần dư GRU dự báo.**

So sánh 4 mô hình:
- **Naive:** giá hôm nay = giá hôm qua (mốc phải vượt)
- **ARIMA:** (p,d,q) chọn bằng `auto_arima` chỉ trên train
- **GRU:** dự báo thẳng giá (để thấy hybrid có thêm gì)
- **Hybrid ARIMA–GRU**

## 2. Thiết lập
- Chia theo thời gian 80/20, 90/10, 92.5/7.5, 95/5 (cùng cách chia với Khoa; Naive ra trùng từng số với báo cáo của Khoa).
- ADF: giá không dừng (p = 0.81), sai phân bậc 1 dừng (p < 0.001) → d = 1.
- ARIMA: tham số ước lượng trên train rồi giữ nguyên; dự báo phiên t chỉ dùng dữ liệu tới t−1 (notebook có ô tự kiểm tra rò rỉ).
- ARIMA được chọn: (0,1,0) ở 80/20 và 90/10 — tức bước ngẫu nhiên, trùng Naive; (2,1,2) ở 92.5/7.5 và 95/5.
- GRU: 1 lớp 64 đơn vị + Dense, Adam lr 0.001, batch 64, tối đa 100 epoch, early stopping patience 10 trên 10% cuối của train (không dùng test). Chuẩn hoá bằng thống kê của train.
- 5 seed (41–45, giống Khoa). Kiểm định Diebold–Mariano so với Naive, loss L1 và L2, phương sai HAC Newey–West.

## 3. Kết quả trên tập test (trung bình 5 seed)

| Split | Mô hình | MAE ($/lb) | ΔMAE vs Naive | RMSE | MAPE (%) | R² |
|---|---|---:|---:|---:|---:|---:|
| 80/20 | Naive | 0.069001 | — | 0.108000 | 1.289 | 0.9833 |
| | ARIMA(0,1,0) | 0.069001 | 0.00% | 0.108000 | 1.289 | 0.9833 |
| | GRU | 0.225625 | +226.99% | 0.307197 | 3.788 | 0.8641 |
| | Hybrid | 0.069225 | +0.32% | 0.108274 | 1.294 | 0.9832 |
| 90/10 | Naive | 0.077077 | — | 0.101033 | 1.278 | 0.9579 |
| | ARIMA(0,1,0) | 0.077077 | 0.00% | 0.101033 | 1.278 | 0.9579 |
| | GRU | 0.299463 | +288.53% | 0.337178 | 4.777 | 0.5289 |
| | Hybrid | **0.076758** | −0.41% | 0.100747 | 1.272 | 0.9581 |
| 92.5/7.5 | Naive | 0.075116 | — | 0.098399 | 1.208 | 0.9238 |
| | ARIMA(2,1,2) | 0.074556 | −0.75% | 0.098132 | 1.199 | 0.9242 |
| | GRU | 0.339846 | +352.43% | 0.366764 | 5.331 | −0.0969 |
| | Hybrid | **0.074411** | −0.94% | 0.097906 | 1.197 | 0.9246 |
| 95/5 | Naive | 0.072799 | — | 0.096612 | 1.130 | 0.7359 |
| | ARIMA(2,1,2) | 0.072576 | −0.31% | 0.097057 | 1.126 | 0.7334 |
| | GRU | 0.347387 | +377.19% | 0.361895 | 5.333 | −2.8257 |
| | Hybrid | **0.072478** | −0.44% | 0.096616 | 1.125 | 0.7358 |

Độ lệch chuẩn MAE của Hybrid giữa 5 seed chỉ 0.00006–0.00029 → kết quả ổn định. Chi tiết từng seed: `result/results_by_seed.csv`.

## 4. Kiểm định Diebold–Mariano so với Naive (loss L1)

| Split | ARIMA p | Hybrid: số seed p < 0.05 | Hybrid: p trung bình |
|---|---|---|---|
| 80/20 | — (trùng Naive) | 0/5 | 0.39 |
| 90/10 | — (trùng Naive) | 2/5 (seed 42: 0.047; seed 45: 0.004) | 0.08 |
| 92.5/7.5 | 0.25 | 0/5 | 0.12 |
| 95/5 | 0.75 | 0/5 | 0.54 |

Với loss L2 chỉ 1/20 lần chạy (90/10, seed 45) có p < 0.05. GRU dự báo thẳng giá kém hơn Naive có ý nghĩa ở cả 20/20 lần chạy (p < 0.001).

## 5. Nhận xét
1. **Hybrid chưa thắng Naive chắc chắn.** MAE thấp hơn Naive ở 3/4 split nhưng chỉ 0.4–0.9%, và chỉ 2/20 lần chạy có ý nghĩa thống kê ở mức 5%. Kết luận giống GRU Residual của Khoa (V20/V22): mô hình học chênh lệch giá cho sai số rất sát Naive.
2. **Phần cải thiện chủ yếu đến từ ARIMA**, không phải GRU: ở 92.5/7.5, ARIMA đã −0.75%, GRU học phần dư chỉ thêm khoảng −0.2%. Ở 80/20 và 90/10, ARIMA(0,1,0) chính là Naive nên Hybrid thực chất là "Naive + GRU học chênh lệch giá".
3. **GRU dự báo thẳng giá thất bại:** đường dự báo nằm thấp hơn giá thật (xem hình) vì giá trên tập test cao hơn mọi mức giá trong train; mạng chuẩn hoá theo train không ngoại suy được. Đây cũng là lý do các mô hình Direct của Khoa kém.
4. **Ý nghĩa:** giá đồng theo ngày hành xử gần như bước ngẫu nhiên; mô hình phức tạp không vượt được "giá mai = giá hôm nay". Vì vậy khi so với các bài báo báo mô hình lai thắng, cần xem họ có so với Naive và kiểm định thống kê hay không.

## 6. Giới hạn
- Chỉ dùng chuỗi giá đồng, chưa thử thêm biến ngoại sinh cho GRU.
- Holdout cố định (không walk-forward); các split có giai đoạn test lồng nhau, không độc lập.
- Siêu tham số GRU cố định, chưa dò; chọn bằng validation, không dùng test.
- `pmdarima` khác phiên bản giữa máy và Kaggle có thể chọn trend khác (thấy ở split 92.5/7.5); số trong báo cáo là bản chạy trên Kaggle.

## 7. File
- `result/results.csv` — trung bình qua 5 seed, kèm ΔMAE, độ lệch chuẩn, DM
- `result/results_by_seed.csv` — từng mô hình × seed × split
- `result/actual_vs_pred_<split>.png` — thực tế vs dự báo (GRU/Hybrid trung bình 5 seed)

Chạy lại: trên Kaggle thêm dataset có file `commodities_9_daily.csv` (chính là `data/raw/dataset_9_commodities_daily.csv`) rồi Run All; chạy trên máy thì mở notebook trong thư mục `huy task/` (cần TensorFlow).

## Tài liệu tham khảo
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159–175.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
