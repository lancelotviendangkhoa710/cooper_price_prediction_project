# Báo cáo thực nghiệm Phương pháp 5: Hybrid ARIMA–GRU trên 6 bộ biến (FS01–FS06)

- **Người làm:** Phan Ngọc Đức Huy · **Ngày:** 23/11/2026
- **Notebook:** `huy task/copper_hybrid_arima_gru_huy.ipynb` (đã chạy trên Kaggle, giữ nguyên output)
- **Dữ liệu:** `data/raw/dataset_9_commodities_daily.csv`; 2.710 phiên, 13/01/2016 – 13/11/2026
- **Bài toán:** dự báo giá đồng ($/lb) 1 phiên tới từ cửa sổ 22 phiên, theo khung chung của nhóm

Bản này thay cho bản 20/11 (chỉ có FS01): chạy đủ 6 bộ biến, dùng kiến trúc GRU và cách cắt validation chung
của nhóm, thêm DA và 2 chiến lược phụ ở 90/10.

## 1. Ý tưởng
Mô hình lai theo Zhang (2003): ARIMA đơn biến trên giá đồng bắt phần tuyến tính, GRU học phần dư ARIMA để lại.
**Dự báo cuối = dự báo ARIMA + phần dư GRU dự báo.**

- ARIMA không phụ thuộc bộ biến → tính một lần mỗi split, dùng chung cho 6 FS.
- Bộ biến chỉ đổi **đầu vào của GRU**: với FS có Copper, cột Copper được thay bằng phần dư ARIMA; các biến khác ở dạng
  mức giá chuẩn hoá MinMax. FS03/FS05 không có Copper: GRU chỉ nhận biến ngoại sinh, nhưng dự báo vẫn dựa trên ARIMA
  của Copper → **không phải "hoàn toàn không dùng lịch sử Copper"**.
- Đối chứng: **Naive** (giá hôm qua), **ARIMA**, **GRU Direct** (dự báo thẳng giá, đầu vào là mức giá các biến của FS).

## 2. Thiết lập
- 4 split 80/20, 90/10, 92.5/7.5, 95/5; `split = int(n·r)`, validation = 10% cuối phần trước test
  (`val_start = int(split·0.9)`); scaler fit trên phần train trước validation. Naive khớp bảng nhóm đến 6 chữ số.
- ARIMA: `auto_arima` chọn bậc chỉ trên phần trước test, tham số giữ nguyên; dự báo phiên t chỉ dùng dữ liệu tới t−1
  (notebook có ô tự kiểm tra rò rỉ). Bậc được chọn: **(0,1,0)** ở 80/20 và 90/10 (= Naive), **(2,1,2) trend=n** ở
  92.5/7.5 và 95/5.
- GRU: GRU 64 (return_sequences) → Dropout 0.2 → GRU 32 → Dropout 0.2 → Dense 16 ReLU → Dense 1; Adam 1e-3, MSE,
  tối đa 65 epoch, EarlyStopping patience 9 (restore best), batch 64, không xáo trộn. Cấu hình cố định từ trước,
  không chọn theo test.
- Chiến lược train (chỉ áp cho GRU): **Expanding** ở cả 4 split; thêm **Rolling 756** và **Recency** (half-life 252)
  ở 90/10.
- 5 seed (41–45) → 360 lần train GRU. Metric: MAE, RMSE, MAPE, R², DA (%) (dấu Δ dự báo = dấu Δ thật). DA kiểm định
  nhị thức so với 50%. Diebold–Mariano so với Naive, loss L1 và L2, HAC Newey–West lag 5 + hiệu chỉnh HLN.

## 3. Naive và ARIMA

| Split | Naive MAE | ARIMA | ARIMA MAE | ΔMAE | DM L1 p |
|---|---:|---|---:|---:|---:|
| 80/20 | 0.069001 | (0,1,0) | 0.069001 | 0.00% | — |
| 90/10 | 0.077077 | (0,1,0) | 0.077077 | 0.00% | — |
| 92.5/7.5 | 0.075116 | (2,1,2) | 0.074556 | −0.75% | 0.25 |
| 95/5 | 0.072799 | (2,1,2) | 0.072576 | −0.31% | 0.76 |

## 4. Hybrid ARIMA–GRU, Expanding (trung bình 5 seed)

**ΔMAE so với Naive (%)** — âm là tốt hơn Naive; trong ngoặc: số seed **kém Naive có ý nghĩa** (DM L1, p < 0.05)

| FS | 80/20 | 90/10 | 92.5/7.5 | 95/5 |
|---|---:|---:|---:|---:|
| FS01_CU | +5.43 (5) | +0.48 (2) | **−0.48** (0) | +0.06 (0) |
| FS02_PAPER_CU | +13.57 (4) | +5.21 (4) | +1.82 (1) | +1.74 (0) |
| FS03_PAPER_EXO | +20.49 (5) | +8.13 (3) | +15.72 (2) | +13.88 (2) |
| FS04_METALS_CU | +9.21 (5) | +6.20 (4) | +1.70 (0) | +3.74 (3) |
| FS05_METALS_EXO | +5.95 (2) | +4.72 (2) | +1.87 (1) | +0.48 (1) |
| FS06_WTI_CU | +4.08 (4) | +1.77 (2) | +4.94 (3) | +3.16 (2) |
| **TB 6 FS** | +9.79 | +4.42 | +4.26 | +3.84 |

- Không seed nào **tốt hơn** Naive có ý nghĩa (L1 lẫn L2) ở Expanding; 57/120 lần chạy **kém hơn** có ý nghĩa (L1).
- **DA (%)** dao động 44.7–54.0%. Chỉ 13/180 lần chạy Hybrid có DA khác 50% có ý nghĩa, trong đó 12 lần là **thấp
  hơn** 50% (FS01 ở 80/20: DA 44.7%, cả 5 seed).
- MAE Hybrid theo FS (90/10): FS01 0.07745 · FS02 0.08110 · FS03 0.08334 · FS04 0.08185 · FS05 0.08071 ·
  FS06 0.07844 → trung bình 6 FS 0.08048. Độ lệch chuẩn giữa seed: FS01 0.0002–0.0008, FS03 tới 0.018.

## 5. Ba chiến lược ở 90/10 (ΔMAE so với Naive, %)

| FS | Expanding | Recency | Rolling 756 |
|---|---:|---:|---:|
| FS01_CU | +0.48 | +0.32 | +1.94 |
| FS02_PAPER_CU | +5.21 | +9.78 | +17.36 |
| FS03_PAPER_EXO | +8.13 | +16.96 | +101.22 |
| FS04_METALS_CU | +6.20 | +5.25 | +16.95 |
| FS05_METALS_EXO | +4.72 | +2.34 | +46.95 |
| FS06_WTI_CU | +1.77 | +2.42 | +10.28 |
| **TB 6 FS** | +4.42 | +6.18 | +32.45 |

Rolling 756 (chỉ train trên khoảng 3 năm gần nhất) kém rõ, nhất là ở FS03/FS05. Cả 3 chiến lược chỉ có đúng 1/90 lần
chạy Hybrid tốt hơn Naive có ý nghĩa (Recency, FS01, seed 42, MAE 0.07652).

## 6. GRU Direct
Kém Naive ở cả 180/180 lần chạy (p < 0.05, cả L1 và L2): ΔMAE từ +151% (FS01, 80/20) tới +814% (FS03, 80/20);
MAE FS01 ở 90/10 = 0.338. Đường dự báo nằm thấp hơn giá thật vì giá test vượt mọi mức giá trong train, mạng chuẩn hoá
MinMax theo train không ngoại suy được (xem `actual_vs_pred_FS01_*.png`). Cùng mức với các mô hình Direct khác
trong bảng nhóm (0.20–1.76 ở FS01, 90/10).

## 7. So với bảng tổng hợp nhóm (90/10 Expanding, MAE)

| FS | Naive | Hybrid ARIMA–GRU | Tốt nhất trong các mô hình Residual khác |
|---|---:|---:|---:|
| FS01 | 0.07708 | 0.07745 | 0.07681 |
| FS02 | 0.07708 | 0.08110 | 0.07701 |
| FS03 | 0.07708 | 0.08334 | 0.07693 |
| FS04 | 0.07708 | 0.08185 | 0.07687 |
| FS05 | 0.07708 | 0.08071 | 0.07720 |
| FS06 | 0.07708 | 0.07844 | 0.07704 |

Dòng dán vào bảng nhóm: `result/group_format_90_10.csv` (cùng cột với `group_ablation_6_feature_sets_all_members.csv`).

## 8. Nhận xét
1. **Hybrid ngang hoặc kém Naive, không thắng ở đâu.** Tốt nhất là FS01 (sát Naive ở 3/4 split, −0.48% ở 92.5/7.5
   nhưng không có ý nghĩa). Ở 80/20 và với các bộ có biến ngoại sinh, Hybrid kém Naive có ý nghĩa ở nhiều seed.
2. **Thêm biến ngoại sinh làm kết quả xấu đi**: FS02–FS06 đều kém FS01; FS03 (không có Copper trong đầu vào GRU) kém
   nhất, và độ lệch giữa các seed cũng lớn hơn hẳn: biến ngoại sinh không giúp GRU dự báo phần dư ARIMA.
3. **Phần cải thiện nhỏ ở 92.5/7.5 và 95/5 đến từ ARIMA**, không phải GRU: ARIMA đã −0.75% / −0.31%, GRU học phần dư
   cộng vào làm nhích lên hoặc kém đi.
4. Ở 80/20 và 90/10, ARIMA(0,1,0) chính là Naive nên Hybrid lúc này = "Naive + GRU học chênh lệch giá"; kết quả sát
   các mô hình Residual khác của nhóm, đúng như kỳ vọng.
5. Giá đồng theo ngày hành xử gần như bước ngẫu nhiên; khi DM không có ý nghĩa, kết luận đúng là "ngang Naive".

## 9. Giới hạn
- 4 tập test lồng nhau (test 95/5 nằm trong test 90/10...) nên không phải 4 bằng chứng độc lập.
- Nhiều phép DM (6 FS × 4 split × 5 seed × 2 loss) chưa hiệu chỉnh so sánh bội; số "thua có ý nghĩa" nên đọc như xu
  hướng.
- Siêu tham số GRU cố định theo khung chung, không dò riêng cho Hybrid.
- `pmdarima` khác phiên bản giữa máy và Kaggle có thể chọn trend khác ở 92.5/7.5 (máy: trend=t; Kaggle: trend=n).
  Số trong báo cáo là bản Kaggle (chạy CPU, khoảng 3 giờ).

## 10. File trong `result/`
- `results.csv` — trung bình qua 5 seed theo Split × Scheme × FS × mô hình: MAE (± std), RMSE, MAPE, R², DA,
  ΔMAE vs Naive, DM L1/L2 (thống kê, p, số seed thắng/thua), số seed có DA khác 50%
- `results_by_seed.csv` — từng lần chạy (368 dòng)
- `predictions_<FS>_<split>_<scheme>.csv` — 36 file dự báo từng phiên: Actual, Naive, ARIMA, GRU Direct và Hybrid
  của từng seed (dùng cho DM chéo)
- `group_format_90_10.csv` — 2 dòng (Hybrid, GRU Direct) theo khuôn bảng nhóm
- `schemes_90_10.csv` — ΔMAE của 3 chiến lược ở 90/10
- `hybrid_delta_mae_by_fs.png`, `actual_vs_pred_FS01_<split>.png`

Chạy lại: trên Kaggle thêm dataset có file `dataset_9_commodities_daily.csv` (hoặc `commodities_9_daily.csv`), bật GPU
rồi Run All; trên máy thì mở notebook trong thư mục `huy task/` (cần TensorFlow, `pmdarima`).

## Tài liệu tham khảo
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159–175.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
