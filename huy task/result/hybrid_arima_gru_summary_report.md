# Báo cáo thực nghiệm Phương pháp 5: Hybrid ARIMA–GRU trên 6 bộ biến (FS01–FS06)

- **Người làm:** Phan Ngọc Đức Huy · **Ngày:** 27/11/2026
- **Notebook:** `huy task/copper_hybrid_arima_gru_huy.ipynb` (chạy trên Kaggle GPU)
- **Dữ liệu:** `data/raw/dataset_9_commodities_daily.csv`; 2.710 phiên, 13/01/2016 – 13/11/2026
- **Bài toán:** dự báo giá đồng ($/lb) 1 phiên tới từ cửa sổ 22 phiên, theo khung chung của nhóm

Bản này thay cho bản 23/11. Bản 23/11 chỉ có **holdout**: mỗi split train một lần rồi dự báo cả tập test. Bản này
thêm **walk-forward** theo đúng giao thức walk-forward của Module1 (notebook 03): mỗi gập 63 phiên chọn lại bậc,
fit lại ARIMA và train lại GRU. Phần holdout được chạy lại cùng lần (GPU), số lệch rất nhỏ so với bản 23/11.

## 1. Ý tưởng
Mô hình lai theo Zhang (2003): ARIMA đơn biến trên giá đồng bắt phần tuyến tính, GRU học phần dư ARIMA để lại.
**Dự báo cuối = dự báo ARIMA + phần dư GRU dự báo.**

- ARIMA không phụ thuộc bộ biến → dùng chung cho 6 FS.
- Bộ biến chỉ đổi **đầu vào của GRU**: với FS có Copper, cột Copper được thay bằng phần dư ARIMA; các biến khác ở dạng
  mức giá chuẩn hoá MinMax. FS03/FS05 không có Copper: GRU chỉ nhận biến ngoại sinh, nhưng dự báo vẫn dựa trên ARIMA
  của Copper → **không phải "hoàn toàn không dùng lịch sử Copper"**.
- Đối chứng: **Naive** (giá hôm qua), **ARIMA**, **GRU Direct** (dự báo thẳng giá, đầu vào là mức giá các biến của FS).

## 2. Thiết lập
- GRU: GRU 64 (return_sequences) → Dropout 0.2 → GRU 32 → Dropout 0.2 → Dense 16 ReLU → Dense 1; Adam 1e-3, MSE,
  EarlyStopping (restore best), batch 64, không xáo trộn. Cấu hình cố định từ trước, không chọn theo test.
- ARIMA: `auto_arima` chọn bậc chỉ trên dữ liệu trước điểm bắt đầu dự báo; dự báo phiên t chỉ dùng dữ liệu tới t−1
  (notebook có ô tự kiểm tra rò rỉ).
- 5 seed (41–45). Metric: MAE, RMSE, MAPE, R², DA (%) (dấu Δ dự báo = dấu Δ thật), DA kiểm định nhị thức so với 50%.
  Diebold–Mariano so với Naive, loss L1 và L2, HAC Newey–West lag 5 + hiệu chỉnh HLN.

**Hai giao thức đánh giá**

| | Holdout | Walk-forward (`walk_forward_63`) |
|---|---|---|
| Điểm bắt đầu test | `split = int(n·r)`, r ∈ {80, 90, 92.5, 95}% | mốc 80% (phiên 2168, 07/10/2024) |
| Train lại | **một lần** mỗi split | **mỗi gập 63 phiên**: 9 gập (8×63 + 1×38) = 542 phiên |
| ARIMA | chọn bậc + fit một lần, giữ tham số | chọn lại bậc + fit lại ở đầu mỗi gập |
| GRU | train một lần; validation = 10% cuối trước test | train lại từ đầu mỗi gập; validation = 10% cuối trước gập |
| Scaler | fit trên phần train trước validation | fit lại mỗi gập trên phần trước validation |
| Epoch / patience | 65 / 9 | 45 / 7 |
| Chiến lược dữ liệu | Expanding ở 4 split; thêm Rolling 756, Recency (half-life 252) ở 90/10 | Expanding |
| Số lần train GRU | 360 | 540 |

`expanding`, `rolling_756`, `recency_weighted` ở holdout chỉ là phạm vi/trọng số dữ liệu của **lần train duy nhất**,
không phải cửa sổ trượt trong lúc test. Walk-forward được tính trên cả 542 phiên, và trên **đúng đoạn test của từng
split** (đoạn test 90/10, 92.5/7.5, 95/5 là phần đuôi của 542 phiên) để đặt cạnh holdout.

## 3. Naive và ARIMA

| Split | Naive MAE | ARIMA holdout (bậc) | ΔMAE | DM L1 p | ARIMA walk-forward | ΔMAE | DM L1 p |
|---|---:|---|---:|---:|---:|---:|---:|
| 80/20 | 0.069001 | 0.069001 (0,1,0) | 0.00% | — | 0.068936 | −0.10% | 0.71 |
| 90/10 | 0.077077 | 0.077077 (0,1,0) | 0.00% | — | 0.076945 | −0.17% | 0.71 |
| 92.5/7.5 | 0.075116 | 0.074556 (2,1,2) | −0.75% | 0.25 | 0.075049 | −0.09% | 0.88 |
| 95/5 | 0.072799 | 0.072576 (2,1,2) | −0.31% | 0.76 | 0.072787 | −0.02% | 0.98 |

Walk-forward: `auto_arima` chọn (0,1,0) = Naive ở gập 1–5; từ gập 6 (30/12/2025) chọn bậc khác. ARIMA không khác Naive
có ý nghĩa ở giao thức nào.

## 4. Hybrid ARIMA–GRU (trung bình 5 seed)

**ΔMAE so với Naive (%)**: âm là tốt hơn Naive. Trong ngoặc: số seed **kém Naive có ý nghĩa** (DM L1, p < 0.05).

### 4.1. Walk-forward

| FS | 80/20 (542 phiên) | 90/10 | 92.5/7.5 | 95/5 |
|---|---:|---:|---:|---:|
| FS01_CU | +3.01 (4) | +1.05 (1) | +1.46 (2) | +1.94 (1) |
| FS02_PAPER_CU | +7.06 (5) | +8.46 (4) | +10.35 (4) | +2.82 (1) |
| FS03_PAPER_EXO | +12.83 (5) | +19.59 (5) | +25.14 (5) | +16.30 (3) |
| FS04_METALS_CU | +5.41 (5) | +5.74 (5) | +7.57 (5) | +7.49 (4) |
| FS05_METALS_EXO | +3.74 (4) | +5.09 (4) | +6.83 (4) | +3.10 (2) |
| FS06_WTI_CU | +3.82 (5) | +3.65 (5) | +4.63 (5) | +5.14 (5) |
| **TB 6 FS** | +5.98 | +7.26 | +9.33 | +6.13 |

- Không lần chạy nào **tốt hơn** Naive có ý nghĩa (L1 lẫn L2); 93/120 lần **kém hơn** có ý nghĩa (L1 và L2).
- Theo từng gập (MAE trung bình 5 seed, `wf_fold_mae.csv`): Hybrid thấp hơn Naive ở 4/54 cặp FS × gập.
  FS01 trên 542 phiên: Hybrid 0.07123, Naive 0.06899.
- DA 42.6–57.4%; chỉ 1/120 lần chạy khác 50% có ý nghĩa.
- MAE ở 90/10: FS01 0.07789 · FS02 0.08360 · FS03 0.09218 · FS04 0.08150 · FS05 0.08100 · FS06 0.07989
  → trung bình 6 FS 0.08268. Độ lệch chuẩn giữa seed: FS01 0.0005–0.0008, FS03 tới 0.0082.

### 4.2. Holdout Expanding

| FS | 80/20 | 90/10 | 92.5/7.5 | 95/5 |
|---|---:|---:|---:|---:|
| FS01_CU | +5.43 (5) | +0.48 (2) | **−0.48** (0) | +0.05 (0) |
| FS02_PAPER_CU | +13.57 (4) | +5.21 (4) | +1.65 (1) | +1.74 (0) |
| FS03_PAPER_EXO | +20.49 (5) | +8.13 (3) | +15.55 (2) | +13.02 (1) |
| FS04_METALS_CU | +9.21 (5) | +6.97 (4) | +1.50 (0) | +3.34 (2) |
| FS05_METALS_EXO | +5.95 (2) | +4.72 (2) | +1.87 (1) | +0.48 (1) |
| FS06_WTI_CU | +4.08 (4) | +1.77 (2) | +4.96 (3) | +3.29 (2) |
| **TB 6 FS** | +9.79 | +4.55 | +4.17 | +3.65 |

- Không lần chạy nào tốt hơn Naive có ý nghĩa; 55/120 (L1), 66/120 (L2) lần kém hơn có ý nghĩa.
- DA 41.9–58.1%; 6/120 lần chạy khác 50% có ý nghĩa, cả 6 đều **thấp hơn** 50% (FS01 80/20: 44.7%, cả 5 seed).
- MAE ở 90/10: FS01 0.07745 · FS02 0.08110 · FS03 0.08334 · FS04 0.08245 · FS05 0.08071 · FS06 0.07844
  → trung bình 6 FS 0.08058.

### 4.3. Holdout và walk-forward
Walk-forward đỡ hơn holdout ở 80/20 (TB 6 FS +5.98% so với +9.79%), nhưng kém hơn ở 90/10, 92.5/7.5 và 95/5.
Ở 80/20, holdout dùng một mô hình cho cả 542 phiên còn walk-forward cập nhật 9 lần. Ở các split sau, holdout train trên
nhiều dữ liệu hơn mốc 80% của walk-forward và (ở 92.5/95) dùng ARIMA(2,1,2) có lợi hơn. Cả hai giao thức đều cho cùng
kết luận: Hybrid không thắng Naive.

## 5. Bốn chiến lược ở 90/10 (Hybrid, ΔMAE so với Naive, %)

| FS | Holdout Expanding | Holdout Recency | Holdout Rolling 756 | Walk-forward |
|---|---:|---:|---:|---:|
| FS01_CU | +0.48 | +0.32 | +1.94 | +1.05 |
| FS02_PAPER_CU | +5.21 | +9.78 | +17.36 | +8.46 |
| FS03_PAPER_EXO | +8.13 | +16.96 | +101.22 | +19.59 |
| FS04_METALS_CU | +6.97 | +5.26 | +16.95 | +5.74 |
| FS05_METALS_EXO | +4.72 | +2.34 | +46.95 | +5.09 |
| FS06_WTI_CU | +1.77 | +2.42 | +10.28 | +3.65 |
| **TB 6 FS** | +4.55 | +6.18 | +32.45 | +7.26 |

Rolling 756 (chỉ train trên khoảng 3 năm gần nhất) kém rõ, nhất là ở FS03/FS05. Cả 4 chiến lược chỉ có đúng
1/120 lần chạy Hybrid tốt hơn Naive có ý nghĩa (holdout Recency, FS01, seed 42, MAE 0.07652).

## 6. GRU Direct
Kém Naive có ý nghĩa ở mọi lần chạy (L1 và L2), ở cả hai giao thức:
- Holdout Expanding: ΔMAE từ +151% (FS01, 80/20) tới +814% (FS03, 80/20); MAE FS01 ở 90/10 = 0.338.
- Walk-forward: từ +134% (FS01, 80/20) tới +692% (FS03, 92.5/7.5); MAE FS01 ở 90/10 = 0.202.

Đường dự báo nằm thấp hơn giá thật vì giá test vượt mọi mức giá trong train, mạng chuẩn hoá MinMax theo train không
ngoại suy được. Train lại mỗi gập chỉ giảm bớt chứ không hết. Cùng mức với các mô hình Direct khác trong bảng nhóm.

## 7. So với bảng tổng hợp nhóm (90/10, MAE)

| FS | Naive | Hybrid holdout | Hybrid walk-forward | Tốt nhất trong các mô hình Residual khác |
|---|---:|---:|---:|---:|
| FS01 | 0.07708 | 0.07745 | 0.07789 | 0.07681 |
| FS02 | 0.07708 | 0.08110 | 0.08360 | 0.07701 |
| FS03 | 0.07708 | 0.08334 | 0.09218 | 0.07693 |
| FS04 | 0.07708 | 0.08245 | 0.08150 | 0.07687 |
| FS05 | 0.07708 | 0.08071 | 0.08100 | 0.07720 |
| FS06 | 0.07708 | 0.07844 | 0.07989 | 0.07704 |

Dòng dán vào bảng nhóm: `result/group_format_90_10.csv`, gồm 4 dòng: Hybrid và GRU Direct, mỗi mô hình một dòng
Holdout Expanding và một dòng Walk-forward 63 (cùng cột với `group_ablation_6_feature_sets_all_members.csv`).

## 8. Nhận xét
1. **Hybrid ngang hoặc kém Naive, không thắng ở giao thức nào.** Ở walk-forward, Hybrid kém Naive ở mọi FS × split và
   kém có ý nghĩa ở 93/120 lần chạy. FS01 sát Naive nhất (+1.0% tới +3.0%).
2. **Thêm biến ngoại sinh làm kết quả xấu đi**: FS02–FS06 đều kém FS01; FS03 (không có Copper trong đầu vào GRU) kém
   nhất và dao động giữa seed lớn nhất. Biến ngoại sinh không giúp GRU dự báo phần dư ARIMA.
3. **Phần cải thiện nhỏ ở holdout 92.5/7.5 và 95/5 đến từ ARIMA**, không phải GRU. Khi ARIMA được fit lại mỗi gập thì
   lợi thế này gần như mất (−0.09% / −0.02%).
4. Khi ARIMA chọn (0,1,0) (holdout 80/20, 90/10; walk-forward gập 1–5), Hybrid = "Naive + GRU học chênh lệch giá";
   kết quả sát các mô hình Residual khác của nhóm, đúng như kỳ vọng.
5. Giá đồng theo ngày hành xử gần như bước ngẫu nhiên; khi DM không có ý nghĩa, kết luận đúng là "ngang Naive".

## 9. Giới hạn
- 4 tập test lồng nhau (test 95/5 nằm trong test 90/10...) nên không phải 4 bằng chứng độc lập. Với walk-forward, 4
  cột của bảng 4.1 là 4 đoạn đuôi của **cùng một** chuỗi dự báo 542 phiên.
- Nhiều phép DM chưa hiệu chỉnh so sánh bội; số "thua có ý nghĩa" nên đọc như xu hướng.
- Siêu tham số GRU cố định theo khung chung, không dò riêng cho Hybrid (cũng không dò lại ở mỗi gập).
- Walk-forward chỉ chạy chiến lược Expanding (giống giao thức Module1), không có Rolling/Recency.
- Chạy GPU nên số lệch nhỏ so với bản CPU 23/11 (ví dụ holdout 90/10 TB 6 FS +4.55% so với +4.42%).
  `pmdarima` khác phiên bản có thể chọn trend khác ở 92.5/7.5.

## 10. File trong `result/`
- `results.csv`: trung bình qua 5 seed theo Split × Scheme × FS × mô hình: MAE (± std), RMSE, MAPE, R², DA,
  ΔMAE vs Naive, DM L1/L2 (thống kê, p, số seed thắng/thua), số seed có DA khác 50%.
  `Scheme`: `expanding`/`rolling_756`/`recency_weighted` = holdout, `walk_forward_63` = walk-forward.
- `results_by_seed.csv`: từng lần chạy (612 dòng).
- `predictions_<FS>_<split>_<scheme>.csv`: 36 file dự báo holdout từng phiên (Actual, Naive, ARIMA, GRU Direct và
  Hybrid của từng seed).
- `wf_predictions_<FS>.csv`: 6 file dự báo walk-forward 542 phiên, có cột `Fold` (1–9) và ARIMA fit lại mỗi gập.
- `wf_fold_mae.csv`: MAE từng gập (Naive, ARIMA, Hybrid, GRU Direct; trung bình 5 seed).
- `group_format_90_10.csv`: 4 dòng theo khuôn bảng nhóm.
- `schemes_90_10.csv`: ΔMAE của 4 chiến lược ở 90/10.
- `hybrid_delta_mae_by_fs.png`, `actual_vs_pred_FS01_<split>.png`, `wf_actual_vs_pred_FS01.png`.

Chạy lại: trên Kaggle thêm dataset có file `dataset_9_commodities_daily.csv` (hoặc `commodities_9_daily.csv`), bật GPU,
Internet On rồi Run All (900 lần train GRU); trên máy thì mở notebook trong thư mục `huy task/` (cần TensorFlow,
`pmdarima`).

## Tài liệu tham khảo
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159–175.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
