# Báo cáo thực nghiệm iTransformer dự báo giá Copper

## 1. Tóm tắt

Sáu notebook FS dùng thực nghiệm 5 seed (41–45). Mỗi bộ có đủ 120/120 lượt neural và 12/12 baseline. FS01–FS05 cùng fingerprint dữ liệu; FS06 hoàn tất nhưng dùng fingerprint khác nên được phân tích riêng.

Trên FS01–FS05, MAE trung bình của Direct là **1,478248**, Residual **0,075400**, Naive **0,073500**. Residual thắng baseline ở 68/300 lượt seed–cấu hình. Trên FS06 riêng, MAE là Direct **1,225048167**, Residual **0,079050344**, Naive **0,073498386**; Residual thắng ở 16/60 lượt.

## 2. Dữ liệu và phạm vi

Các notebook ghi dataset `dataset_9_commodities_daily.csv`, 2.710 phiên từ 13/01/2016 đến 13/11/2026. FS01–FS05 dùng SHA-256 `a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83`; ZIP FS06 dùng `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600`. Cùng số dòng và ngày đầu/cuối không chứng minh dữ liệu từng ngày giống nhau, nên không gộp FS06 với FS01–FS05. CSV không có trong repository.

Mỗi mẫu dùng 22 phiên quá khứ để dự báo Copper phiên kế tiếp. Chia theo thời gian, không shuffle; scaler fit trên train. Grid gồm bốn tỷ lệ test (80%, 90%, 92,5%, 95%), ba chiến lược (`expanding`, `rolling_756`, `recency_weighted`), hai mục tiêu (Direct, Residual), năm seed. Mỗi FS có 120 lượt neural và 12 Naive.

## 3. Feature set

| FS | Biến đầu vào X | K |
|---|---|---:|
| FS01_CU | Copper | 1 |
| FS02_PAPER_CU | Copper, WTI, Gold, Silver | 4 |
| FS03_PAPER_EXO | WTI, Gold, Silver | 3 |
| FS04_METALS_CU | Copper, Aluminium, Tin, Silver, Gold | 5 |
| FS05_METALS_EXO | Aluminium, Tin, Silver, Gold | 4 |
| FS06_WTI_CU | Copper, Aluminium, Tin, WTI | 4 |

FS03/FS05 không đưa Copper vào X; Residual vẫn dùng Copper(t−1) khi chuyển dự báo thay đổi thành mức giá.

## 4. Cấu hình mô hình

Lookback 22, horizon 1, inverted embedding 48, hai encoder với 4 attention head và FFN 96→48. Head: Flatten → Dense(32, GELU) → Dropout(0,1) → Dense(1). Adam learning rate 0,0003, clipnorm 1,0, MSE, batch size 64, tối đa 65 epochs, EarlyStopping patience 9.

## 5. FS01–FS05: cùng fingerprint

MAE Direct/Residual trung bình qua 60 lượt mỗi mục tiêu (12 cấu hình × 5 seed); Naive qua 12 cấu hình. Log notebook hiển thị MAE làm tròn 5 chữ số trước khi tính trung bình.

| Feature set | Trạng thái | Direct MAE TB | Residual MAE TB | Naive MAE TB | Residual thắng Naive |
|---|---|---:|---:|---:|---:|
| FS01_CU | Hoàn tất | 1,518896 | 0,073990 | 0,073500 | 23/60 |
| FS02_PAPER_CU | Hoàn tất | 1,424573 | 0,076788 | 0,073500 | 13/60 |
| FS03_PAPER_EXO | Hoàn tất | 1,484225 | 0,077026 | 0,073500 | 4/60 |
| FS04_METALS_CU | Hoàn tất | 1,487418 | 0,074665 | 0,073500 | 14/60 |
| FS05_METALS_EXO | Hoàn tất | 1,476126 | 0,074533 | 0,073500 | 14/60 |

Direct kém xa baseline Naive. Residual tốt hơn Direct nhưng MAE gộp cao hơn Naive khoảng 2,6%, chỉ thắng ở 68/300 lượt so sánh. FS01 có Residual MAE thấp nhất, nhưng chưa đủ để kết luận Copper-only tổng quát hóa tốt nhất. Với K=1, attention không học quan hệ liên biến.

## 6. FS06: kết quả ZIP hoàn tất

ZIP [iTransformer_FS06_5Seeds_0bdcbf893b7a.zip](../outputs/archives/iTransformer_FS06_5Seeds_0bdcbf893b7a.zip) ghi `complete`: 120/120 lượt neural, 12/12 baseline, 132 dòng metrics, 38.049 dòng raw predictions, 120 histories và một hình tổng hợp.

| Feature set | Trạng thái | Direct MAE TB | Residual MAE TB | Naive MAE TB | Residual thắng Naive |
|---|---|---:|---:|---:|---:|
| FS06_WTI_CU | Hoàn tất | 1,225048167 | 0,079050344 | 0,073498386 | 16/60 |

Metrics full-precision khớp với `metrics_mean_std.csv` tính lại từ metrics theo seed (sai khác tối đa 2,3×10⁻¹⁶). MAE tính lại từ `actual`/`pred` khớp metric trong sai số tối đa khoảng 3×10⁻⁸; số dòng từng nhóm khớp `n_test`. Có đủ metrics và predictions cho 132 job, cùng history cho 120 lượt neural.

Output nhúng trong notebook là checkpoint cũ `running`, 80/120 neural và 8/12 baseline. ZIP có cùng run key `itransformer_fs06_5seed_0bdcbf893b7a_sha9580ede1` và ghi kết quả cuối `complete`. Vì fingerprint FS06 khác FS01–FS05, chỉ diễn giải MAE FS06 trong chính nhóm dữ liệu đó, không xếp hạng trực tiếp sáu feature set.

## 7. Artifact kết quả

- [Tóm tắt FS01–FS05](../outputs/tables/five_seed/feature_set_summary_fs01_fs05.csv)
- [Tóm tắt riêng FS06](../outputs/tables/five_seed/feature_set_summary_fs06_full_run.csv)
- [MAE theo job đủ sáu FS](../outputs/tables/five_seed/job_mae_seed41_45.csv)
- [Tình trạng chạy](../outputs/tables/five_seed/run_status.csv)
- [Metrics full-precision, mean/std, DM và data quality FS06](../outputs/tables/five_seed/fs06_full_run/)
- [ZIP có raw predictions, histories và toàn bộ artifact FS06](../outputs/archives/iTransformer_FS06_5Seeds_0bdcbf893b7a.zip)
- [Biểu đồ MAE FS01–FS05](../outputs/figures/five_seed/mean_mae_by_feature_set_fs01_fs05.png)
- [Danh mục hình dự báo](../outputs/tables/five_seed/figure_manifest.csv)

FS06 có kiểm định DM theo seed và tổng hợp mô tả trong ZIP. Do có nhiều cấu hình/phép so sánh, không suy rộng một p-value thành kết luận tổng quát. Artifact lần chạy một seed cũ được giữ trong `archive_seed42`.

## 8. Benchmark iTransformer với SA-LSTM

Notebook `iTransformer_vs_SA_LSTM_K5_Fair_FINAL_Colab.ipynb` là thí nghiệm riêng, không bị thay đổi. Output benchmark nằm trong [nhận xét pipeline K=5](../Nhan_xet_iTransformer_Pipeline_Input_Output.md) và `outputs/tables/cell_*.html`; không gộp với grid 5 seed FS.

## 9. Kết luận

Trên FS01–FS05, Residual bám Naive nhưng chưa thắng ổn định. FS06 cũng có MAE Residual cao hơn Naive trung bình trên fingerprint riêng. Chưa có bằng chứng iTransformer cải thiện dự báo Copper ổn định. Muốn so sánh sáu nhóm biến trực tiếp, cần chạy lại FS06 trên đúng CSV canonical của FS01–FS05 và cùng ngày test.
