# Báo cáo thực nghiệm iTransformer dự báo giá Copper

## 1. Tóm tắt

Sáu notebook FS được thay bằng phiên bản thực nghiệm 5 seed (41–45). FS01–FS05 có đủ 120/120 lượt neural và 12/12 baseline mỗi notebook. FS06 mới có checkpoint chạy dở và không có bảng tổng hợp cuối.

Trên năm FS hoàn tất, MAE trung bình của iTransformer Direct là **1,478248**, Residual là **0,075400**, còn Naive là **0,073500**. Residual thắng baseline ở 68/300 so sánh seed–cấu hình; hiện chưa có bằng chứng rằng mô hình cải thiện ổn định so với Naive.

## 2. Dữ liệu và phạm vi

FS01–FS05 dùng `dataset_9_commodities_daily.csv`, 2.710 phiên từ 13/01/2016 đến 13/11/2026, fingerprint SHA-256 `a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83`. CSV không nằm trong repository. FS06 có cùng số dòng và ngày đầu/cuối nhưng fingerprint `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600`, nên không gộp khi so sánh.

Mỗi mẫu dùng 22 phiên quá khứ để dự báo Copper tại phiên tiếp theo. Chia theo thời gian, không shuffle; scaler được fit trên phần train. Notebook chạy bốn mốc test (80%, 90%, 92,5%, 95%), ba chiến lược (`expanding`, `rolling_756`, `recency_weighted`), Direct và Residual. Mỗi cấu hình neural được huấn luyện với năm seed 41–45; mỗi FS dự kiến 120 lượt neural và 12 Naive baseline.

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

Các notebook dùng lookback 22, horizon 1, inverted embedding 48, hai encoder với 4 attention head và FFN 96→48. Prediction head Flatten → Dense(32, GELU) → Dropout(0,1) → Dense(1). Adam learning rate 0,0003, clipnorm 1,0, loss MSE, batch size 64, tối đa 65 epochs, EarlyStopping patience 9.

## 5. Kết quả từ 5 seed

Bảng dưới tổng hợp 60 MAE cho mỗi mục tiêu (12 cấu hình × 5 seed), và 12 baseline cho mỗi feature set. Số MAE được lấy từ log notebook đã lưu, làm tròn theo log đến 5 chữ số thập phân trước khi tính trung bình.

| Feature set | Trạng thái | Direct MAE TB | Residual MAE TB | Naive MAE TB | Residual thắng Naive |
|---|---|---:|---:|---:|---:|
| FS01_CU | Hoàn tất | 1,518896 | 0,073990 | 0,073500 | 23/60 |
| FS02_PAPER_CU | Hoàn tất | 1,424573 | 0,076788 | 0,073500 | 13/60 |
| FS03_PAPER_EXO | Hoàn tất | 1,484225 | 0,077026 | 0,073500 | 4/60 |
| FS04_METALS_CU | Hoàn tất | 1,487418 | 0,074665 | 0,073500 | 14/60 |
| FS05_METALS_EXO | Hoàn tất | 1,476126 | 0,074533 | 0,073500 | 14/60 |

### Nhận xét

1. Direct có MAE khoảng 1,42–1,52, lớn hơn nhiều so với Naive khoảng 0,0735.
2. Residual tốt hơn Direct rõ rệt nhưng MAE trung bình vẫn cao hơn Naive khoảng 2,6%; thắng ở 68/300 lượt seed–cấu hình.
3. FS01 có MAE Residual thấp nhất trong năm FS hoàn tất. Đây là mô tả trên các cấu hình hiện có, không đủ để kết luận thêm biến làm kết quả kém đi.
4. Với FS01 (K=1), attention qua biến chỉ có một token, nên không học quan hệ giữa nhiều biến.
5. Các tỷ lệ test dùng những giai đoạn và số lượng quan sát khác nhau. Trung bình tổng hợp hữu ích để mô tả, không thay cho đánh giá trên một tập test cố định hoặc kiểm định thống kê.

Các độ lệch chuẩn theo seed trong CSV được tính từng cấu hình rồi lấy trung bình qua 12 cấu hình; không phải độ lệch chuẩn gộp giữa mọi MAE.

## 6. FS06 chưa hoàn tất

Cell trạng thái cuối được lưu trong notebook báo `running`, 80/120 lượt neural và 8/12 baseline. Transcript notebook có 40 dòng MAE Direct, 37 Residual và 7 baseline; số dòng transcript không khớp checkpoint. Cell tổng hợp cuối và hình dự báo chưa có output. Ngoài ra, FS06 có fingerprint dataset khác FS01–FS05. Do đó báo cáo không đưa MAE FS06 vào bảng so sánh; cần resume trên cùng CSV chuẩn và lưu đủ output cuối trước khi kết luận.

## 7. Bảng, log và biểu đồ

- [Bảng MAE từng job (744 dòng log trích được)](../outputs/tables/five_seed/job_mae_seed41_45.csv)
- [Tóm tắt MAE theo feature set](../outputs/tables/five_seed/feature_set_summary_fs01_fs05.csv)
- [Tóm tắt từng split × chiến lược × mục tiêu](../outputs/tables/five_seed/condition_summary_fs01_fs05.csv)
- [Trạng thái từng notebook](../outputs/tables/five_seed/run_status.csv)
- [Manifest nguồn trích xuất](../outputs/tables/five_seed/extraction_manifest.json)
- [Biểu đồ MAE tổng hợp](../outputs/figures/five_seed/mean_mae_by_feature_set_fs01_fs05.png)
- [Hình dự báo 5 seed](../outputs/tables/five_seed/figure_manifest.csv)

Notebook có code tính so sánh Diebold–Mariano, nhưng bảng p-value và raw predictions không nằm trong các file nguồn đưa vào lần cập nhật này. Vì vậy các nhận xét trên là mô tả, không kết luận ý nghĩa thống kê. Artifact lần chạy một seed trước đây được giữ trong thư mục `archive_seed42`.

## 8. Benchmark iTransformer với SA-LSTM

Notebook `iTransformer_vs_SA_LSTM_K5_Fair_FINAL_Colab.ipynb` là một thí nghiệm riêng và không bị thay đổi. Các kết quả benchmark đã lưu vẫn được ghi trong [nhận xét pipeline K=5](../Nhan_xet_iTransformer_Pipeline_Input_Output.md) và các bảng `outputs/tables/cell_*.html`; không gộp các số liệu đó với grid 5 seed FS01–FS06.

## 9. Kết luận

Trên năm feature set chạy đủ, Residual bám baseline tốt hơn Direct nhưng chưa thắng Naive ổn định. Hiện chưa có cơ sở để tuyên bố iTransformer cải thiện dự báo Copper trên pipeline này. Bước tiếp theo là thống nhất dataset cho FS06, resume các job còn thiếu, tải raw predictions và các bảng metric đầy đủ, rồi so sánh trên cùng giai đoạn test.
