# Nhận xét sáu notebook iTransformer – kết quả 5 seed

## Phạm vi

Đã rà soát code và output đã lưu trong sáu notebook FS01–FS06. Năm notebook FS01–FS05 lưu đủ 120 lượt neural và 12 baseline mỗi bộ. FS06 đang dở. Các bảng dưới đây được trích từ log MAE trong notebook; không chạy lại mô hình.

## Thiết kế

Mỗi lượt dùng 22 phiên Copper/biến đầu vào để dự báo Copper phiên kế tiếp. Grid gồm bốn tỷ lệ chia theo thời gian (0,80; 0,90; 0,925; 0,95), ba chiến lược (`expanding`, `rolling_756`, `recency_weighted`), hai mục tiêu (Direct, Residual) và năm seed (41–45). Mỗi FS có 120 lượt neural và 12 baseline Naive.

FS01–FS05 có cùng fingerprint CSV `a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83`. FS06 có fingerprint `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600`; cùng số dòng/ngày đầu-cuối không đủ chứng minh dữ liệu từng dòng giống nhau.

## MAE trên các FS hoàn tất

MAE Direct/Residual là trung bình 60 lượt (12 cấu hình × 5 seed); MAE Naive là trung bình 12 cấu hình. Cột “Residual thắng Naive” đếm mỗi seed–cấu hình.

| FS | Direct MAE TB | Residual MAE TB | Naive MAE TB | Residual thắng Naive |
|---|---:|---:|---:|---:|
| FS01_CU | 1,518896 | 0,073990 | 0,073500 | 23/60 |
| FS02_PAPER_CU | 1,424573 | 0,076788 | 0,073500 | 13/60 |
| FS03_PAPER_EXO | 1,484225 | 0,077026 | 0,073500 | 4/60 |
| FS04_METALS_CU | 1,487418 | 0,074665 | 0,073500 | 14/60 |
| FS05_METALS_EXO | 1,476126 | 0,074533 | 0,073500 | 14/60 |

Gộp năm FS: Direct MAE 1,478248; Residual 0,075400; Naive 0,073500. Residual chỉ thắng ở 68/300 lượt so sánh; MAE tổng hợp cao hơn baseline khoảng 2,6%.

## Diễn giải

- Direct sai số lớn hơn Naive rất nhiều trong các log này. Cần kiểm tra kỹ nhánh Direct và khâu khôi phục dự báo trước khi xem xét dùng cho dự báo giá.
- Residual cải thiện rõ so với Direct, nhưng chưa tốt hơn Naive ổn định qua các seed và cấu hình.
- FS01 có Residual MAE trung bình thấp nhất, nhưng chênh lệch nhỏ và chưa đủ để kết luận Copper-only tổng quát hóa tốt nhất. Với K=1, attention qua biến chỉ có một token nên không học quan hệ liên biến.
- Trong 300 kết quả Residual, chỉ 68 kết quả có MAE thấp hơn baseline ghép cùng cấu hình. Số lần thắng đơn lẻ không thay thế kiểm định thống kê.
- Các tỷ lệ test khác nhau có giai đoạn và độ dài test khác nhau, nên bảng tổng hợp là mô tả qua grid, không phải một ước lượng duy nhất trên cùng tập test cố định.

## FS06 cần làm tiếp

Checkpoint gần nhất được in trong notebook là `running`, 80/120 neural và 8/12 Naive. Transcript lưu được 40 dòng Direct, 37 Residual và 7 Naive, không khớp trạng thái checkpoint; cell tổng hợp và hình dự báo cuối không có output. FS06 cũng dùng SHA khác với FS01–FS05. Vì vậy không lấy các MAE đang thấy trong log để xếp hạng FS06. Cần thống nhất CSV, resume notebook và lưu đủ bảng tổng hợp cuối.

## Artifact đã xuất

- [Tóm tắt MAE theo FS](../outputs/tables/five_seed/feature_set_summary_fs01_fs05.csv)
- [744 dòng MAE từ log đã lưu](../outputs/tables/five_seed/job_mae_seed41_45.csv)
- [Tóm tắt từng tỷ lệ × chiến lược × mục tiêu](../outputs/tables/five_seed/condition_summary_fs01_fs05.csv)
- [Tiến độ từng notebook](../outputs/tables/five_seed/run_status.csv)
- [Biểu đồ MAE](../outputs/figures/five_seed/mean_mae_by_feature_set_fs01_fs05.png)
- [Danh mục biểu đồ dự báo lấy từ cell output](../outputs/tables/five_seed/figure_manifest.csv)

MAE đã làm tròn 5 chữ số trong log. Raw predictions, full-precision metrics và bảng p-value DM không nằm trong bộ file này, do đó báo cáo không kết luận ý nghĩa thống kê. Artifact của lần chạy FS một seed cũ được giữ trong `outputs/figures/archive_seed42/` và `outputs/tables/archive_seed42/`.
