# Thực nghiệm iTransformer dự báo giá Copper

Thư mục này chứa bộ notebook iTransformer FS01–FS06 phiên bản 5 seed (41–45), kết quả được trích từ log đã lưu, biểu đồ và tài liệu tiếng Việt. Các notebook FS01–FS05 đã chạy đủ; FS06 còn dở. Việc đóng gói này không chạy lại mô hình hay tạo dự báo mới.

## Mục tiêu

Dùng 22 phiên lịch sử để dự báo giá Copper ở phiên kế tiếp, rồi so sánh sáu nhóm biến đầu vào, hai mục tiêu iTransformer (Direct và Residual) cùng baseline Naive. Notebook benchmark iTransformer–SA-LSTM là thí nghiệm riêng.

Dataset được ghi nhận có 2.710 phiên từ 13/01/2016 đến 13/11/2026. FS01–FS05 dùng cùng fingerprint SHA-256 `a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83`. Dữ liệu không được chép vào repository.

## Notebook

| Notebook | Input X | Trạng thái |
|---|---|---|
| [FS01 Copper-only](notebooks/itransformer-fs01-cu-kaggle-final.ipynb) | Copper | Hoàn tất 120/120 lượt mô hình, 12/12 baseline |
| [FS02 Paper + Copper](notebooks/itransformer-fs02-paper-cu-kaggle-final.ipynb) | Copper, WTI, Gold, Silver | Hoàn tất |
| [FS03 Paper external](notebooks/itransformer-fs03-paper-exo-kaggle-final.ipynb) | WTI, Gold, Silver | Hoàn tất |
| [FS04 Metals + Copper](notebooks/itransformer-fs04-metals-cu-kaggle-final.ipynb) | Copper, Aluminium, Tin, Silver, Gold | Hoàn tất |
| [FS05 Metals external](notebooks/itransformer-fs05-metals-exo-kaggle-final.ipynb) | Aluminium, Tin, Silver, Gold | Hoàn tất |
| [FS06 WTI + Copper](notebooks/itransformer-fs06-wti-cu-kaggle-final.ipynb) | Copper, Aluminium, Tin, WTI | Đang chạy dở; không dùng để xếp hạng |
| [Benchmark K=5 iTransformer vs SA-LSTM](iTransformer_vs_SA_LSTM_K5_Fair_FINAL_Colab.ipynb) | Aluminium, Tin, Silver, Gold, Copper | Thí nghiệm riêng, không gộp vào bảng FS |

Trong FS03/FS05, Copper không nằm trong tensor X. Nhánh Residual vẫn dùng Copper(t−1) để khôi phục dự báo mức giá.

## Cấu hình FS01–FS06

- Lookback 22 phiên, horizon 1; input mỗi mẫu có shape 22 × K.
- Inverted embedding 48; hai encoder, mỗi encoder có 4 attention head và FFN 96 → 48.
- Adam, learning rate 0,0003, clipnorm 1,0; MSE; batch size 64; tối đa 65 epochs; EarlyStopping patience 9.
- Seeds 41–45. Bốn mốc chia thời gian 80%, 90%, 92,5%, 95%; ba chiến lược train `expanding`, `rolling_756`, `recency_weighted`.
- Mỗi FS có 120 lượt neural (4 mốc × 3 chiến lược × 2 mục tiêu × 5 seed) và 12 baseline Naive dùng chung.

## Kết quả 5 seed

MAE là trung bình qua 60 lượt của mỗi mục tiêu (12 cấu hình × 5 seed); baseline là trung bình qua 12 cấu hình. Đơn vị là giá Copper trong dữ liệu.

| Feature set | Direct MAE | Residual MAE | Naive MAE | Residual thắng Naive |
|---|---:|---:|---:|---:|
| FS01_CU | 1,518896 | 0,073990 | 0,073500 | 23/60 |
| FS02_PAPER_CU | 1,424573 | 0,076788 | 0,073500 | 13/60 |
| FS03_PAPER_EXO | 1,484225 | 0,077026 | 0,073500 | 4/60 |
| FS04_METALS_CU | 1,487418 | 0,074665 | 0,073500 | 14/60 |
| FS05_METALS_EXO | 1,476126 | 0,074533 | 0,073500 | 14/60 |

Trung bình năm FS hoàn tất: Direct 1,478248; Residual 0,075400; Naive 0,073500. Residual thắng Naive ở 68/300 so sánh seed–cấu hình; vì vậy kết quả hiện tại chưa chứng minh mô hình tốt hơn baseline ổn định. FS01–FS05 dùng cùng fingerprint dữ liệu. FS06 ghi fingerprint khác (`9580ede1…`) và chưa có bảng tổng hợp cuối.

- [Báo cáo thực nghiệm](docs/BAO_CAO_THUC_NGHIEM_ITRANSFORMER.md)
- [Nhận xét kết quả 5 seed](docs/NHAN_XET_iTransformer_5SEEDS.md)
- [Mô tả input, output và kiến trúc](docs/MO_TA_INPUT_OUTPUT_VA_KIEN_TRUC.md)
- [Bảng tóm tắt theo feature set](outputs/tables/five_seed/feature_set_summary_fs01_fs05.csv)
- [MAE từng lượt chạy trích từ notebook](outputs/tables/five_seed/job_mae_seed41_45.csv)
- [Tình trạng hoàn tất từng FS](outputs/tables/five_seed/run_status.csv)
- [Tóm tắt từng cấu hình](outputs/tables/five_seed/condition_summary_fs01_fs05.csv)
- [Biểu đồ MAE](outputs/figures/five_seed/mean_mae_by_feature_set_fs01_fs05.png)
- [Danh mục hình dự báo](outputs/tables/five_seed/figure_manifest.csv)

Các log hiển thị MAE đã làm tròn 5 chữ số. CSV ở đây là phần trích từ log, không thay cho raw predictions/full-precision metrics hay kiểm định thống kê. Log FS06 lưu 40 Direct, 37 Residual và 7 baseline; checkpoint gần nhất ghi 80/120 neural và 8/12 baseline, nên transcript không khớp checkpoint. Không diễn giải các dòng FS06 như kết quả hoàn chỉnh.

Artifact của lần chạy FS cũ một seed được lưu riêng trong `outputs/figures/archive_seed42/` và `outputs/tables/archive_seed42/`. Hình/bảng benchmark K=5 ở `outputs/figures/` và `outputs/tables/cell_*` vẫn thuộc thí nghiệm riêng, không bị trộn với kết quả 5 seed.

## Cập nhật bảng và hình từ notebook

Từ thư mục `Vi Task`, chạy:

```powershell
python scripts/export_fiveseed_results.py
```

Script chỉ đọc log và hình đã lưu trong sáu notebook; không cần TensorFlow, không huấn luyện lại và không tạo dự báo mới. Notebook được thiết kế chạy trên Kaggle: gắn dataset CSV, bật GPU nếu muốn, rồi chạy notebook từ đầu đến cuối để tiếp tục các job còn thiếu.

Không đưa dataset, checkpoint `.keras`, Kaggle ZIP hay API key vào repository.
