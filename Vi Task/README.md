# Thực nghiệm iTransformer dự báo giá Copper

Thư mục này chứa bộ notebook FS01–FS06 phiên bản 5 seed (41–45), báo cáo và kết quả đã lưu. Cả sáu bộ đều hoàn tất 120/120 lượt neural và 12/12 baseline. ZIP FS06 là output cuối cùng sau checkpoint cũ trong notebook. Không chạy huấn luyện lại khi đóng gói.

## Mục tiêu

Dùng 22 phiên lịch sử để dự báo giá Copper phiên kế tiếp, so sánh sáu nhóm biến đầu vào, hai mục tiêu (Direct, Residual) và baseline Naive. Benchmark iTransformer–SA-LSTM là thí nghiệm riêng.

Dataset gồm 2.710 phiên từ 13/01/2016 đến 13/11/2026. FS01–FS05 dùng SHA-256 `a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83`; FS06 dùng `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600`. Do fingerprint khác nhau, FS06 được báo cáo riêng, không gộp để xếp hạng feature set.

## Notebook

| Notebook | Input X | Trạng thái |
|---|---|---|
| [FS01 Copper-only](notebooks/itransformer-fs01-cu-kaggle-final.ipynb) | Copper | Hoàn tất 120/120, baseline 12/12 |
| [FS02 Paper + Copper](notebooks/itransformer-fs02-paper-cu-kaggle-final.ipynb) | Copper, WTI, Gold, Silver | Hoàn tất |
| [FS03 Paper external](notebooks/itransformer-fs03-paper-exo-kaggle-final.ipynb) | WTI, Gold, Silver | Hoàn tất |
| [FS04 Metals + Copper](notebooks/itransformer-fs04-metals-cu-kaggle-final.ipynb) | Copper, Aluminium, Tin, Silver, Gold | Hoàn tất |
| [FS05 Metals external](notebooks/itransformer-fs05-metals-exo-kaggle-final.ipynb) | Aluminium, Tin, Silver, Gold | Hoàn tất |
| [FS06 WTI + Copper](notebooks/itransformer-fs06-wti-cu-kaggle-final.ipynb) | Copper, Aluminium, Tin, WTI | Hoàn tất; kết quả tách riêng do fingerprint khác |
| [Benchmark K=5 iTransformer vs SA-LSTM](iTransformer_vs_SA_LSTM_K5_Fair_FINAL_Colab.ipynb) | Aluminium, Tin, Silver, Gold, Copper | Thí nghiệm riêng |

Trong FS03/FS05, Copper không nằm trong tensor X. Nhánh Residual vẫn dùng Copper(t−1) để khôi phục dự báo mức giá.

## Cấu hình FS01–FS06

- Lookback 22, horizon 1; mỗi mẫu có shape 22 × K.
- Inverted embedding 48; hai encoder, mỗi encoder có 4 attention head và FFN 96 → 48.
- Adam, learning rate 0,0003, clipnorm 1,0; MSE; batch size 64; tối đa 65 epochs; EarlyStopping patience 9.
- Seeds 41–45; bốn mốc chia thời gian 80%, 90%, 92,5%, 95%; ba chiến lược `expanding`, `rolling_756`, `recency_weighted`.
- Mỗi FS có 120 lượt neural (4 mốc × 3 chiến lược × 2 mục tiêu × 5 seed) và 12 baseline Naive.

## Kết quả FS01–FS05

MAE Direct/Residual là trung bình 60 lượt (12 cấu hình × 5 seed); Naive là trung bình 12 cấu hình. Đơn vị là giá Copper trong dữ liệu.

| Feature set | Direct MAE | Residual MAE | Naive MAE | Residual thắng Naive |
|---|---:|---:|---:|---:|
| FS01_CU | 1,518896 | 0,073990 | 0,073500 | 23/60 |
| FS02_PAPER_CU | 1,424573 | 0,076788 | 0,073500 | 13/60 |
| FS03_PAPER_EXO | 1,484225 | 0,077026 | 0,073500 | 4/60 |
| FS04_METALS_CU | 1,487418 | 0,074665 | 0,073500 | 14/60 |
| FS05_METALS_EXO | 1,476126 | 0,074533 | 0,073500 | 14/60 |

Trung bình FS01–FS05: Direct 1,478248; Residual 0,075400; Naive 0,073500. Residual thắng Naive 68/300 lượt so sánh seed–cấu hình.

## Kết quả FS06 riêng

Trên fingerprint FS06, MAE trung bình là Direct **1,225048167**, Residual **0,079050344**, Naive **0,073498386**; Residual thắng baseline 16/60 lượt. Không gộp các số này với FS01–FS05 do khác fingerprint.

- [Báo cáo thực nghiệm](docs/BAO_CAO_THUC_NGHIEM_ITRANSFORMER.md)
- [Nhận xét kết quả 5 seed](docs/NHAN_XET_iTransformer_5SEEDS.md)
- [Mô tả input, output và kiến trúc](docs/MO_TA_INPUT_OUTPUT_VA_KIEN_TRUC.md)
- [Bảng tóm tắt FS01–FS05](outputs/tables/five_seed/feature_set_summary_fs01_fs05.csv)
- [Bảng tóm tắt FS06](outputs/tables/five_seed/feature_set_summary_fs06_full_run.csv)
- [MAE từng job, đủ sáu FS](outputs/tables/five_seed/job_mae_seed41_45.csv)
- [Trạng thái chạy](outputs/tables/five_seed/run_status.csv)
- [Bảng full-precision và DM của FS06](outputs/tables/five_seed/fs06_full_run/)
- [ZIP đầy đủ FS06: metrics, raw predictions, histories](outputs/archives/iTransformer_FS06_5Seeds_0bdcbf893b7a.zip)
- [Biểu đồ MAE FS01–FS05](outputs/figures/five_seed/mean_mae_by_feature_set_fs01_fs05.png)
- [Danh mục hình dự báo](outputs/tables/five_seed/figure_manifest.csv)

Notebook FS06 còn hiển thị checkpoint cũ 80/120 và 8/12; ZIP cùng run key ghi lần chạy cuối đã hoàn tất 120/120 và 12/12. ZIP có 132 metric rows, 38.049 dòng predictions, 120 histories, 24 nhóm mean/std và bảng DM. MAE tính lại từ predictions khớp metrics trong sai số tối đa khoảng 3×10⁻⁸; số dòng dự báo mỗi job khớp `n_test`.

Artifact chạy một seed cũ được lưu trong `outputs/figures/archive_seed42/` và `outputs/tables/archive_seed42/`. Các artifact benchmark K=5 ở `outputs/figures/` và `outputs/tables/cell_*.html` thuộc thí nghiệm riêng.

## Cập nhật bảng và hình

Từ thư mục `Vi Task`, chạy:

```powershell
python scripts/export_fiveseed_results.py
```

Script đọc log notebook và ZIP FS06; không cần TensorFlow, không train lại và không tạo dự báo mới. Notebook được thiết kế chạy trên Kaggle: gắn đúng dataset của FS cần chạy, bật GPU nếu muốn, rồi chạy từ đầu đến cuối.
