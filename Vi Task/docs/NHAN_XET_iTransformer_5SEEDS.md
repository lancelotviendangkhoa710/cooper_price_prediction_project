# Nhận xét sáu notebook iTransformer – kết quả 5 seed

## Phạm vi và thiết kế

Sáu FS dùng grid 4 tỷ lệ chia thời gian × 3 chiến lược train × 2 mục tiêu × 5 seed (41–45), tương đương 120 lượt neural và 12 baseline Naive mỗi bộ. FS01–FS05 có đủ output notebook. FS06 có ZIP kết quả cuối đủ metrics, predictions, histories và bảng DM.

FS01–FS05 dùng fingerprint CSV `a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83`. FS06 dùng `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600`. Cùng số dòng/ngày đầu-cuối không đủ chứng minh dữ liệu giống nhau, nên giữ FS06 riêng.

## Kết quả FS01–FS05

MAE Direct/Residual là trung bình 60 lượt mỗi mục tiêu; Naive là trung bình 12 cấu hình.

| FS | Direct MAE TB | Residual MAE TB | Naive MAE TB | Residual thắng Naive |
|---|---:|---:|---:|---:|
| FS01_CU | 1,518896 | 0,073990 | 0,073500 | 23/60 |
| FS02_PAPER_CU | 1,424573 | 0,076788 | 0,073500 | 13/60 |
| FS03_PAPER_EXO | 1,484225 | 0,077026 | 0,073500 | 4/60 |
| FS04_METALS_CU | 1,487418 | 0,074665 | 0,073500 | 14/60 |
| FS05_METALS_EXO | 1,476126 | 0,074533 | 0,073500 | 14/60 |

Trung bình năm FS: Direct 1,478248; Residual 0,075400; Naive 0,073500. Residual thắng 68/300 lượt so sánh seed–cấu hình.

## Kết quả FS06 – fingerprint riêng

ZIP ghi `complete`, 120/120 neural và 12/12 baseline; có 132 metrics, 38.049 raw prediction rows, 120 histories, 24 nhóm mean/std và bảng DM. Raw prediction có đủ số dòng `n_test`; MAE tính lại khớp metrics trong sai số khoảng 3×10⁻⁸.

| FS | Direct MAE TB | Residual MAE TB | Naive MAE TB | Residual thắng Naive |
|---|---:|---:|---:|---:|
| FS06_WTI_CU | 1,225048167 | 0,079050344 | 0,073498386 | 16/60 |

Residual MAE FS06 cao hơn Naive trung bình. Không xếp hạng FS06 cùng FS01–FS05. Notebook còn hiển thị checkpoint cũ; [ZIP cuối](../outputs/archives/iTransformer_FS06_5Seeds_0bdcbf893b7a.zip) cùng run key ghi đủ 120/120 và 12/12.

## Nhận xét

- Direct có MAE lớn hơn Naive nhiều lần trên hai nhóm dữ liệu.
- Residual cải thiện rõ so với Direct nhưng chưa tốt hơn Naive ổn định.
- FS01 có Residual MAE thấp nhất trong FS01–FS05, nhưng chưa đủ để kết luận Copper-only tốt nhất. K=1 cũng không cho attention học quan hệ nhiều biến.
- Các tỷ lệ test khác nhau có giai đoạn và độ dài khác nhau; số trung bình chỉ mang tính mô tả.
- FS06 có kiểm định DM, nhưng có nhiều phép so sánh; không suy rộng một p-value thành kết luận chung.

## Artifact

- [Bảng MAE FS01–FS05](../outputs/tables/five_seed/feature_set_summary_fs01_fs05.csv)
- [Tóm tắt full-precision FS06](../outputs/tables/five_seed/feature_set_summary_fs06_full_run.csv)
- [792 dòng MAE đủ sáu FS](../outputs/tables/five_seed/job_mae_seed41_45.csv)
- [ZIP FS06: raw predictions, metrics, DM và histories](../outputs/archives/iTransformer_FS06_5Seeds_0bdcbf893b7a.zip)
- [Biểu đồ MAE FS01–FS05](../outputs/figures/five_seed/mean_mae_by_feature_set_fs01_fs05.png)
- [Danh mục hình dự báo](../outputs/tables/five_seed/figure_manifest.csv)

FS01–FS05 trích MAE từ log notebook đã làm tròn 5 chữ số; FS06 dùng metrics full-precision từ ZIP. Artifact lần chạy một seed cũ được giữ trong `archive_seed42`.
