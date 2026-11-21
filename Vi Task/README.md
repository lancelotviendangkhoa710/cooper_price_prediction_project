# Thực nghiệm iTransformer dự báo giá Copper

Thư mục này lưu notebook benchmark iTransformer–SA-LSTM, sáu ablation feature set FS01–FS06, báo cáo tiếng Việt và những output đã kiểm chứng. Nội dung được trích từ notebook/code hiện có; không thay đổi thuật toán, hyperparameter hay chạy lại training khi đóng gói.

## Mục tiêu

Dùng 22 phiên lịch sử để dự báo giá Copper ở phiên kế tiếp, sau đó:

- so sánh sáu nhóm biến đầu vào với cùng bộ dựng iTransformer;
- đối chiếu Direct, Residual và baseline Naive;
- so sánh iTransformer với SA-LSTM trong notebook benchmark riêng.

Dataset được ghi trong các output là dataset_9_commodities_daily.csv, gồm 2.710 phiên từ 13/01/2016 đến 13/11/2026. Fingerprint SHA-256 được notebook ghi nhận là a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83. Dataset không được chép vào repository.

## Notebook

Notebook benchmark đã có sẵn ở thư mục này; sáu notebook feature set được lưu trong thư mục notebooks.

| Notebook | Input theo thứ tự code | Dùng để |
|---|---|---|
| [iTransformer_vs_SA_LSTM_K5_Fair_FINAL_Colab.ipynb](iTransformer_vs_SA_LSTM_K5_Fair_FINAL_Colab.ipynb) | Aluminium, Tin, Silver, Gold, Copper | Benchmark iTransformer với SA-LSTM và Naive |
| [FS01 Copper-only](notebooks/itransformer-fs01-cu-kaggle-final.ipynb) | Copper | Ablation chỉ dùng lịch sử Copper |
| [FS02 Paper + Copper](notebooks/itransformer-fs02-paper-cu-kaggle-final.ipynb) | Copper, WTI, Gold, Silver | Ba biến kiểu paper cộng lịch sử Copper |
| [FS03 Paper external](notebooks/itransformer-fs03-paper-exo-kaggle-final.ipynb) | WTI, Gold, Silver | Chỉ ba biến ngoại sinh |
| [FS04 Metals + Copper](notebooks/itransformer-fs04-metals-cu-kaggle-final.ipynb) | Copper, Aluminium, Tin, Silver, Gold | Năm biến kim loại gồm Copper |
| [FS05 Metals external](notebooks/itransformer-fs05-metals-exo-kaggle-final.ipynb) | Aluminium, Tin, Silver, Gold | Bốn biến kim loại, không Copper trong X |
| [FS06 WTI + Copper](notebooks/itransformer-fs06-wti-cu-kaggle-final.ipynb) | Copper, Aluminium, Tin, WTI | Tổ hợp biến thay thế |

Trong FS03/FS05, Copper không nằm trong tensor X. Nhánh Residual vẫn dùng Copper(t−1) khi dựng lại dự báo mức giá; xem giải thích đầy đủ trong báo cáo kiến trúc.

## Cấu hình chung FS01–FS06

- Lookback 22, horizon 1; input tensor mỗi mẫu có shape 22 × K.
- Inverted Embedding 48; hai encoder, mỗi encoder có 4 attention head và FFN 96 → 48.
- Head: Flatten(K × 48) → Dense(32, GELU) → Dropout(0,1) → Dense(1, linear).
- Adam, learning rate 0,0003, clipnorm 1,0; MSE; batch size 64.
- Tối đa 65 epochs, EarlyStopping patience 9, seed 42.
- Chia theo thời gian tại 80%, 90%, 92,5%, 95%; chạy Expanding, Rolling-756 và Recency Weighted.
- Mỗi feature set có 36 dòng kết quả: Direct, Residual và Naive qua 4 tỷ lệ × 3 chiến lược. Không shuffle thời gian; scaler fit trên train.

Notebook benchmark dùng cấu hình riêng: output ghi 84/84 job hoàn thành, bao gồm iTransformer, SA-LSTM cố định, SA-LSTM tuned bằng SA và Naive. Benchmark cho phép tối đa 100 epochs với SA-LSTM, nên không gộp ngân sách này với 65 epochs của FS01–FS06.

## Kết quả và giới hạn diễn giải

Ở split 90/10–Expanding, mỗi FS có 2.173 mẫu train, 244 validation và 271 test. Bảng notebook đã lưu cho thấy:

- Direct có MAE khoảng 1,14–1,81, cao hơn nhiều so với Naive khoảng 0,077.
- Residual gần Naive hơn. FS01, FS02 và FS04 thấp hơn baseline một lượng nhỏ; FS03, FS05 và FS06 cao hơn.
- Mức chênh lệch nhỏ, chỉ có seed 42; không đủ căn cứ để khẳng định một feature set thắng ổn định.

Trong benchmark K=5, tại 90/10–Expanding, iTransformer Residual có MAE 0,078119; Naive 0,077077; SA-LSTM cố định Residual 0,082928; SA-LSTM tuned Residual 0,085346. iTransformer không vượt Naive theo MAE ở cấu hình này, dù thấp hơn hai SA-LSTM Residual trong output benchmark.

Các kết quả của nhánh Khoa chưa được gộp vào bảng so sánh: báo cáo V20 ghi fingerprint 9580ede1…, còn notebook iTransformer ghi a761a908…. Cùng số dòng/ngày đầu-cuối không chứng minh dữ liệu khớp theo từng phiên. Chi tiết và giới hạn được nêu trong báo cáo thực nghiệm.

## Tài liệu

- [Báo cáo thực nghiệm và nhận xét kết quả](docs/BAO_CAO_THUC_NGHIEM_ITRANSFORMER.md)
- [Giải thích Input, Output và kiến trúc](docs/MO_TA_INPUT_OUTPUT_VA_KIEN_TRUC.md)
- [Nhận xét pipeline benchmark K=5](Nhan_xet_iTransformer_Pipeline_Input_Output.md)
- [Bảng metrics trích từ output notebook, 216 dòng](outputs/tables/itransformer_fs01_fs06_saved_metrics.csv)
- [Manifest trích xuất output](outputs/tables/itransformer_fs_notebook_output_extraction.json)
- Các bảng HTML và execution log của benchmark nằm trong [outputs/tables](outputs/tables/) và [outputs/execution_log.txt](outputs/execution_log.txt).

Một số hình đại diện ở split 90/10–Expanding: [FS01](outputs/figures/feature_sets/FS01_CU_prices_r0900_expanding.png), [FS04](outputs/figures/feature_sets/FS04_METALS_CU_prices_r0900_expanding.png), [FS05](outputs/figures/feature_sets/FS05_METALS_EXO_prices_r0900_expanding.png). Mỗi notebook FS còn lưu output hình cho đủ các tỷ lệ chia và chiến lược train.

## Chạy lại trên Kaggle

1. Mở notebook FS cần chạy và chọn accelerator GPU nếu muốn dùng GPU.
2. Add Input chứa dataset_9_commodities_daily.csv; các notebook tìm dataset ở các đường dẫn Kaggle thông dụng. Nếu cần, điền CUSTOM_CSV_PATH trong cell cấu hình.
3. Chạy notebook từ đầu đến cuối. Mỗi notebook xuất grid_results.csv, grid_predictions.csv, manifest và ZIP vào /kaggle/working.
4. Tải ZIP từ Kaggle Output để lưu raw predictions và full-precision CSV. Các ZIP/CSV đó hiện không có trong repo này; bảng metrics đi kèm là bản trích đã làm tròn từ output notebook.

Không upload dataset, API key, checkpoint .keras hoặc ZIP output trùng lặp vào GitHub.
