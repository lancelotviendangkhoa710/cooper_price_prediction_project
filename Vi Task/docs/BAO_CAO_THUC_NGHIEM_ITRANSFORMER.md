# Báo cáo thực nghiệm iTransformer cho dự báo giá Copper

## 1. Tóm tắt

Báo cáo tổng hợp bảy notebook hiện có trong thư mục Vi Task: một notebook benchmark iTransformer với SA-LSTM và sáu notebook ablation feature set FS01–FS06. Nội dung được trích từ code và output đã lưu trong notebook; không chạy lại training và không tạo số liệu dự đoán mới.

Kết quả chính trên bộ sáu feature set: dự báo trực tiếp mức giá Copper có MAE cao hơn rất nhiều so với baseline Naive. Nhánh Residual cho kết quả gần baseline. Ở cấu hình đại diện 90/10–Expanding, FS01, FS02 và FS04 thấp hơn MAE của Naive một lượng nhỏ; FS03, FS05 và FS06 cao hơn. Vì các lần chạy chỉ có seed 42 và dùng chung một giai đoạn test, đây là kết quả mô tả, chưa đủ để kết luận feature set nào tổng quát hóa tốt hơn.

## 2. Phạm vi, dữ liệu và nguồn kiểm chứng

Các notebook FS01–FS06 ghi nhận tệp đầu vào **dataset_9_commodities_daily.csv**, 2.710 phiên từ **13/01/2016 đến 13/11/2026**, không loại dòng nào trong bước làm sạch. SHA-256 được ghi trong output là:

**a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83**

Tệp CSV gốc và ZIP kết quả Kaggle không có trong thư mục nguồn E:/Task/iTransformer tại thời điểm hoàn thiện báo cáo. Vì vậy, các metric của FS01–FS06 được trích từ bảng HTML đã lưu trong output notebook, với độ chính xác đúng như notebook hiển thị, thường là 5 chữ số thập phân. Bảng trích xuất gồm 216 dòng được lưu tại [itransformer_fs01_fs06_saved_metrics.csv](../outputs/tables/itransformer_fs01_fs06_saved_metrics.csv); provenance của việc trích xuất nằm trong [itransformer_fs_notebook_output_extraction.json](../outputs/tables/itransformer_fs_notebook_output_extraction.json).

### 2.1. Phạm vi dự báo và quy trình chia chuỗi

- Mỗi mẫu dùng 22 phiên quá khứ để dự báo Copper ở phiên kế tiếp: lookback = 22, horizon = 1.
- Chia theo thứ tự thời gian, không shuffle. Các tỷ lệ 80%, 90%, 92,5% và 95% là vị trí bắt đầu phần test ở cuối chuỗi.
- Validation nằm ngay trước test; code dùng khoảng 10% của đoạn đứng trước mốc test làm validation.
- Ba cách tạo tập train là Expanding, Rolling-756 và Recency Weighted. Recency Weighted giảm trọng số quan sát cũ với half-life 252 phiên.
- MinMaxScaler được fit trên phần train của từng split rồi mới áp dụng lên các phần còn lại.
- Đánh giá là rolling one-step-ahead: tại mỗi thời điểm, cửa sổ đầu vào chỉ chứa các quan sát đến phiên trước đó. Đây không phải dự báo đệ quy nhiều bước từ một điểm xuất phát cố định.
- Mỗi notebook FS ghi 36 dòng metric: 4 tỷ lệ chia × 3 chiến lược train × (iTransformer Direct, iTransformer Residual, Naive). Sáu notebook có tổng cộng 216 dòng metric và mỗi notebook báo cáo 10.377 bản ghi dự báo.

Các bước này hạn chế leakage do xáo trộn thời gian hoặc fit scaler trên validation/test. Tuy vậy, việc xác minh rằng các giá ngoại sinh đã thực sự được công bố trước thời điểm dự báo chưa được thực hiện; cần phân biệt ngày trong CSV với thời điểm dữ liệu có thể dùng trong vận hành.

## 3. Sáu feature set

Thứ tự input dưới đây được đọc từ biến FEATURES trong code, không sắp xếp lại theo tên.

| Mã | Input theo đúng thứ tự trong code | K | Ý nghĩa của ablation |
|---|---|---:|---|
| FS01_CU | Copper | 1 | Chỉ lịch sử Copper |
| FS02_PAPER_CU | Copper, WTI, Gold, Silver | 4 | Ba biến ngoại sinh kiểu paper cộng lịch sử Copper |
| FS03_PAPER_EXO | WTI, Gold, Silver | 3 | Chỉ ba biến ngoại sinh; Copper không nằm trong tensor X |
| FS04_METALS_CU | Copper, Aluminium, Tin, Silver, Gold | 5 | Năm kim loại, gồm lịch sử Copper |
| FS05_METALS_EXO | Aluminium, Tin, Silver, Gold | 4 | Bốn kim loại ngoại sinh; Copper không nằm trong tensor X |
| FS06_WTI_CU | Copper, Aluminium, Tin, WTI | 4 | Copper, Aluminium, Tin và WTI |

FS03 và FS05 không đưa Copper vào tensor X. Tuy nhiên, với objective Residual, code vẫn lấy Copper ở phiên trước để cộng lại sau khi mô hình dự báo phần thay đổi. Vì vậy, “không có Copper trong input” chỉ mô tả tensor đưa vào encoder; toàn bộ phép khôi phục dự báo Residual vẫn sử dụng giá Copper quá khứ.

## 4. Cấu hình mô hình và ngân sách

Sáu notebook FS dùng cùng builder iTransformer, thay đổi số biến K theo feature set:

- Input có dạng (batch, 22, K).
- Inverted Embedding hoán đổi trục thành K variable-token; Dense chiếu mỗi chuỗi 22 giá trị thành embedding 48 chiều.
- Encoder có 2 block; mỗi block có Multi-Head Attention 4 head, attention dropout 0,1, dropout sau attention 0,1, residual connection + LayerNormalization, Feed-Forward Dense(96, GELU) → Dropout(0,1) → Dense(48), rồi residual connection + LayerNormalization.
- Prediction head: Flatten(K × 48) → Dense(32, GELU) → Dropout(0,1) → Dense(1, linear).
- Optimizer: Adam, learning rate 0,0003, clipnorm 1,0. Loss: MSE.
- Batch size 64, tối đa 65 epochs, EarlyStopping theo val_loss với patience 9 và khôi phục trọng số tốt nhất. Seed 42.

| Feature set | K | Dạng Flatten | Số tham số theo model.summary() |
|---|---:|---:|---:|
| FS01_CU | 1 | 1 × 48 = 48 | 40.625 |
| FS02_PAPER_CU | 4 | 4 × 48 = 192 | 45.233 |
| FS03_PAPER_EXO | 3 | 3 × 48 = 144 | 43.697 |
| FS04_METALS_CU | 5 | 5 × 48 = 240 | 46.769 |
| FS05_METALS_EXO | 4 | 4 × 48 = 192 | 45.233 |
| FS06_WTI_CU | 4 | 4 × 48 = 192 | 45.233 |

FS01 chỉ có một variable-token. Khi K = 1, attention không thể học quan hệ giữa nhiều biến; vì vậy FS01 hữu ích như một ablation Copper-only nhưng không thể hiện lợi thế attention liên biến. Output notebook có cảnh báo softmax trên trục kích thước 1 ở cấu hình này. Cảnh báo đó không dừng quá trình chạy, nhưng cần được giải thích khi trình bày.

## 5. Cách đọc ba nhánh dự báo

**Direct** huấn luyện mạng để dự báo trực tiếp mức Copper đã chuẩn hóa tại t. **Residual** huấn luyện mạng dự báo chênh lệch giữa Copper chuẩn hóa ở t và t−1; code cộng dự báo chênh lệch vào Copper chuẩn hóa ở t−1 rồi inverse-transform về giá gốc. Về ý tưởng, có thể viết:

**ŷₜ = yₜ₋₁ + Δŷₜ**

Direct và Residual là hai lượt huấn luyện độc lập, không phải hai tầng nối tiếp. **Naive** không huấn luyện mạng; dự báo Copper(t) bằng Copper(t−1).

Tất cả metric giá được tính sau khi khôi phục về thang giá trong CSV. MAE, RMSE, MAPE càng thấp càng tốt; R² càng cao càng tốt. Directional Accuracy so dấu của thay đổi dự báo với dấu thay đổi thật. Vì Naive luôn dự báo không đổi, Directional Accuracy của Naive gần 0% trong bảng không đồng nghĩa với việc baseline dự báo mức giá kém: baseline này được thiết kế để giữ giá phiên trước.

## 6. So sánh sáu feature set trên cùng cấu hình test

Bảng sau trích từ output thật ở tỷ lệ 90/10, chiến lược Expanding; cùng có 2.173 mẫu train, 244 validation và 271 test. Các giá trị được làm tròn theo bảng notebook. “Skill Residual so với Naive” dương nghĩa là MAE Residual thấp hơn MAE Naive.

| Feature set | Input | Direct MAE | Direct RMSE | Residual MAE | Residual RMSE | Naive MAE | Residual skill so với Naive |
|---|---|---:|---:|---:|---:|---:|---:|
| FS01_CU | Copper | 1,75765 | 1,82462 | 0,07681 | 0,10097 | 0,07708 | +0,34374% |
| FS02_PAPER_CU | Copper, WTI, Gold, Silver | 1,14381 | 1,29730 | 0,07701 | 0,10232 | 0,07708 | +0,08760% |
| FS03_PAPER_EXO | WTI, Gold, Silver | 1,54427 | 1,62622 | 0,09027 | 0,11728 | 0,07708 | −17,11429% |
| FS04_METALS_CU | Copper, Aluminium, Tin, Silver, Gold | 1,63235 | 1,70891 | 0,07687 | 0,10045 | 0,07708 | +0,27188% |
| FS05_METALS_EXO | Aluminium, Tin, Silver, Gold | 1,81230 | 1,87994 | 0,07924 | 0,10419 | 0,07708 | −2,80427% |
| FS06_WTI_CU | Copper, Aluminium, Tin, WTI | 1,38313 | 1,48639 | 0,07794 | 0,10152 | 0,07708 | −1,12237% |

### 6.1. Nhận xét

1. Direct có MAE 1,14–1,81, trong khi Naive có MAE xấp xỉ 0,077. Trên lần chạy này, các feature set không cho thấy Direct học mức giá tốt hơn phép giữ giá phiên trước.
2. Residual gần Naive hơn nhiều. Ba cấu hình FS01, FS02 và FS04 có MAE thấp hơn baseline một lượng rất nhỏ; mức cải thiện lớn nhất trong bảng là khoảng 0,34%. FS03, FS05 và FS06 không vượt baseline theo MAE.
3. FS04 có RMSE Residual thấp nhất trong bảng 90/10–Expanding, còn FS01 có MAE Residual thấp nhất. Chênh lệch nhỏ và chỉ có một seed, nên không nên tuyên bố một bộ biến là tốt nhất.
4. Những quan sát này chỉ áp dụng cho split và chiến lược được chọn để trình bày. Chọn cấu hình tốt nhất sau khi xem test là phân tích hậu nghiệm và có selection bias. File CSV kèm theo lưu đủ 216 dòng để xem cả bốn tỷ lệ, ba chiến lược và ba nhánh.

Các biểu đồ giá và phần thay đổi đã lưu được trích riêng cho split 90/10–Expanding. Ví dụ: [FS01 giá](../outputs/figures/feature_sets/FS01_CU_prices_r0900_expanding.png), [FS01 thay đổi](../outputs/figures/feature_sets/FS01_CU_delta_r0900_expanding.png), [FS04 giá](../outputs/figures/feature_sets/FS04_METALS_CU_prices_r0900_expanding.png). Mỗi notebook vẫn giữ các biểu đồ đã lưu cho đủ 12 tổ hợp tỷ lệ × chiến lược.

## 7. Benchmark iTransformer với SA-LSTM

Notebook benchmark ở K = 5 dùng cùng dataset fingerprint a761… và có output xác nhận **84/84 job hoàn thành**. Với tỷ lệ 90/10–Expanding, bảng output báo cáo:

| Model | Objective | MAE trên test |
|---|---|---:|
| Naive | Copper(t−1) | 0,077077 |
| iTransformer | Direct | 1,768602 |
| iTransformer | Residual | 0,078119 |
| SA-LSTM cố định | Direct | 0,798291 |
| SA-LSTM cố định | Residual | 0,082928 |
| SA-LSTM tối ưu bằng SA | Direct | 0,562087 |
| SA-LSTM tối ưu bằng SA | Residual | 0,085346 |

So sánh Direct với Direct và Residual với Residual: trong cấu hình này, iTransformer có MAE thấp hơn hai SA-LSTM ở mục tiêu Residual, nhưng vẫn hơi cao hơn Naive. Kết quả Directional Accuracy của iTransformer Residual là 47,23%, thấp hơn SA-LSTM cố định Residual 49,08% trong cùng output. Bảng kiểm định HAC/DM-style đã lưu cho chênh lệch absolute error Residual giữa iTransformer và SA-LSTM cố định báo p = 0,024835; với SA-LSTM tuned báo p = 0,000142. Đây là kết quả của một giai đoạn test cụ thể, không chứng minh mô hình luôn tốt hơn trên các giai đoạn/dataset khác.

Không nên nhập các con số trên với bảng FS01–FS06: đây là notebook benchmark riêng, có SA-LSTM, số epoch tối đa 100 cho notebook benchmark thay vì 65 ở sáu notebook FS, và bảng FS báo metric đến độ chính xác khác nhau. Notebook benchmark cùng đường dẫn, các bảng lưu và execution log được liệt kê trong README.

## 8. Đối chiếu với các kết quả của Khoa

Báo cáo V20 trong repository ghi SHA-256 dataset là **9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600** ([báo cáo V20](../../Module1/copper-forecasting/results/17-11-Copper-NonSA-Fair-Full-Protocol-V20/BAO_CAO_DANH_GIA_17-11-COPPER_NON_SA_FAIR_FULL_PROTOCOL_V20.md), [file fingerprint](../../Module1/copper-forecasting/results/17-11-Copper-NonSA-Fair-Full-Protocol-V20/dataset_fingerprint.txt)), còn notebook iTransformer ghi **a761a908…**. Hai báo cáo cùng ghi 2.710 dòng và khoảng ngày 13/01/2016–13/11/2026, nhưng khác fingerprint. Chỉ trùng số dòng và ngày đầu/cuối không chứng minh các giá trị ở từng ngày giống nhau.

Do chưa có CSV chuẩn và file dự báo gốc của cả hai nhánh để ghép theo timestamp, báo cáo không xếp hạng iTransformer với các mô hình Khoa và không gộp số liệu vào một bảng “fair comparison”. Để thực hiện đối chiếu, cần xác định cùng một CSV canonical, xác nhận fingerprint, dùng cùng các ngày test, target, objective, đơn vị metric và quy trình dự báo; sau đó mới tạo bảng paired errors và kiểm định thống kê.

## 9. Kiểm kê artifact và giới hạn tái lập

- Sáu notebook FS có bảng kết quả 36 dòng và ghi 10.377 prediction records mỗi notebook. Log FS02 có 36 job SKIP và FS06 có 1 job SKIP; code dùng checkpoint đã xác thực theo run key, số dòng, model và timestamp. Các bảng kết quả cuối vẫn có đủ 36 cấu hình.
- Notebook benchmark có output ghi 84/84 job hoàn thành, cùng các bảng benchmark và biểu đồ đã lưu.
- Các notebook nguồn được sao chép nguyên byte, gồm output cell, vào Vi Task/notebooks. Một bản benchmark đã có sẵn ở Vi Task/iTransformer_vs_SA_LSTM_K5_Fair_FINAL_Colab.ipynb.
- CSV/ZIP Kaggle gốc chứa predictions, histories, manifests và full-precision metrics được notebook chỉ đến đường dẫn /kaggle/working hoặc Google Drive; các tệp đó không có trong thư mục nguồn máy tính. Vì vậy không thể tuyên bố đã đưa chúng vào GitHub. Bảng FS được xuất ở đây là bản trích từ bảng hiển thị, không thay thế raw predictions hoặc ZIP.
- Không thêm dataset, checkpoint mô hình hay ZIP trùng lặp vào repository.

## 10. Kết luận

Thực nghiệm cho thấy cách đặt bài toán có ảnh hưởng lớn: iTransformer Direct dự báo mức Copper không đạt baseline Naive trong các output đã kiểm tra; Residual bám baseline tốt hơn nhưng mức cải thiện nhỏ và chưa ổn định qua các feature set. Vì vậy, kết luận phù hợp là **chưa có bằng chứng mạnh rằng iTransformer cải thiện dự báo trên dataset và quy trình hiện tại**. Các bước tiếp theo nên ưu tiên lấy lại raw predictions/ZIP từ Kaggle, thống nhất fingerprint dataset với nhánh Khoa, và đánh giá nhiều seed hoặc nhiều đoạn test theo thời gian trước khi chọn feature set hay tuyên bố ưu thế.
