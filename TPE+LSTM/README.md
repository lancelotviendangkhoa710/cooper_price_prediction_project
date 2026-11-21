# Tổng kết thí nghiệm SA–LSTM và TPE–LSTM

Tài liệu này tổng hợp những gì **đã được thực hiện và có kết quả lưu trong thư mục `TPE+LSTM`**. Mã thực nghiệm nằm ở [`final-tpe-lstm.ipynb`](final-tpe-lstm.ipynb). Bộ kết quả được đối chiếu là lần chạy `FULL_MODE` tại [`results/copper_sa_vs_tpe_six_feature_sets_no_naive/20261121T004726844166Z/`](results/copper_sa_vs_tpe_six_feature_sets_no_naive/20261121T004726844166Z/). Ngoài ra có [file ZIP](results/copper_sa_vs_tpe_six_feature_sets_no_naive/20261121T004726844166Z_results.zip) chứa bộ kết quả của lần chạy này.

## Mục tiêu và phạm vi

Thí nghiệm dự báo giá Copper phiên kế tiếp (`horizon = 1`) bằng LSTM, so sánh **Simulated Annealing (SA)** và **Tree-structured Parzen Estimator (TPE)** để chọn siêu tham số. Hai phương pháp dùng cùng kiến trúc mạng, mục tiêu học residual, cách chia dữ liệu và ngân sách tìm kiếm; khác nhau ở thuật toán tìm siêu tham số.

Sáu bộ biến đầu vào cố định đã được thử:

| Mã | Biến đưa vào mạng |
| --- | --- |
| FS01 | Copper |
| FS02 | Copper, WTI, Gold, Silver |
| FS03 | WTI, Gold, Silver |
| FS04 | Copper, Aluminium, Tin, Silver, Gold |
| FS05 | Aluminium, Tin, Silver, Gold |
| FS06 | Copper, Aluminium, Tin, WTI |

FS03 và FS05 không đưa Copper vào *đầu vào mạng*, nhưng phép hoàn nguyên residual vẫn dùng giá Copper thật của phiên trước. Vì vậy không thể diễn giải hai bộ này là hoàn toàn không sử dụng lịch sử Copper. Các bộ biến được định nghĩa trước, không được chọn tự động bằng Spearman.

## Những việc đã hoàn thành

1. **Chuẩn bị và kiểm tra dữ liệu.** Đọc dữ liệu ngày của 9 hàng hóa, làm sạch trên hợp các cột cần cho sáu bộ biến để mọi bộ dùng cùng ngày quan sát. Bản ghi kiểm tra cho thấy 2.710 dòng từ **13/01/2016 đến 13/11/2026**, không có dòng bị loại trong lần chạy này. Lưu bản chụp dữ liệu, thông tin chất lượng và mã băm SHA-256 trong kết quả.
2. **Tạo mẫu dự báo theo thời gian.** Dùng cửa sổ 22 phiên, chỉ chứa quan sát đến `t−1`, để dự báo Copper tại `t`. Tách train, validation và test theo thứ tự thời gian; validation chiếm 10% phần dữ liệu trước test. `MinMaxScaler` cho đầu vào và Copper chỉ được fit trên train. Notebook có các kiểm tra về thứ tự thời gian, hình dạng mẫu, scaler và phép hoàn nguyên residual.
3. **Xây dựng mô hình chung.** Hai tầng LSTM với dropout 0,20 sau mỗi tầng, Dense(32, ReLU) và đầu ra tuyến tính; tối ưu bằng Adam, loss MSE. Mạng học `Copper(t) − Copper(t−1)` trên thang đã chuẩn hóa. Khi đánh giá, cộng lại Copper phiên trước rồi đổi về thang giá gốc.
4. **Tìm siêu tham số độc lập cho từng bộ biến.** SA và TPE đều tìm `units_1` trong 16–96, `units_2` trong 24–160 và learning rate trong `1e−5`–`3e−3`. Tìm kiếm diễn ra trên train/validation của bối cảnh **80/20 expanding**, với **16 trial cho mỗi phương pháp và mỗi bộ biến**. Kết quả cấu hình tốt nhất được khóa trước khi đánh giá test và chuyển sang các bối cảnh còn lại.
5. **Huấn luyện và đánh giá đủ sáu bối cảnh.** Các tỷ lệ dữ liệu trước test là 80%, 90%, 92,5% và 95%; tỷ lệ 90% có thêm `rolling_756` và `recency_weighted`, tạo thành sáu bối cảnh. Chạy với seed **42**, tổng cộng **192 search fits + 72 final fits**, tương ứng **36 cặp SA/TPE**. Hai mô hình trong mỗi cặp được kiểm tra là có cùng ngày test, giá thật và giá Copper phiên trước.
6. **Lưu kết quả để kiểm tra lại.** Xuất MAE, RMSE, MSE, MASE, R², directional accuracy, thống kê biến động giá và dự báo theo từng ngày; kèm thông số tốt nhất, lịch sử trial, loss, mô hình `.keras`, scaler và biểu đồ.

`rolling_756` ở đây dùng cửa sổ train 756 phiên tại mỗi lần chia; `recency_weighted` gán trọng số lớn hơn cho quan sát gần đây. Các mô hình **không được tái huấn luyện định kỳ trong giai đoạn test**. Việc tìm kiếm dùng tối đa 15 epoch/trial; lần huấn luyện cuối cho phép tối đa 100 epoch, có early stopping.

## Kết quả đã ghi nhận

Bảng dưới là **MAE trung bình của sáu bối cảnh** cho mỗi bộ biến và phương pháp, tính lại từ [`metrics_by_run.csv`](results/copper_sa_vs_tpe_six_feature_sets_no_naive/20261121T004726844166Z/metrics_by_run.csv). MAE thấp hơn là tốt hơn. Các tập test theo tỷ lệ chia có phần thời gian chồng lấn; trung bình này chỉ để mô tả các lần chạy, không phải đánh giá trên sáu mẫu độc lập.

| Bộ biến | SA–LSTM | TPE–LSTM | TPE có MAE thấp hơn ở |
| --- | ---: | ---: | ---: |
| FS01 | **0,07462** | 0,07618 | 0/6 bối cảnh |
| FS02 | 0,08655 | **0,07710** | 6/6 |
| FS03 | 0,08435 | **0,07578** | 4/6 |
| FS04 | **0,07675** | 0,09648 | 2/6 |
| FS05 | **0,08000** | 0,08401 | 3/6 |
| FS06 | **0,07554** | 0,07734 | 1/6 |

Trên cả **36 cặp**, TPE có MAE thấp hơn trong **16** cặp, SA thấp hơn trong **20** cặp. Nếu lấy trung bình đều 36 lần chạy của mỗi phương pháp, MAE là khoảng **0,0796** cho SA và **0,0811** cho TPE. MAE nhỏ nhất trong các dòng kết quả là **0,06883**, thuộc **FS01 + SA, tỷ lệ 80/20 expanding**. Các số này không chứng minh một phương pháp luôn tốt hơn phương pháp kia; kết quả thay đổi theo bộ biến và bối cảnh.

## Vị trí các tệp quan trọng

Trong [thư mục kết quả của lần chạy](results/copper_sa_vs_tpe_six_feature_sets_no_naive/20261121T004726844166Z/):

| Tệp | Nội dung |
| --- | --- |
| `manifest.json`, `data_quality.json`, `dataset_snapshot.csv` | Cấu hình chạy, phiên bản thư viện, dấu vết và dữ liệu dùng cho thí nghiệm. |
| `data_split_summary.csv` | Số mẫu và khoảng ngày train/validation/test theo bộ biến, tỷ lệ và cách huấn luyện. |
| `all_search_trials.csv`, `search_comparison.csv`, `locked_best_hyperparameters.json` | Toàn bộ 192 trial, tổng hợp tìm kiếm và siêu tham số đã khóa. |
| `final_training.csv`, `final_model_recipes.json` | Thông tin 72 lần huấn luyện cuối và công thức mô hình. |
| `metrics_by_run.csv`, `paired_comparison_SA_vs_TPE_by_mode.csv`, `feature_set_comparison_mae.csv` | Chỉ số từng lần chạy, so sánh theo cặp và bảng MAE giữa các bộ biến. |
| `test_predictions.csv` | Giá thật, giá dự báo, giá phiên trước và delta theo ngày test. |
| `FS01/` đến `FS06/` | Mô hình, scaler, loss và biểu đồ riêng từng bộ biến. |

## Cách chạy lại và giới hạn diễn giải

Mở notebook, đặt `CSV_PATH` đến [`data/processed/dataset_9_commodities_daily.csv`](../data/processed/dataset_9_commodities_daily.csv) nếu chạy tại máy cục bộ, giữ `MODE = 'FULL_MODE'`, rồi chạy các cell theo thứ tự. Đường dẫn Kaggle trong `manifest.json` là đường dẫn **của lần chạy đã lưu**, không phải đường dẫn bắt buộc khi chạy lại. Lưu ý: SHA-256 của CSV cục bộ là `9580ede12d474c8f990556842d551dac176b3c16b2d6a4bffb7dc41dd54bf600`, khác `dataset_sha256` (`a761a9082de36ad8724597ff1518e24bbf1a61676bd3164676e7f4218d275d83`) trong manifest. Vì vậy chưa thể khẳng định đầu vào cục bộ giống hệt theo từng byte; [`dataset_snapshot.csv`](results/copper_sa_vs_tpe_six_feature_sets_no_naive/20261121T004726844166Z/dataset_snapshot.csv) lưu sáu cột đã làm sạch để đối chiếu giá trị. Notebook có nhánh cài `optuna` khi môi trường chưa có thư viện này; các phiên bản Python và thư viện của lần chạy được ghi trong manifest.

Kết quả hiện chỉ có **một seed (42)** nên chưa đánh giá độ ổn định theo khởi tạo ngẫu nhiên; độ lệch chuẩn giữa các seed trong `summary_mean_std.csv` không có ý nghĩa. Không có mô hình naive riêng trong lần chạy này; MASE dùng độ biến thiên tuyệt đối trung bình một bước của train làm mẫu số. Vì vậy cần chạy thêm seed và một bộ test độc lập nếu muốn kết luận chắc hơn về phương pháp hoặc chọn bộ biến để triển khai. Phần mô tả đầu notebook còn câu nói chưa có kết quả/đang mặc định QUICK_MODE từ lúc soạn; **cấu hình thực chạy phải đọc từ `manifest.json`**, xác nhận là `FULL_MODE`.
