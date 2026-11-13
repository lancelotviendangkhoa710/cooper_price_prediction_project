# XGBoost dự báo giá đồng

Notebook: [xgboost_copper_forecast.ipynb](xgboost_copper_forecast.ipynb)  
Dữ liệu: [copper_8_indicators_2006_2026.csv](copper_8_indicators_2006_2026.csv)

Notebook dự báo giá Copper ở **phiên giao dịch kế tiếp**, sử dụng giá đồng, volume, các chỉ báo trong CSV và đặc trưng trễ. Nó áp dụng cách chia theo thời gian của `copper-lstm-sa_result.ipynb`: train/test 90/10, 92.5/7.5, 95/5; lấy 10% cuối của train làm validation; không shuffle. Test tương lai không tham gia chọn số cây.

## Vì sao chọn các tham số này? (trước khi train)

Đây là bộ **giá trị khởi đầu có regularization**, không phải kết quả tối ưu hóa đã được chứng minh. Dataset có khoảng 5.2 nghìn phiên, vì vậy cấu hình giới hạn độ phức tạp để tránh XGBoost ghi nhớ nhiễu từ các đặc trưng trễ.

| Tham số | Giá trị | Lý do chọn |
|---|---:|---|
| `objective` | `reg:squarederror` | Copper là giá liên tục; hàm mất mát bình phương phù hợp với RMSE được báo cáo. |
| `learning_rate` | `0.03` | Bước cập nhật tương đối nhỏ để nhiều cây cộng dồn ổn định, giảm nguy cơ học quá nhanh. |
| `n_estimators` | tối đa `1200` | Cung cấp đủ cây khi learning rate nhỏ. Early stopping sẽ dừng trước giới hạn nếu validation không còn cải thiện. |
| `early_stopping_rounds` | `60` | Bỏ qua dao động validation ngắn; dừng sau 60 vòng liên tiếp không tốt lên. |
| `max_depth` | `4` | Cây nông hạn chế các tương tác phức tạp, vốn dễ khớp nhiễu với dữ liệu vài nghìn điểm. |
| `min_child_weight` | `5` | Đòi hỏi đủ trọng lượng mẫu ở nhánh trước khi tách, tránh lá được quyết định bởi quá ít phiên. |
| `subsample` | `0.8` | Mỗi cây lấy ngẫu nhiên 80% hàng train, giúp giảm phụ thuộc giữa các cây. |
| `colsample_bytree` | `0.8` | Mỗi cây lấy 80% cột, hạn chế mô hình dựa quá nhiều vào một biến/nhóm biến. |
| `reg_alpha` | `0.05` | L1 regularization nhẹ, có thể loại bớt đóng góp nhỏ không hữu ích. |
| `reg_lambda` | `5.0` | L2 regularization tương đối mạnh để làm leaf score thận trọng hơn trên chuỗi nhỏ. |
| `gamma` | `0.0` | Không thêm ngưỡng phạt tách cây ban đầu; độ phức tạp đã được khống chế bởi depth, child weight, sampling và L1/L2. |
| `tree_method`, `device` | `hist`, `cuda:0` | Huấn luyện histogram trên GPU NVIDIA đầu tiên. |
| `random_state` | `42` | Cố định random sampling để dễ tái lập. |

Notebook dành riêng validation từ cuối phần train để early stopping; không dùng test để điều chỉnh tham số. XGBoost dạng cây không cần chuẩn hóa feature như LSTM. Sau khi xem kết quả có thể tinh chỉnh các giá trị trên bằng validation hoặc walk-forward validation, nhưng không chọn tham số dựa trên test cuối.

## Đặc trưng và chống rò rỉ dữ liệu

Mỗi hàng đầu vào chỉ dùng thông tin tới ngày *t*; nhãn là giá đồng ở ngày giao dịch kế tiếp. Đặc trưng gồm Copper lags (0, 1, 2, 3, 5, 10, 20), rolling mean/std 5/10/20 phiên, độ trễ 0/1/5 và phần trăm thay đổi 1/5 phiên của chỉ báo vĩ mô, volume trễ và thành phần mùa vụ. Cửa sổ cuộn chỉ gồm dữ liệu hiện tại và quá khứ. Không chuẩn hóa và không fit preprocessing trên test.

Notebook báo RMSE, MAE, MAPE theo USD/lb và RMSE persistence: dự báo giá ngày mai bằng giá hôm nay. Kết quả được lưu thành `xgboost_copper_metrics.csv` và một CSV dự báo test cho mỗi tỷ lệ chia.

## Tìm cấu hình tốt nhất

Notebook có thêm phần grid search đầy đủ cho mô hình dự báo biến động giá một phiên. Nó thử toàn bộ **9.216** tổ hợp sau bằng 3 validation folds theo thứ tự thời gian, tương đương 27.648 lần fit validation:

| Tham số | Giá trị được thử |
|---|---|
| `max_depth` | 2, 3, 4, 5 |
| `min_child_weight` | 5, 10, 20 |
| `learning_rate` | 0.01, 0.03, 0.05 |
| `subsample`, `colsample_bytree` | 0.7, 0.8, 0.9, 1.0 |
| `reg_lambda` | 1, 5, 10, 20 |
| `reg_alpha` | 0, 0.05, 0.5, 1.0 |

Mỗi lần fit dùng tối đa 250 cây và early stopping 30 vòng. Grid search GPU ghi tiến độ sau mỗi 10 cấu hình vào `xgboost_grid_search_delta_cuda0.csv`; chạy lại cell sẽ bỏ qua phần đã hoàn tất. File `xgboost_grid_search_delta.csv` cũ là kết quả CPU và không được dùng lại cho lần chạy GPU. Sau khi grid search xong, cell tiếp theo dùng cấu hình có RMSE validation trung bình thấp nhất để fit lại trên toàn bộ 90% dữ liệu đầu và chỉ khi đó đánh giá 10% dữ liệu test cuối.

Target của phần này là `Copper(t+1) - Copper(t)` thay vì giá tuyệt đối. Giá dự báo cuối cùng bằng `Copper(t) + dự báo biến động`. Cách biểu diễn này phù hợp hơn khi mức giá cuối chuỗi cao hơn các mức giá đã thấy trong lịch sử huấn luyện.

## Chạy

Cần Python cùng `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost` bản hỗ trợ CUDA, `jupyter` và GPU NVIDIA. Notebook kiểm tra GPU ngay ở cell đầu và báo lỗi nếu XGBoost chuyển về CPU. Mở notebook trong thư mục này và chọn **Run All**. Kết quả được ghi vào thư mục làm việc hiện tại.

## Giới hạn

Giá đồng trong CSV là proxy hợp đồng tương lai Yahoo. Đánh giá một phiên không tính phí giao dịch và không đảm bảo hiệu quả đầu tư. Các chỉ báo được giả định có sẵn tại thời điểm dự báo. Đây chưa phải backtest chiến lược. Metric khác nghiên cứu LSTM nên không so sánh trực tiếp với MSE của bài báo.
