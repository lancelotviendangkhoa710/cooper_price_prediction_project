# ĐÁNH GIÁ CHIẾN LƯỢC DỰ ĐOÁN GIÁ COPPER — 15-11

## Tóm tắt

Báo cáo này so sánh hai cách xây dựng tập huấn luyện theo thời gian (expanding và rolling), kiểm tra lại bảy mô hình ngày 14-11 trên cùng các khoảng test, bổ sung Ridge dự đoán phần dư và seasonal naive lag 5, đồng thời tìm kiếm CNN-LSTM/CNN-GRU. Ba mốc train/test là 90/10, 92,5/7,5 và 95/5. Notebook không lưu outputs; các số liệu báo cáo được lấy từ bốn CSV kết quả trong thư mục này.

Ridge có MSE thấp nhất trong từng chế độ và từng mốc khi tính trên giá gốc: expanding lần lượt 0,009247; 0,008811; 0,006234, còn rolling là 0,006982; 0,008522; 0,007751. Ridge rolling tốt hơn expanding ở 90/10 và 92,5/7,5, nhưng kém hơn ở 95/5. Persistence đứng thứ hai; seasonal lag 5 kém persistence rõ rệt. Kết quả ủng hộ Ridge trong phạm vi các tập test đã quan sát, nhưng Ridge ở đây là ước lượng cùng ngày (nowcast), không phải dự báo giá ngày kế tiếp.

CNN được đánh giá qua ba seed 17, 42, 91; bảng dùng trung bình và khoảng min–max, không chọn seed có kết quả thuận lợi nhất. Hai CNN có MSE_scaled đặc biệt cao ở expanding 90/10. MSE_scaled không dùng để so sánh trực tiếp expanding với rolling vì scaler được fit riêng cho từng tập train.

## I. Mục tiêu và phạm vi

Thí nghiệm ngày 15-11 tiếp nối đánh giá ngày 14-11 bằng cách kiểm tra ảnh hưởng của chế độ cửa sổ huấn luyện, lựa chọn biến theo từng mốc thời gian và hai kiến trúc CNN. Các dự đoán của bảy mô hình ngày 14-11 được tái sử dụng làm đối chứng; chúng **không được huấn luyện lại**. Trong nhánh rolling, dự đoán đã có chỉ được đổi thang theo scaler fit trên train rolling để tính chỉ số cùng thang với các mô hình mới. Vì vậy, đây không phải thử nghiệm rolling retrain cho các mô hình cũ.

Dữ liệu có 2.710 quan sát Copper từ 13/01/2016 đến 13/11/2026. Bài toán trong notebook căn chỉnh cửa sổ 22 phiên để ước lượng Copper(t), sử dụng đặc trưng hàng hóa cùng ngày t. Đây là ước lượng cùng ngày/nowcast, không phải dự báo t+1. Ridge dự đoán delta Copper(t)−Copper(t−1) rồi cộng phần dư ước lượng vào Copper(t−1); các đặc trưng cũng bao gồm thông tin cùng ngày t. Persistence lấy Copper(t−1). Cả hai do đó là đối chứng cho ước lượng Copper(t) trong cách căn chỉnh này, không nên diễn giải như mô hình dự báo tương lai tại thời điểm t−1.

## II. Thiết kế thực nghiệm

### 2.1. Chia tập và chọn đặc trưng

Các mốc test được giữ như báo cáo trước: 90/10 có 271 điểm test (29/10/2025–13/11/2026), 92,5/7,5 có 204 điểm (30/01/2026–13/11/2026), 95/5 có 136 điểm (07/05/2026–13/11/2026). Đây là các khoảng lồng nhau, không phải ba lần lặp độc lập. Với expanding, vùng dữ liệu trước test mở rộng từ đầu chuỗi; với rolling, vùng này gồm 756 hàng gần nhất rồi chia 90% train và 10% validation. Lookback là 22 phiên.

Bốn biến được chọn bằng tương quan Spearman chỉ trên train của từng mốc và chế độ. Expanding chọn Aluminium, Tin, Nickel, WTI ở hai mốc 90/10 và 92,5/7,5; tại 95/5 thay WTI bằng Silver. Rolling chọn Tin, Silver, Gold, Aluminium ở cả ba mốc. Điều này cho thấy tập biến đã thay đổi theo đoạn train/chế độ, nhưng tự nó không chứng minh nguyên nhân làm hiệu năng khác nhau.

| Tỷ lệ | Chế độ | Bốn biến được chọn |
|---|---|---|
| 90/10 | Expanding | Aluminium, Tin, Nickel, WTI |
| 90/10 | Rolling | Tin, Silver, Gold, Aluminium |
| 92,5/7,5 | Expanding | Aluminium, Tin, Nickel, WTI |
| 92,5/7,5 | Rolling | Tin, Silver, Gold, Aluminium |
| 95/5 | Expanding | Aluminium, Tin, Nickel, Silver |
| 95/5 | Rolling | Silver, Tin, Gold, Aluminium |

### 2.2. Mô hình và quy trình

Bảy mô hình từ 14-11 gồm persistence, LSTM, Paper SA-LSTM, Reoptimized SA-LSTM, GRU, CNN1D-LSTM và SA-GRU. Các tệp dự đoán cũ được nạp và căn cùng ngày test. Mô hình Ridge mới hồi quy delta Copper; seasonal naive lag 5 dùng giá Copper cách năm phiên làm dự đoán tham chiếu.

CNN-LSTM và CNN-GRU được tìm kiếm trên 16 cấu hình cho mỗi tổ hợp tỷ lệ × chế độ. Tổng cộng có 192 lượt fit ứng viên (16 × 2 kiến trúc × 6 tổ hợp) và 36 lượt fit cuối (2 × 6 × 3 seed). MSE validation đã chuẩn hóa dùng để chọn cấu hình; test không tham gia chọn tham số. Sau đó cấu hình được chọn huấn luyện cuối trên train + validation đủ 100 epoch với ba seed 17, 42, 91. Thời gian được ghi trong CSV là 786,2 giây tìm kiếm + 487,6 giây fit cuối cho CNN-LSTM, và 713,3 giây tìm kiếm + 480,8 giây fit cuối cho CNN-GRU. Đây là tổng thời gian fit được ghi, không phải wall-clock toàn notebook. Quy trình CNN này khác với các mô hình ngày 14-11, vốn dùng quy trình/thiết lập riêng; không nên xem các so sánh là một phép thử đồng nhất về huấn luyện.

### 2.3. Thang đo

CSV `metrics_scaled.csv` báo MSE trên thang MinMax được fit riêng trên train của từng tỷ lệ và chế độ. Vì scaler expanding và rolling khác nhau, các MSE_scaled giữa hai chế độ **không so sánh trực tiếp**. Để so sánh chế độ, MSE được tính lại từ `actual` và `prediction` trong `predictions_comparison.csv`, tức trên thang giá gốc. Đơn vị gốc của dataset chưa được xác nhận trong báo cáo; do đó không gán USD hoặc đơn vị tiền tệ khác.

## III. Kết quả

### 3.1. So sánh trên thang giá gốc

Bảng dưới dùng MSE bình phương của sai số `actual − prediction` trên cùng các điểm test. Ba mốc test chồng lấn, nên không xem các cột như ba mẫu độc lập.

| Mô hình | Expanding 90/10 | Rolling 90/10 | Expanding 92,5/7,5 | Rolling 92,5/7,5 | Expanding 95/5 | Rolling 95/5 |
|---|---:|---:|---:|---:|---:|---:|
| Ridge residual delta | **0,009247** | **0,006982** | **0,008811** | **0,008522** | **0,006234** | **0,007751** |
| Persistence | 0,010208 | 0,010208 | 0,009682 | 0,009682 | 0,009334 | 0,009334 |
| Seasonal naive lag 5 | 0,037009 | 0,037009 | 0,040273 | 0,040273 | 0,040891 | 0,040891 |

Ridge đứng đầu ở cả sáu tổ hợp. So với persistence, Ridge giảm MSE khoảng 9,4% ở expanding 90/10, 9,0% ở expanding 92,5/7,5 và 33,2% ở expanding 95/5. Ở rolling, mức giảm tương ứng khoảng 31,6%, 12,0% và 17,0%. Đây là mô tả so sánh trên đúng các tập test, không phải bằng chứng về hiệu quả ở các giai đoạn khác.

Rolling làm Ridge giảm MSE khoảng 24,5% so với expanding ở 90/10 và 3,3% ở 92,5/7,5; tại 95/5, MSE rolling cao hơn khoảng 24,3%. Persistence hầu như trùng nhau giữa hai chế độ trên giá gốc, do cùng chuỗi dự đoán và cùng điểm test; seasonal lag 5 cũng như vậy. Kết quả không cho thấy rolling thắng nhất quán.

Seasonal lag 5 có MSE khoảng 3,6–4,4 lần persistence tùy mốc. Vì vậy, trong thử nghiệm này, dự đoán mùa vụ đơn giản theo năm phiên trước không cải thiện baseline persistence.

### 3.2. MSE_scaled của Ridge và persistence

Các số dưới đây được giữ theo từng scaler tương ứng; dùng để đối chiếu mô hình trong cùng chế độ/tỷ lệ, không dùng để suy ra chế độ tốt hơn chế độ kia.

| Mô hình | Expanding 90/10 | Expanding 92,5/7,5 | Expanding 95/5 | Rolling 90/10 | Rolling 92,5/7,5 | Rolling 95/5 |
|---|---:|---:|---:|---:|---:|---:|
| Ridge residual delta | **0,000925** | **0,000881** | **0,000581** | **0,001222** | **0,001657** | **0,001230** |
| Persistence | 0,001021 | 0,000968 | 0,000870 | 0,001786 | 0,001883 | 0,001481 |

Trong từng ô, Ridge vẫn thấp hơn persistence. Giá trị MSE_scaled chênh giữa expanding và rolling một phần phản ánh scaler khác nhau, nên kết luận về chế độ dựa trên bảng giá gốc ở mục 3.1.

### 3.3. CNN qua ba seed

Bảng ghi trung bình MSE_scaled và khoảng min–max của ba seed cho mỗi tỷ lệ/chế độ. Đây là số trên scaler riêng từng train; so sánh CNN với các mô hình khác chỉ nên thực hiện trong cùng chế độ và tỷ lệ. Giá trị giữa expanding và rolling không cùng thang đo.

| Tỷ lệ | Chế độ | CNN-LSTM: mean (min–max) | CNN-GRU: mean (min–max) |
|---|---|---:|---:|
| 90/10 | Expanding | 0,304580 (0,272888–0,325373) | 0,321406 (0,315276–0,325233) |
| 90/10 | Rolling | 0,184076 (0,137661–0,237417) | 0,099308 (0,029491–0,149639) |
| 92,5/7,5 | Expanding | 0,149405 (0,131293–0,170040) | 0,119998 (0,078615–0,151132) |
| 92,5/7,5 | Rolling | 0,076618 (0,061279–0,095924) | 0,062663 (0,033136–0,094357) |
| 95/5 | Expanding | 0,087167 (0,074863–0,095696) | 0,078944 (0,066183–0,085800) |
| 95/5 | Rolling | 0,092656 (0,087964–0,096779) | 0,075151 (0,071951–0,077223) |

Hai CNN đều có MSE_scaled rất cao tại expanding 90/10; mức trung bình cao hơn nhiều so với persistence và Ridge trong cùng ô. Ở các ô còn lại, CNN vẫn không vượt persistence hay Ridge theo MSE_scaled. Độ trải giữa seed đáng kể ở một số trường hợp, nhất là rolling 90/10 CNN-GRU, nên không nên chọn riêng seed 42 hay seed nào khác để đại diện. Đây là một lần chạy gồm ba seed cho CNN, chưa phải đánh giá lặp độc lập trên nhiều cửa sổ walk-forward.

### 3.4. Kết quả test của bảy mô hình ngày 14-11

Bảng đầu là **MSE trên thang giá gốc** tính lại từ các dự đoán đã lưu, trên cùng ngày test của từng mốc. Mỗi mô hình có một chuỗi dự đoán cho mỗi mốc; vì vậy kết quả giá gốc giống nhau ở expanding và rolling (khác biệt tính toán rất nhỏ do đổi scaler). Giá trị Ridge và persistence tương ứng có trong bảng 3.1 để đối chiếu.

| Mô hình 14-11 | 90/10 (271 điểm) | 92,5/7,5 (204 điểm) | 95/5 (136 điểm) |
|---|---:|---:|---:|
| Naive persistence | **0,010208** | **0,009682** | **0,009334** |
| LSTM | 0,894793 | 1,492472 | 0,795625 |
| Paper SA-LSTM | 0,510056 | 0,437793 | 0,375335 |
| Reoptimized SA-LSTM | 1,340949 | 0,736595 | 2,349559 |
| GRU | 0,276846 | 0,499064 | 0,509255 |
| CNN1D-LSTM | 1,049356 | 1,366580 | 1,129441 |
| SA-GRU | 0,799990 | 0,355595 | 0,558512 |

Bảng tiếp theo ghi **MSE_scaled trên test** từ `metrics_scaled.csv`. Mỗi cột dùng scaler của train tương ứng; chỉ so sánh các mô hình trong cùng cột.

| Mô hình 14-11 | Expanding 90/10 | Expanding 92,5/7,5 | Expanding 95/5 | Rolling 90/10 | Rolling 92,5/7,5 | Rolling 95/5 |
|---|---:|---:|---:|---:|---:|---:|
| Naive persistence | **0,001021** | **0,000968** | **0,000870** | **0,001786** | **0,001883** | **0,001481** |
| LSTM | 0,089467 | 0,149226 | 0,074180 | 0,156583 | 0,290276 | 0,126237 |
| Paper SA-LSTM | 0,050998 | 0,043773 | 0,034994 | 0,089257 | 0,085148 | 0,059552 |
| Reoptimized SA-LSTM | 0,134076 | 0,073649 | 0,219060 | 0,234658 | 0,143263 | 0,372791 |
| GRU | 0,027681 | 0,049899 | 0,047480 | 0,048446 | 0,097065 | 0,080801 |
| CNN1D-LSTM | 0,104921 | 0,136639 | 0,105303 | 0,183631 | 0,265791 | 0,179202 |
| SA-GRU | 0,079988 | 0,035554 | 0,052073 | 0,139993 | 0,069161 | 0,088616 |

Naive persistence có MSE thấp nhất trong bảy mô hình tại mọi mốc. Trong sáu mạng học sâu, GRU tốt nhất ở 90/10, SA-GRU ở 92,5/7,5 và Paper SA-LSTM ở 95/5. Ridge mới vẫn có MSE thấp hơn naive persistence theo bảng 3.1. Bảy mô hình cũ **không được fit lại trên rolling**: notebook chỉ đổi thang các dự đoán đã có. Vì thế MSE_scaled rolling ở đây không đo lợi ích của việc tái huấn luyện bằng cửa sổ rolling.

## IV. Giới hạn diễn giải

1. **Notebook không lưu outputs.** Các số liệu và bảng được lập từ bốn CSV xuất sẵn (`metrics_scaled.csv`, `predictions_comparison.csv`, `cnn_parameter_search.csv`, `feature_selection_by_split.csv`): metrics có 96 dòng, predictions có 19.552 dòng. Notebook mô tả quy trình nhưng không có đầu ra cell được lưu để tái kiểm chứng trực tiếp từ lần chạy đó.
2. **Nowcast, không phải dự báo trước.** Cả mục tiêu Copper(t) lẫn delta Copper(t) dùng đặc trưng cùng ngày t. Khi chưa chứng minh các biến cùng ngày có sẵn tại thời điểm ra quyết định, không thể xem đây là dự báo vận hành t+1.
3. **MSE_scaled phụ thuộc scaler.** Không so sánh trực tiếp expanding với rolling trên chỉ số chuẩn hóa; so sánh chế độ dùng giá gốc trong predictions CSV. Không gán đơn vị tiền tệ cho giá gốc vì nguồn hiện có chưa xác nhận đơn vị.
4. **CNN chỉ có ba seed trong một lần chạy.** Grid được chọn theo validation, rồi fit cuối train+validation đủ 100 epoch, không early stopping. Điều này khác quy trình các mô hình 14-11; kết quả không bao hàm biến thiên theo nhiều mốc walk-forward hoặc nhiều lần tìm kiếm.
5. **Tập test lồng nhau.** Các mốc 90/10, 92,5/7,5, 95/5 cùng kết thúc ngày 13/11/2026 nhưng có nhiều ngày chung, nên không phải ba phép lặp độc lập. Việc quan sát rolling có lợi ở hai mốc đầu và bất lợi ở mốc cuối không xác định nguyên nhân.
6. **Đối chứng mô hình cũ chỉ là tái sử dụng dự đoán.** So sánh rolling của bảy mô hình 14-11 đánh giá chuyển scaler, không phải tái huấn luyện rolling.

## V. Kết luận và hướng tiếp theo

Trong các điểm test và cách căn chỉnh hiện tại, Ridge residual delta đạt MSE thấp nhất trong cả hai chế độ và ba tỷ lệ; rolling có ưu thế tại 90/10, 92,5/7,5 nhưng không tại 95/5. Persistence là baseline hữu ích; seasonal lag 5 kém hơn rõ. CNN có độ biến động seed và kết quả expanding 90/10 yếu, chưa có cơ sở để coi CNN là lựa chọn ổn định hơn.

Trước khi diễn giải như một hệ thống dự báo, cần xác định thời điểm khả dụng của từng biến và chuyển nhãn sang Copper(t+1) nếu mục tiêu là dự báo ngày kế tiếp. Nên tiếp tục đánh giá bằng walk-forward trên các origin độc lập hơn, báo cáo thêm MAE/RMSE ở thang giá gốc, và chỉ huấn luyện lại các đối chứng khi thiết kế thử nghiệm yêu cầu. Việc chọn expanding hay rolling nên được xác minh trên nhiều giai đoạn, thay vì quy nguyên nhân cho một chế độ dựa trên các tập test lồng nhau.

## Tệp và tài liệu liên quan

- [Báo cáo đánh giá ngày 14-11](../14-11-copper_7_models/BAO_CAO_DANH_GIA_14-11-COPPER_7_MODELS.md)
- [Notebook ngày 15-11](../../15-11-copper_same_day_walk_forward.ipynb)
- [Chỉ số chuẩn hóa](metrics_scaled.csv)
- [Dự đoán và giá thực tế](predictions_comparison.csv)
- [Lịch sử tìm kiếm tham số CNN và ba seed](cnn_parameter_search.csv)
- [Lựa chọn biến theo từng tập train](feature_selection_by_split.csv)
