# Cấu trúc slide thuyết trình

> Bản dàn ý 26 slide. Mỗi slide có nội dung ngắn, gợi ý hình minh họa và thông điệp nói. Các chỉ số chưa có kết quả xác nhận được để dạng `[RMSE]`, `[MAE]`, `[MAPE]`, `[R²]`, `[Spearman]` để bổ sung sau.

## Mạch trình bày

**Bài toán → dữ liệu → thiết kế đặc trưng → phương pháp → tối ưu và so sánh → iTransformer → kiểm tra độ ổn định → kết luận**

---

## Phần I — Bài toán và dữ liệu

### Slide 1 — Dự báo giá đồng: từ dữ liệu thị trường đến mô hình chuỗi thời gian

- Đề tài: dự báo giá đồng bằng dữ liệu thị trường theo ngày.
- Trọng tâm: lựa chọn biến, tối ưu mô hình và kiểm tra khả năng dự báo ngoài mẫu.
- So sánh các hướng tiếp cận: mô hình nền, mô hình tối ưu và iTransformer.

**Gợi ý hình:** Ảnh đồng/copper cathode kết hợp đường giá đồng theo thời gian.

**Thông điệp cần nhấn mạnh:** Đây là bài toán dự báo giá tài sản, nơi sai số nhỏ và tính ổn định theo thời gian đều có ý nghĩa.

### Slide 2 — Vì sao dự báo giá đồng khó?

- Giá chịu tác động từ cung cầu công nghiệp, năng lượng, kim loại khác và kỳ vọng thị trường.
- Quan hệ giữa biến giải thích và giá có thể thay đổi theo giai đoạn.
- Chuỗi có xu hướng, biến động mạnh và các cú sốc khó dự báo.
- Tương quan quan sát được không đồng nghĩa với quan hệ nhân quả.

**Gợi ý hình:** Đường giá đồng với các vùng biến động mạnh; bên cạnh là sơ đồ các nhóm yếu tố tác động.

**Thông điệp cần nhấn mạnh:** Một mô hình tốt cần được đánh giá trên dữ liệu tương lai theo đúng thứ tự thời gian.

### Slide 3 — Mục tiêu và câu hỏi nghiên cứu

- Có thể tái tạo quy trình dự báo tham khảo của Chen và cộng sự trên bộ dữ liệu hiện tại đến mức nào?
- Việc thay đổi tập đặc trưng ảnh hưởng ra sao đến chất lượng dự báo?
- Simulated Annealing (SA) và Tree-structured Parzen Estimator (TPE) cho kết quả và chi phí tìm kiếm như thế nào?
- iTransformer Direct và iTransformer Residual có cải thiện so với các mô hình nền không?

**Gợi ý hình:** Bốn câu hỏi đặt quanh một pipeline nghiên cứu.

**Thông điệp cần nhấn mạnh:** Mỗi câu hỏi sẽ được trả lời bằng một nhóm thực nghiệm, không gộp các kết quả khác thiết kế thành một phép so sánh duy nhất.

### Slide 4 — Nghiên cứu tham khảo và phạm vi tái lập

- Bộ dữ liệu và hướng xây dựng biến tham khảo nghiên cứu của Chen và cộng sự.
- Nguồn dữ liệu chủ yếu: Investing.com.
- Nghiên cứu hiện tại cập nhật dữ liệu đến 13/11/2026.
- Không sao chép nguyên trạng đặc trưng: các tập FS được thiết kế và thử nghiệm lại trên dữ liệu hiện tại.

**Gợi ý hình:** Sơ đồ “nghiên cứu tham khảo → cập nhật dữ liệu → thử nghiệm đặc trưng → thực nghiệm hiện tại”.

**Thông điệp cần nhấn mạnh:** Đây là tái lập có điều chỉnh, không phải tuyên bố tái tạo nguyên xi mọi thiết lập của bài báo.

### Slide 5 — Dữ liệu sử dụng

- 2.710 phiên giao dịch, từ 13/01/2016 đến 13/11/2026.
- Tần suất theo phiên giao dịch; đồng bộ dữ liệu theo ngày.
- Biến mục tiêu: giá đồng.
- Dữ liệu EDA chính thức trong báo cáo trước có 1.951 dòng, kết thúc 06/12/2023; cần phân biệt với bộ dữ liệu cập nhật.

**Gợi ý hình:** Bảng tóm tắt số dòng, khoảng thời gian, tần suất, biến mục tiêu; đường giá đồng toàn kỳ.

**Thông điệp cần nhấn mạnh:** Nêu rõ mốc thời gian và phiên bản dữ liệu để người nghe không nhầm bộ EDA cũ với dữ liệu thực nghiệm cập nhật.

### Slide 6 — Các nhóm biến và quy tắc chia dữ liệu

- Nhóm biến gồm giá đồng trễ, kim loại liên quan, năng lượng và các biến vĩ mô/thị trường đã thu thập.
- Spearman correlation được dùng để khảo sát quan hệ đơn điệu giữa biến và giá đồng.
- Chia theo thời gian: train/validation/test; không xáo trộn thứ tự phiên.
- Scaling và lựa chọn biến phải được fit trên phần dữ liệu huấn luyện tương ứng để hạn chế data leakage.

**Gợi ý hình:** Timeline train → validation → test và bảng nhóm biến; dành chi tiết các tập FS cho slide kế tiếp.

**Thông điệp cần nhấn mạnh:** Spearman là bằng chứng hỗ trợ lựa chọn biến, không phải tiêu chí duy nhất và không thay thế kiểm định ngoài mẫu.

---

## Phần II — Thiết kế đặc trưng

### Slide 7 — Spearman correlation: biến nào đi cùng giá đồng?

- Al–Copper: `[Spearman = 0.9170]`.
- Tin–Copper: `[Spearman = 0.8427]`.
- WTI–Copper: `[Spearman = 0.7546]`.
- Gold–Copper: `[Spearman = 0.6831]`; Silver–Copper: `[Spearman = 0.6695]`.
- Các hệ số trên tính trên mức giá trong EDA; cần ghi rõ khoảng dữ liệu và kiểm tra lại trên đúng phiên bản dùng cho từng thực nghiệm.

**Gợi ý hình:** Biểu đồ thanh ngang các hệ số Spearman; tô màu theo nhóm hàng hóa/năng lượng.

**Thông điệp cần nhấn mạnh:** Tương quan cao gợi ý biến có thông tin đồng biến, nhưng chưa chứng minh biến đó cải thiện dự báo tương lai.

### Slide 8 — Sáu tập đặc trưng FS01–FS06

- **FS01:** tập cơ sở, tập trung vào lịch sử giá đồng.
- **FS02:** mở rộng tập cơ sở bằng nhóm kim loại liên quan.
- **FS03:** bổ sung nhóm năng lượng/thị trường.
- **FS04:** tập đa biến đã dùng trong các baseline residual của Module 1.
- **FS05:** biến thể mở rộng để kiểm tra đóng góp của nhóm biến bổ sung.
- **FS06:** tập đặc trưng được chọn cho các phân tích iTransformer và walk-forward trong phạm vi hiện tại.

**Gợi ý hình:** Ma trận FS × nhóm biến, đánh dấu biến có/không có thay vì liệt kê một danh sách dài.

**Thông điệp cần nhấn mạnh:** Tên FS chỉ có ý nghĩa khi đi kèm bảng ánh xạ cột thực tế; trước khi bảo vệ, hãy đối chiếu từng tập với file cấu hình/dataset đang chạy.

### Slide 9 — Vì sao chọn FS06 cho các phân tích cuối?

- FS06 là tập được dùng thống nhất trong thực nghiệm iTransformer và walk-forward đang báo cáo.
- Cho phép kiểm tra mô hình trên tập biến mở rộng trong một quy trình đánh giá theo thời gian.
- Kết quả lựa chọn phải dựa trên validation/ngoài mẫu, không chỉ dựa trên Spearman.
- Ablation study sẽ kiểm tra xem các nhóm biến liên quan trực tiếp đến đồng có thực sự đóng góp hay không.

**Gợi ý hình:** Pipeline chọn đặc trưng: ứng viên FS → validation theo thời gian → chọn FS06 → kiểm tra ablation.

**Thông điệp cần nhấn mạnh:** Không nói “FS06 tốt nhất” nếu chưa có bảng đối chiếu đầy đủ các FS theo cùng split và metric; nói chính xác “FS06 được chọn cho các phân tích cuối”.

---

## Phần III — Thiết kế mô hình và tối ưu

### Slide 10 — Quy trình thực nghiệm tổng thể

- Tạo các tập FS từ dữ liệu đã đồng bộ.
- Chia dữ liệu theo thời gian và biến đổi dữ liệu chỉ bằng thông tin train.
- Huấn luyện baseline và mô hình tối ưu trên cùng thiết kế split.
- Đánh giá bằng RMSE, MAE, MAPE, R² và thời gian tối ưu khi có log tương ứng.

**Gợi ý hình:** Sơ đồ pipeline từ dữ liệu thô đến bảng kết quả.

**Thông điệp cần nhấn mạnh:** Phép so sánh chỉ công bằng khi cùng mục tiêu dự báo, split, metric và quy trình tiền xử lý.

### Slide 11 — Mô hình nền trong Module 1

- Module 1 có các mô hình dự báo residual như Residual-GRU và Residual-LSTM.
- Baseline naïve dự báo giá kỳ kế tiếp bằng giá hiện tại.
- Reoptimized SA-LSTM là một cấu hình được đánh giá trong walk-forward Module 1.
- **ARIMA-GRU không thuộc nhóm mô hình đã huấn luyện trong Module 1**; đây là phương pháp benchmark riêng của nhóm.

**Gợi ý hình:** Hai khối tách biệt: “Module 1” và “Benchmark nhóm: ARIMA-GRU”.

**Thông điệp cần nhấn mạnh:** Phân biệt nguồn và phạm vi thực nghiệm; không gán ARIMA-GRU cho Module 1.

### Slide 12 — Simulated Annealing (SA) tìm siêu tham số như thế nào?

- SA bắt đầu từ một cấu hình ứng viên và tạo cấu hình lân cận.
- Cấu hình tốt hơn được nhận; cấu hình kém hơn đôi khi vẫn được nhận khi “nhiệt độ” còn cao.
- Nhiệt độ giảm dần, khiến việc chấp nhận bước kém trở nên ít thường xuyên hơn.
- Cơ chế này giúp thoát khỏi vùng tối ưu cục bộ trong không gian tìm kiếm.

**Gợi ý hình:** Sơ đồ trạng thái: cấu hình hiện tại → lân cận → đánh giá → chấp nhận/từ chối → giảm nhiệt.

**Thông điệp cần nhấn mạnh:** SA không đảm bảo tìm được tối ưu toàn cục; đây là chiến lược tìm kiếm có kiểm soát để khảo sát không gian siêu tham số.

### Slide 13 — Không gian tìm kiếm và điều kiện so sánh SA–TPE

- Hai siêu tham số số neuron: `units_1 ∈ [16, 96]`, `units_2 ∈ [24, 160]`.
- Learning rate: `[1e-5, 3e-3]`.
- Ngân sách so sánh: 16 trials cho mỗi phương pháp theo cấu hình đã chạy.
- Hàm mục tiêu: validation loss/metric đã định trong thực nghiệm; báo cáo đúng cấu hình thực tế.
- Giữ cùng dữ liệu, split, ngân sách thử và cách đánh giá cho SA và TPE.

**Gợi ý hình:** Bảng không gian tìm kiếm + sơ đồ hai optimizer dùng chung ngân sách.

**Thông điệp cần nhấn mạnh:** Tính công bằng của benchmark phụ thuộc thiết kế thử nghiệm ngang nhau, không chỉ phụ thuộc tên thuật toán tối ưu.

### Slide 14 — TPE hoạt động ra sao?

- TPE xây dựng mô hình xác suất từ các cấu hình đã thử và điểm số quan sát được.
- Chia các cấu hình thành nhóm kết quả tốt và nhóm còn lại.
- Đề xuất cấu hình mới có khả năng rơi vào vùng tốt dựa trên phân bố đã học.
- Qua từng trial, TPE cập nhật hiểu biết về vùng siêu tham số hứa hẹn.

**Gợi ý hình:** Các điểm trial trên trục learning rate/loss, minh họa vùng tốt và đề xuất tiếp theo.

**Thông điệp cần nhấn mạnh:** TPE dùng kết quả thử trước để hướng lần thử sau; nó không đơn giản là quét lưới ngẫu nhiên.

### Slide 15 — Kết quả tối ưu: SA so với TPE

- Cấu hình hiện tại: Residual-LSTM trên FS06, split expanding 90/10.
- SA: MAE `0.0787349 ± 0.0021033`; RMSE trung bình `0.1028049`.
- TPE: MAE `0.0789145 ± 0.0025630`; RMSE trung bình `0.1026001`.
- Thời gian tối ưu: `[thời gian SA]` so với `[thời gian TPE]`; bổ sung từ log nếu có.

**Gợi ý hình:** Biểu đồ MAE/RMSE theo phương pháp, kèm thanh sai số qua các seed.

**Thông điệp cần nhấn mạnh:** SA có MAE trung bình thấp hơn một chút trong kết quả này, còn TPE có RMSE trung bình thấp hơn; cần xét cả độ phân tán và thời gian, không tuyên bố một phương pháp thắng tuyệt đối.

---

## Phần IV — iTransformer Direct và Residual

### Slide 16 — iTransformer nhìn chuỗi đa biến theo cách nào?

- iTransformer đảo cách tổ chức đầu vào: biến được biểu diễn thành các token để mô hình hóa quan hệ giữa biến.
- Attention học quan hệ giữa các chuỗi biến, còn embedding/MLP xử lý thông tin theo chiều thời gian.
- Mục tiêu là khai thác tương tác đa biến trong chuỗi thời gian.
- Cần giữ cùng quy trình tạo cửa sổ và đánh giá khi so sánh Direct với Residual.

**Gợi ý hình:** Sơ đồ đầu vào ma trận thời gian × biến → embedding biến → attention → đầu ra.

**Thông điệp cần nhấn mạnh:** Đây là kiến trúc được thiết kế cho chuỗi đa biến; kết quả phụ thuộc mạnh vào cách đặt bài toán và dữ liệu đầu vào.

### Slide 17 — iTransformer Direct: dự báo trực tiếp giá đồng

- Đầu vào: cửa sổ lịch sử của các biến.
- Đầu ra: giá đồng ở bước dự báo kế tiếp.
- Mô hình phải học đồng thời mức giá và biến động trong một chuỗi có xu hướng.
- Kết quả FS06: MAE `1.0795466 ± 0.2988469` qua 5 seed.

**Gợi ý hình:** Mũi tên từ lịch sử đa biến trực tiếp đến Copper(t+1), cùng biểu đồ dự báo–thực tế.

**Thông điệp cần nhấn mạnh:** Direct là dự báo trực tiếp biến mục tiêu; chất lượng cần so với baseline cùng đơn vị và cùng split.

### Slide 18 — iTransformer Residual: dự báo phần thay đổi còn lại

- Mục tiêu mô hình là mức thay đổi của giá đồng so với phiên trước, không phải residual/error của ARIMA-GRU.
- Mô hình dự báo `ΔCopper(t) = Copper(t) − Copper(t−1)`.
- Khôi phục giá: `Copper_hat(t) = Copper(t−1) + ΔCopper_hat(t)`.
- Kết quả FS06: MAE `0.0773802 ± 0.0004121` qua 5 seed; naïve khoảng `0.07708`.

**Gợi ý hình:** Phân rã giá thành “mức hiện tại + thay đổi dự báo”, minh họa dự báo biến động rồi cộng lại.

**Thông điệp cần nhấn mạnh:** Residual ở đây là cách định nghĩa mục tiêu theo sai phân quanh giá phiên trước; không gọi nhầm là lỗi của một mô hình nền khác.

### Slide 19 — Vì sao Residual có thể tốt hơn Direct?

- Giá đồng có tính liên tục: giá phiên sau thường gần giá phiên hiện tại.
- Direct phải học cả mức giá và biến động; mục tiêu này có thể khó nếu mức giá có xu hướng.
- Dự báo thay đổi rồi cộng với giá hiện tại giúp mô hình tập trung vào phần thông tin mới.
- Nhưng kết quả gần baseline naïve cho thấy lợi ích tăng thêm của mô hình cần được kiểm tra cẩn thận.

**Gợi ý hình:** So sánh hai luồng: “lịch sử → mức giá” và “lịch sử → Δgiá → cộng giá hiện tại”.

**Thông điệp cần nhấn mạnh:** Residual không tự động tốt hơn. Nó phù hợp khi phần biến động dễ học hơn mức giá, nhưng phải chứng minh bằng benchmark ngoài mẫu.

### Slide 20 — So sánh iTransformer Direct, Residual và baseline gần nhất

- iTransformer Direct: MAE `1.0795466 ± 0.2988469`.
- iTransformer Residual: MAE `0.0773802 ± 0.0004121`.
- Naïve: MAE khoảng `0.07708`.
- Mức cải thiện Residual so với Direct: báo cáo bằng chênh lệch tuyệt đối và phần trăm sau khi xác nhận cùng đơn vị/split.
- Residual hiện gần naïve; không được diễn giải chênh lệch nhỏ là thắng rõ rệt nếu chưa có kiểm định thích hợp.

**Gợi ý hình:** Biểu đồ MAE với trục phù hợp, hiển thị sai số giữa các seed và đường chuẩn naïve.

**Thông điệp cần nhấn mạnh:** Kết quả nổi bật là chênh lệch lớn giữa hai cách đặt mục tiêu, nhưng Residual chưa cho thấy vượt trội rõ so với dự báo giữ nguyên giá.

### Slide 21 — Benchmark toàn nhóm: đặt các phương pháp vào đúng phạm vi

- So sánh iTransformer với kết quả Module 1 khi split, target và metric tương thích.
- Module 1 residual FS04: GRU MAE `0.076951`, RMSE `0.100810`; LSTM MAE `0.076978`, RMSE `0.100948`.
- Naïve trong thiết lập liên quan: MAE khoảng `0.07708`.
- ARIMA-GRU được trình bày ở nhóm benchmark riêng; bổ sung `[MAE]`, `[RMSE]`, `[MAPE]`, `[R²]` theo đúng kết quả thực nghiệm.
- Không xếp hạng trực tiếp nếu khác FS, split, horizon hoặc tiền xử lý.

**Gợi ý hình:** Bảng nhỏ chia cột “Module 1” và “Benchmark nhóm”; đánh dấu rõ các điều kiện so sánh.

**Thông điệp cần nhấn mạnh:** Kết nối ARIMA-GRU với câu chuyện so sánh toàn đề tài, nhưng minh bạch rằng nó không thuộc tập mô hình Module 1.

---

## Phần V — Kiểm tra độ ổn định và đóng góp đặc trưng

### Slide 22 — Walk-forward validation trên FS06

- Cửa sổ test tổng cộng: 542 phiên; quy trình gồm 9 folds.
- Dịch chuyển/cập nhật mô hình mỗi `S = 63` phiên.
- Có 8 folds đủ 63 phiên và fold cuối 38 phiên.
- Đánh giá Residual-GRU, Residual-LSTM và Reoptimized SA-LSTM qua seeds 41–45.

**Gợi ý hình:** Timeline 9 folds, đánh dấu mỗi 63 phiên và phần 38 phiên cuối.

**Thông điệp cần nhấn mạnh:** Với Module 1, bước dịch chuyển là 63; không nhầm với thiết lập walk-forward khác `W=756, S=40`.

### Slide 23 — Walk-forward: hiệu năng và độ ổn định theo thời gian

- Residual-GRU: MAE `0.069100 ± 0.000154`.
- Residual-LSTM: MAE `0.069179 ± 0.000167`.
- Reoptimized SA-LSTM: MAE `0.207858 ± 0.019601`.
- So sánh với split cố định chỉ khi dữ liệu, target, metric và quy trình tiền xử lý tương ứng.
- Phân tích độ ổn định theo fold bên cạnh điểm trung bình tổng thể.

**Gợi ý hình:** Biểu đồ MAE từng fold hoặc boxplot theo seed; thêm đường ngang cho trung bình.

**Thông điệp cần nhấn mạnh:** Walk-forward cho thấy hiệu năng qua nhiều mốc thời gian và lần cập nhật, giúp phát hiện mô hình phụ thuộc vào một giai đoạn thuận lợi.

### Slide 24 — Ablation study: nhóm biến nào thực sự giúp ích?

- Giữ nguyên split, cửa sổ, mô hình, seed và optimizer; chỉ thay đổi nhóm biến.
- So sánh cấu hình FS06 đầy đủ với cấu hình bỏ nhóm biến liên quan trực tiếp đến đồng.
- Đánh giá bằng `[MAE]`, `[RMSE]` và biến động qua seed/fold.
- Nếu thêm biến không cải thiện, nguyên nhân có thể là dư thừa thông tin, nhiễu hoặc tương quan mạnh giữa đầu vào.

**Gợi ý hình:** Bảng ablation “đầy đủ / bỏ nhóm A / bỏ nhóm B” và biểu đồ thay đổi MAE.

**Thông điệp cần nhấn mạnh:** Ablation đo đóng góp trong một cấu hình cụ thể; nó không chứng minh biến đó có hay không có quan hệ nhân quả.

---

## Phần VI — Kết luận

### Slide 25 — Trả lời các câu hỏi nghiên cứu

- Spearman correlation hỗ trợ mô tả liên hệ đơn điệu và sàng lọc ứng viên; cần đánh giá bằng validation theo thời gian.
- SA và TPE cho kết quả gần nhau trong cấu hình Residual-LSTM FS06; metric thắng khác nhau.
- iTransformer Residual tốt hơn Direct trên kết quả đang có, nhưng gần baseline naïve.
- Walk-forward Module 1 với `S=63` cho thấy GRU/LSTM residual ổn định hơn Reoptimized SA-LSTM trong thực nghiệm này.

**Gợi ý hình:** Bốn câu hỏi nghiên cứu và một kết luận ngắn dưới mỗi câu.

**Thông điệp cần nhấn mạnh:** Trả lời bằng đúng mức độ bằng chứng; phân biệt “quan sát được trong thực nghiệm” với kết luận tổng quát.

### Slide 26 — Hạn chế và hướng phát triển

- Các kết quả khác FS/split/horizon chưa chắc so sánh trực tiếp được.
- Tập dữ liệu thị trường và quan hệ giữa biến có thể thay đổi theo thời gian.
- Cần báo cáo thêm chi phí tính toán và kiểm định chênh lệch khi có đủ log.
- Hướng tiếp theo: đánh giá rolling/walk-forward nhất quán, kiểm tra ablation đầy đủ và thử nhiều horizon dự báo.

**Gợi ý hình:** Ba cột “Hạn chế”, “Việc cần làm”, “Ý nghĩa tiếp theo”.

**Thông điệp cần nhấn mạnh:** Đóng bài bằng giới hạn cụ thể và kế hoạch kiểm tra có thể thực hiện, không khẳng định mô hình đã sẵn sàng cho ứng dụng giao dịch.

---

## Ghi chú rà soát trước khi đưa vào slide

- Đối chiếu lại định nghĩa cột trong FS01–FS06 với notebook/config thực nghiệm trước khi trình bày tên nhóm biến.
- Spearman `0.9170, 0.8427, 0.7546, 0.6831, 0.6695` thuộc bảng EDA mức giá; xác nhận lại trên bộ dữ liệu và khoảng thời gian cuối cùng sẽ báo cáo.
- Bổ sung thời gian tối ưu SA/TPE và các metric còn thiếu từ log; không tự suy diễn số liệu.
- Khi so sánh, ghi rõ target Direct/Residual, split, FS, horizon và đơn vị giá.
- Cụm “Residual” trong iTransformer nghĩa là sai phân so với Copper phiên trước; không phải residual của ARIMA-GRU.
- Walk-forward Module 1 dùng `S=63`; cấu hình `W=756,S=40` thuộc thiết lập khác.

