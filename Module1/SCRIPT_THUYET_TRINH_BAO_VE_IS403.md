# Script thuyết trình bảo vệ IS403 - Dự báo giá đồng một phiên trước

**Thời lượng gợi ý:** 18-22 phút, chưa tính hỏi đáp.  
**Mục tiêu:** trình bày được mạch nghiên cứu từ bài báo gốc, tái lập trên dữ liệu mới, EDA và Spearman, thiết kế sáu tập đặc trưng, đến cơ chế Direct/Residual và kết quả thực nghiệm.  
**Quy ước:** phần trong dấu ngoặc vuông là chỉ dẫn cho người thuyết trình, không đọc thành lời.

---

## Slide 1 - Mở đầu: bài toán và câu hỏi nghiên cứu

### Lời thoại

“Kính thưa cô và các bạn, đề tài của nhóm em nghiên cứu bài toán dự báo giá đóng cửa của đồng trước một phiên giao dịch.

Tại thời điểm dự báo phiên (t), mô hình chỉ được nhìn 22 phiên đã xảy ra, từ (t-22) đến (t-1), và phải dự báo giá Copper tại (t). Như vậy đây là dự báo một bước trước, không phải dự báo đồng thời bằng dữ liệu của chính ngày cần dự báo.

Điểm xuất phát của nhóm là bài báo của Chen và cộng sự năm 2023 về mô hình SA-LSTM. Tuy nhiên, nhóm không chỉ chạy lại kiến trúc của bài báo. Nhóm đặt ba câu hỏi mở rộng.

Thứ nhất, khi chuyển sang dữ liệu cập nhật giai đoạn 2016-2026, các biến được bài báo chọn còn giữ quan hệ giống trước hay không?

Thứ hai, ngoài WTI, Gold và Silver của bài báo, Aluminium và Tin có đem lại thông tin hữu ích hay không?

Thứ ba, thay vì bắt mạng học trực tiếp toàn bộ mức giá, nếu cho mạng học phần thay đổi so với giá phiên trước - tức residual - thì sai số ngoài mẫu có giảm không?

Toàn bộ phần trình bày của em sẽ đi theo đúng ba câu hỏi đó: bài báo gốc, tái lập trên dữ liệu hiện tại, thiết kế biến và cuối cùng là Direct so với Residual.”

### Câu chuyển

“Trước hết, em xin tách rõ bài báo gốc đã làm gì, để tránh gọi mọi thí nghiệm sau đó là tái lập nguyên xi.”

---

## Slide 2 - Bài báo gốc của Chen et al. (2023)

### Lời thoại

“Bài báo có tên *Copper price prediction using LSTM recurrent neural network integrated simulated annealing algorithm*, công bố trên PLOS ONE năm 2023.

Bài báo sử dụng dữ liệu từ Investing.com trong giai đoạn 01/01/1990 đến 31/12/2009, sau khi đồng bộ ngày còn 4.870 quan sát. Từ tám chỉ báo kinh tế ban đầu, tác giả dùng tương quan Spearman trên mức giá để chọn ba biến ngoại sinh: WTI, Gold và Silver. Hệ số Spearman được báo cáo lần lượt là 0,89; 0,84 và 0,91.

Lý do tác giả chọn Spearman là phân phối giá Copper không chuẩn theo kiểm định KS. Khi trình bày, em xin diễn đạt chặt hơn: Pearson vẫn có thể được tính trên dữ liệu không chuẩn, nhưng Spearman phù hợp hơn với mục tiêu đo quan hệ đơn điệu theo thứ hạng và ít nhạy hơn với phân phối lệch hoặc ngoại lệ. Vì vậy, nói ‘Pearson hoàn toàn không dùng được’ là quá mạnh.

Về mô hình, tác giả dùng hai tầng LSTM có 39 và 111 units, Dense 32, dropout 0,2, batch size 64, 100 epochs và learning rate (5\times10^{-5}). Simulated Annealing được dùng để tìm số units và learning rate. Bài báo đánh giá ở ba tỷ lệ train là 90%, 92,5% và 95%, rồi báo cáo MSE trên thang chuẩn hóa lần lượt khoảng 0,00195; 0,00190 và 0,00097.

Đóng góp cốt lõi của bài báo gồm hai phần: dùng Spearman để chọn biến và dùng Simulated Annealing để tìm siêu tham số LSTM.”

### Điểm phải nói nếu bị hỏi

“Các con số MSE của bài báo không được so trực tiếp với MAE hoặc RMSE theo USD/lb của nhóm, vì khác dữ liệu, khác giai đoạn và khác thang đo.”

---

## Slide 3 - Simulated Annealing tìm siêu tham số như thế nào?

### Lời thoại

“Simulated Annealing, viết tắt là SA, mô phỏng quá trình làm nguội vật liệu.

Ở mỗi bước, thuật toán tạo một bộ siêu tham số mới, ví dụ thay đổi số units hoặc learning rate. Nếu bộ mới có validation loss thấp hơn, thuật toán chấp nhận. Nếu loss cao hơn, thuật toán vẫn có thể chấp nhận với xác suất phụ thuộc vào mức tăng loss và nhiệt độ hiện tại.

Ở nhiệt độ cao, xác suất chấp nhận một nghiệm xấu còn lớn. Mục đích là tránh mắc kẹt quá sớm tại cực tiểu cục bộ. Khi nhiệt độ giảm, thuật toán trở nên bảo thủ hơn và hội tụ quanh nghiệm tốt đã tìm thấy.

Điều quan trọng là SA không trực tiếp dự báo giá. SA đứng bên ngoài, đề xuất siêu tham số; còn LSTM bên trong vẫn được huấn luyện bằng lan truyền ngược. Hàm mục tiêu của SA là validation loss của LSTM tương ứng.

Trong bài báo, bộ cuối cùng là 39-111 units với learning rate (5\times10^{-5}). Trong dự án hiện tại, nhóm giữ bộ này làm cấu hình Paper SA-LSTM, đồng thời có nhánh Reoptimized SA-LSTM để tìm lại siêu tham số trên dữ liệu mới.”

### Câu chuyển

“Như vậy, bước tiếp theo không phải là hỏi nhóm có tạo ra đúng ba con số của paper hay không, mà là hỏi mức độ tái lập nào đã được thực hiện.”

---

## Slide 4 - Tái lập hay mở rộng?

### Lời thoại

“Nhóm em gọi đây là tái lập theo phương pháp, kết hợp mở rộng thực nghiệm; không phải tái lập số liệu tuyệt đối.

Phần được giữ lại gồm: bài toán dự báo Copper, cửa sổ 22 phiên, cấu trúc Paper SA-LSTM 39-111, learning rate (5\times10^{-5}), batch 64, dropout 0,2, các tỷ lệ 90%, 92,5%, 95%, và nhóm biến WTI-Gold-Silver.

Phần thay đổi gồm bốn điểm.

Một là dữ liệu mới có 2.710 phiên từ 13/01/2016 đến 13/11/2026, thay cho dữ liệu 1990-2009.

Hai là nhóm thêm split 80/20 và đánh giá bằng MAE, RMSE, MAPE, (R^2), directional accuracy khi có đủ artifact, thay vì chỉ nhìn MSE chuẩn hóa.

Ba là nhóm xây dựng sáu tập đặc trưng để thực hiện đối chứng và ablation.

Bốn là nhóm bổ sung nhiều mô hình, ba chiến lược lịch sử và hai cách đặt target Direct/Residual.

Vì vậy, nếu hội đồng hỏi ‘có reproduce đúng paper không?’, câu trả lời chính xác là: nhóm tái dựng cấu hình và logic của paper trên dữ liệu mới, sau đó mở rộng giao thức. Do dữ liệu và metric khác, nhóm không tuyên bố tái tạo đúng các giá trị lỗi của paper.”

---

## Slide 5 - Dữ liệu hiện tại và chống rò rỉ

### Lời thoại

“Dataset hiện tại có 2.710 phiên và chín mặt hàng ngoài cột ngày: Copper, Aluminium, Nickel, Tin, Iron Ore 62%, Silver, Gold, WTI và Brent.

Trong giao thức chính thức, EDA dùng 1.951 dòng đầu, từ 13/01/2016 đến 06/12/2023. Mốc này nằm trước cả validation và test của split 80/20. Artifact kiểm toán xác nhận EDA không chạm test bắt đầu ngày 07/10/2024.

Bốn split 80/20, 90/10, 92,5/7,5 và 95/5 có lần lượt 542, 271, 204 và 136 phiên test. Các test cùng kết thúc ở cuối dữ liệu nên chúng lồng nhau; do đó không được coi bốn MAE là bốn mẫu độc lập.

Nhóm fit hai scaler riêng trên train: `xsc` cho các biến đầu vào và `ysc` cho Copper mục tiêu. Validation và test chỉ được transform bằng scaler đã fit. Mỗi mẫu có tensor kích thước (22\times F), trong đó (F) là số biến của feature set.

Tách theo thời gian là điều kiện cần nhưng chưa đủ. Dữ liệu mới chỉ có ngày, chưa xác minh giờ đóng cửa và thời điểm công bố của từng thị trường. Vì vậy, nhóm chưa khẳng định đã loại hết look-ahead ở mức thông tin intraday.”

### Ghi chú kiểm toán

“Báo cáo Mythos hiện còn một đoạn EDA cũ ghi 2.195 dòng đến 12/11/2024. Khi bảo vệ phải dùng artifact official mới: 1.951 dòng đến 06/12/2023. Nếu slide đang dùng 2.195, cần sửa trước khi nộp.”

---

## Slide 6 - EDA dẫn đến giả thuyết Residual

### Lời thoại

“Hai kiểm định quan trọng nhất để dẫn đến residual là ADF và KPSS.

Với ADF, giả thuyết không là chuỗi có nghiệm đơn vị. Với KPSS, giả thuyết không lại là chuỗi dừng quanh hằng số. Vì hai giả thuyết không đối nghịch nhau, ta đọc kết hợp.

Đối với mức giá Copper, ADF có p khoảng 0,432 nên chưa bác bỏ nghiệm đơn vị; KPSS có p ở biên 0,01 nên bác bỏ tính dừng. Với sai phân Copper, ADF có p khoảng (6,31\times10^{-30}), còn KPSS ở biên 0,10. Mẫu hình này ủng hộ nhận định: mức giá có dấu hiệu không dừng, còn sai phân ổn định hơn theo đặc tả kiểm định.

Đây không phải bằng chứng residual chắc chắn thắng. Nó chỉ tạo ra giả thuyết hợp lý: thay vì bắt mô hình học cả mặt bằng giá đang trôi theo thời gian, ta thử cho mô hình học phần thay đổi quanh giá gần nhất.

Ngoài ra, Brown-Forsythe cho thấy độ lệch chuẩn return Copper giữa hai nửa EDA tăng từ khoảng 1,185% lên 1,521%, p sau hiệu chỉnh khoảng (1,35\times10^{-9}). Điều này tạo lý do thử Rolling và Recency Weighting, vì dữ liệu gần đây có thể đại diện tốt hơn cho regime hiện tại.”

### Câu chuyển

“Sau khi xác định cách biểu diễn target, câu hỏi tiếp theo là chọn biến nào cho đầu vào.”

---

## Slide 7 - Spearman và các biến mới

### Lời thoại

“Spearman đo tương quan giữa thứ hạng của hai biến. Nếu (R(X)) và (R(Y)) là rank, Spearman chính là Pearson trên hai dãy rank. Với trường hợp không có rank trùng, có thể viết công thức (\rho_s=1-\frac{6\sum d_i^2}{n(n^2-1)}).

Trong 1.951 dòng EDA chính thức, tương quan Spearman mức giá với Copper là: Aluminium 0,9170; Tin 0,8427; WTI 0,7546; Gold 0,6831 và Silver 0,6695.

So với dữ liệu lịch sử của paper, WTI-Gold-Silver không còn giữ đúng thứ tự và độ lớn cũ. Aluminium và Tin nổi lên như hai biến có đồng biến thứ hạng rất cao với Copper. Vì vậy, hai biến mới mà nhóm ưu tiên khảo sát là Aluminium và Tin.

Tuy nhiên, phải nói rõ ba giới hạn.

Thứ nhất, đây là tương quan đồng thời trên mức giá, không phải tương quan giữa thông tin ở (t-1) và Copper tại (t).

Thứ hai, các chuỗi mức giá đều có xu hướng và dấu hiệu không dừng. Hai chuỗi cùng tăng dài hạn có thể cho Spearman cao dù không có giá trị dự báo gia tăng.

Thứ ba, tương quan không phải nhân quả và cũng không đo đóng góp có điều kiện khi nhiều biến cùng đi vào mạng.

Khi kiểm tra return ngoại sinh trễ, chỉ Gold lag 1 còn Spearman khoảng 0,0931 với p sau Benjamini-Hochberg khoảng 0,00116. Độ lớn 0,0931 vẫn rất nhỏ. WTI, Aluminium và Tin không có tương quan return trễ mạnh sau hiệu chỉnh trong lưới lag đã xét.

Vì vậy, Spearman được dùng để đề xuất giả thuyết và thiết kế tập biến; quyết định cuối phải dựa trên validation và kết quả ngoài mẫu.”

### Nếu bị hỏi: “Dataset có chín hàng hóa, sao không chọn top-k tuyệt đối?”

“Nếu tính Spearman mức giá cho toàn bộ chín hàng hóa trên cùng 1.951 dòng, Nickel cũng có hệ số khoảng 0,8365, Iron Ore khoảng 0,7562 và Brent khoảng 0,7307. Sáu FS hiện tại được thiết kế theo tính liên tục với paper và các ablation đã định trước, không phải thuật toán top-k toàn bộ dataset. Việc chưa đưa Nickel, Iron Ore và Brent vào ablation là một giới hạn và là hướng mở rộng hợp lý. Vì thế nhóm không tuyên bố sáu FS là tập biến tối ưu toàn cục.”

---

## Slide 8 - Vì sao có sáu feature set?

### Lời thoại

“Sáu feature set không phải sáu dataset khác nhau. Chúng dùng chung ngày, chung target Copper và chung test; chỉ thay các kênh đầu vào.

FS01 là `Copper`. Đây là baseline tự hồi quy: chỉ lịch sử của chính Copper. Nó trả lời câu hỏi liệu biến ngoại sinh có thực sự cần thiết.

FS02 là `Copper, WTI, Gold, Silver`. Đây là bộ ba của paper cộng lịch sử Copper. Lý do thêm Copper là tạo đối chứng công bằng cho mô hình Direct, vì giá của chính Copper thường là tín hiệu mạnh nhất cho mức giá kế tiếp.

FS03 là `WTI, Gold, Silver`, tức bộ paper nhưng bỏ Copper khỏi tensor đầu vào. So FS02 với FS03 là ablation vai trò của kênh Copper.

FS04 là `Copper, Aluminium, Tin, Silver, Gold`. Đây là giả thuyết nhóm kim loại: giữ Copper, thêm hai biến mới Aluminium-Tin có Spearman cao và giữ Gold-Silver để so với thiết kế trước.

FS05 là `Aluminium, Tin, Silver, Gold`, tức FS04 bỏ Copper. So FS04 với FS05 là ablation thứ hai về lịch sử Copper.

FS06 là `Copper, Aluminium, Tin, WTI`. Bộ này kết hợp Copper, hai biến mới có Spearman mức giá cao, và WTI - biến năng lượng được paper sử dụng. So FS04 với FS06 kiểm tra cấu hình thay Gold và Silver bằng WTI, đồng thời giảm từ năm xuống bốn biến.

Điểm cần nói rất chặt: FS04 sang FS06 không phải ablation riêng của WTI. Ta thay hai biến bằng một biến và đổi số chiều, nên mọi chênh lệch là hiệu ứng của cả cấu hình. Muốn kết luận tác động riêng của WTI, cần các cặp chỉ khác đúng một biến.”

### Cảnh báo về FS03/FS05

“Trong nhánh Residual, ‘không Copper trong X’ không có nghĩa toàn hệ thống không dùng Copper. Giá Copper tại (t-1) vẫn được dùng làm mốc để hoàn nguyên dự báo mức giá.”

---

## Slide 9 - Cách tạo một mẫu học

### Lời thoại

“Với mỗi thời điểm (t), đầu vào là:

\[
X_t=[x_{t-22},x_{t-21},\ldots,x_{t-1}]\in\mathbb{R}^{22\times F}.
\]

Ví dụ FS06 có bốn biến nên một mẫu có shape (22\times4). Qua batch, tensor là `(B, 22, 4)`.

Scaler được fit ở đoạn train trước validation. `xsc` biến đổi từng kênh đầu vào. `ysc` biến đổi riêng Copper thành:

\[
z_t=\frac{P_t-P_{\min,train}}{P_{\max,train}-P_{\min,train}}.
\]

Mạng không shuffle thứ tự mẫu trong khi fit. Ở mỗi epoch, nó forward qua 22 bước, tạo một đầu ra, tính MSE, rồi dùng backpropagation through time để đưa gradient ngược qua Dense và hai tầng recurrent. Adam cập nhật các trọng số dùng chung ở mọi bước thời gian. Early stopping theo dõi validation loss và khôi phục bộ trọng số tốt nhất.

Dropout chỉ ngẫu nhiên bỏ một phần activation khi train để giảm đồng thích nghi; khi predict, dropout được tắt.”

---

## Slide 10 - Direct học cái gì?

### Lời thoại

“Trong Direct, target là mức giá đã scale:

\[
y_t^{direct}=z_t.
\]

Mạng nhận 22 phiên và phải sinh trực tiếp toàn bộ tọa độ của giá ngày kế tiếp trên thang chuẩn hóa. Loss của một mẫu là:

\[
L_{direct}=(\hat z_t-z_t)^2.
\]

Gradient tại đầu ra là (2(\hat z_t-z_t)). Từ gradient này, mạng điều chỉnh các tầng Dense và recurrent để dự báo mức giá chính xác hơn.

Khó khăn là mức giá chứa một thành phần rất lớn và chậm - mặt bằng giá - cộng với một thành phần nhỏ là biến động ngày kế tiếp. Mạng phải tự học cả phép sao chép mức gần nhất lẫn phần điều chỉnh. Khi train và test nằm ở các mặt bằng giá khác nhau, đầu ra Direct dễ bị lệch level hoặc co về vùng giá thường thấy trong train.”

---

## Slide 11 - Residual học cái gì, từng phép tính một

### Lời thoại

“Residual thay target bằng chênh lệch giữa hai mức giá đã scale:

\[
r_t=z_t-z_{t-1}.
\]

Vì (z_t) và (z_{t-1}) dùng cùng `ysc`, nên:

\[
r_t=\frac{P_t-P_{t-1}}{P_{\max,train}-P_{\min,train}}.
\]

Nói cách khác, đây là sai phân theo đơn vị giá, chỉ được chia cho range của train; không phải phần trăm return.

Mạng học:

\[
\hat r_t=f_\theta(X_t).
\]

Trong lúc train, loss là:

\[
L_{res}=(\hat r_t-r_t)^2.
\]

Khi dự báo, code không inverse-transform residual riêng. Code cộng residual vào giá trước trên thang scaled:

\[
\hat z_t=z_{t-1}+\hat r_t,
\]

sau đó mới inverse-transform:

\[
\hat P_t=P_{t-1}+\widehat{\Delta P_t}.
\]

Nếu mạng dự báo (\hat r_t=0), kết quả chính xác bằng giá phiên trước. Do đó Residual có thể được hiểu là học phần hiệu chỉnh cho Naive Persistence.”

### Ví dụ nói miệng

“Nếu hôm qua Copper là 4,50 và mạng dự báo thay đổi +0,03, giá ngày mai là 4,53. Nếu mạng chưa học được gì và dự báo thay đổi bằng 0, đầu ra vẫn là 4,50 chứ không rơi về trung bình của train.”

---

## Slide 12 - Residual thay đổi việc học như thế nào?

### Lời thoại

“Điểm sâu nhất nằm ở phép tái tham số hóa.

Sai số mức giá sau khi hoàn nguyên trên thang scaled là:

\[
\hat z_t-z_t=(z_{t-1}+\hat r_t)-(z_{t-1}+r_t)=\hat r_t-r_t.
\]

Như vậy, vì (z_{t-1}) là hằng số không có tham số học, gradient của loss mức giá đối với đầu ra residual vẫn là (2(\hat r_t-r_t)). Skip connection từ giá hôm qua truyền thẳng phần level mà không bắt mạng học lại.

Direct và Residual có thể dùng cùng kiến trúc. Khác biệt nằm ở target và đường tắt cộng (z_{t-1}). Residual tạo một inductive bias: giả định tốt ban đầu là ngày mai gần hôm nay, còn mạng chỉ cần học phần lệch khỏi giả định đó.

Target residual thường tập trung quanh 0, có dynamic range nhỏ hơn và theo EDA có tính dừng thuận lợi hơn mức giá. Vì vậy optimization dễ hơn và ít nhạy hơn với việc mặt bằng giá dịch chuyển.

Trong LSTM hoặc GRU, mạng vẫn học quan hệ phi tuyến từ 22 phiên. Residual không biến mạng thành phép lấy sai phân đơn giản. Các gate vẫn quyết định thông tin nào trong 22 phiên được giữ, quên hoặc trộn để tạo ra (\widehat{\Delta P_t}).”

---

## Slide 13 - Tại sao Residual thường tốt hơn Direct?

### Lời thoại

“Có bốn lý do chính.

Một, nó cho sẵn baseline mạnh. Với chuỗi giá tần suất ngày, giá ngày mai thường gần giá hôm nay. Direct phải học phép sao chép này từ dữ liệu; Residual có nó trong cấu trúc.

Hai, residual làm target gần 0 và ổn định hơn, nên mạng không phải dành nhiều năng lực để biểu diễn xu hướng level.

Ba, nếu test bước sang một vùng giá cao hơn train, skip connection dùng giá thật (P_{t-1}) tại thời điểm dự báo, nên level được cập nhật ngay. Direct có thể bị neo vào range đã học.

Bốn, gradient tập trung vào sai số của phần thay đổi nhỏ, thay vì vừa học mặt bằng vừa học dao động.

Nhưng residual không luôn tốt hơn. Nếu có cú nhảy lớn, nếu giá trước bị lỗi, nếu bài toán nhiều bước không có giá thật trung gian, hoặc nếu thay đổi thực sự có cấu trúc level dài hạn mà sai phân làm mất, Direct hoặc mô hình khác có thể tốt hơn.

Do đó, ‘Residual tốt hơn Direct’ là kết quả thực nghiệm trong các điều kiện đã chạy, không phải định lý áp dụng cho mọi chuỗi.”

---

## Slide 14 - Cái bẫy: thắng Direct chưa chắc học được biến động

### Lời thoại

“Đây là điểm em muốn chủ động nêu trước khi hội đồng hỏi.

Ở FS06, split 90/10 Expanding, kết quả năm seed của SA-LSTM cho MAE Direct khoảng 0,3590, còn Residual khoảng 0,0787. TPE-LSTM Direct khoảng 0,4081, Residual khoảng 0,0789. Residual thắng Direct rất xa.

Tuy nhiên Naive Persistence ở cùng đoạn test có MAE khoảng 0,07708. Như vậy Residual gần như bám vào baseline giá hôm qua và chưa chắc đã thắng Naive.

Audit delta cho thấy độ lệch chuẩn thay đổi thực khoảng 0,1010 USD/lb, trong khi độ lệch chuẩn delta dự báo residual chỉ khoảng 0,0089 với SA-LSTM và 0,0077 với TPE-LSTM. Tương quan giữa delta dự báo và delta thật chỉ khoảng 0,025 và 0,010. Nghĩa là dự báo residual bị co rất mạnh về 0.

Vì vậy kết luận đúng là: residual cung cấp neo level rất mạnh nên tránh lỗi level lớn của Direct. Nhưng để khẳng định mạng học được biến động, phải kiểm tra thêm skill so với Naive, directional accuracy, delta correlation và độ phân tán của delta dự báo.

Nếu residual có MAE thấp nhưng không thắng Naive, đóng góp chính đến từ cấu trúc persistence chứ chưa phải khả năng đoán thay đổi.”

### Câu chốt mạnh

“Direct là đối chứng về cách đặt target; Naive mới là đối chứng tối thiểu về giá trị dự báo.”

---

## Slide 15 - Kết quả Module 1 ở split 90/10

### Lời thoại

“Trong kết quả Module 1 lịch sử tại split 90/10, GRU Residual với FS04 Expanding đạt MAE 0,076951; LSTM Residual cùng FS04 đạt 0,076978. LSTM Direct tốt nhất được liệt kê ở điều kiện này khoảng 0,247409 và GRU Direct khoảng 0,240815.

So với Direct, lợi thế của residual là rất lớn. Nhưng Naive cùng test khoảng 0,07708, nên mức cải thiện tốt nhất chỉ cỡ vài phần nghìn tương đối. Đây là lý do nhóm không chỉ nhìn bảng Direct-Residual.

Ở bốn split, cấu hình Residual-GRU đứng đầu bảng hậu nghiệm của Module 1: 80/20 là 0,068780; 90/10 là 0,076951; 92,5/7,5 là 0,075028; 95/5 là 0,072357. Tuy nhiên mỗi split chọn một FS và strategy khác nhau sau khi đã xem kết quả, nên có selection bias. Ta chỉ được nói ‘đứng đầu trong lưới đã quan sát’, không được nói ‘tốt nhất tổng quát’.”

### Nếu có biểu đồ

“Khi chỉ vào biểu đồ, hãy chỉ cả đường Naive hoặc ghi MAE Naive cạnh cột residual. Không trình bày residual một mình.”

---

## Slide 16 - Paper SA-LSTM và Reoptimized SA-LSTM trên dữ liệu mới

### Lời thoại

“Trên FS06 Direct Expanding, Fresh SA của nhóm chọn 47-127 units và learning rate khoảng (5,05\times10^{-5}), thay cho cấu hình paper 39-111.

Ở split 80/20, Paper SA-LSTM có MAE khoảng 0,29895, còn Reoptimized khoảng 0,22381, giảm khoảng 25,1%.

Nhưng ở ba split còn lại, cấu hình paper lại thấp hơn: 90/10 là 0,18463 so với 0,35892; 92,5/7,5 là 0,19487 so với 0,36608; 95/5 là 0,20080 so với 0,32543.

Điều này cho thấy SA đã giảm validation loss tại mốc search, nhưng lợi thế không ổn định khi đổi giai đoạn test. Có ba nguyên nhân có thể: siêu tham số overfit vào validation reference, regime thị trường thay đổi, hoặc ngân sách search 16 trials chưa đủ khám phá không gian.

Kết luận đúng là Fresh SA có ích ở một điều kiện, nhưng chưa chứng minh bộ reoptimized tổng quát hơn bộ paper.”

---

## Slide 17 - Ba chiến lược sử dụng lịch sử

### Lời thoại

“Expanding dùng toàn bộ lịch sử từ đầu đến ranh giới train. Ưu điểm là nhiều mẫu; nhược điểm là dữ liệu rất cũ được xem quan trọng như dữ liệu mới.

Rolling 756 chỉ giữ 756 phiên gần nhất trước validation. Nó thích ứng với regime mới hơn nhưng bỏ dữ liệu cũ và giảm kích thước train.

Recency-weighted vẫn giữ toàn bộ lịch sử, nhưng trọng số giảm theo tuổi mẫu. Half-life 252 nghĩa là mẫu cách 252 phiên có trọng số bằng một nửa mẫu mới nhất trước khi chuẩn hóa trọng số.

Các chiến lược holdout này fit một lần cho mỗi điều kiện. Chúng không tự động retrain mỗi ngày. Chỉ walk-forward mới tái fit qua các fold.

Brown-Forsythe và bằng chứng phương sai thay đổi tạo động cơ thử Rolling và Recency. Nhưng chiến lược nào tốt hơn vẫn phải được quyết định bằng validation theo thời gian, không suy ra trực tiếp từ một kiểm định.”

---

## Slide 18 - Hạn chế và tính hợp lệ

### Lời thoại

“Nhóm ghi nhận sáu hạn chế chính.

Thứ nhất, Spearman mức giá có thể cao vì xu hướng chung và không chứng minh giá trị dự báo.

Thứ hai, sáu FS là thiết kế a priori theo paper và các ablation, chưa phải kết quả tìm kiếm tối ưu trên cả chín hàng hóa. Nickel, Iron Ore và Brent chưa được ablation đầy đủ.

Thứ ba, FS04 sang FS06 thay nhiều yếu tố cùng lúc; không cô lập tác động riêng của WTI.

Thứ tư, các split có test lồng nhau và nhiều cấu hình được xếp hạng sau khi xem test, nên có selection bias. Cần nested walk-forward hoặc test cuối chưa từng dùng để chọn model.

Thứ năm, ngày trong CSV chưa chứng minh các giá ngoại sinh đã có sẵn đúng thời điểm forecast origin.

Thứ sáu, Residual có thể đạt MAE đẹp chủ yếu nhờ persistence. Vì vậy phải báo cáo Naive, skill, DA và audit delta cùng MAE/RMSE.

Ngoài ra, kết quả Granger trong EDA V26 cũ bị lỗi 20 trên 20 lượt do tham số `verbose`; không được diễn giải thành ‘không có quan hệ Granger’. Hiện chưa có kết quả Granger hợp lệ trong phạm vi đó.”

---

## Slide 19 - Kết luận

### Lời thoại

“Em xin kết luận bằng bốn ý.

Một, nhóm đã tái dựng logic của Chen và cộng sự trên dữ liệu 2016-2026, nhưng đây là tái lập theo phương pháp kết hợp mở rộng, không phải tái tạo số liệu tuyệt đối.

Hai, Spearman trên tập EDA không rò rỉ cho thấy Aluminium và Tin có tương quan mức giá với Copper cao hơn WTI, Gold và Silver trong phạm vi biến được khảo sát. Điều đó tạo lý do xây dựng các FS mới, nhưng không đủ chứng minh khả năng dự báo.

Ba, residual hoạt động bằng cách cho mạng học phần hiệu chỉnh so với giá phiên trước. Skip connection truyền level (P_{t-1}), còn mạng học (\Delta P_t). Cách đặt target này phù hợp hơn với EDA về tính dừng và giải thích vì sao residual tránh được lỗi level lớn của Direct.

Bốn, residual thắng Direct rõ rệt trong nhiều thí nghiệm, nhưng thường chỉ ngang hoặc hơn Naive rất ít. Vì vậy phát biểu khoa học nhất không phải ‘mạng dự báo biến động rất tốt’, mà là ‘biểu diễn residual tạo baseline mạnh và ổn định; giá trị gia tăng vượt persistence còn nhỏ và cần kiểm định trên test độc lập’.

Đóng góp của nghiên cứu không chỉ là tìm một MAE thấp, mà là xây dựng chuỗi lập luận có kiểm toán: EDA tạo giả thuyết, feature set kiểm tra giả thuyết, Direct/Residual tạo đối chứng, và Naive xác định mô hình có thực sự tạo thêm giá trị hay không.

Em xin cảm ơn cô và các bạn.”

---

# Bộ câu hỏi phản biện khó và câu trả lời mẫu

## 1. “Tương quan Spearman 0,917 có nghĩa Aluminium dự báo Copper rất tốt phải không?”

“Không. Đó là tương quan đồng thời trên mức giá và có thể bị chi phối bởi xu hướng chung của hai chuỗi không dừng. Khả năng dự báo cần quan hệ có độ trễ và kiểm tra ngoài mẫu. Spearman chỉ dùng để hình thành giả thuyết feature.”

## 2. “Tại sao không dùng Pearson?”

“Pearson vẫn tính được khi dữ liệu không chuẩn; không chuẩn không làm hệ số Pearson vô nghĩa. Nhóm ưu tiên Spearman vì cần quan hệ đơn điệu theo rank và muốn giảm nhạy với phân phối lệch, đuôi dày, ngoại lệ. Nếu suy luận p-value, vẫn phải xét phụ thuộc chuỗi thời gian.”

## 3. “Tại sao chọn Aluminium và Tin mà bỏ Nickel dù Nickel cũng tương quan cao?”

“Sáu FS được xây theo continuity với paper và các ablation đã định trước, không phải top-k tối ưu toàn bộ chín biến. Nickel cao là bằng chứng rằng phạm vi feature hiện tại chưa đầy đủ. Muốn kết luận bộ tối ưu, cần thêm Nickel, Iron Ore, Brent vào feature selection train-only và ablation ngoài mẫu.”

## 4. “FS06 tốt hơn FS04 có chứng minh WTI tốt hơn Gold và Silver không?”

“Không. FS04 sang FS06 đồng thời bỏ Gold-Silver, thêm WTI và giảm số kênh từ năm xuống bốn. Đây là so sánh hai cấu hình, không phải hiệu ứng riêng của WTI. Cần factorial ablation chỉ thay một biến mỗi lần.”

## 5. “Residual có phải là ResNet không?”

“Không theo nghĩa kiến trúc residual block trong CNN. Ở đây residual là cách định nghĩa target: dự báo (P_t-P_{t-1}), rồi cộng lại (P_{t-1}). Điểm giống là đều có đường tắt; nhưng đối tượng và kiến trúc khác nhau.”

## 6. “Residual học return hay sai phân?”

“Nó học sai phân giá trên thang MinMax: (z_t-z_{t-1}=(P_t-P_{t-1})/(P_{max}-P_{min})). Nó không học percentage return (P_t/P_{t-1}-1).”

## 7. “Tại sao residual có MAE tốt nhưng delta correlation gần 0?”

“Vì dự báo mức giá được neo vào giá thật hôm trước. Chỉ cần dự báo delta gần 0 đã tạo một dự báo mức rất mạnh. MAE level thấp không đồng nghĩa dự báo đúng độ lớn hoặc hướng thay đổi. Vì vậy phải so với Naive và audit delta.”

## 8. “Nếu residual chỉ ngang Naive thì dùng deep learning để làm gì?”

“Kết quả đó cho thấy trong giao thức hiện tại deep learning chưa tạo giá trị gia tăng ổn định so với persistence. Đây là một kết quả khoa học có ý nghĩa: nó ngăn overclaim và chỉ ra cần feature có khả năng dự báo trễ tốt hơn, loss phù hợp hơn hoặc đánh giá theo regime.”

## 9. “FS03 và FS05 không có Copper, vậy có thật sự không dùng Copper?”

“Trong tensor (X), đúng là không có kênh Copper. Nhưng với residual, hệ thống vẫn dùng (P_{t-1}) để hoàn nguyên mức giá. Vì vậy phải gọi là ‘bỏ Copper khỏi tensor đầu vào’, không gọi là ‘hệ thống hoàn toàn không dùng lịch sử Copper’.”

## 10. “Có được so MSE 0,00195 của paper với MAE 0,07 của nhóm không?”

“Không. Khác metric, scale, dữ liệu và kỳ test. Muốn so công bằng phải chạy hai model trên cùng timestamp, cùng preprocessing, cùng metric sau inverse-transform.”

## 11. “Tại sao dùng 22 phiên?”

“Đây là độ dài theo thiết kế paper và xấp xỉ một tháng giao dịch. Trong phạm vi kết quả hiện tại, nhóm chưa chứng minh 22 là tối ưu. Cần tuning lookback trên validation theo thời gian.”

## 12. “Bốn split có chứng minh độ bền không?”

“Chúng cho góc nhìn qua nhiều mốc bắt đầu test, nhưng các test lồng nhau nên không độc lập. Đây là kiểm tra độ nhạy mô tả; để xác nhận độ bền cần nhiều cửa sổ walk-forward không chồng lấn hoặc kiểm định ghép cặp đúng timestamp.”

## 13. “Tại sao Direct tệ đến vậy ở một số mô hình?”

“Direct phải học và ngoại suy level. Khi test nằm ở regime giá khác, đầu ra có thể co về vùng train. Residual dùng giá thật (t-1) làm level nên tránh lệch này. Tuy nhiên cần kiểm tra thêm implementation, scaler và loss để chắc chênh lệch không đến từ bất công giao thức.”

## 14. “Kết luận một câu về đề tài?”

“Trên dữ liệu 2016-2026, residual là cách biểu diễn ổn định hơn direct cho dự báo mức giá Copper một phiên trước, nhưng phần lớn lợi thế đến từ neo persistence; khả năng tạo skill vượt Naive còn nhỏ và chưa ổn định.”

---

# Checklist trước khi lên trình bày

- Dùng số EDA official: 1.951 dòng, kết thúc 06/12/2023; không dùng 2.195 dòng trong đoạn Mythos cũ.
- Ghi rõ đơn vị metric: MAE/RMSE trên giá gốc USD/lb; MSE paper trên thang chuẩn hóa.
- Mỗi biểu đồ residual nên có Naive hoặc ít nhất ghi MAE Naive.
- Không gọi Spearman là causal hoặc predictive importance.
- Không nói FS06 cô lập tác động của WTI.
- Không nói FS03/FS05 residual hoàn toàn không dùng Copper.
- Không gọi bốn split là bốn test độc lập.
- Không nói Granger “không ý nghĩa”; kết quả V26 cũ là lỗi kỹ thuật.
- Khi nói “tốt nhất”, thêm phạm vi: trong lưới đã quan sát, tại split/FS/strategy cụ thể.
- Học thuộc ba công thức: (r_t=z_t-z_{t-1}), (\hat z_t=z_{t-1}+\hat r_t), và (\hat z_t-z_t=\hat r_t-r_t).
