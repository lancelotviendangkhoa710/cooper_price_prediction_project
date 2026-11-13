# Đánh giá kết quả notebook LSTM–SA

**Notebook được đánh giá:** [`13-11-2026_tai_hien_gia_dong_LSTM_SA.ipynb`](13-11-2026_tai_hien_gia_dong_LSTM_SA.ipynb)  
**Trạng thái:** Chưa có kết quả huấn luyện trong workspace. Notebook chưa được chạy trên Kaggle/TensorFlow, vì vậy tài liệu này đánh giá thiết kế và khả năng kiểm chứng; không bịa chỉ số mô hình.

## 1. Kết luận hiện tại

Notebook có quy trình phù hợp để so sánh bốn phương pháp dự báo: naive persistence, Vanilla LSTM, cấu hình LSTM theo tham số bài báo, và SA-LSTM tối ưu lại. SA dùng validation MSE để chọn siêu tham số; tập test chỉ được dùng ở bước đánh giá cuối. So sánh chính thức cần chạy `FULL_RESEARCH_MODE`; `QUICK_MODE` chỉ xác nhận pipeline.

Chưa thể kết luận SA cải thiện LSTM. Kết luận đó cần số đo test sau khi hoàn tất full run, cùng chênh lệch so với cả Vanilla LSTM và cấu hình bài báo. Một lần chạy với một seed và một giai đoạn test chỉ cung cấp bằng chứng mô tả trên tập dữ liệu này, không chứng minh hiệu quả tổng quát.

## 2. Bằng chứng dữ liệu đã kiểm tra tại máy

CSV hiện có gồm 5.231 dòng từ 03/01/2006 đến 29/10/2026, không có ngày trùng hoặc giá trị khuyết. Với split 90/10 và lookback 22, pipeline tạo 4.214 mẫu huấn luyện, 471 mẫu validation và 524 mẫu test.

Spearman giữa Copper và ba biến đầu vào, tính trên phần train của split 90/10:

| Biến | Spearman với Copper |
|---|---:|
| WTI | 0,711 |
| Silver | 0,614 |
| Gold | 0,413 |

Các tương quan này thấp hơn các giá trị được bài báo báo cáo cho dữ liệu Investing.com 1990–2009. Đây là khác biệt về giai đoạn và nguồn dữ liệu; không phải bằng chứng lỗi mô hình.

## 3. K?t qu? th?c nghi?m ???c cung c?p

B?ng d??i ??y ch?p t? k?t qu? b?n g?i. MSE gi??, RMSE v? MAE d?ng ??n v? gi? ??ng g?c; MSE chu?n h?a d?ng thang MinMax.

| Train/test | M? h?nh | N test | Val MSE chu?n h?a | Test MSE chu?n h?a | Test MSE gi?? | RMSE gi? | MAE gi? | R? | Th?i gian (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 90/10 | Persistence | 524 | ? | 0,000880 | 0,01193 | 0,1092 | 0,0693 | 0,9827 | 0 |
| 90/10 | Vanilla LSTM | 524 | 0,04573 | 0,04969 | 0,67347 | 0,8206 | 0,7146 | 0,0241 | 63,08 |
| 90/10 | Paper SA-LSTM | 524 | 0,007071 | 0,08242 | 1,11704 | 1,0569 | 0,9619 | -0,6186 | 63,34 |
| 90/10 | New SA-LSTM | 524 | 0,01159 | 0,02242 | 0,30392 | 0,5513 | 0,4921 | 0,5596 | 63,17 |
| 92,5/7,5 | Persistence | 393 | ? | 0,001023 | 0,01387 | 0,1178 | 0,0743 | 0,9744 | 0 |
| 92,5/7,5 | Vanilla LSTM | 393 | 0,01873 | 0,02301 | 0,31189 | 0,5585 | 0,4931 | 0,4241 | 63,79 |
| 92,5/7,5 | Paper SA-LSTM | 393 | 0,02027 | 0,09541 | 1,29317 | 1,1372 | 1,0500 | -1,3879 | 65,40 |
| 92,5/7,5 | New SA-LSTM | 393 | 0,02114 | 0,01788 | 0,24231 | 0,4923 | 0,4356 | 0,5526 | 64,41 |
| 95/5 | Persistence | 262 | ? | 0,000759 | 0,01029 | 0,1014 | 0,0767 | 0,9629 | 0 |
| 95/5 | Vanilla LSTM | 262 | 0,02514 | 0,17790 | 2,41110 | 1,5528 | 1,4652 | -7,6868 | 64,47 |
| 95/5 | Paper SA-LSTM | 262 | 0,03020 | 0,05211 | 0,70626 | 0,8404 | 0,7054 | -1,5445 | 65,10 |
| 95/5 | New SA-LSTM | 262 | 0,01158 | 0,11171 | 1,51403 | 1,2305 | 1,1398 | -4,4548 | 64,96 |

## 4. Di?n gi?i

- **Persistence t?t nh?t tr?n c? ba test period.** So v?i persistence, test MSE gi?? c?a New SA cao h?n kho?ng 25,5 l?n ? 90/10, 17,5 l?n ? 92,5/7,5 v? 147 l?n ? 95/5. V? v?y kh?ng LSTM n?o v??t m?c d? b?o Copper b?ng gi? phi?n tr??c.
- New SA t?t h?n Vanilla v? Paper SA ? 90/10 v? 92,5/7,5: MSE gi?m t??ng ?ng 54,9% v? 22,3% so v?i Vanilla; gi?m 72,8% v? 81,3% so v?i Paper SA.
- ? 95/5, Paper SA t?t h?n New SA (MSE 0,7063 so v?i 1,5140 gi??). New SA c?ng kh?ng ?n ??nh qua c?c t? l?.
- Validation kh?ng lu?n d? b?o ??ng th? h?ng test. Ch?ng h?n 90/10, Paper SA c? validation MSE th?p nh?t trong ba LSTM (0,00707) nh?ng test MSE cao nh?t (0,08242 chu?n h?a). C? th? c? d?ch chuy?n ph?n ph?i ho?c qu? kh?p validation; b?ng n?y ch?a ?? x?c ??nh nguy?n nh?n.
- R? ?m l?n cho th?y m?t s? LSTM d? b?o c?n k?m h?n d? ?o?n b?ng gi? tr? trung b?nh c?a test period.

**K?t lu?n:** SA m?i c?i thi?n so v?i hai c?u h?nh LSTM ? hai split ??u, nh?ng ch?a cho th?y gi? tr? d? b?o th?c t? v? v?n thua persistence ? c? ba split. ??y l? m?t seed (42) v? c?c giai ?o?n test c? th?, n?n kh?ng th? kh?i qu?t th?nh hi?u qu? chung.

## 5. Giới hạn cần giữ khi diễn giải

Notebook dự báo phiên chung tiếp theo từ 22 phiên trước của WTI, Gold và Silver. Thời điểm công bố/đóng cửa chính xác của từng thị trường không có trong CSV, nên giả định khả dụng của các đầu vào cần được hiểu là quyết định sau khi dữ liệu phiên trước đã có.

Bài báo không mô tả rõ horizon, activation của Dense, và có bất nhất về số đầu vào: phần văn bản nêu ba biến, còn số tham số trong Hình 12 hàm ý bốn. Notebook theo ba biến được nêu và ghi rõ đây là lựa chọn triển khai. Dữ liệu hiện tại khác dữ liệu và giai đoạn của bài báo; các MSE công bố không thể so trực tiếp khi thang đo cũng chưa được xác định nhất quán.

**Đánh giá cuối cùng:** quy trình đã có thể kiểm tra rò rỉ và so sánh công bằng theo thiết kế, nhưng hiệu quả dự báo và lợi ích của SA vẫn **chưa được xác minh bằng lần chạy full**.
