# Audit notebook LSTM–SA (13/11/2026)

## Phạm vi và bằng chứng

Đã đọc toàn bộ source và output lưu trong `copper-lstm-sa_result_lasted.ipynb` (19 cells), notebook tiền nhiệm `Copper_LSTM_SA.ipynb` (16 cells), và PDF `../Copper price prediction using LSTM recurrent.pdf` (19 trang). Output lưu là bằng chứng của **lần chạy cũ**, không phải kết quả của framework mới. Không có CSV dữ liệu gốc trong workspace để kiểm tra lại các con số hoặc thời điểm công bố của từng chỉ báo. Notebook cũ được giữ nguyên.

## Đối chiếu phương pháp

| Thành phần | Phân loại | Bằng chứng / ảnh hưởng |
|---|---|---|
| 8 biến ứng viên, chọn WTI/Gold/Silver bằng Spearman dương > 0.8 | Confirmed by paper | PDF tr. 7–8. Bài báo tính trên dữ liệu Investing.com 1990–2009; notebook tính lại trên Yahoo 2006–2026. |
| Xóa ngày thiếu, MinMax [0,1], ba tỷ lệ 90/10, 92.5/7.5, 95/5 | Confirmed by paper | PDF tr. 8, 12. Chi tiết ranh giới validation và thời điểm fit scaler không được bài báo mô tả rõ. |
| 2 tầng LSTM, 39/111 neurons, LR 5e-5, batch 64, 100 epochs, dropout .2 sau Dense đầu | Confirmed by paper | PDF tr. 9, 12, 15 (Table 4). Dense(32), activation ReLU, output tanh, Adam là implementation assumptions; Table 4 không xác nhận các giá trị đó. |
| Baseline 32/32, LR .001; lookback 30; validation là 10% cuối phần train; 15 epochs/trial; giới hạn neurons 8–128 | Implementation assumption | Lựa chọn của notebook. Không được gọi là tái hiện chính xác bài báo. |
| SA 30 bước, cooling .95, bắt đầu LR .0005/15/30 | Deviation from paper | PDF tr. 7 ghi cooling .97; tr. 9 ghi .95, T đầu/cuối 1/.1; tr. 10 ghi 50 iterations/process và tổng 650 scenarios. Các mô tả bài báo tự mâu thuẫn. Notebook có 31 evaluations/ratio kể cả điểm đầu. |
| Learning rate được SA tìm trong [1e-6,.005] | Deviation from paper | PDF tr. 10 nói phạm vi (0,.005), ngoài phạm vi reset về 5e-5; tr. 11 lại nói LR luôn là 5e-5. Thí nghiệm tối ưu LR mới chỉ là phần mở rộng. |
| Ngưỡng Spearman >.8 rồi fallback top 3 | Deviation from paper, documented | Output cũ: không biến nào >.8 ở cả 3 split; WTI/Silver/Gold được chọn bằng fallback. Cần ghi quy tắc fallback trong mọi run. |
| Metropolis | Potential bug in paper | PDF tr. 7 in `exp(ΔE/T)` cho nghiệm kém, cho xác suất >1 với ΔE>0; code dùng `exp(-ΔE/T)` hợp lệ. Scale loss riêng của notebook là implementation assumption. |
| MSE | Implementation assumption | PDF tr. 9 định nghĩa MSE trên test nhưng không nêu chắc chắn đơn vị của các số công bố. Notebook báo cả MSE scaled và USD², không so sánh trực tiếp với số bài báo. PDF tr. 12 ghi loss .000569, tr. 16 ghi .00569: mâu thuẫn dấu thập phân. |

## Kiểm tra tính đúng đắn của notebook hiện tại

- **Nhãn t+1:** `X[t] = x[t-30:t]`, `y[t] = Copper[t]`; đầu vào cuối là ngày t-1, nên đây là dự báo ngày kế tiếp sau ngày quan sát cuối. Không có Copper quá khứ trong X. Chỉ báo WTI/Gold/Silver ở ngày t-1 có thể có múi giờ và giờ công bố khác giá đóng cửa đồng; cần quy định thời điểm ra quyết định sau khi tất cả giá đóng cửa đã có. Không thể xác minh availability thực tế từ CSV cũ.
- **Horizon của bài báo không được xác nhận:** PDF không cho một chỉ số lag/cửa sổ rõ ràng để chứng minh tác giả dự báo t hay t+1. Tương quan Spearman của bài báo là giữa các biến cùng ngày; đó không tự xác định thời điểm dự báo. Framework mới định nghĩa t+1 một cách tường minh, nên là giả định triển khai khi so với bài báo.
- **Split/window/scaler:** chia theo thứ tự, scaler và Spearman fit trên phần train trước validation; cửa sổ validation/test chỉ lấy hàng quá khứ, có thể đi qua ranh giới tập. Không thấy leakage trực tiếp từ test vào SA. Các ngày trùng được bỏ sau inner join nhưng notebook không ghi số lượng bị bỏ; output chỉ có tổng 5,232 hàng hoàn chỉnh.
- **Mối nguy lớn nhất:** output `tanh` bị chặn trong [-1,1] trong khi target MinMax có thể vượt 1 ở validation/test. Output cũ có MSE test cao dù val SA thấp, phù hợp với bão hòa hoặc distribution shift nhưng chưa đủ bằng chứng để kết luận nguyên nhân. Cần ablation `linear/sigmoid/tanh` và tỷ lệ nhãn >1.
- **Selection bias:** SA chọn min của 31 validation evaluations trên đúng một seed và 15 epochs, sau đó train 100 epochs. Ranking trial ngắn không đảm bảo giữ nguyên ở 100 epochs. Cần log val 100 epochs cho paper preset/baseline và so sánh với SA trên cùng ngân sách; chỉ dùng test một lần sau chốt lựa chọn.
- **Không có persistence baseline:** với giá hàng hóa, điều này có thể làm mọi MSE LSTM trông tốt dù không vượt dự báo `Copper[t-1]`.
- **So sánh split không đồng nhất:** 90/10, 92.5/7.5, 95/5 có các kỳ test khác nhau; không suy ra tác động thuần túy của tỷ lệ train từ MSE giữa chúng.
- **Output lưu:** SA mới chọn `(34,128,.000917)`, `(88,114,.000904)`, `(83,128,.000467)` cho ba split; MSE USD² test lần lượt 1.119, 1.875, 2.501, lớn hơn paper preset 0.856, 1.084, 1.688. Đây là kết quả cũ của một seed trên tập mới, không phải bằng chứng SA kém hơn nói chung. Notebook tiền nhiệm có output tuyến tính và mặc định dùng paper preset; cấu hình đã đổi ở notebook mới và phải được gắn nhãn tách biệt.
- **Reproducibility:** không có manifest, version đầy đủ, checkpoint/resume trial, dữ liệu nguồn đóng băng, GPU time, hoặc tập kiểm thử. TensorFlow 2.21.0 và yfinance .2.66 chỉ xuất hiện trong output cũ.

## Thiết kế triển khai

Tạo notebook điều phối từng phase, module `research.py` giữ split/preprocess/model/search/logging, `experiment_config.json` là nguồn cấu hình, `tests/` kiểm tra alignment/leakage/shape. Mỗi fold có scaler và Spearman riêng. Screening, SA/TPE và robustness dùng validation; test chỉ chạy trong phase final với cấu hình đã chốt. Mọi trial ghi ngay xuống `/kaggle/working/`, resume theo experiment ID và kiểm tra fingerprint dữ liệu/cấu hình. Chạy benchmark ngắn trước khi tăng ngân sách GPU. Giữ notebook cũ để đối chiếu, không sửa ngầm output lịch sử.
