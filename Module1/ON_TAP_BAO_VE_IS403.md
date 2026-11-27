# Ghi chú ôn bảo vệ IS403 Module 1

## Mạch trình bày chính

**Dữ liệu → EDA → Direct và Residual → GRU → Paper và Reoptimized SA-LSTM → đánh giá.** Dùng GRU để giải thích kiến trúc. Giữ Paper SA-LSTM và Reoptimized SA-LSTM cạnh nhau để trả lời câu hỏi về việc tìm lại siêu tham số. LSTM thường, SimpleRNN và Ridge có thể đưa vào phụ lục. Dành iTransformer cho phần benchmark toàn nhóm. Phạm vi chính thức gồm 9 mô hình được huấn luyện; không còn các biến thể kết hợp CNN hoặc mô hình không huấn luyện.

## 1. Dữ liệu và chống rò rỉ

- Mục tiêu là dự báo giá đóng cửa Copper trước **một phiên**, từ **22 quan sát** đã có. CSV có **2.710 phiên**, từ 13/01/2016 đến 13/11/2026.
- EDA chính thức dùng **1.951 dòng đầu đến 06/12/2023**, là phần train ban đầu của split 80/20 sau khi tách validation. Từ 1.951 dòng tạo được **1.929 mẫu train** vì 22 dòng đầu cung cấp lịch sử. Split 80/20 Expanding có 217 mẫu validation và 542 mẫu test.
- Bốn split 80/20, 90/10, 92,5/7,5 và 95/5 có lần lượt **542, 271, 204, 136** phiên test. Validation là khoảng 10% cuối phần trước test. Các tập test lồng nhau và cùng ngày kết thúc.
- Với phiên test đầu tiên **07/10/2024**, lấy 22 dòng từ **05/09 đến 04/10/2024**. Giá Copper phiên trước là **4,5740**, giá thật là **4,5665**. Sai phân là **−0,0075 USD/lb**, lợi suất đơn khoảng **−0,164%**. Sai số dự báo là giá thật trừ giá dự báo.
- 22 là số quan sát, xấp xỉ một tháng giao dịch. Chưa có thí nghiệm trong phần trình bày chứng minh đây là độ dài tối ưu. Khoảng cách lớn nhất trong CSV là **66 ngày**, từ **13/03/2023 đến 18/05/2023**. Return qua khoảng này dùng hai giá quan sát liên tiếp.
- Fit riêng `xsc` và `ysc` trên train. Validation/test chỉ transform. Giá hôm qua trong test được đưa vào cửa sổ hôm nay nếu đã công bố tại giờ dự báo. Holdout giữ trọng số cố định. Audit theo ngày trong CSV chưa xác nhận giờ công bố từng thị trường.

Nguồn [Notebook 01](offical-notebooks/01_EDA_va_Kiem_Dinh_Thong_Ke.ipynb) · [Notebook 02](offical-notebooks/02_Train_SA_LSTM_Series.ipynb) · [CSV nguồn](../data/processed/dataset_9_commodities_daily.csv)

## 2. Tính dừng, H0/H1 và các kiểm định EDA

**Cách nhớ tính dừng.** Chuỗi vẫn lên xuống. Với tính dừng yếu, trung bình và phương sai ổn định theo thời gian, còn tương quan phụ thuộc vào khoảng cách giữa các phiên. Hình dung giá dao động quanh 4 với biên độ tương tự qua các tháng. Nếu mặt bằng chuyển dần từ 4 lên 5 rồi 6, đó là dấu hiệu không dừng. Biên độ thay đổi rõ cũng là dấu hiệu cần xem xét.

**H0** là giả thuyết gốc. **H1** là giả thuyết đối. Ở mức ý nghĩa 5%, p < 0,05 thì bác bỏ H0. p ≥ 0,05 nghĩa là chưa đủ bằng chứng bác bỏ. H0 của **ADF** là có nghiệm đơn vị. H0 của **KPSS** là dừng quanh hằng số. Mẹo đọc khi hai test nhất quán là **ADF p nhỏ và KPSS p lớn** ủng hộ tính dừng. Nếu hai test đưa kết luận trái chiều, xem lại lag, đặc tả xu hướng và khả năng có điểm gãy.

| Phép kiểm tra | Số liệu cần nhớ | Câu giải thích |
|---|---|---|
| ADF và KPSS | Giá Copper ADF **0,432**, KPSS **0,01**. Sai phân ADF **6,31 × 10⁻³⁰**, KPSS **0,10** | Mức giá có dấu hiệu không dừng. Kết quả ủng hộ tính dừng của sai phân theo đặc tả kiểm định. 0,01 và 0,10 là biên bảng tra KPSS. |
| Spearman mức giá | Copper với Aluminium **0,9170**, Tin **0,8427**, WTI **0,7546**, Gold **0,6831**, Silver **0,6695** | Cùng xu hướng giá có thể cho tương quan lớn. Giá trị này chưa xác nhận tín hiệu dự báo. |
| Tương quan return có độ trễ | Gold lag 1, Spearman **0,09309**, p sau BH **0,00116**. Pearson **0,06595**, p sau BH **0,06882** | Liên hệ hạng nhỏ. Qua ngưỡng 5% với Spearman. BH hiệu chỉnh riêng 30 phép Pearson và 30 phép Spearman, gồm 5 biến ngoại sinh × 6 lag. |
| Brown–Forsythe | **975 + 975** returns, chuẩn sai **1,185% → 1,521%**, tăng **28,33%**, p sau BH **1,35 × 10⁻⁹** | Hai nửa theo thời gian có độ phân tán khác nhau. Điểm chia do vị trí giữa tập EDA xác định. Đây là lý do thử rolling và recency. |
| Ljung–Box trên returns | p tại lag 5, 10, 22 lần lượt **0,46872 / 0,03336 / 0,03574** | Thử đồng thời các tự tương quan đến lag đang xét. Các ngưỡng này là danh nghĩa. |
| Ljung–Box trên returns² và ARCH-LM | Tại lag 5, p lần lượt **9,32 × 10⁻⁸** và **1,20 × 10⁻⁶** | Cú biến động lớn có thể tập trung gần nhau. Điều này khác câu hỏi về nghiệm đơn vị và khác khả năng đoán chiều. |
| Jarque–Bera và robust-z | JB **498,45**, p **5,80 × 10⁻¹⁰⁹**, excess kurtosis **2,454**, có **23 phiên** vượt ngưỡng robust-z **3,5** | Bằng chứng chống phân phối chuẩn. Robust-z là quy tắc gắn cờ để xem lại phiên. Boxplot dùng râu **1,5 × IQR** trong từng nửa. |
| Granger V26 | **20/20 lượt lỗi** vì `verbose` không được hàm chấp nhận | V26 dùng **2.195 dòng đến 12/11/2024**. Chưa có kết quả Granger hợp lệ để diễn giải. |

**Hình dùng khi bảo vệ.** Boxplot trên 1.950 simple returns Copper, nửa đầu **14/01/2016–12/11/2019**, nửa sau **13/11/2019–06/12/2023**. Đường trong hộp là median, mép hộp là Q1/Q3, chấm ngoài râu là phiên cần xem lại. Lưới **3 × 2** vẽ mức giá MinMax của sáu FS cùng trục ngày. MinMax tính từ `train_eda` để so hình dạng các chuỗi. Notebook 01 đã chạy đủ cell trong Linux. Windows gặp lỗi Application Control khi nạp DLL SciPy. ZIP đã được kiểm tra và chứa hai PNG.

Nguồn [Kết quả EDA](results/01_eda_results/) · [ZIP EDA](../01_eda_results.zip) · [Notebook 01, cell 6–7 và 15–16](offical-notebooks/01_EDA_va_Kiem_Dinh_Thong_Ke.ipynb)

## 3. Từ EDA đến Residual

1. **Quan sát.** Mức giá có dấu hiệu không dừng, còn ADF và KPSS ủng hộ tính dừng của sai phân. Điều này gợi ý thử cho mô hình học mức biến động thay vì học toàn bộ mặt bằng giá.
2. **Thiết kế.** Direct học mức giá scaled. Residual học `y - prev`, tức sai phân trên cùng thang của `ysc`. Lấy đầu ra cộng `prev` trên thang scaled rồi mới inverse-transform ra USD/lb. Nếu dự báo sai phân bằng 0, giá dự báo bằng giá phiên trước.
3. **Kiểm chứng.** So MAE/RMSE ngoài mẫu với Direct cùng backbone, FS, ngày test và giao thức. EDA tạo giả thuyết. Sai số test đánh giá giả thuyết.

**FS theo đúng thứ tự cột.** FS01 Copper. FS02 Copper, WTI, Gold, Silver. FS03 WTI, Gold, Silver. FS04 Copper, Aluminium, Tin, Silver, Gold. FS05 Aluminium, Tin, Silver, Gold. FS06 Copper, Aluminium, Tin, WTI. FS03/FS05 bỏ Copper khỏi `X`, còn `prev` lấy từ Copper phiên trước ở bước dựng giá.

## 4. Vẽ LSTM và GRU

- **Unit/node.** `LSTM(64)` hoặc `GRU(64)` tạo trạng thái ẩn 64 chiều ở mỗi phiên. Các trọng số dùng lại qua cả 22 phiên. Mỗi unit có thể đọc cả vector biến đầu vào và trạng thái trước.
- **LSTM.** Có hai trạng thái `h` và `c`. Forget gate quyết định giữ bao nhiêu bộ nhớ cũ. Input gate điều tiết phần ứng viên ghi vào bộ nhớ. Output gate điều tiết phần bộ nhớ đưa ra trạng thái ẩn. Vẽ `c_mới = forget × c_cũ + input × ứng_viên` và `h_mới = output × tanh(c_mới)`.
- **GRU.** Có một trạng thái `h`. Reset gate điều tiết lịch sử khi tạo ứng viên. Update gate trộn trạng thái cũ và ứng viên. Theo Keras, `h_mới = z × h_cũ + (1 − z) × ứng_viên`. Khi z = 0,9 trên một tọa độ, giữ 90% phần cũ.
- **Shape FS06.** `(B,22,4) → (B,22,64) → (B,32) → (B,16) → (B,1)`. Tầng đầu bật `return_sequences`, tầng hai đọc cả 22 trạng thái rồi trả trạng thái cuối. Dropout 0,20 sau hai tầng recurrent. Dense 16 dùng ReLU, Dense 1 dùng linear. Với FS03, input là `(B,22,3)`. Lưới Holdout chính thức của Notebook 03 chạy đủ FS01–FS06; Walk-Forward chỉ chạy FS06.
- **Đếm tham số.** Với d kênh vào, u units, LSTM có `4u(d+u+1)`, GRU Keras mặc định `reset_after=True` có `3u(d+u+2)`. Số 4 và 3 tính cả phép chiếu tạo ứng viên. Tầng đầu FS06 có **17.664** tham số LSTM và **13.440** tham số GRU.
- **Các mô hình phụ.** SimpleRNN cập nhật một trạng thái bằng tanh và không có cổng. Ridge Residual duỗi `(B,22,F)` thành `(B,22F)` và hồi quy sai phân scaled với alpha 1,0.

Nguồn [Notebook 02, kiến trúc SA-series](offical-notebooks/02_Train_SA_LSTM_Series.ipynb) · [Notebook 03, kiến trúc Non-SA](offical-notebooks/03_Train_NonSA_Baselines_va_WalkForward.ipynb)

## 5. Paper và Reoptimized SA-LSTM — số liệu ưu tiên học

**Paper SA-LSTM** dùng **39 → 111 units** và learning rate **5 × 10⁻⁵** từ Chen và cộng sự. Module 1 fit lại trọng số trên dữ liệu nhóm, dùng Direct trong Notebook 02 và ghi rõ Dense(32, relu) là giả định tái dựng. Bản Colab Fixed/Tuned và SA-LSTM Residual của Công là các cấu hình khác trong repo.

**Reoptimized SA-LSTM** chạy Fresh Simulated Annealing để tìm `units_1`, `units_2`, `learning_rate`. Nhóm dùng **16 trial/FS**, **tối đa 15 epoch/trial**, patience **4**. Tìm trên validation **80/20 Expanding**, chọn theo MSE scaled, rồi cố định bộ chọn được cho lần fit cuối. Notebook 02 mặc định Fresh SA cho sáu FS, tức **96 trial và tối đa 1.440 epoch cho search**. Chi phí fit cuối và walk-forward tính riêng.

**Số đo V22 FS06.** Có **239 epoch thực chạy** trong 16 trial, tổng `fit_seconds` **121 giây**, khoảng **2 phút**. Toàn lượt chạy V22 trên FS06 mất **11,6 phút**. Đây là log của một phiên chạy V22. Toàn lưới Notebook 02 còn phụ thuộc máy, early stopping và các lượt tái huấn luyện.

**Kết quả quan trọng.** Fresh SA chọn **47 → 127 units**, learning rate khoảng **5,05 × 10⁻⁵**. MSE scaled trên validation 80/20 giảm **0,003729 → 0,003001**, khoảng **19,5%** so với trial đầu dùng bộ paper.

| FS06 · Direct · Expanding | MAE Paper, USD/lb | MAE Reoptimized, USD/lb | Bản thấp hơn |
|---|---:|---:|---|
| 80/20 | 0,29895 | **0,22381** | Reoptimized, giảm **25,1%** |
| 90/10 | **0,18463** | 0,35892 | Paper |
| 92,5/7,5 | **0,19487** | 0,36608 | Paper |
| 95/5 | **0,20080** | 0,32543 | Paper |

**Câu nói.** “Fresh SA giảm lỗi validation và giảm MAE test khoảng 25% ở mốc 80/20 trên FS06. Paper có MAE thấp hơn ở ba mốc còn lại. Lợi thế của Reoptimized chưa ổn định khi đổi mốc chia.”

Paper gốc báo MSE chuẩn hóa **0,00195 / 0,00190 / 0,00097** trên dữ liệu và các test window của paper. MAE/RMSE của nhóm tính trên USD/lb sau inverse-transform. Muốn so hai model công bằng, đo cả hai trên cùng dữ liệu, cùng thang đo và cùng giao thức.

Nguồn [PDF Chen et al.](<../doc/Copper price prediction using LSTM recurrent.pdf>) · [Log search V22](<results/18-11- Copper_SA-LSTM-standalone-V22/copper_sa_lstm_wti_only_v22/sa_search_trials.csv>) · [Runtime V22](<results/18-11- Copper_SA-LSTM-standalone-V22/copper_sa_lstm_wti_only_v22/run_summary.md>) · [MAE test V22](<results/18-11- Copper_SA-LSTM-standalone-V22/copper_sa_lstm_wti_only_v22/ratio_summary.csv>)

## 6. Chiến lược train, thời gian và seed

- **Expanding holdout** lấy lịch sử từ đầu, fit một lần. **Rolling 756 holdout** giới hạn lịch sử fit ở 756 dòng, còn 734 mẫu khi mất 22 dòng tạo cửa sổ. **Recency-weighted** vẫn dùng lịch sử expanding, nhưng mẫu cách 252 phiên nhận nửa trọng số tương đối so với mẫu mới nhất.
- **Walk-forward Module 1** ở FS06, 80/20, 542 phiên, 9 fold, tái fit mỗi **63 phiên**. Neural model chạy seed **41–45**. Ridge chạy một lần mỗi fold. Lịch này khác WF của Quân trong bảng nhóm, dùng `W=756, S=40`.
- **Mạng** dùng MSE scaled, Adam, batch 64, early stopping theo `val_loss` và khôi phục trọng số tốt nhất. Holdout Notebook 02 tối đa **100 epoch**, Notebook 03 **65**, patience **9**. Walk-forward tối đa **45 epoch**, patience **7**.
- Báo giờ riêng cho **search**, **fit cuối**, **retrain**. Notebook 02/03 bấm giờ quanh fit mạng. K5 Colab tính cả fit, predict và lưu model khi bật. Khi đo tốc độ, ghép lượt chạy cùng máy, phiên bản, dữ liệu, seed, batch, chính sách dừng và phạm vi bấm giờ.
- Wilcoxon signed-rank dùng khi có đủ lượt đo **ghép cặp và độc lập giữa các lượt**. Đặt chênh lệch là giây iTransformer trừ giây SA-LSTM. Dấu âm nghĩa là iTransformer nhanh hơn ở đại lượng so sánh. Repo thiếu log lặp ghép cặp để kết luận tốc độ K5. Tham số và epoch chỉ giúp giải thích chi phí.

## 7. Đọc benchmark toàn nhóm và DA

**MAE** đo mức sai số tuyệt đối trung bình. **RMSE** nhạy hơn với lỗi lớn. **DA_pct** đo tỷ lệ đúng dấu khi so `P_t − P_(t−1)` và `P̂_t − P_(t−1)`. Ví dụ, giá trước 4,00, giá thật 4,10, dự báo 4,02 thì đúng hướng dù sai 0,08 USD/lb. Dự báo đứng giá chỉ tính đúng khi giá thật đứng yên. DA quanh **45–55%** thường cho thấy tín hiệu hướng yếu. Cần so với tỷ lệ tăng/giảm thật và mốc luôn chọn hướng phổ biến. Nếu 56% phiên tăng, luôn đoán tăng đã đạt 56%. Trên 271 phiên, 50% lên 55% tương ứng khoảng 14 phiên đúng thêm.

### MAE thấp nhất theo từng FS tại 90/10

| FS | Residual | MAE | Direct | MAE |
|---|---|---:|---|---:|
| FS01 | Vĩ iTransformer | **0,07681** | Quân WF Expanding | 0,20265 |
| FS02 | Khoa LSTM và Vĩ iTransformer, đồng hạng ở số hiển thị | **0,07701** | Quân WF Expanding | 0,23696 |
| FS03 | Quân WF Expanding | **0,07693** | Quân Static Train-Only | 0,31990 |
| FS04 | Vĩ iTransformer | **0,07687** | Khoa GRU | 0,24082 |
| FS05 | Quân WF Expanding | **0,07720** | Quân WF Expanding | 0,36182 |
| FS06 | Khoa LSTM | **0,07704** | Khoa Paper SA-LSTM | 0,18463 |

- **Module 1 ở FS04.** GRU Residual đạt **0,07695**, thấp hơn SA-LSTM Residual của Công **0,08097** trên cùng FS04, khoảng **4,97%**. Cấu hình tốt nhất của Công là FS01 **0,07695**, bằng GRU FS04 ở độ chính xác hiển thị. Muốn tuyên bố ý nghĩa cho cặp GRU–SA-LSTM phải đọc kết quả kiểm định của đúng cặp dự báo trên cùng timestamp.
- **Cross-ratio.** Các test ở bốn mốc chia lồng nhau, nên chỉ mô tả độ ổn định của từng mô hình qua các mốc. ARIMA của Huy có MAE trung bình bốn mốc **0,07331**, còn SA-LSTM của Công **0,07334**.
- **Ablation biến.** iTransformer Residual **FS02 → FS03** tăng MAE **0,07701 → 0,09027**, **FS04 → FS05** tăng **0,07687 → 0,07924** khi bỏ Copper khỏi `X`. Các ô thiếu FS03/FS05 ở Non-SA không tham gia trung bình. Khi so trung bình giữa các mô hình, dùng giao của các FS.
- **Tính công bằng.** Bảng nhóm phối hợp holdout của Khoa, Công, Vĩ, walk-forward của Quân và trung bình 5 seed theo giao thức Huy. Công và Vĩ dùng seed 42. Quân có kiểm tra 5 seed ở FS02 90/10. Module 1 có walk-forward 5 seed riêng. Bảng sáu FS cho biết **cấu hình có MAE thấp nhất đã quan sát**. Để cô lập tác động kiến trúc, cần cùng objective, FS, ngày test, lịch retrain và seed.

Nguồn [Báo cáo nhóm](../group_benchmark_summary/THONG_KE_CHISO_SIGNIFICANCE_ABLATION.md) · [Ablation sáu FS](../group_benchmark_summary/group_ablation_6_feature_sets_all_members.csv) · [Cross-ratio](../group_benchmark_summary/group_cross_ratio_4_splits_all_members.csv) · [CSV DM](../group_benchmark_summary/group_significance_dm_tests_complete.csv)

## 8. iTransformer và benchmark Colab K5

iTransformer đảo trục từ `(B,22,F)` sang `(B,F,22)`, rồi chiếu mỗi lịch sử mặt hàng thành token 48 chiều. Hai encoder dùng attention **giữa F token**, **4 head × 12 chiều**, FFN 48 → 96 → 48 và LayerNorm trên 48 chiều của mỗi token. Flatten đưa `(B,48F)` vào Dense 32 rồi Dense 1. FS03 có 3 token, head nhận 144 chiều. FS06 có 4 token, head nhận 192 chiều. FS01 có một token, softmax của mỗi head bằng 1 khi suy luận. Lịch sử Copper vẫn đi qua embedding, value projection và head. Phép cộng trong encoder và phép cộng `prev` ở mục tiêu Residual thực hiện ở hai chỗ khác nhau.

**K5 Colab** dùng `Aluminium, Tin, Silver, Gold, Copper`, seed 42, 84/84 job. Lịch 90/10 có 271 phiên. CSV log khác SHA do LF/CRLF, chuẩn hóa xuống dòng thì hai fingerprint khớp.

| 90/10 Expanding | iTransformer MAE | SA-LSTM Fixed MAE | Kết quả |
|---|---:|---:|---|
| Direct | 1,768602 | **0,798291** | Fixed thấp hơn |
| Residual | **0,078119** | 0,082928 | iTransformer thấp hơn **5,80%**, p HAC **0,024835** |

Ở nhánh Residual, Tuned có MAE **0,085346** và p khi so với iTransformer **0,000142**. DA của iTransformer Residual **47,23%**, Fixed **49,08%**. Đây là ví dụ MAE tốt hơn nhưng DA thấp hơn. Giá trị **0,07681** thuộc **iTransformer FS01 chỉ có Copper** trong bảng toàn nhóm.

**DM-HAC.** Ghép theo từng ngày và lấy `d_t = |e_i| − |e_SA|`. H0 là lỗi tuyệt đối trung bình bằng nhau. Dấu âm là iTransformer tốt hơn theo MAE. Với Residual K5, trung bình chênh **−0,004808 USD/lb**, p danh nghĩa **0,024835**. P-value là xác suất thu được thống kê cực đoan ít nhất như đã quan sát nếu H0 và giả định test đúng. Bảng có nhiều kiểm định và p chưa hiệu chỉnh. Repo chưa có file dự báo từng ngày K5 để tự tính lại HAC từ checkout, còn kết quả HAC được lưu trong notebook Colab.

Nguồn [Benchmark K5, cell 7, 9, 18, 20 và 24](<../Vi Task/iTransformer_vs_SA_LSTM_K5_Fair_FINAL_Colab.ipynb>)

## 9. Câu trả lời nhanh trước hội đồng

1. **Tách theo thời gian đã đủ chống rò rỉ chưa?** Chưa. Vẫn phải kiểm tra biên cửa sổ, chỉ fit scaler trên train và xác minh giờ dữ liệu ngoại sinh sẵn có.
2. **Vì sao cần hai scaler?** `xsc` chuẩn hóa các biến đầu vào, còn `ysc` chuẩn hóa riêng Copper mục tiêu. Cả hai chỉ fit trên train để giữ đúng đơn vị khi inverse-transform.
3. **Vì sao giữ Direct khi EDA ủng hộ sai phân?** Direct cho đối chứng cùng backbone khi đổi mục tiêu, để kết quả test kiểm tra lợi ích thực tế của Residual.
4. **Tại sao giữ cả Paper và Reoptimized SA-LSTM?** Paper là cấu hình tham chiếu từ nghiên cứu trước; Reoptimized kiểm tra liệu tìm lại siêu tham số trên dữ liệu nhóm có cải thiện ngoài mẫu hay không.
5. **Một token thì bốn head có ích thế nào?** Phép chiếu value và các tầng tiếp theo vẫn hoạt động. Attention không có token hàng hóa thứ hai để lựa chọn.
6. **Boxplot và robust-z có phải cùng danh sách ngoại lệ?** Boxplot dùng IQR từng nửa, robust-z dùng median và MAD toàn chuỗi tiền kiểm. Mỗi quy tắc có thể gắn cờ khác nhau.
7. **Giá dự báo sát đường thật có đủ chứng minh mô hình tốt?** Xem MAE/RMSE cùng ngày test và so với giá phiên trước. Check thêm sai phân dự báo để thấy mô hình học được gì về thay đổi.
8. **Ít tham số đã chứng minh nhanh?** Số tham số là kết quả đếm. Tốc độ cần log thời gian cùng phạm vi và điều kiện.
9. **Notebook 04 chọn mô hình thế nào?** `best_per_split` chọn MAE test thấp nhất để mô tả kết quả đã quan sát. Quy trình chọn mô hình trước test phải dùng validation, như `validation_winner` trong Colab.
10. **Kết luận gọn cho Module 1?** “EDA cho lý do thử học sai phân và ưu tiên dữ liệu gần đây. Một số cấu hình Residual có MAE thấp hơn nhẹ ở từng mốc. Với FS06 Direct, Fresh SA cải thiện ở 80/20 và Paper giữ lợi thế ở ba mốc còn lại.”

Nguồn [Notebook 04, tổng hợp và đánh giá](offical-notebooks/04_Evaluate_va_Tong_Hop_Ket_Qua.ipynb)
