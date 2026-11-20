# Module 1 Copper Price Forecasting — Hướng dẫn học và bảo vệ

> Phạm vi: giải thích đúng những gì repository đã chạy, không phải giáo trình tổng quát. Cell được đánh số 0-based theo JSON notebook. Chỉ mục nguồn dùng chung: [`RESEARCH_INDEX.md`](RESEARCH_INDEX.md). Ngày đối chiếu: 20/11/2026.

## 0. Bản đồ phát biểu an toàn

- Bài toán là dự báo `Copper(t)` trước một phiên từ 22 phiên `t−22…t−1`; 2.710 quan sát từ 13/01/2016 đến 13/11/2026. Nguồn: report §1.1; V20 cell 2/8; V22-SA `experiment_manifest.json`.
- “Tốt nhất” chỉ có nghĩa MAE holdout thấp nhất trong các cấu hình fitted đã chạy. Naive được báo riêng và không tham gia xếp hạng fitted. Nguồn: report §2, §9.3.
- Residual-GRU đứng đầu fitted holdout ở bốn split, nhưng trên walk-forward V23 nó gần Naive và không có bằng chứng cải thiện ổn định ở mức 5%. Nguồn: report §2, §8; V23 `wf_seed_summary.csv`, `wf_dm_hac_by_seed.csv`.
- Không nói “WTI gây cải thiện”, “SA luôn tốt hơn”, “chuỗi chắc chắn random walk”, hay “đã loại bỏ mọi leakage”. Các thí nghiệm chưa đủ cho những kết luận đó.

Ký hiệu nguồn ngắn: **V20** = `17-11-Copper-NonSA-Fair-Full-Protocol-V20.ipynb`; **V21** = `17-11-SA-LSTM-standalone-V21.ipynb`; **V22-N** = `18-11- Copper_NonSA_V22_WTI_Only_Kaggle (2).ipynb`; **V22-S** = `18-11-Copper_SA_LSTM_V22_Fresh_SA_WTI_Only_Kaggle.ipynb`; **V23–V26** là notebook top-level cùng số phiên bản.

---

# A. Kiến trúc và phương pháp

## 1. CNN-LSTM và CNN-GRU

### What

Đây là hai kiến trúc nối một bộ trích xuất mẫu cục bộ theo thời gian với mạng recurrent. Tensor `(N,22,F)` đi qua `Conv1D(32, kernel_size=3, padding='causal', activation='relu')`, rồi `MaxPooling1D(2)`. Vì causal convolution không dùng thời điểm tương lai và pooling giảm chiều thời gian từ 22 xuống 11, đầu ra CNN là `(N,11,32)`. Sau đó:

- CNN-LSTM: LSTM 48 có `return_sequences=True` → Dropout 0,20 → LSTM 24.
- CNN-GRU: GRU 48 có `return_sequences=True` → Dropout 0,20 → GRU 24.
- Cả hai: Dropout 0,20 → Dense 16 ReLU → Dense 1 linear.

Nguồn: V20 cell 10, hàm `build_net`; cùng logic ở V22-N cell 10, V23 cell 7 và V25 cell 6. Shape có thể suy trực tiếp từ các layer với input 22; parameter thực tế được kiểm ở `results/model_parameter_counts_v20_v21.csv`.

### Why

Conv1D thử học các mẫu ngắn ba phiên trước khi recurrent layer tổng hợp quan hệ dài hơn trong cửa sổ 22 phiên. Pooling giảm số bước recurrent và độ phức tạp. Đây là giả thuyết kiến trúc, không phải bằng chứng rằng ba phiên là chu kỳ thật của giá đồng.

### How

`build_net(arch, shape)` dùng cùng kiến trúc cho objective Direct và Residual; chỉ target đổi. Adam `lr=1e-3`, loss MSE, batch 64, tối đa 65 epoch, EarlyStopping trên `val_loss`, patience 9, `restore_best_weights=True`. Nguồn: V20 cell 2/10, hàm `train_one`; cell 12 điều phối grid.

### Input/Output

- Input: `(batch,22,F)`, F thuộc 1, 4 hoặc 5 ở V20; F=4 ở V22-N.
- Sau Conv1D causal: `(batch,22,32)`; sau pool: `(batch,11,32)`.
- Recurrent thứ nhất giữ chuỗi `(batch,11,48)`; recurrent thứ hai xuất vector `(batch,24)`.
- Output: `(batch,1)`. Với Direct là Copper scaled; với Residual là delta scaled theo cách pipeline tạo target.

### Configuration và results

| Model | F=1 | F=4 | F=5 | Nguồn parameter |
|---|---:|---:|---:|---|
| CNN-LSTM Direct/Residual | 23.105 | 23.393 | 23.489 | `results/model_parameter_counts_v20_v21.csv` |
| CNN-GRU Direct/Residual | 17.681 | 17.969 | 18.065 | cùng nguồn |

Ví dụ tốt nhất ở split 80/20: Residual-CNN-GRU FS06 Recency MAE 0,068947/RMSE 0,107682; Residual-CNN-LSTM FS01 Recency MAE 0,068841/RMSE 0,108019. Nguồn: report §2.1, dòng V22-N/V20. Không được lấy hai hàng hậu nghiệm này để tuyên bố CNN luôn vượt recurrent thuần.

### Limitations

Kernel 3 và max-pool 2 được cố định, chưa có ablation riêng. Pooling có thể làm mất định vị thời gian; causal padding chỉ ngăn look-ahead trong layer, không tự bảo đảm toàn pipeline không leakage. Kết quả chọn cấu hình tốt nhất sau khi nhìn test có selection bias.

### Defense Script (≈60 giây)

“Trong CNN-LSTM và CNN-GRU, em không đưa CNN theo nghĩa ảnh vào dữ liệu. Em dùng Conv1D dọc trục 22 phiên để trích mẫu cục bộ ba phiên. Padding causal bảo đảm output tại một vị trí không nhìn sang tương lai. Tensor 22 nhân F thành 22 nhân 32, rồi max-pooling giảm còn 11 nhân 32. Hai tầng LSTM hoặc GRU 48 và 24 units tổng hợp chuỗi đó, sau cùng Dense cho một dự báo. Direct và Residual dùng đúng cùng capacity; điểm khác là target. Vì vậy, chênh MAE giữa hai objective không thể giải thích bằng số tham số. Tuy nhiên kernel và pooling chưa được ablation, nên em chỉ trình bày đây là kiến trúc đã thử, không khẳng định nó tối ưu.”

### Potential Questions

1. **Causal padding có đủ chống leakage không?** Không. Nó chỉ bảo vệ convolution; scaler, split, window và availability của ngoại sinh vẫn phải audit. Nguồn: V20 cell 8; report §10.
2. **Vì sao length 22 thành 11?** Conv causal giữ length 22; MaxPooling1D(2) lấy cửa sổ/stride mặc định 2 nên còn 11.
3. **Tại sao GRU ít parameter hơn LSTM?** GRU có ba nhóm cổng, LSTM có bốn; số kiểm toán F=4 là 17.969 so với 23.393 ở biến thể CNN.

## 2. Residual Learning, Direct và Naive

### What

- Direct học `ŷ_t = f(X_{t−22:t−1})`.
- Residual học `Δŷ_t = f(X)` với `Δy_t=y_t−y_{t−1}`, sau đó `ŷ_t=y_{t−1}+Δŷ_t`.
- Naive không fit: `ŷ_t=y_{t−1}`.

Nguồn: report §1.1; V20 cell 8 (`make_context`) và cell 10 (`train_one`, `evaluate`); kiểm toán `residual_audit.csv`.

### Why

Mức giá có bằng chứng không dừng trong mẫu V26, còn sai phân/returns gần dừng hơn theo ADF–KPSS. Residual biến Naive thành mốc 0: mạng chỉ cần học phần điều chỉnh so với giá phiên trước. Đây là động cơ hợp lý, không phải định lý Residual sẽ thắng.

### How, Input/Output và Configuration

Windows luôn kết thúc ở `t−1`; `prev` là Copper scaled của `t−1`. Direct dùng target `y_t`; Residual dùng `y_t−prev_t`. Khi đánh giá, prediction được khôi phục và inverse-transform về đơn vị giá gốc trước khi tính metric. Kiến trúc, Adam, MSE, epochs, batch và seed giống nhau trong từng cặp Direct/Residual V20. Nguồn: V20 cell 8/10 (`make_context`, `train_one`, `evaluate`); V23 cell 3–6.

### Results

Holdout fitted tốt nhất là Residual-GRU: MAE 0,068780 (80%, FS02 Expanding), 0,076951 (90%, FS04 Expanding), 0,075028 (92,5%, FS06 Recency), 0,072357 (95%, FS02 Recency). Naive tương ứng 0,069001; 0,077077; 0,075116; 0,072799. Nguồn: report §2 và §11. Các cách biệt rất nhỏ, và cấu hình được chọn hậu nghiệm.

V23 walk-forward 542 phiên: Residual-GRU mean MAE 0,069100, SD giữa seeds 0,000154, chỉ 2/5 seeds thắng Naive; tất cả DM-HAC p>0,05. Nguồn: report §8.1–8.3; V23 CSV.

### Tại sao Residual-GRU gần Naive dù là neural network?

Naive tương đương `Δŷ=0`. Nếu mạng học delta có trung bình/độ lớn nhỏ, giá reconstructed tự nhiên sát Naive. Copper return không có tự tương quan tuyến tính rõ đến lag 22 trong V26, nên tín hiệu mean change có thể yếu. Nhưng không được khẳng định mạng “chỉ copy” nếu chưa kiểm delta dự báo; repository có `delta_pred_std_raw`, `delta_true_std_raw` và `residual_audit.csv` để audit.

### Tại sao cùng kiến trúc mà MAE Direct và Residual khác?

Hai objective tạo bề mặt loss và baseline khác nhau. Direct phải học mức giá scaled; Residual học thay đổi quanh 0 rồi cộng một anchor quan sát được. Sai số tối ưu MSE trên thang target, phân phối target và conditioning đều khác. Cùng parameter count không có nghĩa cùng bài toán tối ưu.

### Limitations

Sai phân một bước có thể làm target nhiễu; cộng lại `y_{t−1}` khiến prediction bám Naive và Directional Accuracy yếu. Kết quả holdout lồng nhau chưa là xác nhận độc lập.

### Defense Script (≈60 giây)

“Naive dự báo ngày mai bằng giá phiên trước. Residual learning biến baseline này thành điểm xuất phát: mạng chỉ dự báo delta, rồi em cộng delta vào giá phiên trước. Nếu delta dự báo nhỏ, mô hình neural vẫn cho giá gần Naive; đó chính là điều quan sát trong walk-forward. Direct và Residual dùng cùng kiến trúc nhưng học target khác, nên MAE khác do distribution và objective chứ không do capacity. Trên holdout Residual-GRU đứng đầu fitted, nhưng ở V23 MAE trung bình khoảng 0,06910 so với Naive 0,06900 và DM-HAC không cho bằng chứng thắng ổn định. Vì vậy em tách ‘đứng đầu fitted’ khỏi ‘vượt baseline có ý nghĩa’.”

### Potential Questions

1. **Residual có phải skip connection trong network?** Không; đây là target transformation và reconstruction ngoài model.
2. **Nếu delta=0 thì sao?** Prediction đúng bằng Naive.
3. **Có thể kết luận Residual tốt hơn Direct?** Chỉ trong những cấu hình/đoạn test đã chạy; chưa có kết luận phổ quát.

## 3. Simulated Annealing (SA)

### What và Why

SA là tìm kiếm stochastic: đôi khi nhận proposal xấu để thoát cực trị cục bộ, xác suất `exp(−Δ/T)`. Dự án dùng SA để chọn **hyperparameters** của LSTM — `units_1`, `units_2`, learning rate — không tối ưu neural weights. Neural weights vẫn được Adam fit.

### How thực tế

Nguồn chuẩn là V22-S cell 12:

- Search space: `units_1∈[16,96]`, `units_2∈[24,160]`, LR `[1e−5,3e−3]` trên log-space.
- Start: 39/111, LR `5e−5`; RNG seed 2026.
- Objective: validation MSE trên target scaled của reference 80/20, Expanding.
- Mỗi candidate train tối đa 15 epochs; patience search 4; không dự báo test trong search (`skip_test=True`).
- Budget 16 trials.
- Temperature trial k: `max(0.05, 1.0×0.88^k)`.
- Neighbor thay units trong kiến trúc-space và LR trong log-space; code clamp về search bounds.
- Acceptance: luôn nhận nếu score tốt hơn; nếu xấu hơn nhận với `exp(−(score−current_loss)/effective_temp)`, trong đó effective temperature được scale theo độ lớn loss.
- Best FS06: 47/127, LR `5.0478806754686805e−05`.

Nguồn artifact: `results/18-11- Copper_SA-LSTM-standalone-V22/.../sa_search_trials.csv`, `fresh_sa_best_hp.json`, `experiment_manifest.json`.

### Input/Output và final configuration

Input `(N,22,4)` FS06. Builder: LSTM u1 tanh/sigmoid, return sequences; LSTM u2; Dense 32 ReLU; Dense 1 linear; Adam(best LR), MSE. Final holdout cap 100 epochs, batch 64, EarlyStopping trong FULL_MODE. Paper-SA dùng 39/111, LR `5e−5`; Dense(32) là **giả định triển khai** được notebook ghi rõ, không phải code bài báo được xác minh. Nguồn: V22-S cell 2/8/10/20.

### Results

Ở 80/20, best observed Reoptimized SA-LSTM FS06 Expanding MAE 0,223809/RMSE 0,297888, kém Naive 0,069001; Paper-SA best observed FS06 Recency MAE 0,270155. Nguồn: report §2.1. Trong V23 WF, Reoptimized SA-LSTM MAE mean 0,207858, 0/5 seed thắng Naive và DM-HAC cho chênh lỗi dương có ý nghĩa. Không kết luận SA vô dụng: đây là một search budget, target/model family và giai đoạn cụ thể.

### Tại sao SA-LSTM chưa chắc tốt hơn LSTM?

SA chỉ tìm trong space/budget/objective đã định; validation có nhiễu, 16 trials nhỏ, mỗi trial 15 epoch, và architecture SA khác baseline. Tuning không bảo đảm test generalization. So sánh V21/V22 còn đổi feature lẫn HP, nên không phải ablation SA thuần.

### Limitations

Search chỉ trên một reference split/scheme; chưa nested walk-forward; budget tuning không đồng đều giữa họ model. Các HP V23/V24 là cố định từ V22, không SA lại từng fold. Paper configuration là tái dựng một phần.

### Defense Script (≈75 giây)

“SA trong dự án không thay Adam và không tối ưu từng trọng số. Nó đề xuất số units hai tầng và learning rate. Mỗi proposal được train ngắn trên train, objective là validation MSE scaled; test không được dùng trong search. Nhiệt độ giảm theo 0,88 mũ k nhưng có sàn 0,05. Proposal tốt luôn nhận, proposal xấu có thể nhận theo xác suất Boltzmann để tránh kẹt cục bộ. Sau 16 trials trên FS06, cấu hình tốt nhất là 47 và 127 units, learning rate khoảng 5,048×10⁻⁵. Tuy nhiên search budget nhỏ và chỉ dùng reference split 80/20 Expanding; kết quả test và walk-forward không cho SA-LSTM tốt hơn baseline. Vì vậy em gọi đây là hyperparameter search đã triển khai, không phải chứng minh SA ưu việt.”

### Potential Questions

1. **SA tối ưu weights hay HP?** HP; Adam tối ưu weights.
2. **Tại sao đôi khi nhận lời giải xấu?** Để khám phá và tránh cực trị cục bộ; xác suất giảm khi Δ lớn hoặc T thấp.
3. **Có dùng test chọn HP không?** Code search đặt `skip_test=True`; objective là validation MSE. Nhưng model cuối vẫn được so hậu nghiệm trên test.
4. **V24 có chạy SA lại không?** Không; dùng HP đã chọn V21/V22.

## 4. Expanding, Rolling 756 và Recency Weighting

### What/Why/How

| Strategy | Thực thi | Lý do thử | Nguồn |
|---|---|---|---|
| Expanding | `fit_start=0`, dùng toàn lịch sử trước validation/origin | Nhiều dữ liệu nhất | V20 cell 8; V21 cell 6 |
| Rolling 756 | `fit_start=max(0,val_start−756)` | Khoảng ba năm phiên, giảm ảnh hưởng regime cũ | cùng nguồn |
| Recency | Giữ toàn lịch sử, sample weight `0.5^(age/252)` | Nhấn mạnh năm gần nhất; half-life 252 | V21 cell 10 `sample_weights`; V20 cell 10 |

Scaler chỉ fit trên đoạn fit trước validation. Trong holdout, strategy là một lần fit, không refit mỗi phiên. Trong walk-forward, model được fit lại mỗi block 63 phiên.

### Input/Output, Configuration, Results

Shape không đổi `(N,22,F)→(N,1)`; chỉ lịch sử hoặc trọng số train đổi. Trong ma trận SA 48 ô, Expanding có MAE thấp nhất ở 6/8 cặp split-feature cho cả Paper và Reoptimized; Recency 2/8; Rolling 0/8. Nguồn: report §7.5. Đây là đếm mô tả trên test lồng nhau, không phải kiểm định chiến lược độc lập.

### Limitations

756 và 252 là lựa chọn cố định, chưa tuning nested; rolling bỏ dữ liệu, recency phụ thuộc sample-weight hỗ trợ của model. So sánh giữa split khác nhau bị confound bởi test periods khác nhau.

### Defense Script (≈50 giây)

“Em kiểm tra ba cách dùng lịch sử. Expanding dùng toàn bộ quá khứ; rolling chỉ giữ 756 phiên gần nhất; recency vẫn giữ toàn bộ nhưng trọng số giảm một nửa sau 252 phiên. Scaler luôn fit trước validation. Cần phân biệt đây là strategy của một lần holdout fit với lịch walk-forward: holdout không retrain từng phiên. Trong bảng SA, Expanding thắng mô tả nhiều nhất, nhưng các test period lồng nhau nên em không tuyên bố nó tối ưu tuyệt đối.”

### Potential Questions

1. **Half-life 252 nghĩa gì?** Quan sát cách 252 phiên có trọng số 0,5 so với gần nhất.
2. **Rolling 756 có phải retrain 756 lần?** Không; đó là chiều dài lịch sử dùng fit.
3. **Tại sao không chọn strategy bằng test?** Làm vậy gây selection bias; cần validation/nested WF.

## 5. Holdout, walk-forward, retrain 63 phiên và multi-seed

### What/Why/How

Holdout chia theo thứ tự thời gian: train 80/90/92,5/95%, test 542/271/204/136 phiên. Validation là phần cuối train; scaler fit trước validation. Walk-forward chia test thành các block; mỗi origin fit lại model từ đầu bằng dữ liệu khả dụng, dự báo block kế tiếp, rồi tiến 63 phiên. `63` là khoảng lịch cập nhật model, **không liên quan số epochs**: epoch là số lượt optimizer quét train trong một lần fit.

V23 dùng cùng 542 timestamps 07/10/2024–13/11/2026, 9 folds (8×63+38), seeds 41–45, ba model, 135 fits. Nguồn: report §8; V23 cell 1/11. V20 WF dùng 2.214 phiên 2018–2026, Residual-GRU FS02, nên không so trực tiếp MAE với V23.

### Input/Output và Results

Mỗi fold xuất prediction theo Date, actual, pred, naive và fold/seed; metric tổng hợp trên cùng 542 ngày. Multi-seed đo độ nhạy stochastic của training, không tạo năm tập test độc lập. V23 Residual-GRU mean MAE 0,069100 (2/5 thắng Naive), Residual-LSTM 0,069179 (0/5), SA-LSTM 0,207858. Nguồn: `wf_predictions.csv`, `wf_seed_summary.csv`.

### Vì sao model không có WF đầy đủ vẫn được báo holdout?

Holdout và WF trả lời hai mức bằng chứng khác nhau. Artifact holdout vẫn là kết quả thực nghiệm hợp lệ nếu ghi đúng scope; thiếu WF làm yếu kết luận về ổn định thời gian, không làm kết quả holdout biến mất. Report §10.2 đánh dấu rõ coverage chưa đầy đủ.

### Vì sao không so WF 2018–2026 với 2024–2026?

MAE phụ thuộc volatility/regime, độ dài mẫu, origin và có thể protocol/feature khác. Không cùng timestamp thì chênh MAE vừa chứa hiệu ứng model vừa chứa độ khó giai đoạn.

### Limitations

Các seed chia sẻ test dates; retrain mỗi 63 phiên có thể bỏ qua drift bên trong block; chưa có nested tuning đầy đủ; không phải mọi model/FS có coverage WF.

### Defense Script (≈75 giây)

“Holdout giữ nguyên thứ tự thời gian và cho bốn đoạn test. Walk-forward chặt hơn: tại mỗi origin em chỉ dùng quá khứ, fit lại rồi dự báo block tiếp theo. Bước 63 nghĩa là cập nhật khoảng mỗi quý giao dịch; nó không phải 63 epochs. Một lần fit vẫn có tối đa 45 epochs trong V23. Năm seeds đánh giá độ nhạy khởi tạo, nhưng cùng dự báo một tập ngày nên không xem là năm thí nghiệm độc lập. V23 dùng 542 phiên 2024–2026, còn V20 bắt đầu từ 2018 với 2.214 phiên; do khác giai đoạn và feature/protocol, em giữ hai kết quả riêng.”

### Potential Questions

1. **63 và 45 khác gì?** 63 là số phiên giữa hai lần refit; 45 là epoch cap mỗi lần fit.
2. **Multi-seed có tăng cỡ mẫu kiểm định lên 5 lần?** Không, dates trùng nhau và predictions phụ thuộc.
3. **Holdout có vô dụng nếu chưa WF?** Không; nó là bằng chứng hẹp hơn, phải ghi đúng phạm vi.

## 6. EDA: Pearson, Spearman, lagged correlations và feature choice

### What/Why/How

Spearman đo quan hệ đơn điệu theo rank; Pearson đo tuyến tính. V20/V22 tính Spearman mức giá và lagged correlations trên phần train. V26 phân biệt tương quan mức giá với tương quan simple returns có lag, hiệu chỉnh FDR cho nhiều phép thử. Nguồn: V20 cell 6; V22-N cell 6; V26 cell 11; ZIP V26 `05_*`.

### Input/Output, Configuration, Results

V26 dùng 2.195 dòng đầu đến 12/11/2024; lag 1–22 trên returns, Pearson và Spearman, FDR. Spearman mức giá cao có thể do cùng trend/non-stationarity. Report §1.3.6–1.3.7 ghi lagged correlations nhỏ; Gold lag 1 là tín hiệu giữ được sau FDR, nhưng chưa chứng minh incremental predictive value. FS06 thay đồng thời Gold+Silver bằng WTI và giảm F từ 5 xuống 4; không phải ablation từng biến.

### Tại sao Spearman mức giá cao không bảo đảm dự báo tốt?

Hai chuỗi không dừng có thể cùng tăng và tạo rank correlation cao mà thay đổi ngày mai không liên hệ. Forecast cần thông tin có điều kiện, đúng lag và đúng availability. Correlation không đo đóng góp sau khi đã biết lịch sử Copper, không xét tương tác mạng và không chứng minh nhân quả.

### Limitations

EDA V26 kết thúc sau ngày bắt đầu test 80/20, nên không được dùng như train-only feature selection cho split này. Granger V26 thất bại kỹ thuật 20/20 do tham số `verbose`, không được diễn giải “không có ý nghĩa”. Availability theo giờ đóng cửa chưa xác minh.

### Defense Script (≈55 giây)

“Em không chọn WTI chỉ vì tương quan mức giá cao. Giá các hàng hóa đều có trend và không dừng nên Spearman level có thể là spurious. V26 vì thế còn tính Pearson/Spearman trên returns có lag và FDR. Các hệ số lag khá nhỏ; Gold lag một còn tín hiệu, nhưng correlation vẫn không chứng minh giá trị dự báo gia tăng. FS04 sang FS06 còn thay hai biến và số chiều, nên em gọi đó là so sánh feature set, không phải causal effect của WTI.”

### Potential Questions

1. **Pearson và Spearman khác nhau?** Pearson tuyến tính trên giá trị; Spearman tuyến tính trên rank, nhạy với quan hệ đơn điệu.
2. **Correlation âm/dương cao có nghĩa feature hữu ích?** Không nhất thiết; cần validation forecast đúng thời gian.
3. **Granger kết luận gì?** Chưa có kết quả hợp lệ; file chứa lỗi kỹ thuật.

## 7. Các kiểm định thống kê V26

Nguồn chung: V26 cells 5/7/9/15; ZIP `results/copper_v26_comprehensive_eda_results.zip`; report §1.3. V26 dùng 2.195 dòng đầu, không phải train thuần của 80/20.

| Test | H0 và triển khai | Kết quả thật | Diễn giải đúng/giới hạn |
|---|---|---|---|
| Jarque–Bera | returns có skew 0, kurtosis 3 | 6/6 bác bỏ; Copper p=8,53e−108; WTI skew 5,181, excess kurtosis 172,438 | Đuôi nặng; cờ outlier không đồng nghĩa dữ liệu lỗi. `02_returns_distribution_JB.csv` |
| ADF | có unit root; regression `c` | Copper level p=0,4318 không bác bỏ; diff p=3,64e−30 và return p=5,47e−30 bác bỏ | Hỗ trợ level không dừng, biến đổi gần dừng; không chứng minh tuyệt đối I(1). `03_adf_kpss.csv` |
| KPSS | level-stationary; regression `c` | Copper level p=0,01 bác bỏ; diff/return p=0,10 không bác bỏ | Kết hợp với ADF; 0,01/0,10 là biên bảng nội suy. |
| Ljung–Box | không autocorrelation đến lag h | Copper returns p=0,5578/0,1239/0,0635 ở lag 5/10/22; không bác bỏ | Không thấy linear mean autocorrelation; không chứng minh “không dự báo được”. `04_lb_arch.csv` |
| LB trên returns² | không autocorrelation trong bình phương returns | Copper p rất nhỏ ở 5/10/22 | Volatility magnitude có cấu trúc. |
| ARCH-LM | không ARCH effect | Copper p=1,17e−6; 4,95e−6; 0,0005 | Có bằng chứng conditional variance thay đổi; không tự chứng minh GARCH đúng. |
| CUSUM | mean-return intercept ổn định | Copper p FDR=0,8264; 6/6 không bác bỏ | Chưa phát hiện đổi mean theo test này, không chứng minh không regime. `07_cusum_mean_returns.csv` |
| Zivot–Andrews | unit root; H1 stationary quanh một break, `ct` | tất cả p>0,05 | Candidate break date không phải break đã chứng minh; chỉ một break. `07_zivot_andrews_break_unit_root.csv` |
| Brown–Forsythe | dispersion hai nửa bằng nhau; Levene median | Copper std 0,01212→0,01505, +24,1%, p FDR<1e−8 | Bằng chứng variance khác giữa hai nửa cố định; dependence/ARCH làm p cần thận trọng. `07_brown_forsythe_variance_halves.csv` |
| Engle–Granger | không cointegration cặp Copper–exo | mọi cặp không bác bỏ ở 5% sau FDR | Không chứng minh không có quan hệ đa biến/phi tuyến; test cặp, specification cụ thể. `07_cointegration_exploratory.csv` |

### Why/Input/Output/Configuration

Các test kiểm tra assumptions khác nhau: stationarity của level/transformation, dependence của mean/variance, distribution, structural stability và long-run pair relation. Input là levels, first differences hoặc simple returns tùy test; output gồm statistic, p-value và FDR khi có nhiều chuỗi. Không hoán đổi kết quả giữa transformations.

### Limitations chung

P-value phụ thuộc specification, sample và temporal dependence. Nhiều test không tạo causal evidence. CUSUM mean không mâu thuẫn Brown–Forsythe variance: chúng kiểm hai đối tượng khác nhau. V26 EDA overlap đầu test 80/20 27 phiên.

### Defense Script (≈90 giây)

“Em dùng cặp ADF–KPSS vì hai giả thuyết gốc đối nghịch. Với Copper, level không bác bỏ unit root ở ADF và bác bỏ stationary ở KPSS, trong khi difference và return cho mẫu hình gần dừng. Jarque–Bera bác bỏ normality và cho thấy đuôi nặng. Ljung–Box trên return chưa phát hiện tự tương quan tuyến tính đến lag 22, nhưng Ljung–Box trên return bình phương và ARCH-LM rất có ý nghĩa, nên cấu trúc variance rõ hơn mean. CUSUM không phát hiện đổi mean, còn Brown–Forsythe phát hiện variance hai nửa khác; hai kết quả không mâu thuẫn. Zivot–Andrews và Engle–Granger đều không bác bỏ H0 ở 5%, nhưng ‘không bác bỏ’ không phải chứng minh H0 đúng. Cuối cùng, mẫu V26 overlap đầu test 80/20 nên em không dùng nó để tuyên bố feature selection 80/20 hoàn toàn unseen.”

### Potential Questions

1. **ADF và KPSS có mâu thuẫn?** Không; H0 đối nghịch, kết hợp cho chẩn đoán mạnh hơn.
2. **CUSUM không đổi mà Brown–Forsythe đổi vì sao?** Một test mean stability, một test dispersion.
3. **ARCH-LM có nghĩa phải dùng GARCH?** Không; chỉ cho bằng chứng ARCH effect.
4. **Không bác bỏ cointegration H0 nghĩa không có quan hệ?** Chỉ là chưa đủ bằng chứng trong test cặp/specification này.

## 8. Metrics: MAE, RMSE, MAPE, R², Directional Accuracy, DM-HAC

### What/How/Input/Output

| Metric | Công thức/ý nghĩa trong code | Nguồn |
|---|---|---|
| MAE | mean `|y−ŷ|`, đơn vị giá gốc | V20 cell 10 `evaluate`; report §2 |
| RMSE | sqrt mean squared error, phạt lỗi lớn hơn | cùng nguồn |
| MAPE | mean `|y−ŷ|/|y|×100`; hợp lệ vì actual không 0 trong mẫu | V21/V22-S cell 10; report §2.1 |
| R² | `1−SSE/SST` trên level Copper cùng test | V21/V22-S cell 10; report §2.1 |
| Directional Accuracy | `% sign(ŷ−y_{t−1}) = sign(y−y_{t−1})` | V20 cell 10, dòng `directional_accuracy_pct` |
| DM-HAC | kiểm mean loss difference với Newey–West/HAC lag 5 | V20 cell 20 `hac_test`; V23 cell 13 `dm_hac` |

Tất cả metric giá phải tính sau inverse scaling. V23 DM dùng `d_t=absolute_error_model−absolute_error_naive`; d âm nghiêng về model, p hai phía. HAC xử lý autocorrelation/heteroskedasticity của loss difference tốt hơn standard iid SE.

### Results và câu hỏi trọng yếu

Naive 80/20 MAE 0,069001/RMSE 0,108000. Best fitted holdout Residual-GRU MAE 0,068780/RMSE 0,107853, R² 0,9833. R² cao vì mức giá có persistence và variance lớn; nó không đo riêng việc đoán dấu delta. Một prediction gần `y_{t−1}` có thể giải thích level rất tốt nhưng Directional Accuracy yếu.

V23 DM-HAC: Residual-GRU p từ 0,0567 đến 0,9315 ở 5 seeds, không seed nào <0,05; mean loss difference đổi dấu. Nguồn: `wf_dm_hac_by_seed.csv`.

### Limitations

MAPE khó so khi denominator gần 0 (không xảy ra ở sample này). R² có thể âm và không trực tiếp là trading utility. DM nhiều seeds chưa chỉnh multiple testing và seeds chia sẻ dates. MAE/RMSE nhỏ không tự chứng minh khác biệt có ý nghĩa.

### Defense Script (≈60 giây)

“Em báo MAE vì dễ đọc theo đơn vị giá, RMSE để nhấn mạnh lỗi lớn, MAPE theo tỷ lệ và R² cho mức độ giải thích level. Nhưng R² cao không đồng nghĩa dự báo biến động tốt: chỉ cần bám giá hôm qua đã theo level rất sát. Vì thế code còn tính Directional Accuracy trên dấu delta. Để so model với Naive trên cùng ngày, em dùng loss difference tuyệt đối và DM-style test với HAC lag 5. Trong V23, p-value của Residual-GRU đều trên 0,05, nên chưa có bằng chứng thắng Naive ổn định dù nó đứng đầu fitted.”

### Potential Questions

1. **Tại sao R² cao nhưng direction kém?** Level persistence chi phối SST; delta nhỏ và khó đoán.
2. **RMSE khác MAE?** Bình phương khiến RMSE nhạy hơn lỗi lớn.
3. **DM-HAC kiểm gì?** Mean chênh loss trên cùng timestamp, với SE điều chỉnh dependence; không kiểm model parameter.

---

# B. Model Specification — đúng theo từng protocol

## B0. Ma trận đủ trường bắt buộc

Mỗi hàng dưới đây là một cấu hình thực sự trong code; dấu “—” nghĩa trường không áp dụng, không phải bị thiếu. `ES p9` là EarlyStopping theo `val_loss`, patience 9, restore best weights. Parameter thay đổi theo F nên giữ đúng ba mức đã audit, không nội suy.

| Model | Input Shape | Output | Layers | Activation | Loss | Optimizer | Learning Rate | Batch Size | Max Epochs | EarlyStopping | Parameters | Feature Sets | Source |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
| V20/V22-N LSTM Direct | `(22,F)` | scaled `Copper(t)`, `(1,)` | LSTM64→Dropout.2→LSTM32→Dropout.2→Dense16→Dense1 | LSTM tanh/sigmoid; ReLU; linear | MSE | Adam | 1e−3 | 64 | 65 | ES p9 | F1 29.857; F4 30.625; F5 30.881 | FS01/02/04/06 | V20 c2/10; V22-N c2/10; parameter CSV |
| V20/V22-N LSTM Residual | `(22,F)` | scaled delta, rồi `prev+delta`, `(1,)` | như LSTM Direct | như trên | MSE | Adam | 1e−3 | 64 | 65 | ES p9 | cùng Direct | FS01/02/04/06 | cùng nguồn; V20 `train_one` |
| V20/V22-N GRU Direct | `(22,F)` | scaled `Copper(t)`, `(1,)` | GRU64→Dropout.2→GRU32→Dropout.2→Dense16→Dense1 | GRU tanh/sigmoid; ReLU; linear | MSE | Adam | 1e−3 | 64 | 65 | ES p9 | F1 22.817; F4 23.393; F5 23.585 | FS01/02/04/06 | V20 c2/10; V22-N c2/10; parameter CSV |
| V20/V22-N GRU Residual | `(22,F)` | scaled delta, rồi `prev+delta`, `(1,)` | như GRU Direct | như trên | MSE | Adam | 1e−3 | 64 | 65 | ES p9 | cùng Direct | FS01/02/04/06 | cùng nguồn; V20 `train_one` |
| V20/V22-N CNN-LSTM Direct | `(22,F)` | scaled `Copper(t)`, `(1,)` | Conv32 k3 causal→Pool2→LSTM48→Dropout.2→LSTM24→Dropout.2→Dense16→1 | Conv/Dense ReLU; LSTM tanh/sigmoid; linear | MSE | Adam | 1e−3 | 64 | 65 | ES p9 | F1 23.105; F4 23.393; F5 23.489 | FS01/02/04/06 | V20 c2/10; parameter CSV |
| V20/V22-N CNN-LSTM Residual | `(22,F)` | scaled delta, rồi `prev+delta`, `(1,)` | như CNN-LSTM Direct | như trên | MSE | Adam | 1e−3 | 64 | 65 | ES p9 | cùng Direct | FS01/02/04/06 | cùng nguồn; V20 `train_one` |
| V20/V22-N CNN-GRU Direct | `(22,F)` | scaled `Copper(t)`, `(1,)` | Conv32 k3 causal→Pool2→GRU48→Dropout.2→GRU24→Dropout.2→Dense16→1 | Conv/Dense ReLU; GRU tanh/sigmoid; linear | MSE | Adam | 1e−3 | 64 | 65 | ES p9 | F1 17.681; F4 17.969; F5 18.065 | FS01/02/04/06 | V20 c2/10; parameter CSV |
| V20/V22-N CNN-GRU Residual | `(22,F)` | scaled delta, rồi `prev+delta`, `(1,)` | như CNN-GRU Direct | như trên | MSE | Adam | 1e−3 | 64 | 65 | ES p9 | cùng Direct | FS01/02/04/06 | cùng nguồn; V20 `train_one` |
| V20/V22-N Ridge Residual | flattened `(22F,)` | scaled delta, rồi `prev+delta`, scalar | Ridge `alpha=1` | linear | squared error | Ridge solver | — | — | — | — | F1 23; F4 89; F5 111 gồm intercept | FS01/02/04/06 | V20 c10 `train_one`; parameter CSV |
| V20–V25 Naive | Copper(t−1) | Copper(t−1), scalar | — | — | — | — | — | — | — | — | 0 | mọi FS | V20 c10; các prediction CSV |
| V21/V22-S Baseline LSTM | `(22,F)` | scaled `Copper(t)`, `(1,)` | LSTM64→Dropout.2→LSTM32→Dropout.2→Dense16→1 | LSTM tanh/sigmoid; ReLU; linear | MSE | Adam | 1e−3 | 64 | 100 | ES p9 | F3 30.369; F4 30.625; F5 30.881 | FS02/03/04/05/06 | V21 c2/8/10; V22-S c2/8/10 |
| V21/V22-S SimpleRNN | `(22,F)` | scaled `Copper(t)`, `(1,)` | RNN64→Dropout.2→RNN32→Dropout.2→Dense16→1 | tanh; ReLU; linear | MSE | Adam | 1e−3 | 64 | 100 | ES p9 | F3 8.001; F4 8.065; F5 8.129 | FS02/03/04/05/06 | V21 c8/10; parameter CSV |
| V21/V22-S/V24 Paper SA-LSTM | `(22,F)` | scaled `Copper(t)`, `(1,)` | LSTM39→Dropout.2→LSTM111→Dropout.2→Dense32→1 | tanh/sigmoid; ReLU; linear | MSE | Adam | 5e−5 | 64 | 100 | ES p9 | F3 77.369; F4 77.525; F5 77.681 | FS02/03/04/05/06 | V21/V22-S c8/10; V24 c1/4/5; Dense32 assumed |
| V21/V24 Reopt SA-LSTM Paper groups | `(22,F)` | scaled `Copper(t)`, `(1,)` | LSTM22→Dropout.2→LSTM91→Dropout.2→Dense32→1 | tanh/sigmoid; ReLU; linear | MSE | Adam | 0.000922 | 64 | 100 | ES p9 | F3 46.761; F4 46.849 | FS02/03 | V21 c2/8/12; V24 HP map; parameter CSV |
| V21/V24 Reopt SA-LSTM Legacy groups | `(22,F)` | scaled `Copper(t)`, `(1,)` | LSTM22→Dropout.2→LSTM83→Dropout.2→Dense32→1 | tanh/sigmoid; ReLU; linear | MSE | Adam | 0.000655 | 64 | 100 | ES p9 | F4 40.289; F5 40.377 | FS04/05 | V21 c2/8/12; V24 HP map; parameter CSV |
| V22-S/V23/V24 Reopt SA-LSTM FS06 | `(22,4)` | scaled `Copper(t)`, `(1,)` | LSTM47→Dropout.2→LSTM127→Dropout.2→Dense32→1 | tanh/sigmoid; ReLU; linear | MSE | Adam | 5.047880675e−5 | 64 | 100 holdout; 45 WF | ES p9 holdout; p7 WF | chưa có audit row; không suy đoán | FS06 | V22-S c8/10/12 + HP JSON; V23 c1; V24 c1 |

## B1. V20/V22 Non-SA

Mọi network: output `(N,1)` linear; MSE; Adam 1e−3; batch 64; max 65; EarlyStopping `val_loss`, patience 9, restore best. Direct/Residual cùng layer/parameter. Feature sets V20 FS01/02/04; V22-N FS06. Nguồn: V20 cells 2/10/12; V22-N tương ứng 2/10/12.

| Model thực tế | Input | Layers/activation | Params theo F=1/4/5 | Feature sets | Source |
|---|---|---|---|---|---|
| LSTM Direct / Residual | `(22,F)` | LSTM64→Dropout.2→LSTM32→Dropout.2→Dense16 ReLU→Dense1 linear | 29.857 / 30.625 / 30.881 | FS01/02/04/06 | `build_net`, parameter CSV |
| GRU Direct / Residual | `(22,F)` | GRU64→Dropout.2→GRU32→Dropout.2→Dense16 ReLU→1 | 22.817 / 23.393 / 23.585 | FS01/02/04/06 | cùng nguồn |
| CNN-LSTM Direct / Residual | `(22,F)` | Conv32 k3 causal ReLU→Pool2→LSTM48→Dropout.2→LSTM24→Dropout.2→Dense16→1 | 23.105 / 23.393 / 23.489 | FS01/02/04/06 | cùng nguồn |
| CNN-GRU Direct / Residual | `(22,F)` | Conv32 k3 causal ReLU→Pool2→GRU48→Dropout.2→GRU24→Dropout.2→Dense16→1 | 17.681 / 17.969 / 18.065 | FS01/02/04/06 | cùng nguồn |
| Ridge Residual | flattened `(22F,)` | Ridge linear, `alpha=1.0` | 23 / 89 / 111 coef+intercept | FS01/02/04/06 | V20 cell 10 `train_one`; parameter CSV |
| Naive | Copper(t−1) | không layer/loss/optimizer | 0 | mọi FS, lặp để đối chiếu | V20 cell 10 `train_one`; cell 12 điều phối grid |

## B2. V21/V22/V24 SA-series

| Protocol/model | Input | Layers | Loss/optimizer/LR | Batch; epochs; ES | Params đã kiểm | Features | Source |
|---|---|---|---|---|---|---|---|
| Baseline LSTM V21/V22-S | `(22,F)` | LSTM64→Dropout.2→LSTM32→Dropout.2→Dense16 ReLU→1 | MSE, Adam 1e−3 | 64; max100; ES patience9 | F3 30.369; F4 30.625; F5 30.881 | FS02/03/04/05; FS06 | V21 c2/8/10; V22-S c2/8/10; parameter CSV |
| SimpleRNN Direct | `(22,F)` | RNN64→Dropout.2→RNN32→Dropout.2→Dense16→1 | MSE, Adam 1e−3 | 64; max100; ES patience9 | F3 8.001; F4 8.065; F5 8.129 | same | V21 cell 8/10; parameter CSV |
| Paper SA-LSTM | `(22,F)` | LSTM39 tanh/sigmoid→Dropout.2→LSTM111→Dropout.2→Dense32 ReLU→1 | MSE, Adam 5e−5 | 64; max100; ES patience9 | F3 77.369; F4 77.525; F5 77.681 | FS02/03/04/05/06 | V21/V22-S cell 8/10; Dense32 assumed |
| Reopt V21 Paper groups | `(22,F)` | LSTM22→Dropout.2→LSTM91→Dropout.2→Dense32→1 | MSE, Adam 0.000922 | 64; max100; ES patience9 | F3 46.761; F4 46.849 | FS02/03 | V21 c2/8/12; parameter CSV |
| Reopt V21 Legacy groups | `(22,F)` | LSTM22→Dropout.2→LSTM83→Dropout.2→Dense32→1 | MSE, Adam 0.000655 | 64; max100; ES patience9 | F4 40.289; F5 40.377 | FS04/05 | V21 c2/8/12; report §10.1 |
| Reopt V22 FS06 | `(22,4)` | LSTM47→Dropout.2→LSTM127→Dropout.2→Dense32→1 | MSE, Adam 5.047880675e−5 | 64; final max100; ES patience9 | không có hàng parameter audit V22; không suy đoán | FS06 | V22-S c8/10/12; `fresh_sa_best_hp.json` |
| Naive | previous Copper | none | none | none | 0 | all | V21/V22-S evaluation |

**Không gộp:** V20 max65 khác SA-series max100; V21 Reopt dùng HP cache, V22 Reopt chạy SA mới 16 trials; V24 chỉ fit HP đã có. `PAPER_SA_LSTM` Dense32 là giả định triển khai được ghi trong notebook, không phải reproduction tuyệt đối.

---

# C. Kịch bản thuyết trình

## C1. Bản 5 phút

“Em xin trình bày Module 1: dự báo giá đồng trước một phiên. Dữ liệu có 2.710 phiên từ 2016 đến 2026. Tại ngày t, mô hình chỉ nhận 22 phiên từ t−22 đến t−1 và dự báo Copper tại t. Bốn holdout được chia theo thời gian, tuyệt đối không shuffle.

Em so sánh ba cách nhìn chính. Direct dự báo thẳng mức giá. Residual dự báo chênh lệch so với phiên trước rồi cộng lại. Naive đơn giản dùng giá phiên trước và không được đưa vào xếp hạng mô hình fitted. Ngoài LSTM và GRU, em thử CNN-LSTM/CNN-GRU: Conv1D causal kernel 3 trích mẫu cục bộ, pool giảm 22 còn 11 bước, rồi hai tầng recurrent và Dense cho một output.

Về dữ liệu, EDA V26 cho thấy mức giá có mẫu hình không dừng theo ADF–KPSS, còn difference và return gần dừng hơn. Copper return không có tự tương quan tuyến tính rõ đến lag 22, nhưng return bình phương và ARCH-LM cho bằng chứng volatility clustering. Đây là lý do Residual hợp lý để thử và Naive là baseline bắt buộc; không phải bằng chứng Residual chắc chắn thắng.

Trong bảng holdout fitted, Residual-GRU có MAE thấp nhất đã quan sát ở cả bốn split; ví dụ 80/20 là 0,068780, trong khi Naive 0,069001. Nhưng chênh lệch rất nhỏ và đây là best configuration sau khi nhìn nhiều thử nghiệm. Vì vậy em kiểm bằng walk-forward.

Walk-forward V23 dùng cùng 542 phiên từ 2024 đến 2026, retrain mỗi 63 phiên, năm seeds. Residual-GRU có MAE trung bình 0,06910, trong khi Naive 0,06900; chỉ 2/5 seeds thắng Naive và DM-HAC không có p dưới 0,05. Kết luận trung thực là Residual-GRU đứng đầu nhóm fitted, nhưng chưa chứng minh vượt baseline ổn định.

SA trong dự án chỉ tối ưu số units và learning rate trên validation, không tối ưu neural weights. Search V22 có 16 trials và tìm được 47/127 units, learning rate khoảng 5,05×10⁻⁵. SA-LSTM không tốt hơn trong walk-forward, nên em không tuyên bố SA ưu việt.

Giới hạn chính là các test period lồng nhau, tuning budget chưa hoàn toàn đồng đều, EDA V26 overlap đầu test 80/20, và chưa audit đủ giờ đóng cửa của ngoại sinh. Module 1 vì vậy là benchmark có kiểm toán, không phải kết luận cuối cùng.”

## C2. Bản 10 phút

“Em bắt đầu từ protocol. Mỗi sample có shape 22 nhân F, trong đó F tùy feature set. FS04 gồm Copper, Aluminium, Tin, Silver, Gold; FS06 thay đồng thời Gold và Silver bằng WTI và giảm từ năm xuống bốn biến. Vì vậy đây không phải ablation riêng WTI.

Split theo thời gian gồm 80/20, 90/10, 92,5/7,5 và 95/5. Validation nằm cuối train; MinMaxScaler chỉ fit trước validation. Em thử expanding, rolling 756 phiên và recency weighting half-life 252. Holdout chỉ fit một lần; walk-forward mới refit theo block.

Về architecture, LSTM/GRU thuần dùng 64 rồi 32 units. CNN variants dùng Conv1D 32 filters, kernel 3, causal ReLU, MaxPool 2, rồi recurrent 48 và 24. Tất cả Non-SA dùng Dense16, output linear, Adam 0,001, MSE, batch 64, cap 65 epochs và early stopping patience 9. Direct và Residual trong cùng cặp có cùng parameter count.

Residual không phải residual connection. Mạng học y_t trừ y_{t−1}; prediction cuối bằng y_{t−1} cộng delta dự báo. Khi delta nhỏ, prediction gần Naive. Direct học level nên dù cùng architecture, distribution target và optimization khác, MAE có thể khác.

Nhánh SA-series dùng LSTM. Paper configuration triển khai 39/111 units và learning rate 5×10⁻⁵, nhưng Dense32 là giả định tái dựng được notebook ghi rõ. V22 thực sự chạy SA mới cho FS06: space units 16–96 và 24–160, LR 10⁻⁵ đến 3×10⁻³; 16 trials, 15 epochs/trial, validation MSE scaled, không nhìn test. Proposal xấu vẫn có thể được nhận theo xác suất Boltzmann. Cấu hình tốt nhất là 47/127 và LR 5,0479×10⁻⁵. Neural weights vẫn do Adam học.

EDA V26 dùng 2.195 dòng đầu. Jarque–Bera bác bỏ normality cho sáu chuỗi; WTI có đuôi đặc biệt lớn. ADF và KPSS cùng cho mẫu hình level không dừng, difference/return gần dừng. Với Copper, Ljung–Box return p là 0,558, 0,124 và 0,064 ở lag 5, 10, 22 nên chưa bác bỏ không-autocorrelation; ngược lại returns bình phương và ARCH-LM có p rất nhỏ. Brown–Forsythe cho std Copper hai nửa tăng khoảng 24,1%, trong khi CUSUM mean không có ý nghĩa. Hai kết quả không mâu thuẫn vì kiểm mean và variance khác nhau. Engle–Granger không tìm thấy cointegration cặp ở 5%, nhưng đó không phải bằng chứng không có mọi quan hệ.

Kết quả holdout: Residual-GRU đứng đầu fitted ở bốn split, nhưng khoảng cách với Naive nhỏ. Ở 80/20, MAE 0,068780 so với 0,069001. R² có thể khoảng 0,98 vì level price có persistence; R² cao không chứng minh dự báo dấu delta tốt. Directional Accuracy trong code so dấu prediction change với actual change.

Walk-forward V23 là kiểm tra quan trọng hơn: 542 phiên, chín folds, năm seeds, retrain mỗi 63 phiên. 63 là lịch cập nhật, không phải epochs; mỗi fit có cap 45 epochs. Residual-GRU mean MAE 0,069100 và chỉ 2/5 seeds thắng Naive. DM-HAC trên absolute loss difference không cho p dưới 0,05. SA-LSTM kém rõ hơn. Multi-seed chỉ đo độ nhạy, không phải năm test độc lập.

Em vẫn báo model chỉ có holdout vì đó là evidence có thật, nhưng gắn nhãn coverage. Em không so trực tiếp V20 walk-forward 2018–2026 với V23 2024–2026 vì khác dates, feature và protocol. Cuối cùng, Granger V26 bị lỗi kỹ thuật nên không có kết luận Granger. Những giới hạn này là phần của kết quả, không phải chi tiết cần giấu.”

## C3. Bản 15 phút

“Câu hỏi nghiên cứu của Module 1 là: với dữ liệu giá hàng hóa theo phiên, kiến trúc và cách biểu diễn target nào dự báo Copper trước một phiên tốt hơn, và liệu thay nhóm Gold/Silver bằng WTI có cải thiện ngoài mẫu không?

Dữ liệu có 2.710 dòng. Mỗi sample lấy đúng 22 quan sát trước origin, nên input là N×22×F và output N×1. HORIZON bằng 1. Các ngày không đều hoàn toàn; V26 thấy gap tới 19, 28, 66 ngày, nên một lag là một phiên dữ liệu, không luôn là một ngày lịch. Đây là rủi ro trước khi triển khai thực tế.

Em tổ chức feature set để giữ rõ thứ tự. FS01 chỉ Copper; FS02 Copper, WTI, Gold, Silver; FS04 Copper, Aluminium, Tin, Silver, Gold; FS06 Copper, Aluminium, Tin, WTI. Thay FS04 sang FS06 đồng thời bỏ hai biến, thêm một biến và giảm dimension. Ở nhánh SA còn đổi HP, nên không thể gán toàn bộ hiệu ứng cho WTI.

Pipeline chia thời gian trước, giữ validation ở cuối train và fit scaler trên train-fit בלבד. Em kiểm expanding, rolling 756 và recency half-life 252. Với recency, observation cách 252 phiên còn nửa trọng số. Các window vẫn kết thúc t−1. Output được inverse-transform trước metric.

Về model, LSTM/GRU thuần có 64 và 32 units. CNN-LSTM/CNN-GRU thêm Conv1D causal 32 filters kernel 3 và MaxPool2, sau đó recurrent 48/24. Causal nghĩa convolution không nhìn phải, nhưng leakage protection còn cần scaler/split và point-in-time availability. V20 dùng Adam 10⁻³, MSE, batch64, cap65, early stopping patience9.

Hai objective chính là Direct và Residual. Direct học level. Residual học delta rồi cộng price t−1. Naive chính là residual với delta bằng 0. Vì level không dừng và return khó dự báo, residual thường học correction nhỏ, nên MAE rất gần Naive. Điều này không có nghĩa neural network không học gì; phải xem phân phối predicted delta và audit CSV. Cùng architecture nhưng target distribution khác nên Direct/Residual cho kết quả khác.

SA là lớp search hyperparameter. V22 chọn units và learning rate bằng validation MSE, 16 trials. Temperature giảm 0,88 mỗi trial với floor 0,05; proposal xấu có xác suất được nhận. Search không gọi test prediction. Best HP 47/127 và LR xấp xỉ 5,048×10⁻⁵. Sau đó final model vẫn được Adam huấn luyện. Vì chỉ một reference split, budget nhỏ và tuning không đồng đều, SA-LSTM không mặc nhiên công bằng hơn baseline LSTM.

EDA giúp đặt kết quả vào bối cảnh. Jarque–Bera bác bỏ normality; WTI có skew và kurtosis cực đoan. ADF/KPSS cho level không dừng, difference/return gần dừng. Copper return không có autocorrelation tuyến tính rõ đến 22 lag, nhưng squared return và ARCH-LM cho volatility clustering. CUSUM không phát hiện mean instability, Brown–Forsythe phát hiện variance hai nửa khác nhau; đây là hai đối tượng khác nhau. ZA không bác bỏ unit-root null khi cho một break; Engle–Granger không bác bỏ no-cointegration cho các cặp. Em luôn nói ‘không bác bỏ’, không nói ‘chứng minh H0’.

Spearman level cao không đủ chọn feature vì trend chung tạo correlation. Lagged return correlation gần forecast question hơn nhưng vẫn không đo conditional incremental value. V26 còn có một giới hạn: mốc 12/11/2024 muộn hơn test 80/20 bắt đầu 07/10/2024, nên không dùng EDA này để tuyên bố chọn feature hoàn toàn unseen cho 80/20. Granger bị lỗi API, nên chưa có kết luận.

Kết quả holdout cho thấy Residual-GRU là fitted model có MAE thấp nhất quan sát ở bốn split: 0,068780; 0,076951; 0,075028; 0,072357. Naive lần lượt 0,069001; 0,077077; 0,075116; 0,072799. Chênh lệch rất nhỏ và best được chọn trong nhiều cấu hình. R² cao khoảng 0,98 ở một số residual model chủ yếu phản ánh bám level; không nói riêng khả năng đoán direction.

Để kiểm ổn định, V23 walk-forward dùng cùng 542 dates 2024–2026, seeds41–45, retrain 63 phiên. Residual-GRU mean MAE 0,069100 với SD 0,000154 và 2/5 seeds thắng Naive. Residual-LSTM mean 0,069179; SA-LSTM 0,207858. DM-HAC dùng d bằng absolute error model trừ Naive, lag5. Không p nào của GRU dưới 0,05. Vì vậy, fitted ranking và statistical superiority là hai tuyên bố khác nhau.

Cuối cùng, V20 WF kéo dài 2018–2026 còn V23 chỉ 2024–2026. Không được lấy chênh MAE giữa chúng để nói WTI tốt hơn vì market regime khác. Module 1 có giá trị ở chỗ tạo benchmark audit được, chỉ ra residual methods cạnh tranh nhưng khó thắng Naive, đồng thời phơi bày rủi ro selection bias, coverage và data timing. Bước tiếp theo hợp lý là nested walk-forward, ablation từng biến, cùng tuning budget và test độc lập chưa dùng chọn model.”

---

# D. Mock Defense — 36 câu hỏi và đáp án

## D1. Cơ bản

1. **Bài toán dự báo gì?** Copper(t) trước một phiên từ 22 phiên trước; `LOOKBACK=22`, `HORIZON=1`. Kiểm: V20 cell 2/8.
2. **Một sample có shape gì?** `(22,F)`; batch thành `(N,22,F)`, output `(N,1)`. Kiểm: V20 cell 8/10.
3. **Naive là gì?** Copper(t−1), không fit, 0 parameter. Kiểm: report §1.1; parameter CSV.
4. **Direct khác Residual?** Direct dự báo level; Residual dự báo delta rồi reconstruct. Kiểm: V20 cell 10 `train_one`.
5. **Tại sao không shuffle?** Time series phải giữ causal order; random shuffle làm future/past trộn. Kiểm: `make_context`, V20 cell 8.
6. **FS06 gồm gì?** Copper, Aluminium, Tin, WTI theo thứ tự. Kiểm: V22-N cell 2.
7. **MAE được tính trên scale nào?** Giá gốc sau inverse transform. Kiểm: V20 cell 10 `original`/`evaluate`.
8. **Holdout có bao nhiêu test ở 80/20?** 542 phiên. Kiểm: report §1.1.
9. **EarlyStopping dùng để làm gì?** Dừng khi `val_loss` không cải thiện, restore best weights; giảm overfit/compute. Kiểm: V20 cell 10 `train_one`.
10. **Best holdout fitted là gì?** Theo từng split trong phạm vi đã chạy là Residual-GRU; không đồng nghĩa thắng Naive thống kê. Kiểm: report §2/§11.
11. **Dataset bao nhiêu dòng?** 2.710, 13/01/2016–13/11/2026. Kiểm: manifest V22-S.
12. **Một lag có phải một ngày?** Không luôn; là một phiên có trong dataset, có gap lịch. Kiểm: report §1.3.2.

## D2. Kỹ thuật

13. **Causal Conv1D bảo vệ điều gì?** Kernel không dùng timestep tương lai; không bảo vệ scaler/feature availability. Kiểm: V20 cell 10, report §10.
14. **Output shape sau CNN/pool?** `(N,11,32)` từ `(N,22,F)`. Kiểm bằng cấu hình V20 cell 10.
15. **LSTM V20 có bao nhiêu params với F=4?** 30.625 cho cả Direct/Residual. Kiểm: `model_parameter_counts_v20_v21.csv`.
16. **Tại sao Direct và Residual cùng params?** Builder giống nhau, chỉ target/reconstruction khác. Kiểm: V20 `build_net`, `train_one`.
17. **SA objective là gì?** Validation MSE scaled trên reference 80/20 Expanding. Kiểm: V22-S cell 12.
18. **SA có thay Adam không?** Không; SA chọn HP, Adam fit weights. Kiểm: V22-S cell 8/12.
19. **Cooling schedule?** `max(0.05, 0.88^k)`. Kiểm: V22-S cell 12.
20. **63 phiên liên quan epochs thế nào?** Không liên quan; là WF step, còn V23 mỗi fit cap45 epochs. Kiểm: V23 cell 1.
21. **Recency half-life 252 tính thế nào?** Weight `0.5^(age/252)`. Kiểm: V21 cell 10.
22. **Directional Accuracy đo gì?** Trùng dấu giữa predicted delta và actual delta. Kiểm: V20 cell 10 `evaluate`.
23. **DM-HAC d âm nghĩa gì?** Absolute loss model thấp hơn Naive trung bình. Kiểm: report §8.3.
24. **Tại sao dùng HAC lag5?** Loss difference theo thời gian có thể autocorrelated/heteroskedastic; lag5 là cấu hình code. Kiểm: V23 DM CSV.

## D3. Phản biện nghiên cứu

25. **R² 0,98 có nghĩa model rất tốt?** Chỉ cho level variance explained trên test; persistence có thể tạo R² cao dù direction kém. Kiểm: report §2.1/§9.
26. **Residual-GRU có thắng Naive không?** Holdout MAE hơi thấp ở vài best configs; V23 WF không có bằng chứng ổn định p<0,05. Kiểm: report §8.
27. **Tại sao gọi best nếu không thắng Naive?** “Best fitted” loại baseline không fit; statistical superiority là tiêu chuẩn khác. Kiểm: report §9.3.
28. **Có thể nói WTI tốt hơn Gold/Silver?** Không; FS thay hai biến, dimension và ở SA còn HP. Kiểm: report §7.5/§9.1.
29. **Spearman level cao chứng minh predictive power?** Không; trend/nonstationarity có thể gây spurious association. Kiểm: V26 `05_*`, report §1.3.6.
30. **V26 có hoàn toàn train-only cho 80/20?** Không; mốc 12/11/2024 overlap test bắt đầu 07/10/2024. Kiểm: report §1.3.1.
31. **Kết luận Granger là gì?** Chưa có; 20 phép chạy lỗi tham số `verbose`. Kiểm: `06_granger_returns.csv`, report §10.3.
32. **Tại sao không so V20 WF và V23 WF?** Khác dates 2018–2026 vs 2024–2026 và feature/protocol; độ khó regime khác. Kiểm: report §8.5.
33. **SA-LSTM kém có chứng minh SA vô dụng?** Không; chỉ bác bỏ tuyên bố ưu việt trong setup/budget này. Kiểm: V22/V23 artifacts.
34. **Brown–Forsythe và CUSUM có mâu thuẫn?** Không; variance hai nửa khác nhưng mean-return stability chưa bị bác bỏ. Kiểm: V26 `07_*`.
35. **Có data leakage không?** Code kiểm scaler/window split; nhưng point-in-time availability theo giờ thị trường chưa xác minh, nên không tuyên bố loại bỏ mọi leakage. Kiểm: report §10.
36. **Bước tiếp theo để chứng minh model tốt hơn?** Cố định HP bằng nested validation/WF, cùng timestamp/budget, ablation từng biến, test độc lập chưa dùng chọn model và audit availability. Kiểm: report §10–11.

---

## Checklist trước khi lên bảo vệ

- Nói đúng “phiên”, không tự đổi thành “ngày liên tục”.
- Nói đúng “best fitted holdout”, không nói “mô hình tốt nhất tuyệt đối”.
- Nhớ bộ số cốt lõi: 22→1; 2.710 dòng; holdout 542/271/204/136; WF V23 542 ngày, 9 folds, 5 seeds, step63; Residual-GRU 0,069100 vs Naive 0,069001.
- Khi bị hỏi thông số, trỏ đúng notebook/cell hoặc CSV trong bảng nguồn; không đoán.
- Thừa nhận ngay ba giới hạn: selection bias, EDA overlap 80/20, point-in-time availability chưa audit đầy đủ.
- Granger: **lỗi, chưa kết luận**.
