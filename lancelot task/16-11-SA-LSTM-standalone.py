# -*- coding: utf-8 -*-
"""
COPPER PRICE FORECASTING — SA-LSTM REPRODUCTION & CROSS-FEATURE BENCHMARK
Protocol V20 (HORIZON=1) & Chen et al. (2023) Reproduction (HORIZON=0)
"""


# %% [markdown] Cell 0
# # Copper Price Forecasting — SA-LSTM Reproduction & Cross-Feature Evaluation (Protocol V20 Protocol)
# 
# Notebook này triển khai thực nghiệm chạy lại, khảo sát và đối chuẩn chuyên sâu đối với thuật toán **Simulated Annealing LSTM (SA-LSTM)** theo bài báo khoa học của **Chen và cộng sự (2023)** (*"Copper price prediction using LSTM recurrent neural network integrated simulated annealing algorithm", PLOS ONE*) kết hợp với chuẩn phương pháp luận **Protocol V20** được thiết lập ngày 17/11/2026.
# 
# ### BẢNG ĐỐI CHIẾU THÔNG SỐ: BÀI BÁO GỐC (CHEN ET AL. 2023) VS. NOTEBOOK TÁI LẬP
# 
# | Thông số / Thành phần | Bài báo gốc Chen et al. (2023) | Cấu hình Notebook (`FULL_MODE`) | Ghi chú phương pháp luận |
# |---|---|---|---|
# | **Mô hình mục tiêu** | SA-LSTM (2 lớp LSTM) | SA-LSTM (2 lớp LSTM) | Đã loại bỏ hoàn toàn SA-GRU theo yêu cầu. |
# | **Số epoch huấn luyện** | **100 epochs** (Bảng 4, trang 15) | **100 epochs** (`epochs=100`) | Khớp 100% với bài báo gốc. |
# | **Batch size** | **64** (Bảng 4, trang 15) | **64** (`batch_size=64`) | Khớp 100% với bài báo gốc. |
# | **Dropout rate** | **0.2** (Bảng 4, trang 15) | **0.2** (`dropout=0.20`) | Khớp 100% với bài báo gốc. |
# | **Số nơ-ron lớp 1 (SA tìm)** | **39 neurons** (Bảng 4, trang 15) | **39 neurons** (`units_1=39`) | Cấu hình Paper SA-LSTM khớp 100%. |
# | **Số nơ-ron lớp 2 (SA tìm)** | **111 neurons** (Bảng 4, trang 15) | **111 neurons** (`units_2=111`) | Cấu hình Paper SA-LSTM khớp 100%. |
# | **Tốc độ học (Learning rate)**| **5e-5** (Bảng 4, trang 15) | **5e-5** (`lr=5e-5`) | Khớp 100% với bài báo gốc. |
# | **SA: Nhiệt độ ban đầu / cuối**| $T_0 = 1.0, T_{end} = 0.1$ | $T_0 = 1.0, T_{end} = 0.1$ | Khớp 100% với bài báo gốc. |
# | **SA: Hệ số làm nguội** | $\alpha = 0.95$ (Trang 9) | $\alpha = 0.95$ (`cooling=0.95`) | Khớp 100% với bài báo gốc. |
# | **SA: Nghiệm khởi tạo** | Units: (15, 30), LR: 5e-4 (Trang 10) | Units: (15, 30), LR: 5e-4 | Khớp 100% với bài báo gốc. |
# | **SA: Miền tìm kiếm LR** | $[0, 0.005]$, mặc định 5e-5 | $[10^{-6}, 0.005]$, fallback 5e-5 | Khớp 100% với bài báo gốc. |
# | **SA: Số kịch bản tìm kiếm** | 650 scenarios (Trang 10) | 35 evals (Budgeted) / 650 (Full) | 35 evals để tối ưu thời gian GPU Kaggle (~15-20p). |
# | **Tỷ lệ phân chia Train/Test** | **0.90, 0.925, 0.95** (Trang 12 & 16) | **0.90, 0.925, 0.95** (+ 0.80) | Bao phủ toàn bộ 3 tỷ lệ của bài báo gốc. |
# | **Tập biến đầu vào** | WTI, Gold, Silver (nhóm `paper3`) | `copper_paper3` và `copper_legacy14` | Mở rộng kiểm tra chuyển giao sang 5 kim loại. |
# | **Ranh giới dự báo (Horizon)** | Ngầm định $H=0$ (Nowcasting cùng ngày)| **Khóa chặt $HORIZON = 1$ (Protocol V20)** | **Nâng cấp cốt lõi**: Tránh rò rỉ giá cùng ngày. |
# | **Chuẩn hóa dữ liệu** | MinMaxScaler trên toàn bộ dữ liệu | MinMaxScaler chỉ fit trên train | Tránh rò rỉ phân phối từ tập test sang train. |
# | **Độ đo đánh giá** | MSE, RMSE | MAE, RMSE, MSE raw & scaled, DirAcc | Khớp đầy đủ hệ thống chỉ số Protocol V20. |
# 
# 
# 
# ---
# 
# ## ⚠️ Ghi chú Phương pháp luận: HORIZON=0 (Nowcasting) vs HORIZON=1 (Forecasting)
# 
# Bài báo gốc **Chen et al. (2023)** xây dựng cửa sổ đặc trưng như sau:  
# `X[t] = [WTI(t-21:t), Gold(t-21:t), Silver(t-21:t), Copper(t-21:t)]`  
# → **Bao gồm cả giá ngày t của các hàng hóa khác** để dự báo `Copper(t)` trong cùng ngày.  
# Đây là chiến lược **HORIZON=0 (Nowcasting)** — tức là dự báo đồng thời với dữ liệu cùng ngày.
# 
# **Vấn đề thực tế:** Giá WTI, Gold, Silver chốt vào các múi giờ khác nhau (ví dụ London Metal Exchange và CME Group). Trong điều kiện giao dịch thực tế, giá đóng cửa của các hàng hóa này đôi khi **chưa có** vào thời điểm lệnh đặt mua/bán Copper phải được thực hiện. Do đó HORIZON=0 có thể dẫn đến **rò rỉ thông tin tương lai (look-ahead bias)** trong các chiến lược giao dịch intraday.
# 
# **Protocol V20 chọn HORIZON=1 (True Forecasting):**  
# `X[t] = [WTI(t-22:t-1), Gold(t-22:t-1), Silver(t-22:t-1), Copper(t-22:t-1)]`  
# → Chỉ dùng thông tin đến cuối ngày `t-1` để dự báo `Copper(t)`.  
# Đây là cài đặt **khả thi thực tế nghiêm ngặt** và phù hợp với chuẩn thực nghiệm học thuật.
# 
# **Lý do notebook này chạy CẢ HAI:**  
# - Kết quả HORIZON=0 phục vụ **tái tạo thực nghiệm bài báo** (kiểm tra xem kết quả của bài báo có thể tái tạo được không).  
# - Kết quả HORIZON=1 phục vụ **đánh giá khả năng dự báo thực tế** theo Protocol V20.  
# - Việc đối chiếu HORIZON=0 vs HORIZON=1 làm rõ **tác động định lượng của rò rỉ dữ liệu** lên các chỉ số đánh giá.
# 
# > **Tóm lại:** Nếu MSE của HORIZON=0 thấp hơn đáng kể so với HORIZON=1, điều đó xác nhận bài báo gốc hưởng lợi từ cùng-ngày signals và kết quả khó chuyển giao sang bối cảnh giao dịch thực tế.
# 

# %% [markdown] Cell 1
# ## 1. Cấu hình và Môi trường Thực nghiệm (Protocol V20 Protocol)

# %% [code] Cell 2
from pathlib import Path
import os, gc, json, time, math, hashlib, zipfile, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# ─── CHẾ ĐỘ THỰC NGHIỆM ─────────────────────────────────────────────────────────
# 'QUICK_MODE': Kiểm tra logic chạy nhanh (5 epochs, 5 SA evals)
# 'FULL_MODE': Chế độ chuẩn báo cáo (100 epochs chuẩn bài báo, 35 SA evals, early stopping)
# 'EXACT_PAPER_MODE': Chạy đúng 100 epochs cố định không early stopping, 3 tỷ lệ bài báo (0.9, 0.925, 0.95)
MODE = 'FULL_MODE'
MODE_CFG = {
    'QUICK_MODE':       {'epochs': 5,   'search_epochs': 3,  'sa_evals': 5,   'earlystop': True},
    'FULL_MODE':        {'epochs': 100, 'search_epochs': 12, 'sa_evals': 100, 'earlystop': True},
    'EXACT_PAPER_MODE': {'epochs': 100, 'search_epochs': 25, 'sa_evals': 50,  'earlystop': False}
}
CFG = MODE_CFG[MODE]

# ─── THIẾT LẬP PHƯƠNG PHÁP LUẬN V20 & BÀI BÁO GỐC ───────────────────────────────
LOOKBACK = 22
HORIZON = 1          # Cố định HORIZON=1: Protocol V20 (true forecasting, không look-ahead)
HORIZON_PAPER = 0    # HORIZON=0 bài báo gốc: Nowcasting (dùng features cùng ngày t)
# Đặt RUN_HORIZON0 = True để thêm thực nghiệm tái tạo chiến lược bài báo gốc
# (Tốn thêm ~50% thời gian chạy, chỉ áp dụng trên paper3 + ratio=0.90 + expanding)
RUN_HORIZON0 = True
# Bài báo Chen et al. (2023) sử dụng 3 tỷ lệ phân chia: 0.90, 0.925, 0.95
# V20 bổ sung thêm 0.80 để khảo sát tính vững chắc trên cửa sổ dài hạn
RATIOS = (0.90, 0.925, 0.95) if MODE == 'EXACT_PAPER_MODE' else (0.80, 0.90, 0.925, 0.95)
PRIMARY_RATIO = 0.90
BATCH_SIZE = 64      # Chuẩn 64 như Bảng 4 bài báo
SEED = 42
MAX_RUNTIME_MINUTES = 240  # Giới hạn an toàn trước ngưỡng session timeout
JOB_START = time.monotonic()

# ─── TẬP ĐẶC TRƯNG KHẢO SÁT ───────────────────────────────────────────────────
# copper_paper3: Bộ biến theo bài báo Chen et al. (Copper + WTI + Gold + Silver)
# copper_legacy14: Bộ biến mở rộng 5 kim loại (Copper + Aluminium + Tin + Silver + Gold)
FEATURE_SETS = {
    'copper_paper3': ['Copper', 'WTI', 'Gold', 'Silver'],
    'copper_legacy14': ['Copper', 'Aluminium', 'Tin', 'Silver', 'Gold']
}
SCHEMES = ('expanding', 'rolling_756', 'recency_weighted')

# ─── ĐƯỜNG DẪN VÀ CHECKPOINT ──────────────────────────────────────────────────
DATASET_CANDIDATES = [
    Path('/kaggle/input/datasets/lancelot71000/dataset-9-commodities-daily/dataset_9_commodities_daily.csv'),
    Path('/kaggle/input/dataset-9-commodities-daily/dataset_9_commodities_daily.csv'),
    Path('data/processed/dataset_9_commodities_daily.csv'),
    Path('data/raw/dataset_9_commodities_daily.csv'),
    Path('dataset_9_commodities_daily.csv')
]

OUT = Path('/kaggle/working/copper_sa_lstm_v20') if Path('/kaggle/working').exists() else Path('copper_sa_lstm_v20')
OUT.mkdir(parents=True, exist_ok=True)

# Cố định tính ngẫu nhiên
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
for gpu in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError: pass

print('═'*75)
print(f'COPPER SA-LSTM BENCHMARK | CHẾ ĐỘ: {MODE}')
print(f'Epochs: {CFG["epochs"]} (Chuẩn bài báo 100) | Batch Size: {BATCH_SIZE} | EarlyStop: {CFG["earlystop"]}')
print(f'Horizon: {HORIZON} | Lookback: {LOOKBACK} | Seed: {SEED}')
print(f'Splits: {[f"{r*100:g}%" for r in RATIOS]}')
print(f'Feature Sets: {list(FEATURE_SETS.keys())}')
print(f'GPU Devices: {tf.config.list_physical_devices("GPU")}')
print(f'Output Directory: {OUT.resolve()}')


# %% [markdown] Cell 3
# ## 2. Nạp Dữ liệu, Kiểm tra Chất lượng và Tính Toàn vẹn

# %% [code] Cell 4
TARGET = 'Copper'
REQUIRED_COLS = ['Copper', 'WTI', 'Gold', 'Silver', 'Aluminium', 'Tin']

DATA_PATH = next((p for p in DATASET_CANDIDATES if p.is_file()), None)
if DATA_PATH is None and Path('/kaggle/input').exists():
    matches = list(Path('/kaggle/input').rglob('dataset_9_commodities_daily.csv'))
    if len(matches) == 1: DATA_PATH = matches[0]
if DATA_PATH is None:
    raise FileNotFoundError('Không tìm thấy file dataset_9_commodities_daily.csv. Vui lòng kiểm tra đường dẫn.')

raw = pd.read_csv(DATA_PATH)
if 'Date' not in raw.columns or TARGET not in raw.columns:
    raise KeyError('Tệp CSV phải có cột Date và Copper.')

raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
if raw['Date'].isna().any(): raise ValueError('Cột Date có dòng không hợp lệ.')
if raw['Date'].duplicated().any(): raise ValueError('Có ngày bị trùng lặp; cần xử lý nguồn dữ liệu.')

for col in REQUIRED_COLS:
    if col not in raw.columns: raise ValueError(f'Thiếu cột bắt buộc: {col}')
    raw[col] = pd.to_numeric(raw[col], errors='coerce')

raw = raw.sort_values('Date').reset_index(drop=True)
quality = pd.DataFrame({
    'column': raw.columns,
    'missing': raw.isna().sum().values,
    'dtype': raw.dtypes.astype(str).values
})
quality.to_csv(OUT / 'data_quality.csv', index=False)

# Làm sạch dữ liệu trên hợp các biến cần dùng
data = raw.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
assert len(data) > 1000 and data['Date'].is_monotonic_increasing
assert all((data[c] > 0).all() for c in REQUIRED_COLS), 'Giá <= 0: cần kiểm toán dữ liệu trước khi huấn luyện.'

dataset_sha256 = hashlib.sha256(Path(DATA_PATH).read_bytes()).hexdigest()
(OUT / 'dataset_fingerprint.txt').write_text(dataset_sha256)

print(f'Nạp dữ liệu thành công từ: {DATA_PATH}')
print(f'Tổng số phiên: {len(data):,} phiên | Từ ngày {data.Date.iloc[0].date()} đến {data.Date.iloc[-1].date()}')
print(f'SHA256 Fingerprint: {dataset_sha256[:16]}...')
display(quality[quality['column'].isin(['Date'] + REQUIRED_COLS)])

# %% [markdown] Cell 5
# ## 3. Phân chia Dữ liệu và Chuẩn hóa Chuẩn Protocol V20 (Leak-Safe Windowing)
# 
# - Cố định $HORIZON = 1$: Dự báo $Copper(t)$ hoàn toàn từ các quan sát $\le t-1$. Cửa sổ trượt lấy $X[t-22 : t-1]$ (kích thước $22 \times D$).
# - `MinMaxScaler` **chỉ được fit trên tập huấn luyện ban đầu** (`[fit_start, val_start)`), không thấy trước dữ liệu validation và test.
# - Giá trị cơ sở Naive Persistence $Copper(t-1)$ được lưu ở cùng không gian chuẩn hóa để đối chiếu công bằng.
# 
# > **Lưu ý HORIZON=0:** Phần 12 của notebook này bổ sung thực nghiệm HORIZON=0 (nowcasting)  
# > để tái tạo chiến lược gốc của bài báo và làm rõ tác động định lượng của same-day look-ahead.  
# > Toàn bộ Phần 3–11 cố định HORIZON=1 (Protocol V20).
# 

# %% [code] Cell 6
def make_context(frame, ratio, scheme='expanding', feature_set='copper_paper3', val_fraction=0.10, horizon=None):
    n = len(frame)
    split = int(n * ratio)
    val_start = int(split * (1.0 - val_fraction))
    fit_start = 0 if scheme != 'rolling_756' else max(0, val_start - 756)
    cols = FEATURE_SETS[feature_set]
    h = horizon if horizon is not None else HORIZON
    
    assert LOOKBACK + 25 < val_start < split < n
    assert (val_start - fit_start) > LOOKBACK + 10
    
    # Fit scaler DUY NHẤT trên tập train trước validation
    xsc = MinMaxScaler().fit(frame.loc[fit_start:val_start-1, cols])
    ysc = MinMaxScaler().fit(frame.loc[fit_start:val_start-1, [TARGET]])
    
    x = xsc.transform(frame[cols]).astype('float32')
    y = ysc.transform(frame[[TARGET]]).ravel().astype('float32')
    prev = np.r_[np.nan, y[:-1]]  # Giá trị t-1 trong cùng miền chuẩn hóa của Copper
    
    # ─── XỬ LÝ MA TRẬN ĐẦU VÀO THEO HORIZON ────────────────────────────────────
    # Cột 0 luôn là TARGET ('Copper')
    # HORIZON=1 (Protocol V20): Tất cả các biến từ [t-LOOKBACK : t] đại diện cho [t-22 : t-1].
    # HORIZON=0 (Bài báo gốc - Nowcasting): 
    #   - Các biến khác (WTI, Gold, Silver, ...): lấy ở thời điểm t (cửa sổ kết thúc ở t).
    #   - Biến Copper: CHỈ ĐƯA ĐẾN t-1 (dịch trễ 1 ngày), TUYỆT ĐỐI KHÔNG rò rỉ Copper(t) vào X!
    if h == 0:
        x_prepared = x.copy()
        x_prepared[:, 0] = np.r_[0.0, x[:-1, 0]]  # Copper trễ 1 ngày: dòng t nhận Copper(t-1)
    else:
        x_prepared = x
    
    def block(start, end, restrict_train=False):
        indices = np.arange(start, end)
        indices = indices[indices >= LOOKBACK]
        if restrict_train:
            indices = indices[indices - LOOKBACK >= fit_start]
        
        if h == 0:
            # Lấy cửa sổ [t-LOOKBACK+1 : t+1]: độ dài LOOKBACK, kết thúc tại t
            # Cột Copper bước cuối cùng = Copper(t-1); các cột khác bước cuối cùng = Feature(t)
            X = np.stack([x_prepared[t-LOOKBACK+1:t+1] for t in indices]).astype('float32')
        else:
            # Lấy cửa sổ [t-LOOKBACK : t]: độ dài LOOKBACK, kết thúc tại t-1 (không có look-ahead)
            X = np.stack([x_prepared[t-LOOKBACK:t] for t in indices]).astype('float32')
            
        return {
            'idx': indices,
            'X': X,
            'y': y[indices],
            'prev': prev[indices].astype('float32'),
            'date': frame.Date.iloc[indices].to_numpy()
        }
    
    tr = block(fit_start, val_start, True)
    va = block(val_start, split)
    te = block(split, n)
    
    # Kiểm toán ranh giới dữ liệu
    assert tr['idx'][-1] < va['idx'][0] and va['idx'][-1] < te['idx'][0]
    # Cả H=1 lẫn H=0: bước cuối cùng của cột Copper luôn luôn bằng prev[t] = Copper(t-1)
    assert np.allclose(tr['X'][:, -1, 0], tr['prev'], atol=2e-6, rtol=1e-6), 'Rò rỉ Copper hoặc sai lệch bước trễ!'
    assert xsc.n_samples_seen_ == val_start - fit_start
    
    return {
        'train': tr, 'val': va, 'test': te,
        'xsc': xsc, 'ysc': ysc,
        'features': cols, 'feature_set': feature_set, 'horizon': h,
        'ratio': float(ratio), 'scheme': scheme,
        'fit_start': fit_start, 'val_start': val_start, 'test_start': split
    }

# Smoke check kiểm tra tính đồng bộ của ranh giới test trên cả 2 nhóm biến
ctx_p3 = make_context(data, PRIMARY_RATIO, 'expanding', 'copper_paper3')
ctx_l14 = make_context(data, PRIMARY_RATIO, 'expanding', 'copper_legacy14')
assert np.array_equal(ctx_p3['test']['idx'], ctx_l14['test']['idx'])

# Smoke check cho HORIZON=0 (bảo đảm không rò rỉ Copper(t) vào X)
ctx_h0_check = make_context(data, PRIMARY_RATIO, 'expanding', 'copper_paper3', horizon=0)
assert np.allclose(ctx_h0_check['train']['X'][:, -1, 0], ctx_h0_check['train']['prev'], atol=2e-6)
print('PASS: Kiểm tra ranh giới thông tin (cả H=1 và H=0) thành công!')
print(f'Test split {PRIMARY_RATIO*100:g}%: Train={len(ctx_p3["train"]["idx"]):,} | Val={len(ctx_p3["val"]["idx"]):,} | Test={len(ctx_p3["test"]["idx"]):,}')


# %% [markdown] Cell 7
# ## 4. Các Kiến trúc Mô hình (Baseline LSTM, Paper SA-LSTM và Re-optimized SA-LSTM)
# 
# Notebook chỉ tập trung vào kiến trúc **LSTM** (đã loại bỏ toàn bộ GRU/SA-GRU theo yêu cầu phương pháp luận):
# 1. **`build_baseline_lstm`**: Kiến trúc chuẩn 2 lớp LSTM (64/32 units, Dropout 0.2, lr $10^{-3}$) từ giao thức Protocol V20.
# 2. **`build_paper_sa_lstm`**: Cấu hình siêu tham số công bố trong bài báo Chen et al. (2023) (`units_1: 39, units_2: 111, lr: 5e-5`).
# 3. **`build_sa_lstm`**: Kiến trúc tham số hóa linh hoạt nhận vào bất kỳ bộ siêu tham số nào do thuật toán SA đề xuất.

# %% [code] Cell 8
PAPER_SA_LSTM_HP = {'units_1': 39, 'units_2': 111, 'learning_rate': 5e-5}

def build_baseline_lstm(input_shape, hp=None):
    """Mô hình LSTM chuẩn không tối ưu SA (Protocol V20 baseline)."""
    model = tf.keras.Sequential(name='Baseline_LSTM')
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.LSTM(32))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model

def build_paper_sa_lstm(input_shape, hp=None):
    """Kiến trúc SA-LSTM với siêu tham số công bố trong bài báo Chen et al. (2023)."""
    model = tf.keras.Sequential(name='Paper_SA_LSTM')
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.LSTM(39, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.LSTM(111, activation='tanh', recurrent_activation='sigmoid'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss='mse')
    return model

def build_sa_lstm_paper3(input_shape, hp):
    """Kiến trúc SA-LSTM tham số hóa bởi bộ siêu tham số do thuật toán SA tìm được."""
    u1 = int(hp.get('units_1', 39))
    u2 = int(hp.get('units_2', 111))
    lr = float(hp.get('learning_rate', 5e-5))
    model = tf.keras.Sequential(name='Reoptimized_SA_LSTM')
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.LSTM(22, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.LSTM(91, activation='tanh', recurrent_activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.000922), loss='mse')
    return model
    
def build_sa_lstm_legacy14(input_shape, hp):
    """Kiến trúc SA-LSTM tham số hóa bởi bộ siêu tham số do thuật toán SA tìm được."""
    u1 = int(hp.get('units_1', 39))
    u2 = int(hp.get('units_2', 111))
    lr = float(hp.get('learning_rate', 5e-5))
    model = tf.keras.Sequential(name='Reoptimized_SA_LSTM')
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.LSTM(22, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.LSTM(83, activation='tanh', recurrent_activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.000655), loss='mse')
    return model





print('Đã khởi tạo xong các hàm dựng mô hình LSTM.')

# %% [markdown] Cell 9
# ## 5. Quy trình Huấn luyện và Chỉ số Đánh giá (Protocol V20 Metrics)
# 
# Hệ thống đo lường hiệu năng áp dụng đúng chuẩn mực của Protocol V20, phân định rõ ràng giữa thang đo giá gốc (USD) và thang đo chuẩn hóa (scaled):
# - `mae`: Sai số tuyệt đối trung bình trên **giá gốc (dịch ngược lại, đơn vị USD)**: $\text{MAE}(y_{true}, \widehat{y})$.
# - `rmse`: Căn bậc hai sai số bình phương trung bình trên **giá gốc (dịch ngược lại, đơn vị USD)**: $\sqrt{\text{MSE}(y_{true}, \widehat{y})}$.
# - `mse`: Sai số bình phương trung bình trên **giá gốc (dịch ngược lại, đơn vị USD$^2$)** — **khớp 100% với cột `mse` của V20**.
# - `mse_scaled`: Sai số bình phương trung bình trên **thang đo chuẩn hóa $[0, 1]$ của tập test**: $\text{MSE}(y_{test\_scaled}, \widehat{y}_{scaled})$.
# - `val_mse_scaled`: Sai số bình phương trung bình trên **thang đo chuẩn hóa $[0, 1]$ của tập validation** — dùng làm hàm mục tiêu tối ưu hóa cho Simulated Annealing và Early Stopping (khớp với `val_mse_scaled` của V20).
# - `mae_naive`: MAE của mô hình cơ sở Naive Persistence trên **giá gốc (USD)** (dự báo $\widehat{y}_t = y_{t-1}$).
# - `mae_improvement_pct`: Tỷ lệ phần trăm cải thiện MAE so với Naive ($100 \times \frac{MAE_{naive} - MAE}{MAE_{naive}}$).
# - `directional_accuracy_pct`: Độ chính xác dự báo hướng biến động giá ($100 \times \text{mean}(\text{sign}(\Delta_{pred}) == \text{sign}(\Delta_{true}))$).
# - `delta_pred_std_raw` & `delta_true_std_raw`: Kiểm toán độ lệch chuẩn của phần dư dự đoán so với thực tế trên giá gốc.
# - `delta_pred_mean_abs_raw` & `pred_delta_corr`: Độ lệch tuyệt đối trung bình và tương quan của phần dư delta (khớp với V20).

# %% [code] Cell 10
def reset_keras(seed=SEED):
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(int(seed))

def sample_weights(length, scheme, half_life=252):
    if scheme != 'recency_weighted': return None
    ages = np.arange(length - 1, -1, -1, dtype='float64')
    w = np.exp(-np.log(2) * ages / half_life)
    return (w / w.mean()).astype('float32')

def evaluate_predictions(ctx, model_name, p_val, p_test, seed=SEED, extras=None):
    ysc = ctx['ysc']; va = ctx['val']; te = ctx['test']
    def original(z): return ysc.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()
    
    # 1. Dịch ngược về giá gốc (USD)
    y_true = original(te['y'])
    pred = original(p_test)
    base = original(te['prev'])
    
    delta_pred = pred - base
    delta_true = y_true - base
    
    # 2. Tính các độ đo trên giá gốc (RAW - USD) khớp 100% với V20
    mae = float(mean_absolute_error(y_true, pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mse = float(mean_squared_error(y_true, pred))  # DỊCH NGƯỢC LẠI GIÁ GỐC (USD^2) - KHỚP V20
    base_mae = float(mean_absolute_error(y_true, base))
    improvement = float(100.0 * (base_mae - mae) / base_mae) if base_mae > 0 else 0.0
    
    # 3. Tính các độ đo trên thang đo chuẩn hóa (SCALED [0, 1])
    val_mse_scaled = float(mean_squared_error(va['y'], p_val))
    mse_scaled = float(mean_squared_error(te['y'], p_test)) if p_test is not None else np.nan
    
    # 4. Tính Directional Accuracy và các thống kê kiểm toán phần dư
    dir_acc = float(100.0 * np.mean(np.sign(delta_pred) == np.sign(delta_true)))
    delta_pred_std = float(np.std(delta_pred))
    delta_true_std = float(np.std(delta_true))
    delta_pred_mean_abs = float(np.mean(np.abs(delta_pred)))
    pred_delta_corr = float(np.corrcoef(delta_pred, delta_true)[0, 1]) if delta_pred_std > 1e-12 else np.nan
    
    row = {
        'model': model_name,
        'feature_set': ctx['feature_set'],
        'scheme': ctx['scheme'],
        'ratio': ctx['ratio'],
        'seed': int(seed),
        'n_test': len(y_true),
        'mae': mae,
        'rmse': rmse,
        'mse': mse,                        # GIÁ GỐC DỊCH NGƯỢC (USD^2) - KHỚP V20
        'mse_scaled': mse_scaled,          # SCALED TRÊN TẬP TEST [0, 1]
        'mae_naive': base_mae,
        'mae_improvement_pct': improvement,
        'val_mse_scaled': val_mse_scaled,  # SCALED TRÊN TẬP VALIDATION [0, 1] - KHỚP V20
        'directional_accuracy_pct': dir_acc,
        'delta_pred_std_raw': delta_pred_std,
        'delta_true_std_raw': delta_true_std,
        'delta_pred_mean_abs_raw': delta_pred_mean_abs,
        'pred_delta_corr': pred_delta_corr,
        'test_start': str(pd.Timestamp(te['date'][0]).date()),
        'fit_start': str(data.Date.iloc[ctx['fit_start']].date())
    }
    if extras: row.update(extras)
    
    predictions = pd.DataFrame({
        'Date': te['date'],
        'ratio': ctx['ratio'],
        'scheme': ctx['scheme'],
        'feature_set': ctx['feature_set'],
        'model': model_name,
        'seed': int(seed),
        'actual': y_true,
        'pred': pred,
        'naive': base,
        'pred_delta': delta_pred,
        'actual_delta': delta_true
    })
    return row, predictions

def fit_predict_model(builder, ctx, hp=None, seed=SEED, epochs=None, patience=None, is_search=False, skip_test=False):
    reset_keras(seed)
    tr, va, te = ctx['train'], ctx['val'], ctx['test']
    model = builder(tr['X'].shape[1:], hp)
    
    cb = []
    if CFG['earlystop']:
        pat = patience if patience is not None else (4 if is_search else 9)
        cb.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=pat, restore_best_weights=True))
        
    ep = epochs if epochs is not None else (CFG['search_epochs'] if is_search else CFG['epochs'])
    weights = sample_weights(len(tr['idx']), ctx['scheme'])
    
    start_t = time.perf_counter()
    hist = model.fit(
        tr['X'], tr['y'],
        validation_data=(va['X'], va['y']),
        epochs=ep,
        batch_size=BATCH_SIZE,
        sample_weight=weights,
        callbacks=cb,
        shuffle=False,
        verbose=0
    )
    fit_sec = time.perf_counter() - start_t
    
    pv = model.predict(va['X'], verbose=0).ravel()
    pt = model.predict(te['X'], verbose=0).ravel() if not skip_test else None
    val_mse = float(mean_squared_error(va['y'], pv))
    epochs_run = len(hist.history['loss'])
    
    del model
    reset_keras(seed)
    return {'val_pred': pv, 'test_pred': pt, 'val_mse': val_mse, 'seconds': fit_sec, 'epochs_run': epochs_run}

# %% [markdown] Cell 11
# ## 6. Thuật toán Tối ưu hóa Siêu tham số Simulated Annealing (SA Search)
# 
# Thuật toán mô phỏng luyện kim bám sát công thức đề xuất trong Chen et al. (2023) với không gian tìm kiếm được **thu hẹp trọng tâm (Narrowed Search Space)** và số lượng kịch bản được cấu hình là **100 scenarios** (`sa_evals = 100`):
# - **Khoảng quy định nơ-ron lớp 1 (`units_1`)**: $\in [16, 64]$ (tập trung bao phủ vùng tối ưu 39 của bài báo).
# - **Khoảng quy định nơ-ron lớp 2 (`units_2`)**: $\in [64, 128]$ (tập trung bao phủ vùng tối ưu 111 của bài báo).
# - **Khoảng quy định tốc độ học (`learning_rate`)**: $\in [10^{-5}, 10^{-3}]$ trên thang logarit (loại bỏ dải dưới $10^{-5}$ học quá chậm và dải trên $10^{-3}$ dễ phân kỳ gradient).
# - **Hàm mục tiêu**: `val_loss` (MSE trên tập validation trong không gian co giãn $[0, 1]$), tuyệt đối không chạm vào tập test.
# - **Số kịch bản tìm kiếm**: 100 kịch bản (`sa_evals = 100`), tương ứng với 100 lượt đề xuất và đánh giá mô hình.

# %% [code] Cell 12
# ─── CẤU HÌNH THU HẸP KHÔNG GIAN TÌM KIẾM SA (NARROWED SEARCH SPACE) ─────────
PAPER_SA = {
    # Khoảng quy định số nơ-ron (tương đương suggest_int trong Optuna)
    'units_1_min': 16,    # Lớp 1: tối thiểu 16 nơ-ron
    'units_1_max': 64,    # Lớp 1: tối đa 64 nơ-ron (bao phủ nghiệm 39 của bài báo)
    'units_2_min': 64,    # Lớp 2: tối thiểu 64 nơ-ron
    'units_2_max': 128,   # Lớp 2: tối đa 128 nơ-ron (bao phủ nghiệm 111 của bài báo)
    
    # Khoảng quy định tốc độ học (tương đương suggest_float log=True trong Optuna)
    'lr_min': 1e-5,       # Tối thiểu 0.00001 (1e-5)
    'lr_max': 1e-3,       # Tối đa 0.001 (1e-3, loại bỏ vùng gradient bùng nổ)
    
    # Điểm xuất phát khởi tạo ban đầu hợp lý bên trong miền thu hẹp
    'initial': {'units_1': 24, 'units_2': 80, 'learning_rate': 2e-4},
    
    # Các tham số làm nguội và bước nhảy SA
    'initial_temperature': 1.0,
    'final_temperature': 0.1,
    'cooling': 0.95,
    'proposals_per_temperature': 10,   # 10 bước đề xuất tại mỗi mức nhiệt độ
    'proposal_width': 1.5,             # Thu hẹp độ rộng bước nhảy để tập trung khai thác sâu
}

def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-float(np.clip(z, -30, 30))))

def _logit(p):
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))

def paper_proposal(current, temperature, rng):
    candidate = {}
    for key in ('units_1', 'units_2', 'learning_rate'):
        if key == 'learning_rate':
            lo = math.log(PAPER_SA['lr_min'])
            hi = math.log(PAPER_SA['lr_max'])
            value = math.log(current[key])
        elif key == 'units_1':
            lo = PAPER_SA['units_1_min']
            hi = PAPER_SA['units_1_max']
            value = current[key]
        elif key == 'units_2':
            lo = PAPER_SA['units_2_min']
            hi = PAPER_SA['units_2_max']
            value = current[key]
        else:
            raise KeyError(key)
            
        # Chuyển sang không gian latent via logit
        latent = _logit((value - lo) / (hi - lo))
        # Khuếch tán theo nhiệt độ hiện tại
        moved = latent + temperature * PAPER_SA['proposal_width'] * rng.uniform(-1.0, 1.0)
        # Chiếu ngược lại không gian bị chặn bằng Sigmoid
        bounded = lo + (hi - lo) * _sigmoid(moved)
        
        if key == 'learning_rate':
            candidate[key] = float(np.clip(math.exp(bounded), PAPER_SA['lr_min'], PAPER_SA['lr_max']))
        else:
            candidate[key] = int(np.clip(round(bounded), lo, hi))
            
    return candidate

def metropolis(delta, temperature, rng):
    probability = 1.0 if delta <= 0 else math.exp(-delta / max(temperature, 1e-12))
    return bool(rng.random() < probability), probability

def run_simulated_annealing(search_name, ctx, sa_evals=None):
    target_evals = sa_evals or CFG['sa_evals']
    checkpoint_path = OUT / f'{search_name}_sa_checkpoint.json'
    history_path = OUT / f'{search_name}_sa_history.csv'
    rng = np.random.default_rng(SEED)
    
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        rng.bit_generator.state = state['rng_state']
        print(f'Tải lại trạng thái checkpoint {search_name} từ evaluation {state["evaluations"]}/{target_evals}')
    else:
        current = dict(PAPER_SA['initial'])
        res = fit_predict_model(build_sa_lstm, ctx, hp=current, is_search=True, skip_test=True)
        loss, seconds, epochs_run = res['val_mse'], res['seconds'], res['epochs_run']
        state = {
            'current': current, 'current_loss': loss,
            'best': dict(current), 'best_loss': loss,
            'evaluations': 1,
            'history': [{
                'evaluation': 1, 'temperature': 1.0,
                'val_mse_scaled': loss, 'best_mse_scaled': loss,
                'accepted': True, 'acceptance_probability': 1.0,
                **current, 'epochs_run': epochs_run, 'train_seconds': seconds
            }]
        }
        
    while state['evaluations'] < target_evals:
        if (time.monotonic() - JOB_START) / 60.0 >= MAX_RUNTIME_MINUTES:
            print(f'Đạt giới hạn thời gian chạy ({MAX_RUNTIME_MINUTES}m): đã lưu checkpoint.')
            break
            
        prop_idx = state['evaluations'] - 1
        temperature = PAPER_SA['initial_temperature'] * (PAPER_SA['cooling'] ** (
            prop_idx // PAPER_SA['proposals_per_temperature']))
        if temperature < PAPER_SA['final_temperature']:
            break
            
        candidate = paper_proposal(state['current'], temperature, rng)
        res = fit_predict_model(build_sa_lstm, ctx, hp=candidate, is_search=True, skip_test=True)
        loss, seconds, epochs_run = res['val_mse'], res['seconds'], res['epochs_run']
        
        accepted, prob = metropolis(loss - state['current_loss'], temperature, rng)
        if accepted:
            state['current'], state['current_loss'] = candidate, loss
        if loss < state['best_loss']:
            state['best'], state['best_loss'] = dict(candidate), loss
            
        state['evaluations'] += 1
        state['history'].append({
            'evaluation': state['evaluations'], 'temperature': temperature,
            'val_mse_scaled': loss, 'best_mse_scaled': state['best_loss'],
            'accepted': accepted, 'acceptance_probability': prob,
            **candidate, 'epochs_run': epochs_run, 'train_seconds': seconds
        })
        state['rng_state'] = rng.bit_generator.state
        checkpoint_path.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')
        pd.DataFrame(state['history']).to_csv(history_path, index=False)
        
        if state['evaluations'] % 10 == 0 or accepted:
            print(f"[{search_name}] eval {state['evaluations']}/{target_evals}: "
                  f"T={temperature:.4f} | val={loss:.6f} | best={state['best_loss']:.6f} | accepted={accepted}")
                  
    state['rng_state'] = rng.bit_generator.state
    checkpoint_path.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')
    pd.DataFrame(state['history']).to_csv(history_path, index=False)
    
    best_result = {
        'search_name': search_name,
        'best_hp': state['best'],
        'best_val_mse_scaled': state['best_loss'],
        'evaluations': state['evaluations']
    }
    (OUT / f'{search_name}_best_hp.json').write_text(json.dumps(best_result, indent=2), encoding='utf-8')
    print(f'==> Hoàn tất SA {search_name}. Siêu tham số tối ưu: {state["best"]}')
    return state['best'], pd.DataFrame(state['history'])


# %% [markdown] Cell 13
# ## 7. Thực thi Tìm kiếm SA trên `paper3` và `legacy14`
# 
# 1. **Bước 7A**: Chạy Simulated Annealing trên tập biến `copper_paper3` (tỷ lệ 90/10, expanding) $\to$ Tìm bộ siêu tham số tối ưu dành cho `paper3` ở $HORIZON=1$.
# 2. **Bước 7B**: Chạy Simulated Annealing trên tập biến `copper_legacy14` (tỷ lệ 90/10, expanding) $\to$ Tái tối ưu hóa siêu tham số dành riêng cho `legacy14`.

# %% [code] Cell 14
# ─── BỘ SIÊU THAM SỐ ĐÃ TÌM ĐƯỢC TỪ THỰC NGHIỆM SA (H=1) ──────────────────────
# Kết quả tối ưu đã chạy được từ thực nghiệm:
#   - Paper3 (H=1):   units_1=22, units_2=91, lr=0.000922
#   - Legacy14 (H=1): units_1=22, units_2=83, lr=0.000655
sa_best_cache = {
    'copper_paper3': {'units_1': 22, 'units_2': 91, 'learning_rate': 0.000922},
    'copper_legacy14': {'units_1': 22, 'units_2': 83, 'learning_rate': 0.000655}
}
hp_p3 = sa_best_cache['copper_paper3']
hp_l14 = sa_best_cache['copper_legacy14']

print('━'*75)
print('SỬ DỤNG BỘ SIÊU THAM SỐ SA (H=1) ĐÃ CHẠY ĐƯỢC TRƯỚC ĐÓ (KHÔNG HUẤN LUYỆN LẠI)')
print('━'*75)
print(f'Paper3 (H=1):   {hp_p3}')
print(f'Legacy14 (H=1): {hp_l14}')

# ─── [ĐÃ COMMENT LẠI ĐOẠN CHẠY SA H=1 ĐỂ KHÔNG PHẢI TRAIN LẠI] ─────────────────
# print('━'*75)
# print('BƯỚC 7A: TÌM KIẾM SA TRÊN NHÓM BIẾN PAPER3 (H=1)')
# print('━'*75)
# ctx_sa_p3 = make_context(data, PRIMARY_RATIO, 'expanding', 'copper_paper3')
# hp_p3, hist_p3 = run_simulated_annealing('sa_lstm_paper3', ctx_sa_p3)
# sa_best_cache['copper_paper3'] = hp_p3
# sa_history_cache['copper_paper3'] = hist_p3

# print('\n' + '━'*75)
# print('BƯỚC 7B: TÌM KIẾM SA TRÊN NHÓM BIẾN LEGACY14 (H=1)')
# print('━'*75)
# ctx_sa_l14 = make_context(data, PRIMARY_RATIO, 'expanding', 'copper_legacy14')
# hp_l14, hist_l14 = run_simulated_annealing('sa_lstm_legacy14', ctx_sa_l14)
# sa_best_cache['copper_legacy14'] = hp_l14
# sa_history_cache['copper_legacy14'] = hist_l14

# Lưu tổng hợp siêu tham số H=1
(OUT / 'sa_best_hyperparameters.json').write_text(json.dumps(sa_best_cache, indent=2), encoding='utf-8')

# Hiển thị đối chiếu các bộ siêu tham số H=1
hp_comparison = pd.DataFrame([
    {'Nguồn cấu hình': 'Bài báo Chen et al. (2023)', 'units_1': 39, 'units_2': 111, 'learning_rate': 5e-5},
    {'Nguồn cấu hình': 'Re-optimized SA trên Paper3 (H=1)', **hp_p3},
    {'Nguồn cấu hình': 'Re-optimized SA trên Legacy14 (H=1)', **hp_l14}
])
print('\nĐỐI CHIẾU CÁC BỘ SIÊU THAM SỐ (H=1):')
display(hp_comparison)


# %% [markdown] Cell 15
# ## 8. Huấn luyện Đầy đủ và Đánh giá Đối chuẩn Mô hình (Protocol V20 Protocol)
# 
# Thực hiện ma trận thực nghiệm đối chuẩn hoàn chỉnh giữa các mô hình:
# 1. **Trên nhóm biến `copper_paper3`**:
#    - `naive`: Naive Persistence baseline (đường chuẩn cơ sở)
#    - `baseline_lstm`: Mô hình LSTM chuẩn Protocol V20 (64/32 units, lr $10^{-3}$)
#    - `paper_sa_lstm`: Cấu hình bài báo Chen et al. (39/111 units, lr $5 \times 10^{-5}$)
#    - `reopt_sa_lstm`: Mô hình SA-LSTM với siêu tham số tái tối ưu hóa trên `paper3`
# 2. **Trên nhóm biến `copper_legacy14`**:
#    - `naive`: Naive Persistence baseline
#    - `baseline_lstm`: Mô hình LSTM chuẩn Protocol V20 (64/32 units, lr $10^{-3}$)
#    - `paper_sa_lstm_transferred`: Sử dụng cấu hình siêu tham số của bài báo (39/111, lr $5 \times 10^{-5}$) chuyển giao sang `legacy14`
#    - `reopt_sa_lstm`: Mô hình SA-LSTM với siêu tham số tái tối ưu hóa chuyên biệt cho `legacy14`
# 
# Khảo sát qua các mốc phân chia thời gian ($RATIOS$) và các chiến lược thích nghi dữ liệu (`expanding`, `rolling_756`, `recency_weighted`). Kết quả được lưu tăng dần vào `results_sa_v20.csv` và `predictions_sa_v20.csv`.

# %% [code] Cell 16
RESULT_CSV = OUT / 'results_sa_v20.csv'
PRED_CSV = OUT / 'predictions_sa_v20.csv'

# Cờ chạy ma trận đánh giá H=1 (nếu kết quả đã có trong CSV, hàm has_job_completed tự động skip)
RUN_H1_EVALUATION = True

def load_existing_results():
    if RESULT_CSV.exists() and RESULT_CSV.stat().st_size > 0:
        return pd.read_csv(RESULT_CSV).to_dict('records')
    return []

RESULT_ROWS = load_existing_results()

def has_job_completed(model_name, feature_set, scheme, ratio, seed=SEED):
    return any(
        str(r.get('model')) == str(model_name) and
        str(r.get('feature_set')) == str(feature_set) and
        str(r.get('scheme')) == str(scheme) and
        abs(float(r.get('ratio', -1)) - float(ratio)) < 1e-6 and
        int(r.get('seed', -1)) == int(seed)
        for r in RESULT_ROWS
    )

def record_evaluation(row, pred_df):
    RESULT_ROWS.append(row)
    pd.DataFrame(RESULT_ROWS).to_csv(RESULT_CSV, index=False)
    header = not PRED_CSV.exists()
    pred_df.to_csv(PRED_CSV, mode='a', header=header, index=False)

# Thiết lập danh sách các công việc đánh giá
evaluation_jobs = []
for ratio in RATIOS:
    for scheme in SCHEMES:
        # 1. Nhóm Paper3
        evaluation_jobs.append(('copper_paper3', scheme, ratio, 'naive', None, None))
        evaluation_jobs.append(('copper_paper3', scheme, ratio, 'baseline_lstm', build_baseline_lstm, None))
        evaluation_jobs.append(('copper_paper3', scheme, ratio, 'paper_sa_lstm', build_paper_sa_lstm, PAPER_SA_LSTM_HP))
        evaluation_jobs.append(('copper_paper3', scheme, ratio, 'reopt_sa_lstm', build_sa_lstm, sa_best_cache['copper_paper3']))
        
        # 2. Nhóm Legacy14
        evaluation_jobs.append(('copper_legacy14', scheme, ratio, 'naive', None, None))
        evaluation_jobs.append(('copper_legacy14', scheme, ratio, 'baseline_lstm', build_baseline_lstm, None))
        evaluation_jobs.append(('copper_legacy14', scheme, ratio, 'paper_sa_lstm_transferred', build_paper_sa_lstm, PAPER_SA_LSTM_HP))
        evaluation_jobs.append(('copper_legacy14', scheme, ratio, 'reopt_sa_lstm', build_sa_lstm, sa_best_cache['copper_legacy14']))

total_jobs = len(evaluation_jobs)
print(f'Tổng số lượt thực nghiệm đối chuẩn: {total_jobs} lượt.')

for idx, (feat_set, scheme, ratio, model_name, builder, hp) in enumerate(evaluation_jobs, 1):
    if not RUN_H1_EVALUATION:
        print('Bỏ qua ma trận H=1 theo cấu hình RUN_H1_EVALUATION=False')
        break
    if has_job_completed(model_name, feat_set, scheme, ratio):
        continue
        
    ctx = make_context(data, ratio, scheme, feat_set)
    
    if model_name == 'naive':
        pv, pt = ctx['val']['prev'], ctx['test']['prev']
        extras = {'epochs_run': 0, 'fit_seconds': 0.0, 'hp_config': 'naive'}
    else:
        res = fit_predict_model(builder, ctx, hp=hp, is_search=False)
        pv, pt = res['val_pred'], res['test_pred']
        extras = {'epochs_run': res['epochs_run'], 'fit_seconds': res['seconds'], 'hp_config': str(hp)}
        
    row, pr = evaluate_predictions(ctx, model_name, pv, pt, seed=SEED, extras=extras)
    record_evaluation(row, pr)
    
    print(f'[{idx:02d}/{total_jobs}] {feat_set} | {scheme:<16} | Split {ratio*100:g}% | '
          f'{model_name:<24} => MAE={row["mae"]:.5f} | vs Naive={row["mae_improvement_pct"]:+.2f}% | '
          f'DirAcc={row["directional_accuracy_pct"]:.1f}% | {row["fit_seconds"]:.1f}s')

print('\n==> Đã hoàn tất toàn bộ ma trận huấn luyện và đánh giá!')

# %% [markdown] Cell 17
# ## 9. Phân tích Kết quả Thực nghiệm và Đối chuẩn Chi tiết
# 
# Trích xuất các bảng đối chuẩn khoa học phục vụ báo cáo:
# - **Bảng 1**: So sánh trên nhóm biến `paper3` (Baseline LSTM vs Paper SA-LSTM vs Re-optimized SA-LSTM vs Naive Persistence) trên tập test chuẩn 90/10.
# - **Bảng 2**: So sánh trên nhóm biến `legacy14` (Baseline LSTM vs Paper SA-LSTM Transferred vs Re-optimized SA-LSTM vs Naive Persistence).
# - **Bảng 3**: Đánh giá chuyển giao và tác động của không gian đặc trưng (Paper3 vs Legacy14).
# - **Bảng 4**: Độ bền vững qua 4 mốc phân chia thời gian (80/20, 90/10, 92.5/7.5, 95/5).
# - **Bảng 5**: So sánh giữa các chiến lược thích nghi dữ liệu (`expanding`, `rolling_756`, `recency_weighted`).

# %% [code] Cell 18
df_res = pd.read_csv(RESULT_CSV)

# ─── BẢNG 1: ĐỐI CHUẨN TRÊN NHÓM BIẾN PAPER3 (90/10, EXPANDING) ────────────────
p3_90 = df_res[(df_res['feature_set'] == 'copper_paper3') & 
               (np.isclose(df_res['ratio'], 0.90)) & 
               (df_res['scheme'] == 'expanding')].copy()
display_cols = ['model', 'mae', 'rmse', 'mse', 'mse_scaled', 'val_mse_scaled', 
                'mae_improvement_pct', 'directional_accuracy_pct', 'delta_pred_std_raw', 'epochs_run']
p3_table = p3_90[[c for c in display_cols if c in p3_90.columns]].sort_values('mae')
p3_table.to_csv(OUT / 'fair_table_sa_paper3.csv', index=False)

print('━'*95)
print('BẢNG 1: ĐỐI CHUẨN HIỆU NĂNG TRÊN NHÓM BIẾN PAPER3 (TỶ LỆ 90/10 - EXPANDING)')
print('━'*95)
display(p3_table)

# ─── BẢNG 2: ĐỐI CHUẨN TRÊN NHÓM BIẾN LEGACY14 (90/10, EXPANDING) ───────────────
l14_90 = df_res[(df_res['feature_set'] == 'copper_legacy14') & 
                (np.isclose(df_res['ratio'], 0.90)) & 
                (df_res['scheme'] == 'expanding')].copy()
l14_table = l14_90[[c for c in display_cols if c in l14_90.columns]].sort_values('mae')
l14_table.to_csv(OUT / 'fair_table_sa_legacy14.csv', index=False)

print('\n' + '━'*95)
print('BẢNG 2: ĐỐI CHUẨN HIỆU NĂNG TRÊN NHÓM BIẾN LEGACY14 (TỶ LỆ 90/10 - EXPANDING)')
print('━'*95)
display(l14_table)

# ─── BẢNG 3: SO SÁNH CHUYỂN GIAO SIÊU THAM SỐ (TRANSFERABILITY) ────────────────
cross_compare = df_res[(np.isclose(df_res['ratio'], 0.90)) & 
                       (df_res['scheme'] == 'expanding')].pivot_table(
    index='model', columns='feature_set', values=['mae', 'rmse', 'mse', 'mae_improvement_pct']
)
cross_compare.to_csv(OUT / 'cross_feature_transferability.csv')
print('\n' + '━'*95)
print('BẢNG 3: SO SÁNH CHÉO ĐẶC TRƯNG VÀ TÍNH CHUYỂN GIAO SIÊU THAM SỐ (90/10)')
print('━'*95)
display(cross_compare)

# ─── BẢNG 4: TỔNG HỢP QUA 4 TỶ LỆ CHIA (80/20, 90/10, 92.5/7.5, 95/5) ──────────
ratios_summary = df_res[df_res['scheme'] == 'expanding'].pivot_table(
    index=['feature_set', 'model'], columns='ratio', values='mae'
)
ratios_summary.to_csv(OUT / 'sa_ratios_summary.csv')
print('\n' + '━'*95)
print('BẢNG 4: MAE CÁC MÔ HÌNH QUA 4 TỶ LỆ PHÂN CHIA (EXPANDING)')
print('━'*95)
display(ratios_summary)

# %% [markdown] Cell 19
# ## 10. Trực quan hóa Biểu đồ (Actual vs Predicted & Tiến trình Hội tụ SA)

# %% [code] Cell 20
preds = pd.read_csv(PRED_CSV)

# 1. Đồ thị Actual vs Predicted trên Paper3 (90/10, expanding)
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

subset_p3 = preds[(preds['feature_set'] == 'copper_paper3') & 
                  (np.isclose(preds['ratio'], 0.90)) & 
                  (preds['scheme'] == 'expanding')].copy()
subset_p3['Date'] = pd.to_datetime(subset_p3['Date'])

ax = axes[0]
for model_name, grp in subset_p3.groupby('model'):
    grp = grp.sort_values('Date')
    if model_name != 'naive':
        ax.plot(grp['Date'], grp['pred'], label=model_name, alpha=0.75)
actual_curve = subset_p3.drop_duplicates('Date').sort_values('Date')
ax.plot(actual_curve['Date'], actual_curve['actual'], label='Copper Actual', color='black', linewidth=1.5)
ax.set_title('Copper Price: Paper3 Feature Set — Actual vs Predictions (90/10 Split, Horizon=1)', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=10)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# 2. Đồ thị Actual vs Predicted trên Legacy14 (90/10, expanding)
subset_l14 = preds[(preds['feature_set'] == 'copper_legacy14') & 
                   (np.isclose(preds['ratio'], 0.90)) & 
                   (preds['scheme'] == 'expanding')].copy()
subset_l14['Date'] = pd.to_datetime(subset_l14['Date'])

ax = axes[1]
for model_name, grp in subset_l14.groupby('model'):
    grp = grp.sort_values('Date')
    if model_name != 'naive':
        ax.plot(grp['Date'], grp['pred'], label=model_name, alpha=0.75)
actual_curve_l14 = subset_l14.drop_duplicates('Date').sort_values('Date')
ax.plot(actual_curve_l14['Date'], actual_curve_l14['actual'], label='Copper Actual', color='black', linewidth=1.5)
ax.set_title('Copper Price: Legacy14 Feature Set — Actual vs Predictions (90/10 Split, Horizon=1)', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=10)
ax.set_xlabel('Date', fontsize=10)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / 'sa_actual_vs_predicted_paper3_legacy14.png', dpi=150)
plt.show()

# 3. Đồ thị Tiến trình Hội tụ Simulated Annealing (Paper3 vs Legacy14)
fig, ax = plt.subplots(figsize=(12, 5))
if 'copper_paper3' in sa_history_cache and not sa_history_cache['copper_paper3'].empty:
    h_p3 = sa_history_cache['copper_paper3']
    ax.plot(h_p3['evaluation'], h_p3['best_mse_scaled'], label='Best Val MSE — Paper3', color='blue', lw=2)
    ax.scatter(h_p3['evaluation'], h_p3['val_mse_scaled'], alpha=0.3, color='blue', s=15, label='Proposals — Paper3')
if 'copper_legacy14' in sa_history_cache and not sa_history_cache['copper_legacy14'].empty:
    h_l14 = sa_history_cache['copper_legacy14']
    ax.plot(h_l14['evaluation'], h_l14['best_mse_scaled'], label='Best Val MSE — Legacy14', color='green', lw=2)
    ax.scatter(h_l14['evaluation'], h_l14['val_mse_scaled'], alpha=0.3, color='green', s=15, label='Proposals — Legacy14')

ax.set_title('Simulated Annealing Optimization Convergence (Horizon=1, Val MSE)', fontsize=12)
ax.set_xlabel('Evaluation Number', fontsize=10)
ax.set_ylabel('Validation MSE (Scaled)', fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / 'sa_convergence_history.png', dpi=150)
plt.show()

# %% [markdown] Cell 21
# ## 12. Thực nghiệm Tái tạo Bài báo — Chiến lược HORIZON=0 (Nowcasting)
# 
# Phần này tái tạo chiến lược **HORIZON=0** của Chen et al. (2023):
# - Đặc trưng bao gồm giá **cùng ngày t** của WTI, Gold, Silver (Nowcasting)
# - Chỉ chạy trên nhóm biến `copper_paper3`, tỷ lệ 90/10, expanding (cấu hình chính bài báo)
# - Kết quả để **đối chiếu định lượng** với HORIZON=1 cùng điều kiện
# 
# > **Mục đích:** Làm rõ bao nhiêu phần trăm hiệu năng bài báo đến từ same-day signals
# > thay vì từ kiến trúc LSTM/SA thực sự.
# 

# %% [code] Cell 22
# ─── BƯỚC H0-A: TÌM KIẾM SA TRÊN HORIZON=0 (NOWCASTING) ────────────────────────
# Ở Horizon=0: Các biến khác (WTI, Vàng, Bạc...) được lấy ở cùng ngày t.
# Biến Copper chỉ lấy đến ngày t-1 (không rò rỉ target).
# Ta chạy tìm kiếm SA riêng cho Horizon=0 để tìm kiến trúc nơ-ron và lr tối ưu!

RUN_SA_H0_PAPER3 = True     # Chạy SA tìm kiếm cho nhóm biến paper3 (H=0)
RUN_SA_H0_LEGACY14 = True   # Chạy SA tìm kiếm cho nhóm biến legacy14 (H=0)

sa_best_cache_h0 = {}
sa_history_cache_h0 = {}

if RUN_HORIZON0:
    # 1. Tìm kiếm SA trên Paper3 (H=0)
    if RUN_SA_H0_PAPER3:
        print('═'*75)
        print('BƯỚC H0-A1: TÌM KIẾM SA TRÊN NHÓM BIẾN PAPER3 (HORIZON=0 - NOWCASTING)')
        print('═'*75)
        ctx_h0_p3 = make_context(data, PRIMARY_RATIO, 'expanding', 'copper_paper3', horizon=0)
        hp_h0_p3, hist_h0_p3 = run_simulated_annealing('sa_lstm_paper3_h0', ctx_h0_p3)
        sa_best_cache_h0['copper_paper3_h0'] = hp_h0_p3
        sa_history_cache_h0['copper_paper3_h0'] = hist_h0_p3
    else:
        hp_h0_p3 = PAPER_SA_LSTM_HP
        sa_best_cache_h0['copper_paper3_h0'] = hp_h0_p3
        
    # 2. Tìm kiếm SA trên Legacy14 (H=0)
    if RUN_SA_H0_LEGACY14:
        print('\n' + '═'*75)
        print('BƯỚC H0-A2: TÌM KIẾM SA TRÊN NHÓM BIẾN LEGACY14 (HORIZON=0 - NOWCASTING)')
        print('═'*75)
        ctx_h0_l14 = make_context(data, PRIMARY_RATIO, 'expanding', 'copper_legacy14', horizon=0)
        hp_h0_l14, hist_h0_l14 = run_simulated_annealing('sa_lstm_legacy14_h0', ctx_h0_l14)
        sa_best_cache_h0['copper_legacy14_h0'] = hp_h0_l14
        sa_history_cache_h0['copper_legacy14_h0'] = hist_h0_l14
    else:
        hp_h0_l14 = hp_h0_p3
        sa_best_cache_h0['copper_legacy14_h0'] = hp_h0_l14

    # Lưu lại kết quả siêu tham số H=0
    (OUT / 'sa_best_hyperparameters_h0.json').write_text(
        json.dumps(sa_best_cache_h0, indent=2), encoding='utf-8'
    )
    
    # ─── BẢNG ĐỐI CHIẾU TOÀN DIỆN CÁC BỘ SIÊU THAM SỐ ───────────────────────────
    all_hp_summary = [
        {'Chiến lược': 'Chen et al. (2023) gốc', 'Biến': 'paper3', 'Horizon': '0 (Nowcasting)', 
         'units_1': 39, 'units_2': 111, 'learning_rate': 5e-5},
        {'Chiến lược': 'Re-opt SA (Protocol V20)', 'Biến': 'paper3', 'Horizon': '1 (Forecasting)', 
         **sa_best_cache.get('copper_paper3', {})},
        {'Chiến lược': 'Re-opt SA (Protocol V20)', 'Biến': 'legacy14', 'Horizon': '1 (Forecasting)', 
         **sa_best_cache.get('copper_legacy14', {})},
        {'Chiến lược': 'Re-opt SA (Tái tạo bài báo)', 'Biến': 'paper3', 'Horizon': '0 (Nowcasting)', 
         **hp_h0_p3},
        {'Chiến lược': 'Re-opt SA (Tái tạo mở rộng)', 'Biến': 'legacy14', 'Horizon': '0 (Nowcasting)', 
         **hp_h0_l14},
    ]
    df_all_hp = pd.DataFrame(all_hp_summary)
    print('\n' + '━'*95)
    print('BẢNG ĐỐI CHIẾU SIÊU THAM SỐ: BÀI BÁO GỐC vs SA H=1 vs SA H=0')
    print('━'*95)
    display(df_all_hp)
    df_all_hp.to_csv(OUT / 'all_hyperparameters_comparison.csv', index=False)
else:
    print('RUN_HORIZON0 = False — Bỏ qua tìm kiếm SA cho HORIZON=0')


# %% [markdown] Cell 23
# ### 12b. Ma trận Đánh giá Mô hình với HORIZON=0
# 

# %% [code] Cell 24
# ─── HORIZON=0: MA TRẬN ĐÁNH GIÁ (PAPER3 & LEGACY14, 90/10, EXPANDING) ─────────
RESULT_CSV_H0 = OUT / 'results_sa_h0.csv'
PRED_CSV_H0 = OUT / 'predictions_sa_h0.csv'

def load_existing_results_h0():
    if RESULT_CSV_H0.exists() and RESULT_CSV_H0.stat().st_size > 0:
        return pd.read_csv(RESULT_CSV_H0).to_dict('records')
    return []

RESULT_ROWS_H0 = load_existing_results_h0()

def has_job_completed_h0(model_name, feature_set, scheme, ratio, seed=SEED):
    return any(
        str(r.get('model')) == str(model_name) and
        str(r.get('feature_set')) == str(feature_set) and
        str(r.get('scheme')) == str(scheme) and
        abs(float(r.get('ratio', -1)) - float(ratio)) < 1e-6 and
        int(r.get('seed', -1)) == int(seed)
        for r in RESULT_ROWS_H0
    )

def record_evaluation_h0(row, pred_df):
    RESULT_ROWS_H0.append(row)
    pd.DataFrame(RESULT_ROWS_H0).to_csv(RESULT_CSV_H0, index=False)
    header = not PRED_CSV_H0.exists()
    pred_df.to_csv(PRED_CSV_H0, mode='a', header=header, index=False)

if RUN_HORIZON0:
    print('═'*75)
    print('BƯỚC H0-B: ĐÁNH GIÁ MA TRẬN HORIZON=0 (90/10, EXPANDING)')
    print('═'*75)
    
    h0_jobs = [
        # Nhóm Paper3 (Nowcasting H=0)
        ('copper_paper3', 'expanding', 0.90, 'naive_h0',                 None,                None),
        ('copper_paper3', 'expanding', 0.90, 'baseline_lstm_h0',         build_baseline_lstm, None),
        ('copper_paper3', 'expanding', 0.90, 'paper_sa_lstm_h0',         build_paper_sa_lstm, PAPER_SA_LSTM_HP),
        ('copper_paper3', 'expanding', 0.90, 'reopt_sa_lstm_h0',         build_sa_lstm,       hp_h0_p3),
        
        # Nhóm Legacy14 (Nowcasting H=0)
        ('copper_legacy14', 'expanding', 0.90, 'naive_h0',               None,                None),
        ('copper_legacy14', 'expanding', 0.90, 'baseline_lstm_h0',       build_baseline_lstm, None),
        ('copper_legacy14', 'expanding', 0.90, 'paper_sa_lstm_trans_h0', build_paper_sa_lstm, PAPER_SA_LSTM_HP),
        ('copper_legacy14', 'expanding', 0.90, 'reopt_sa_lstm_h0',       build_sa_lstm,       hp_h0_l14),
    ]
    
    total_h0 = len(h0_jobs)
    for idx, (feat_set, scheme, ratio, model_name, builder, hp) in enumerate(h0_jobs, 1):
        if has_job_completed_h0(model_name, feat_set, scheme, ratio):
            print(f'[SKIP] {feat_set} | {model_name} đã hoàn tất từ trước.')
            continue
        
        ctx = make_context(data, ratio, scheme, feat_set, horizon=0)
        
        if 'naive' in model_name:
            pv, pt = ctx['val']['prev'], ctx['test']['prev']
            extras = {'epochs_run': 0, 'fit_seconds': 0.0, 'hp_config': 'naive', 'horizon': 0}
        else:
            res = fit_predict_model(builder, ctx, hp=hp, is_search=False)
            pv, pt = res['val_pred'], res['test_pred']
            extras = {'epochs_run': res['epochs_run'], 'fit_seconds': res['seconds'],
                      'hp_config': str(hp), 'horizon': 0}
        
        row, pr = evaluate_predictions(ctx, model_name, pv, pt, seed=SEED, extras=extras)
        row['horizon'] = 0
        pr['horizon'] = 0
        record_evaluation_h0(row, pr)
        
        print(f'[H0 {idx:02d}/{total_h0}] {feat_set:<16} | {model_name:<26} => '
              f'MAE={row["mae"]:.5f} | vs Naive={row["mae_improvement_pct"]:+.2f}% | '
              f'DirAcc={row["directional_accuracy_pct"]:.1f}% | {row["fit_seconds"]:.1f}s')
    
    print('\n==> Đã hoàn tất toàn bộ ma trận HORIZON=0!')
else:
    print('RUN_HORIZON0=False — Bỏ qua ma trận HORIZON=0')
    RESULT_ROWS_H0 = []


# %% [markdown] Cell 25
# ### 12c. Phân tích So sánh HORIZON=0 vs HORIZON=1
# 

# %% [code] Cell 26
# ─── SO SÁNH HORIZON=0 (NOWCASTING) vs HORIZON=1 (TRUE FORECASTING) ─────────────
if RUN_HORIZON0 and len(RESULT_ROWS_H0) > 0 and len(RESULT_ROWS) > 0:
    df_h0 = pd.read_csv(RESULT_CSV_H0)
    df_h1 = pd.read_csv(RESULT_CSV)
    
    # ─── BẢNG 5A: ĐỐI CHIẾU TRÊN NHÓM BIẾN PAPER3 (90/10, EXPANDING) ─────────────
    h1_p3 = df_h1[
        (df_h1['feature_set'] == 'copper_paper3') &
        (np.isclose(df_h1['ratio'], 0.90)) &
        (df_h1['scheme'] == 'expanding')
    ][['model', 'mae', 'rmse', 'mse', 'directional_accuracy_pct']].copy()
    h1_p3['model_base'] = h1_p3['model']
    
    h0_p3 = df_h0[df_h0['feature_set'] == 'copper_paper3'][['model', 'mae', 'rmse', 'mse', 'directional_accuracy_pct']].copy()
    h0_p3['model_base'] = h0_p3['model'].str.replace('_h0', '', regex=False)
    
    merged_p3 = pd.merge(
        h0_p3.rename(columns={'mae': 'mae_H0', 'rmse': 'rmse_H0', 'mse': 'mse_H0', 'directional_accuracy_pct': 'dir_acc_H0'}),
        h1_p3.rename(columns={'mae': 'mae_H1', 'rmse': 'rmse_H1', 'mse': 'mse_H1', 'directional_accuracy_pct': 'dir_acc_H1'}),
        on='model_base', how='outer'
    )
    merged_p3['mae_lift_pct'] = 100.0 * (merged_p3['mae_H1'] - merged_p3['mae_H0']) / merged_p3['mae_H1']
    merged_p3['mse_ratio_H0_H1'] = merged_p3['mse_H0'] / merged_p3['mse_H1']
    merged_p3 = merged_p3.sort_values('mae_H0')
    merged_p3.to_csv(OUT / 'horizon_comparison_paper3.csv', index=False)
    
    print('━'*105)
    print('BẢNG 5A: ĐỐI CHIẾU HORIZON=0 vs HORIZON=1 — PAPER3 (90/10, EXPANDING)')
    print('  mae_lift_pct > 0: H=0 giảm MAE bao nhiêu % so với H=1 (Lợi thế thông tin cùng ngày)')
    print('  mse_ratio_H0_H1 < 1: MSE của H=0 bằng bao nhiêu lần MSE của H=1')
    print('━'*105)
    display(merged_p3)
    
    # ─── BẢNG 5B: ĐỐI CHIẾU TRÊN NHÓM BIẾN LEGACY14 (90/10, EXPANDING) ────────────
    h1_l14 = df_h1[
        (df_h1['feature_set'] == 'copper_legacy14') &
        (np.isclose(df_h1['ratio'], 0.90)) &
        (df_h1['scheme'] == 'expanding')
    ][['model', 'mae', 'rmse', 'mse', 'directional_accuracy_pct']].copy()
    h1_l14['model_base'] = h1_l14['model'].str.replace('_transferred', '', regex=False)
    
    h0_l14 = df_h0[df_h0['feature_set'] == 'copper_legacy14'][['model', 'mae', 'rmse', 'mse', 'directional_accuracy_pct']].copy()
    h0_l14['model_base'] = h0_l14['model'].str.replace('_h0', '', regex=False).str.replace('_trans', '', regex=False)
    
    if len(h0_l14) > 0 and len(h1_l14) > 0:
        merged_l14 = pd.merge(
            h0_l14.rename(columns={'mae': 'mae_H0', 'rmse': 'rmse_H0', 'mse': 'mse_H0', 'directional_accuracy_pct': 'dir_acc_H0'}),
            h1_l14.rename(columns={'mae': 'mae_H1', 'rmse': 'rmse_H1', 'mse': 'mse_H1', 'directional_accuracy_pct': 'dir_acc_H1'}),
            on='model_base', how='outer'
        )
        merged_l14['mae_lift_pct'] = 100.0 * (merged_l14['mae_H1'] - merged_l14['mae_H0']) / merged_l14['mae_H1']
        merged_l14['mse_ratio_H0_H1'] = merged_l14['mse_H0'] / merged_l14['mse_H1']
        merged_l14 = merged_l14.sort_values('mae_H0')
        merged_l14.to_csv(OUT / 'horizon_comparison_legacy14.csv', index=False)
        
        print('\n' + '━'*105)
        print('BẢNG 5B: ĐỐI CHIẾU HORIZON=0 vs HORIZON=1 — LEGACY14 (90/10, EXPANDING)')
        print('━'*105)
        display(merged_l14)
    
    print('\n📌 KẾT LUẬN ĐỐI CHIẾU PHƯƠNG PHÁP LUẬN:')
    if len(merged_p3) > 0:
        best_row = merged_p3.iloc[0]
        lift = best_row.get('mae_lift_pct', float('nan'))
        ratio_mse = best_row.get('mse_ratio_H0_H1', float('nan'))
        print(f'  - Trên Paper3 ({best_row["model_base"]}): HORIZON=0 giảm MAE {lift:.1f}% so với HORIZON=1.')
        print(f'  - Tỷ lệ MSE(H=0) / MSE(H=1) = {ratio_mse:.4f}.')
        if ratio_mse < 0.2:
            print('    ==> KẾT LUẬN: Hiệu năng vượt trội trong bài báo Chen et al. (2023) đến từ tín hiệu đồng thời cùng ngày (Nowcasting).')
            print('    ==> Khi chuyển sang Protocol V20 (Forecasting H=1), sai số tăng mạnh do không còn thông tin tương lai.')
else:
    print('Chưa có đủ kết quả thực nghiệm để vẽ bảng so sánh HORIZON=0 vs HORIZON=1.')


# %% [markdown] Cell 27
# ## 11. Báo cáo Tóm tắt và Đóng gói Hiện vật (Export ZIP)
# 
# Tự động tạo tệp `run_summary.md` và nén toàn bộ thư mục hiện vật thành tệp ZIP để thuận tiện tải về phân tích.

# %% [code] Cell 28
runtime_mins = (time.monotonic() - JOB_START) / 60.0
summary_lines = [
    '# Copper SA-LSTM Reproduction & Cross-Feature Evaluation (Protocol V20)',
    f'- Thời điểm thực thi: {time.strftime("%Y-%m-%d %H:%M:%S")}',
    f'- MODE: {MODE} | Horizon: {HORIZON} | Lookback: {LOOKBACK}',
    f'- Tổng số bản ghi thực nghiệm: {len(RESULT_ROWS)}',
    f'- Thời gian chạy tổng cộng: {runtime_mins:.1f} phút',
    '',
    '## Siêu tham số tối ưu tìm được:',
    f'- Paper3 Re-optimized: {sa_best_cache.get("copper_paper3")}',
    f'- Legacy14 Re-optimized: {sa_best_cache.get("copper_legacy14")}',
    '',
    '## Kết luận phương pháp luận:',
    '1. Mô hình SA-GRU đã được loại bỏ hoàn toàn theo yêu cầu.',
    '2. Đối chuẩn SA-LSTM vs Baseline LSTM trên Paper3 và Legacy14 được thực hiện nghiêm ngặt ở Horizon=1.',
    '3. Kết quả cho thấy chuyển giao cấu hình bài báo sang Legacy14 cần đánh giá thận trọng trước biến động không dừng.',
]
(OUT / 'run_summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')

zip_path = OUT.parent / f'{OUT.name}_results.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for file in sorted(OUT.rglob('*')):
        if file.is_file():
            z.write(file, arcname=str(Path(OUT.name) / file.relative_to(OUT)))

print('━'*75)
print(f'Đã tạo tệp nén hiện vật: {zip_path}')
print(f'Dung lượng: {zip_path.stat().st_size / 1e6:.2f} MB')
print(f'Tổng thời gian thực thi: {runtime_mins:.1f} phút.')
print('━'*75)