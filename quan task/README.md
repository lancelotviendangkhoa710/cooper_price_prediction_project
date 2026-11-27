# Phương pháp 4 — Rolling & Expanding Window Walk-Forward Forecasting

Thư mục lưu trữ toàn bộ mã nguồn, kết quả thực nghiệm, biểu đồ và kiểm thử cho bài toán dự báo giá đồng theo giao thức **Walk-Forward (`Expanding` & `Rolling Window`)** đối chứng **Static (`Train-Only`, `Train+Val`)** và **Naive Persistence** trên đủ **6 bộ biến (`FS01–FS06`)**.

## Cấu trúc thư mục

```text
quan task/
├── copper_rolling_expanding_quan.ipynb   # Notebook chính (44 cells, đầy đủ bảng số liệu & 41 biểu đồ nhúng sẵn)
├── notebooks/
│   └── phase2d_80_20_kaggle_gpu.ipynb    # Notebook chạy 60 lượt 6 FS × 5 seeds (80/20) trên Kaggle GPU T4
├── result/
│   ├── rolling_expanding_summary_report.md
│   ├── metrics/                          # 14 tệp CSV bảng chỉ số (Phase 1 -> Phase 5, Ablation, Regime, DM, 5-seed)
│   ├── predictions/                      # 51 tệp CSV dự báo từng phiên (49 pred_*.csv + 90/10 & 80/20 5-seed full)
│   └── figures/                          # 17 biểu đồ tổng hợp (fig1..fig15, fig14b, fig14c)
│       └── per_feature_set/              # 24 biểu đồ chi tiết theo từng bộ biến FS01..FS06 (4 ảnh/bộ)
├── tests/
│   └── test_rolling_expanding.py         # Unit test cấu trúc notebook, chống leakage & tính toàn vẹn dữ liệu
└── README.md
```

## Chạy kiểm thử

```powershell
python "quan task/tests/test_rolling_expanding.py"
```
