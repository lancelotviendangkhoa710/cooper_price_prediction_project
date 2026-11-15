# Dữ liệu hàng hóa Investing.com

Tải dữ liệu **Daily** từ `01/01/2016` đến ngày mới nhất năm 2026 và đặt 9 file CSV vào thư mục này.

Tên file đề nghị:

- `01_copper.csv`
- `02_aluminium.csv`
- `03_nickel.csv`
- `04_tin.csv`
- `05_iron_ore_62.csv`
- `06_silver.csv`
- `07_gold.csv`
- `08_wti.csv`
- `09_brent.csv`

Notebook xử lý nằm tại `lstm/13-11-2026_xu_ly_9_hang_hoa_investing.ipynb`. Notebook cũng nhận diện được tên file mặc định do Investing.com tạo nếu tên có chứa tên hàng hóa.

Output chính sau khi chạy:

- `13-11-2026_dataset_9_commodities_daily.csv`
- `13-11-2026_correlation_summary.csv`
- `13-11-2026_data_coverage.csv`

