from pathlib import Path
import subprocess
import sys


module_dir = Path(__file__).resolve().parent
notebook = module_dir / "offical-notebooks" / "01_EDA_va_Kiem_Dinh_Thong_Ke.ipynb"
output_dir = module_dir / "html"
output = output_dir / notebook.with_suffix(".html").name

if not notebook.is_file():
    raise SystemExit(f"Notebook not found: {notebook}")

output_dir.mkdir(exist_ok=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "html",
        "--embed-images",
        "--output",
        output.name,
        "--output-dir",
        str(output_dir),
        str(notebook),
    ],
    check=True,
)

if not output.is_file() or output.stat().st_size == 0:
    raise SystemExit(f"HTML export failed: {output}")

print(f"Exported: {output}")
