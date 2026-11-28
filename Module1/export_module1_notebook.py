from pathlib import Path
import subprocess
import sys


module_dir = Path(__file__).resolve().parent
output_dir = module_dir / "html"
notebooks = [
    module_dir / "offical-notebooks" / "01_EDA_va_Kiem_Dinh_Thong_Ke.ipynb",
    module_dir / "offical-notebooks" / "04_Evaluate_va_Tong_Hop_Ket_Qua.ipynb",
]

missing = [notebook for notebook in notebooks if not notebook.is_file()]
if missing:
    raise SystemExit(f"Notebook not found: {missing[0]}")

output_dir.mkdir(exist_ok=True)
for notebook in notebooks:
    output = output_dir / notebook.with_suffix(".html").name
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
