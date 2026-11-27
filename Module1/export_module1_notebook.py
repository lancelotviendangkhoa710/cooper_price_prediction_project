from pathlib import Path
import subprocess
import sys


module_dir = Path(__file__).resolve().parent
notebook = module_dir / "offical-notebooks" / "03_Train_NonSA_Baselines_va_WalkForward.ipynb"
output = notebook.with_suffix(".html")

if not notebook.is_file():
    raise SystemExit(f"Notebook not found: {notebook}")

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
        str(output.parent),
        str(notebook),
    ],
    check=True,
)

if not output.is_file() or output.stat().st_size == 0:
    raise SystemExit(f"HTML export failed: {output}")

print(f"Exported: {output}")
