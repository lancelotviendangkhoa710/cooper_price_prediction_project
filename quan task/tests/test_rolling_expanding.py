"""Protocol and integrity tests for Quan's Rolling/Expanding Window notebook."""
import ast
import csv
import datetime as dt
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "quan task" / "copper_rolling_expanding_quan.ipynb"
CSV_PATH = ROOT / "copper_8_indicators_2006_2026.csv"


def compute_boundaries(n: int, ratio: float = 0.90, val_fraction: float = 0.10) -> tuple[int, int]:
    test_start = int(n * ratio)
    val_start = int(test_start * (1.0 - val_fraction))
    if not (0 < val_start < test_start < n):
        raise ValueError("Invalid chronological boundaries")
    return val_start, test_start


def walk_forward_slices(
    total_len: int,
    eval_start: int,
    eval_end: int,
    strategy: str,
    window_size: int,
    stride: int,
    lookback: int = 22,
) -> list[dict]:
    """Pure-Python replica of the notebook's walk-forward index slicing logic."""
    slices = []
    for origin in range(eval_start, eval_end, stride):
        block_end = min(origin + stride, eval_end)
        fit_start = 0 if strategy == "expanding_window" else max(0, origin - window_size)
        train_targets = list(range(fit_start + lookback, origin))
        eval_targets = list(range(origin, block_end))
        slices.append(
            {
                "fit_start": fit_start,
                "origin": origin,
                "block_end": block_end,
                "train_targets": train_targets,
                "eval_targets": eval_targets,
                "scaler_range": (fit_start, origin),
            }
        )
    return slices


class RollingExpandingNotebookTests(unittest.TestCase):
    def test_notebook_schema_and_python_syntax(self):
        self.assertTrue(NOTEBOOK_PATH.is_file(), f"Missing notebook: {NOTEBOOK_PATH}")
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        cells = nb.get("cells", [])
        self.assertGreaterEqual(len(cells), 14)

        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        self.assertGreaterEqual(len(code_cells), 10)
        self.assertGreaterEqual(len(md_cells), 10)

        for idx, cell in enumerate(code_cells):
            src = "".join(cell.get("source", []))
            self.assertTrue(src.strip(), f"Code cell {idx} is empty")
            # Every code cell must parse cleanly as valid Python syntax
            ast.parse(src, filename=f"cell_{idx}")

    def test_notebook_contains_required_components(self):
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        full_code = "\n".join(
            "".join(c.get("source", []))
            for c in nb.get("cells", [])
            if c.get("cell_type") == "code"
        )
        for required_symbol in [
            "make_walk_forward_fold",
            "run_forecasting_strategy",
            "diebold_mariano_test",
            "compute_metrics",
            "static_train_only",
            "static_train_val",
            "expanding_window",
            "rolling_window",
            "naive_persistence",
            "FEATURES_4",
            "FEATURES_3",
            "MASE",
        ]:
            self.assertIn(required_symbol, full_code, f"Missing symbol: {required_symbol}")

    def test_walk_forward_no_leakage_invariants(self):
        n = 5231
        val_start, test_start = compute_boundaries(n, ratio=0.90, val_fraction=0.10)
        self.assertEqual((val_start, test_start), (4236, 4707))

        # Test Rolling Window invariants
        rolling_slices = walk_forward_slices(
            n, test_start, n, strategy="rolling_window", window_size=504, stride=20, lookback=22
        )
        self.assertGreater(len(rolling_slices), 1)
        covered_eval = []
        for s in rolling_slices:
            self.assertEqual(s["origin"] - s["fit_start"], 504)
            self.assertLess(max(s["train_targets"]), min(s["eval_targets"]))
            # Scaler fit range strictly precedes evaluation block
            self.assertEqual(s["scaler_range"][1], s["origin"])
            self.assertLess(s["scaler_range"][1], s["block_end"])
            # Last training input observation is at most origin - 1
            last_train_target = max(s["train_targets"])
            self.assertEqual(last_train_target, s["origin"] - 1)
            covered_eval.extend(s["eval_targets"])
        self.assertEqual(covered_eval, list(range(test_start, n)))

        # Test Expanding Window growth invariants
        expanding_slices = walk_forward_slices(
            n, test_start, n, strategy="expanding_window", window_size=504, stride=20, lookback=22
        )
        train_sizes = [len(s["train_targets"]) for s in expanding_slices]
        self.assertTrue(all(b > a for a, b in zip(train_sizes, train_sizes[1:])))
        self.assertTrue(all(s["fit_start"] == 0 for s in expanding_slices))

    def test_csv_dataset_integrity(self):
        self.assertTrue(CSV_PATH.is_file(), f"Missing dataset: {CSV_PATH}")
        expected_cols = {
            "Date",
            "Copper",
            "Copper_Volume",
            "Dow_Jones",
            "Nasdaq",
            "Dollar_Index",
            "TBill_10Y",
            "Gold",
            "Silver",
            "WTI",
        }
        with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            self.assertTrue(expected_cols.issubset(set(reader.fieldnames or [])))
            prev_date = None
            row_count = 0
            for row in reader:
                d = dt.date.fromisoformat(row["Date"])
                if prev_date is not None:
                    self.assertGreater(d, prev_date, "Dates are not strictly increasing")
                prev_date = d
                for col in expected_cols - {"Date"}:
                    val = float(row[col])
                    self.assertEqual(val, val, f"NaN in {col} at {row['Date']}")
                row_count += 1
        self.assertEqual(row_count, 5231)


if __name__ == "__main__":
    unittest.main(verbosity=2)
