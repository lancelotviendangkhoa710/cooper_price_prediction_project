import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = research.load_config()
        n = 150
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        values = np.arange(n, dtype=float)
        self.raw = pd.DataFrame({"Copper": values, **{c: values + i for i, c in enumerate(
            self.cfg["features"]["candidates"])}}, index=dates)

    def test_chronological_boundaries(self):
        val, test = research.boundaries(150, .9, .1)
        self.assertEqual((val, test), (121, 135))
        c = research.context(self.raw, self.cfg, .9, 7, "paper")
        self.assertLess(c["train"][2].max(), c["val"][2].min())
        self.assertNotIn("test", c)
        self.assertEqual(c["test_range"][0], str(self.raw.index[135].date()))

    def test_scaler_and_spearman_never_see_future(self):
        a = research.context(self.raw, self.cfg, .9, 7, "spearman")
        changed = self.raw.copy()
        changed.iloc[121:, changed.columns.get_loc("Copper")] = 1_000_000
        for col in self.cfg["features"]["candidates"]:
            changed.iloc[121:, changed.columns.get_loc(col)] = -1_000_000
        b = research.context(changed, self.cfg, .9, 7, "spearman")
        np.testing.assert_allclose(a["x_scaler"].data_max_, b["x_scaler"].data_max_)
        np.testing.assert_allclose(a["y_scaler"].data_max_, b["y_scaler"].data_max_)
        self.assertEqual(a["features"], b["features"])
        self.assertTrue((b["val"][1] > 1).all())

    def test_search_context_does_not_transform_test_labels(self):
        future_hidden = self.raw.copy()
        future_hidden.iloc[135:, future_hidden.columns.get_loc("Copper")] = np.nan
        c = research.context(future_hidden, self.cfg, .9, 7, "paper")
        self.assertNotIn("test", c)
        self.assertEqual(len(c["previous_targets"]), 135)
        self.assertTrue(np.isfinite(c["val"][1]).all())

    def test_window_is_previous_observations_only(self):
        x = np.arange(20, dtype=float).reshape(-1, 1)
        y = np.arange(20, dtype=float)
        dates = pd.date_range("2020-01-01", periods=20)
        X, Y, d = research.make_windows(x, y, dates, 7, 10, 3)
        np.testing.assert_array_equal(X[0, :, 0], [4, 5, 6])
        self.assertEqual(Y[0], 7)
        self.assertEqual(d[0], dates[7])
        self.assertEqual(X.shape, (3, 3, 1))

    def test_persistence_uses_previous_copper(self):
        c = research.context(self.raw, self.cfg, .9, 7, "paper")
        m, prediction = research.persistence(c)
        self.assertEqual(len(prediction), len(c["val"][1]))
        np.testing.assert_allclose(prediction.predicted, np.arange(120, 134))
        self.assertAlmostEqual(m["mse_price2"], 1.0)

    @unittest.skipUnless(__import__("importlib").util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_prediction_shape(self):
        c = research.context(self.raw, self.cfg, .9, 7, "paper")
        hp = {"units1": 8, "units2": 8, "lr": .001}
        m = research.train_score(c, hp, self.cfg, epochs=1)
        self.assertTrue(np.isfinite(m["mse_scaled"]))


if __name__ == "__main__":
    unittest.main()
