"""Reproducible copper next-session forecasting experiments; run from Kaggle notebook.

Search and screening only see train/validation. Test arrays are created solely by final().
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent
TICKERS = {"Copper": ["HG=F"], "Dow_Jones": ["^DJI"], "Nasdaq": ["^IXIC"],
           "Dollar_Index": ["DX-Y.NYB", "DX=F"], "TBill_10Y": ["^TNX"],
           "Gold": ["GC=F"], "Silver": ["SI=F"], "WTI": ["CL=F"]}


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_config(mode="QUICK_TEST"):
    cfg = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
    if mode not in cfg["modes"]:
        raise ValueError(mode)
    cfg["mode"] = mode
    cfg["budget"] = cfg["modes"][mode]
    return cfg


def output_dir(cfg):
    p = Path(cfg["output_dir"])
    if not p.parent.exists():
        p = ROOT / "output"
    p = p / cfg["mode"].lower()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download_one(ticker, start, end, dest):
    # Preserve the subprocess workaround that worked in the saved Kaggle notebook.
    code = """import sys, yfinance as yf
d=yf.download(sys.argv[1],start=sys.argv[2],end=sys.argv[3],auto_adjust=False,progress=False,threads=False)
if d.empty: raise RuntimeError('empty Yahoo response')
d.to_pickle(sys.argv[4])
"""
    subprocess.run([sys.executable, "-c", code, ticker, start, end, str(dest)],
                   check=True, timeout=180)
    frame = pd.read_pickle(dest)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    duplicate_dates = int(frame.index.duplicated().sum())
    frame = frame.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
    frame.attrs["duplicate_dates"] = duplicate_dates
    return frame


def load_data(cfg, out):
    source = cfg["data"].get("csv")
    snapshot = out / "copper_data_snapshot.csv"
    using_snapshot = False
    source_duplicate_count = 0
    union_dates = None
    if not source and snapshot.exists():
        source = str(snapshot)
        using_snapshot = True
    if source:
        raw = pd.read_csv(source, parse_dates=["Date"], index_col="Date")
        mapping = {"source": str(source)}
    else:
        import tempfile
        series, mapping = {}, {}
        with tempfile.TemporaryDirectory() as td:
            for name, tickers in TICKERS.items():
                errors = []
                for ticker in tickers:
                    try:
                        frame = _download_one(ticker, cfg["data"]["start"],
                                              cfg["data"]["end_exclusive"], Path(td) / "data.pkl")
                        if frame["Close"].notna().sum() < 1000:
                            raise ValueError("insufficient history")
                        source_duplicate_count += frame.attrs.get("duplicate_dates", 0)
                        series[name] = pd.to_numeric(frame["Close"], errors="coerce").rename(name)
                        if name == "Copper":
                            series["Copper_Volume"] = pd.to_numeric(frame["Volume"], errors="coerce").rename("Copper_Volume")
                        mapping[name] = ticker
                        break
                    except Exception as exc:
                        errors.append(f"{ticker}: {exc}")
                else:
                    raise RuntimeError(f"Cannot download {name}: {'; '.join(errors)}")
        union_dates = len(set().union(*(set(s.index) for s in series.values())))
        raw = pd.concat(series.values(), axis=1, join="inner")
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = raw.sort_index()
    raw = raw.loc[(raw.index >= cfg["data"]["start"]) &
                  (raw.index < cfg["data"]["end_exclusive"])].copy()
    expected = ["Copper"] + cfg["features"]["candidates"]
    if set(expected) - set(raw):
        raise ValueError(f"Missing columns: {set(expected) - set(raw)}")
    duplicate_count = int(raw.index.duplicated().sum())
    raw = raw.loc[~raw.index.duplicated(keep="last"), expected]
    if raw["TBill_10Y"].iloc[:max(1, len(raw)//5)].median() > 15:
        raw["TBill_10Y"] /= 10
    raw = raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing = raw.isna().sum()
    missing_rows = int(raw.isna().any(axis=1).sum())
    raw = raw.dropna()
    if len(raw) < 100 or not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("Invalid or insufficient chronological data")
    max_rows = cfg["budget"].get("max_rows")
    if max_rows:
        raw = raw.tail(max_rows)
    quality = pd.DataFrame([{"column": c, "missing_before_drop": int(missing[c]),
                             "rows_dropped_any_missing": missing_rows, "duplicate_dates": duplicate_count,
                             "source_duplicate_dates": source_duplicate_count,
                             "union_dates_before_inner_join": union_dates,
                             "rows_retained": len(raw), "first_date": str(raw.index.min().date()),
                             "last_date": str(raw.index.max().date())} for c in expected])
    if not using_snapshot:
        quality.to_csv(out / "data_quality_report.csv", index=False)
        raw.to_csv(snapshot, index_label="Date")
        atomic_json(out / "data_sources.json", mapping)
    return raw


def boundaries(n, ratio, val_fraction):
    test_start = int(n * ratio)
    val_start = int(test_start * (1 - val_fraction))
    if not 0 < val_start < test_start < n:
        raise ValueError("Invalid chronological split")
    return val_start, test_start


def choose_features(fit, cfg, method):
    candidates = cfg["features"]["candidates"]
    corr = fit[candidates].corrwith(fit["Copper"], method="spearman").fillna(-1).sort_values(ascending=False)
    fallback = None
    if method == "paper":
        chosen = cfg["features"]["paper"]
    elif method == "all8":
        chosen = candidates
    elif method.startswith("top"):
        chosen = corr.head(int(method[3:])).index.tolist()
    elif method == "spearman":
        chosen = corr[corr > cfg["features"]["spearman_cutoff"]].index.tolist()
        if not chosen:
            if cfg["features"]["fallback"] != "top3_positive_rank":
                raise ValueError("Unsupported Spearman fallback")
            chosen = corr.head(3).index.tolist()
            fallback = "No feature above paper cutoff; train-only top 3 by signed Spearman"
    else:
        raise ValueError(method)
    return chosen, corr, fallback


def make_windows(x, y, dates, start, end, lookback):
    if start < lookback or end <= start:
        raise ValueError("Invalid window range")
    X = np.stack([x[t-lookback:t] for t in range(start, end)]).astype("float32")
    Y = y[start:end].astype("float32")
    assert X.shape == (end-start, lookback, x.shape[1])
    return X, Y, dates[start:end]


def context(raw, cfg, ratio=.9, lookback=30, feature_method="paper", fold=None, include_test=False):
    # fold=(validation_start, validation_end); no test is available to search callers.
    val_start, test_start = fold or boundaries(len(raw), ratio, cfg["data"]["validation_fraction"])
    if val_start <= lookback or test_start > len(raw):
        raise ValueError("Fold is too short")
    fit = raw.iloc[:val_start]
    chosen, corr, fallback = choose_features(fit, cfg, feature_method)
    sx, sy = MinMaxScaler().fit(fit[chosen]), MinMaxScaler().fit(fit[["Copper"]])
    # Transforming future rows is safe: fit was restricted to the training fold.
    visible = raw if include_test else raw.iloc[:test_start]
    x = sx.transform(visible[chosen]).astype("float32")
    y = sy.transform(visible[["Copper"]]).reshape(-1).astype("float32")
    train = make_windows(x, y, raw.index, lookback, val_start, lookback)
    val = make_windows(x, y, raw.index, val_start, test_start, lookback)
    result = {"train": train, "val": val, "features": chosen, "correlations": corr,
              "fallback": fallback, "x_scaler": sx, "y_scaler": sy, "ratio": ratio,
              "lookback": lookback, "feature_method": feature_method,
              "train_range": [str(train[2].min().date()), str(train[2].max().date())],
              "val_range": [str(val[2].min().date()), str(val[2].max().date())],
              "test_range": [str(raw.index[test_start].date()), str(raw.index[-1].date())],
              "n_test": len(raw) - test_start, "split": (val_start, test_start),
              "previous_targets": np.r_[np.nan, y[:-1]]}
    if include_test:
        result["test"] = make_windows(x, y, raw.index, test_start, len(raw), lookback)
    assert train[2].max() < val[2].min()
    return result


def set_seed(seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_model(n_features, lookback, hp, activation, cfg):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input((lookback, n_features)),
        tf.keras.layers.LSTM(int(hp["units1"]), activation="tanh", recurrent_activation="sigmoid", return_sequences=True),
        tf.keras.layers.LSTM(int(hp["units2"]), activation="tanh", recurrent_activation="sigmoid"),
        tf.keras.layers.Dense(cfg["model"]["dense_units"], activation="relu"),
        tf.keras.layers.Dropout(cfg["model"]["dropout"]),
        tf.keras.layers.Dense(1, activation=None if activation == "linear" else activation),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(hp["lr"])), loss="mse")
    return m


def train_score(ctx, hp, cfg, seed=42, activation=None, epochs=None, evaluate="val", predict=False):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    gc.collect()
    set_seed(seed)
    activation = activation or cfg["model"]["activation"]
    epochs = epochs or cfg["budget"].get("epochs", cfg["model"]["epochs"])
    X, y, _ = ctx["train"]
    model = build_model(X.shape[-1], ctx["lookback"], hp, activation, cfg)
    started = time.perf_counter()
    history = model.fit(X, y, validation_data=ctx["val"][:2], epochs=epochs,
                        batch_size=cfg["model"]["batch_size"], shuffle=False, verbose=0)
    duration = time.perf_counter() - started
    Xe, ye, dates = ctx[evaluate]
    yp = model.predict(Xe, batch_size=cfg["model"]["batch_size"], verbose=0).reshape(-1)
    if yp.shape != ye.shape or not np.isfinite(yp).all():
        raise ValueError("Invalid prediction shape or NaN")
    scaler = ctx["y_scaler"]
    actual = scaler.inverse_transform(ye.reshape(-1, 1)).reshape(-1)
    predicted = scaler.inverse_transform(yp.reshape(-1, 1)).reshape(-1)
    metrics = {"mse_scaled": float(mean_squared_error(ye, yp)),
               "mse_price2": float(mean_squared_error(actual, predicted)),
               "rmse_price": float(np.sqrt(mean_squared_error(actual, predicted))),
               "mae_price": float(mean_absolute_error(actual, predicted)),
               "target_above_one_fraction": float(np.mean(ye > 1)),
               "prediction_near_bound_fraction": float(np.mean(np.abs(yp) > .99)) if activation != "linear" else 0.0,
               "extreme_mae_price": float(mean_absolute_error(actual[actual >= np.quantile(actual, .9)], predicted[actual >= np.quantile(actual, .9)])),
               "train_seconds": duration}
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    if predict:
        return metrics, pd.DataFrame({"Date": dates, "actual": actual, "predicted": predicted,
                                      "error": predicted - actual}), history.history
    return metrics


def persistence(ctx, evaluate="val"):
    # Last observed copper price is available when predicting the next session.
    _, y, dates = ctx[evaluate]
    train_count = ctx["split"][0]
    val_end = ctx["split"][1]
    start = train_count if evaluate == "val" else val_end
    # This helper receives the original last-observed target through context().
    previous = ctx["previous_targets"][start:start+len(y)]
    scaler = ctx["y_scaler"]
    actual = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
    pred = scaler.inverse_transform(previous.reshape(-1, 1)).reshape(-1)
    return {"mse_scaled": float(mean_squared_error(y, previous)),
            "mse_price2": float(mean_squared_error(actual, pred)),
            "rmse_price": float(np.sqrt(mean_squared_error(actual, pred))),
            "mae_price": float(mean_absolute_error(actual, pred)), "train_seconds": 0.0}, pd.DataFrame({"Date": dates, "actual": actual, "predicted": pred, "error": pred-actual})


def environment():
    import sklearn
    import tensorflow as tf
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "tensorflow": tf.__version__, "gpu": [x.name for x in tf.config.list_physical_devices("GPU")]}


def manifest(cfg, raw, out):
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(raw, index=True).values.tobytes()).hexdigest()
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    path = out / "experiment_manifest.json"
    item = {"data_sha256": fingerprint, "config_sha256": config_hash, "mode": cfg["mode"],
            "date_range": [str(raw.index.min().date()), str(raw.index.max().date())],
            "rows": len(raw), "environment": environment(),
            "forecast_task": "next joint trading session; inputs available through previous joint session close",
            "metric": "MSE scaled for selection; price squared for final reporting"}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if (old["data_sha256"], old["config_sha256"]) != (fingerprint, config_hash):
            raise RuntimeError("Output directory has a different data/config manifest; choose a new output directory")
    atomic_json(path, item)
    return item


def record(out, eid, phase, label, ctx, hp, cfg, seed, metrics, search_seed=None, evaluate="val", epochs=None):
    row = {"experiment_id": eid, "phase": phase, "model_type": label, "seed": seed,
           "search_seed": search_seed if search_seed is not None else "", "ratio": ctx["ratio"],
           "features": "|".join(ctx["features"]), "feature_rule": ctx["feature_method"],
           "feature_fallback": ctx["fallback"] or "", "lookback": ctx["lookback"],
           "activation": cfg["model"]["activation"] if label == "persistence" else cfg.get("active_activation", cfg["model"]["activation"]),
           "units1": hp.get("units1", ""), "units2": hp.get("units2", ""), "lr": hp.get("lr", ""),
           "epochs": epochs or 0,
           "train_start": ctx["train_range"][0], "train_end": ctx["train_range"][1],
           "val_start": ctx["val_range"][0], "val_end": ctx["val_range"][1],
           "test_start": ctx["test_range"][0], "test_end": ctx["test_range"][1],
           "n_train": len(ctx["train"][1]), "n_val": len(ctx["val"][1]), "n_test": ctx["n_test"],
           "validation_mse": metrics["mse_scaled"] if evaluate == "val" else "",
           "test_mse": metrics["mse_scaled"] if evaluate == "test" else "",
           "mse_price2": metrics["mse_price2"], "rmse_price": metrics["rmse_price"],
           "mae_price": metrics["mae_price"], "train_seconds": metrics["train_seconds"],
           "target_above_one_fraction": metrics.get("target_above_one_fraction", ""),
           "prediction_near_bound_fraction": metrics.get("prediction_near_bound_fraction", ""),
           "extreme_mae_price": metrics.get("extreme_mae_price", ""),
           "environment": json.dumps(environment(), sort_keys=True)}
    append_csv(out / "experiment_results.csv", row)


def completed_ids(out):
    path = out / "experiment_results.csv"
    return set(pd.read_csv(path)["experiment_id"].astype(str)) if path.exists() else set()


def run_candidate(out, eid, phase, label, ctx, hp, cfg, seed=42, activation=None, evaluate="val", search_seed=None, epochs=None):
    if eid in completed_ids(out):
        return
    cfg["active_activation"] = activation or cfg["model"]["activation"]
    if label == "persistence":
        metrics, prediction = persistence(ctx, evaluate)
        history = None
    else:
        metrics, prediction, history = train_score(ctx, hp, cfg, seed, activation, epochs=epochs,
                                                   evaluate=evaluate, predict=True)
    if evaluate == "test":
        prediction.insert(0, "experiment_id", eid)
        prediction.to_csv(out / f"predictions_{eid}.csv", index=False)
    actual_epochs = 0 if label == "persistence" else (epochs or cfg["budget"].get("epochs", cfg["model"]["epochs"]))
    record(out, eid, phase, label, ctx, hp, cfg, seed, metrics, search_seed, evaluate, actual_epochs)
    return metrics


def phase1(raw, cfg, out):
    ratio = cfg["protocol"]["primary_ratio"]
    ctx = context(raw, cfg, ratio, cfg["model"]["lookback"], "paper")
    X, y, _ = ctx["train"]
    assert np.isfinite(X).all() and np.isfinite(y).all()
    assert len(ctx["val"][1]) and X.ndim == 3 and y.ndim == 1
    # Small model; true wall time is used for estimates, never a guessed T4 speed.
    hp = {"units1": 8, "units2": 8, "lr": .001}
    import tensorflow as tf
    tf.keras.backend.clear_session()
    pilot = build_model(X.shape[-1], ctx["lookback"], hp, "linear", cfg)
    with tf.GradientTape() as tape:
        prediction = pilot(tf.convert_to_tensor(X[:8]), training=True)
        loss = tf.reduce_mean(tf.square(prediction[:, 0] - tf.convert_to_tensor(y[:8])))
    gradients = tape.gradient(loss, pilot.trainable_variables)
    if any(g is None or not bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients):
        raise ValueError("Missing or non-finite pilot gradient")
    del pilot
    tf.keras.backend.clear_session()
    metrics = train_score(ctx, hp, cfg, epochs=cfg["budget"].get("epochs", 2))
    benchmark = {"train_seconds": metrics["train_seconds"], "epochs": cfg["budget"].get("epochs", 2),
                 "train_samples": len(y), "validation_mse": metrics["mse_scaled"],
                 "gpu": environment()["gpu"], "note": "8/8 unit pilot; full trials can be slower"}
    atomic_json(out / "benchmark.json", benchmark)
    atomic_json(out / "sanity_check.json", {"shape": X.shape, "nan_count": int(np.isnan(X).sum()),
                                           "finite_gradients": True,
                                           "prediction_shape": [len(ctx["val"][1])], **benchmark})
    print("Sanity check and measured benchmark:", benchmark)


def phase2(raw, cfg, out):
    ratio = cfg["protocol"]["primary_ratio"]
    hp = cfg["model"]["paper"]
    base = cfg["model"]["baseline"]
    screen_epochs = cfg["budget"].get("trial_epochs", cfg["model"]["trial_epochs"])
    methods = ["paper", "spearman", "top3", "top5", "all8"]
    for activation in cfg["protocol"]["activations"]:
        ctx = context(raw, cfg, ratio, cfg["model"]["lookback"], "paper")
        run_candidate(out, f"activation_{activation}", "screening", "activation_ablation", ctx, hp, cfg, activation=activation, epochs=screen_epochs)
    for lb in cfg["protocol"]["lookbacks"]:
        ctx = context(raw, cfg, ratio, lb, "paper")
        run_candidate(out, f"lookback_{lb}", "screening", "lookback_ablation", ctx, hp, cfg, epochs=screen_epochs)
    for method in methods:
        ctx = context(raw, cfg, ratio, cfg["model"]["lookback"], method)
        run_candidate(out, f"features_{method}", "screening", "feature_ablation", ctx, hp, cfg, epochs=screen_epochs)
    ctx = context(raw, cfg, ratio, cfg["model"]["lookback"], "paper")
    run_candidate(out, "baseline_90", "reproduction", "vanilla_lstm", ctx, base, cfg, epochs=screen_epochs)
    run_candidate(out, "paper_preset_90", "reproduction", "paper_configuration_rerun", ctx, hp, cfg, epochs=screen_epochs)
    run_candidate(out, "persistence_90", "screening", "persistence", ctx, {}, cfg)


def _trial(ctx, hp, cfg, training_seed):
    return train_score(ctx, hp, cfg, training_seed,
                       epochs=cfg["budget"].get("trial_epochs", cfg["model"]["trial_epochs"]))


def sa_search(ctx, cfg, out, search_seed, training_seed, steps, name="sa_lr_neurons", optimize_lr=True):
    state_path = out / f"{name}_state.json"
    rng = np.random.default_rng(search_seed)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        history_path = out / "sa_search_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            history = history[(history.search != name) | (history.step <= state["step"])]
            history = history.drop_duplicates(["search", "step"], keep="last")
            history.to_csv(history_path, index=False)
        rng.bit_generator.state = state["rng_state"]
        current, best = state["current"], state["best"]
        current_loss, best_loss, start = state["current_loss"], state["best_loss"], state["step"] + 1
        evaluations = state["evaluations"]
    else:
        current = dict(cfg["search"]["initial"])
        if not optimize_lr:
            current["lr"] = cfg["model"]["paper"]["lr"]
        metrics = _trial(ctx, current, cfg, training_seed)
        current_loss = best_loss = metrics["mse_scaled"]
        best = dict(current)
        evaluations, start = 1, 1
        append_csv(out / "sa_search_history.csv", {"search": name, "step": 0, "temperature": cfg["search"]["temperature"],
                   "candidate_mse": current_loss, "best_mse": best_loss, "accepted": 1, "acceptance_probability": 1,
                   "units1": current["units1"], "units2": current["units2"], "lr": current["lr"],
                   "evaluations": evaluations, "train_seconds": metrics["train_seconds"]})
        atomic_json(state_path, {"current": current, "best": best, "current_loss": current_loss,
                                 "best_loss": best_loss, "step": 0, "evaluations": evaluations,
                                 "rng_state": rng.bit_generator.state})
    for step in range(start, steps + 1):
        T = cfg["search"]["temperature"] * cfg["search"]["cooling"] ** (step-1)
        low, high = cfg["search"]["units_min"], cfg["search"]["units_max"]
        cand = {"units1": int(np.clip(round(current["units1"] + rng.uniform(-1,1)*(16+80*T)),low,high)),
                "units2": int(np.clip(round(current["units2"] + rng.uniform(-1,1)*(16+80*T)),low,high)),
                "lr": current["lr"]}
        if optimize_lr:
            cand["lr"] = float(np.clip(current["lr"] * np.exp(rng.uniform(-1,1)*(0.5+T)*2),
                                       cfg["search"]["lr_min"], cfg["search"]["lr_max"]))
        metrics = _trial(ctx, cand, cfg, training_seed)
        loss = metrics["mse_scaled"]
        evaluations += 1
        delta = loss - current_loss
        probability = 1.0 if delta <= 0 else math.exp(-delta / max(T * max(best_loss, 1e-6), 1e-12))
        accepted = bool(rng.random() < probability)
        if accepted:
            current, current_loss = cand, loss
        if loss < best_loss:
            best, best_loss = dict(cand), loss
        append_csv(out / "sa_search_history.csv", {"search": name, "step": step, "temperature": T,
                   "candidate_mse": loss, "best_mse": best_loss, "accepted": int(accepted),
                   "acceptance_probability": probability, "units1": cand["units1"], "units2": cand["units2"],
                   "lr": cand["lr"], "evaluations": evaluations, "train_seconds": metrics["train_seconds"]})
        atomic_json(state_path, {"current": current, "best": best, "current_loss": current_loss,
                                 "best_loss": best_loss, "step": step, "evaluations": evaluations,
                                 "rng_state": rng.bit_generator.state})
        print(f"{name}: {step}/{steps} val={loss:.6g} best={best_loss:.6g}")
    return best, best_loss, evaluations


def tpe_search(ctx, cfg, out, trials, training_seed, search_seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=search_seed),
                                study_name="copper_tpe", storage=f"sqlite:///{(out / 'optuna.db').as_posix()}",
                                load_if_exists=True)
    def objective(trial):
        hp = {"units1": trial.suggest_int("units1", cfg["search"]["units_min"], cfg["search"]["units_max"]),
              "units2": trial.suggest_int("units2", cfg["search"]["units_min"], cfg["search"]["units_max"]),
              "lr": trial.suggest_float("lr", cfg["search"]["lr_min"], cfg["search"]["lr_max"], log=True)}
        m = _trial(ctx, hp, cfg, training_seed)
        trial.set_user_attr("train_seconds", m["train_seconds"])
        append_csv(out / "optuna_trials.csv", {"trial": trial.number, "validation_mse": m["mse_scaled"],
                   "units1": hp["units1"], "units2": hp["units2"], "lr": hp["lr"],
                   "train_seconds": m["train_seconds"]})
        return m["mse_scaled"]
    remaining = max(0, trials - len(study.trials))
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    rows = [{"trial": t.number, "validation_mse": t.value,
             "units1": t.params["units1"], "units2": t.params["units2"], "lr": t.params["lr"],
             "train_seconds": t.user_attrs.get("train_seconds", np.nan)}
            for t in study.trials if t.value is not None]
    pd.DataFrame(rows).to_csv(out / "optuna_trials.csv", index=False)
    return study.best_params, study.best_value, len(study.trials)


def selected_screening(cfg, out):
    df = pd.read_csv(out / "experiment_results.csv")
    choices = {}
    for prefix, key, convert in [("activation_", "activation", str), ("lookback_", "lookback", int),
                                 ("features_", "feature_method", str)]:
        part = df[df.experiment_id.str.startswith(prefix)].dropna(subset=["validation_mse"])
        if part.empty:
            raise RuntimeError(f"Missing screening results: {prefix}")
        best = part.sort_values("validation_mse").iloc[0]
        choices[key] = convert(best.experiment_id[len(prefix):])
    atomic_json(out / "screening_selection.json", choices)
    return choices


def phase3(raw, cfg, out, allow_large=False):
    choices = selected_screening(cfg, out)
    cfg["model"]["activation"] = choices["activation"]
    ctx = context(raw, cfg, cfg["protocol"]["primary_ratio"], choices["lookback"], choices["feature_method"])
    steps = cfg["budget"].get("sa_iterations", max(cfg["search"]["sa_iterations"]))
    if not (out / "benchmark.json").exists():
        raise RuntimeError("Run phase1 benchmark before hyperparameter optimization")
    bench = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    trial_epochs = cfg["budget"].get("trial_epochs", cfg["model"]["trial_epochs"])
    # Arithmetic projection from a measured small model; actual time may differ.
    per_trial = bench["train_seconds"] * trial_epochs / bench["epochs"]
    estimated_hours = per_trial * (2 * (steps+1) + (steps+1)) / 3600
    atomic_json(out / "compute_estimate.json", {"pilot_seconds": bench["train_seconds"],
                "pilot_epochs": bench["epochs"], "projected_hours_at_pilot_rate": estimated_hours,
                "trial_evaluations": 3 * (steps+1),
                "warning": "Projection uses 8/8 pilot; actual time may be higher or lower"})
    print(f"Measured-pilot projection: {estimated_hours:.2f} GPU wall hours for phase3; actual time may differ")
    if estimated_hours > 4 and not allow_large:
        raise RuntimeError("Large search budget: inspect compute_estimate.json and rerun with allow_large=True")
    seed = cfg["protocol"]["seeds"][0]
    sa_hp, sa_val, sa_n = sa_search(ctx, cfg, out, seed, seed, steps)
    # 30/100/300 are prefixes of a single trajectory, so budgets are comparable.
    sa_hist = pd.read_csv(out / "sa_search_history.csv")
    for budget in cfg["search"]["sa_iterations"]:
        subset = sa_hist[(sa_hist.search == "sa_lr_neurons") & (sa_hist.step <= budget)]
        if len(subset) == budget + 1:
            ix = subset.candidate_mse.idxmin()
            row = subset.loc[ix]
            atomic_json(out / f"sa_best_{budget}.json", {"units1": int(row.units1), "units2": int(row.units2),
                                                       "lr": float(row.lr), "validation_mse": float(row.candidate_mse),
                                                       "evaluations": len(subset),
                                                       "acceptance_rate": float(subset.accepted.iloc[1:].mean()),
                                                       "gpu_wall_seconds": float(subset.train_seconds.sum())})
    fixed_hp, fixed_val, fixed_n = sa_search(ctx, cfg, out, seed+1000, seed, steps,
                                              name="sa_neurons_fixed_lr", optimize_lr=False)
    tpe_hp, tpe_val, tpe_n = tpe_search(ctx, cfg, out, steps+1, seed, seed)
    best = {"sa": {"hp": sa_hp, "validation_mse": sa_val, "evaluations": sa_n},
            "sa_fixed_lr": {"hp": fixed_hp, "validation_mse": fixed_val, "evaluations": fixed_n},
            "tpe": {"hp": tpe_hp, "validation_mse": tpe_val, "evaluations": tpe_n},
            "selection": choices, "training_seed": seed, "search_seed": seed}
    atomic_json(out / "best_hyperparameters.json", best)
    for label in ["sa", "sa_fixed_lr", "tpe"]:
        run_candidate(out, f"hpo_{label}", "hpo", label, ctx, best[label]["hp"], cfg,
                      seed=seed, activation=choices["activation"], search_seed=seed)


def expanding_folds(n, ratio, fold_count):
    _, test_start = boundaries(n, ratio, .1)
    val_length = max(30, int(test_start * .1))
    first = test_start - fold_count * val_length
    if first <= 60:
        raise ValueError("Insufficient history for expanding folds")
    return [(first + i * val_length, first + (i+1) * val_length) for i in range(fold_count)]


def phase4(raw, cfg, out):
    if not (out / "best_hyperparameters.json").exists():
        raise RuntimeError("Run phase3 first")
    best = json.loads((out / "best_hyperparameters.json").read_text(encoding="utf-8"))
    sel = best["selection"]
    cfg["model"]["activation"] = sel["activation"]
    seeds = cfg["protocol"]["seeds"] if cfg["mode"] == "FULL_RESEARCH" else cfg["protocol"]["seeds"][:2]
    for ratio in cfg["data"]["train_ratios"]:
        ctx = context(raw, cfg, ratio, sel["lookback"], sel["feature_method"])
        for seed in seeds:
            for label, hp in [("persistence", {}), ("vanilla_lstm", cfg["model"]["baseline"]),
                              ("paper_configuration_rerun", cfg["model"]["paper"]),
                              ("sa", best["sa"]["hp"]), ("tpe", best["tpe"]["hp"])]:
                run_candidate(out, f"robust_{ratio}_{seed}_{label}", "robustness", label,
                              ctx, hp, cfg, seed=seed, search_seed=best["search_seed"] if label in ("sa", "tpe") else None)
    for i, fold in enumerate(expanding_folds(len(raw), cfg["protocol"]["primary_ratio"],
                                           cfg["protocol"]["walk_forward_folds"])):
        ctx = context(raw, cfg, cfg["protocol"]["primary_ratio"], sel["lookback"],
                      sel["feature_method"], fold=fold)
        for label, hp in [("paper_configuration_rerun", cfg["model"]["paper"]), ("sa", best["sa"]["hp"])]:
            run_candidate(out, f"fold_{i}_{label}", "walk_forward", label, ctx, hp, cfg, seed=seeds[0])
    # Search seed robustness is separate from training seed robustness.
    search_steps = min(30, cfg["budget"].get("sa_iterations", 30))
    ctx = context(raw, cfg, cfg["protocol"]["primary_ratio"], sel["lookback"], sel["feature_method"])
    for search_seed in seeds:
        hp, val, evaluations = sa_search(ctx, cfg, out, search_seed, seeds[0], search_steps,
                                         name=f"sa_seed_{search_seed}")
        path = out / "search_seed_results.csv"
        existing = set(pd.read_csv(path).search_seed) if path.exists() else set()
        if search_seed not in existing:
            append_csv(path, {"search_seed": search_seed,
                       "training_seed": seeds[0], "validation_mse": val, "evaluations": evaluations,
                       "units1": hp["units1"], "units2": hp["units2"], "lr": hp["lr"]})
    seed_report(out)


def seed_report(out):
    from scipy.stats import t
    df = pd.read_csv(out / "experiment_results.csv")
    df = df[df.phase == "robustness"]
    rows = []
    for (ratio, model), part in df.groupby(["ratio", "model_type"]):
        vals = part.mse_price2.to_numpy(dtype=float)
        n = len(vals)
        std = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        margin = float(t.ppf(.975, n-1) * std / np.sqrt(n)) if n > 1 else float("nan")
        rows.append({"ratio": ratio, "model_type": model, "n": n, "mean_mse_price2": np.mean(vals),
                     "std_mse_price2": std, "median_mse_price2": np.median(vals),
                     "best_mse_price2": np.min(vals), "worst_mse_price2": np.max(vals),
                     "ci95_low": np.mean(vals)-margin, "ci95_high": np.mean(vals)+margin,
                     "ci_method": "Student t over independent training seeds; descriptive with n<5"})
    pd.DataFrame(rows).to_csv(out / "seed_summary.csv", index=False)


def phase5(raw, cfg, out):
    if not (out / "best_hyperparameters.json").exists():
        raise RuntimeError("Run phases 1-3 and freeze selection before final test")
    best = json.loads((out / "best_hyperparameters.json").read_text(encoding="utf-8"))
    results_path = out / "experiment_results.csv"
    if not results_path.exists():
        raise RuntimeError("Run phase4 robustness before final selection")
    robust = pd.read_csv(results_path)
    robust = robust[(robust.phase == "robustness") &
                    (np.isclose(robust.ratio, cfg["protocol"]["primary_ratio"]))]
    if robust.empty:
        raise RuntimeError("No primary split robustness results for final selection")
    champion = robust.groupby("model_type").validation_mse.mean().idxmin()
    sel = best["selection"]
    cfg["model"]["activation"] = sel["activation"]
    locked = out / "final_selection.json"
    selection = {"activation": sel["activation"], "lookback": sel["lookback"],
                 "feature_method": sel["feature_method"], "models": best,
                 "champion": champion,
                 "selection_rule": "lowest mean scaled validation MSE across primary-split training seeds; test first accessed in phase5"}
    if locked.exists() and json.loads(locked.read_text(encoding="utf-8")) != selection:
        raise RuntimeError("Final selection changed; do not retune after test")
    atomic_json(locked, selection)
    for ratio in cfg["data"]["train_ratios"]:
        ctx = context(raw, cfg, ratio, sel["lookback"], sel["feature_method"], include_test=True)
        for label, hp in [("persistence", {}), ("vanilla_lstm", cfg["model"]["baseline"]),
                          ("paper_configuration_rerun", cfg["model"]["paper"]),
                          ("sa_transfer_from_primary_split", best["sa"]["hp"]),
                          ("tpe_transfer_from_primary_split", best["tpe"]["hp"] )]:
            run_candidate(out, f"final_{ratio}_{label}", "final", label, ctx, hp, cfg,
                          seed=best["training_seed"], evaluate="test",
                          search_seed=best["search_seed"] if "transfer" in label else None)
    report(out)


def report(out):
    df = pd.read_csv(out / "experiment_results.csv")
    lines = ["# Experiment summary (generated from actual run artifacts)", "",
             "Forecast: next joint trading session. Model selection uses validation; test appears only in final phase.", "",
             "The published 1990–2009 MSE values are not directly comparable with Yahoo 2006–2026 MSE.", ""]
    for phase in ["reproduction", "screening", "hpo", "robustness", "walk_forward", "final"]:
        part = df[df.phase == phase]
        if part.empty:
            continue
        lines += [f"## {phase}", "", "| ID | Model | Ratio | Validation MSE scaled | Test MSE scaled | MSE price² |", "|---|---|---:|---:|---:|---:|"]
        for _, r in part.iterrows():
            lines.append(f"| {r.experiment_id} | {r.model_type} | {r.ratio:.3f} | {r.validation_mse} | {r.test_mse} | {r.mse_price2:.6g} |")
        lines.append("")
    if (out / "seed_summary.csv").exists():
        lines += ["Seed robustness: `seed_summary.csv` includes mean, sample SD, median, range and Student t CI. ",
                  "CIs describe training seed variability on fixed data, not future market uncertainty.", ""]
    if (out / "compute_estimate.json").exists():
        lines += ["Compute estimate: see `compute_estimate.json`; the 8/8 pilot projection is not a T4 guarantee.", ""]
    if (out / "final_selection.json").exists():
        chosen = json.loads((out / "final_selection.json").read_text(encoding="utf-8"))
        lines += [f"Validation-selected champion: **{chosen['champion']}** (mean over primary-split training seeds).", ""]
    lines += ["SA/TPE hyperparameters are optimized on the primary 90% split and transferred to other ratios; those ratios are transfer tests, not independently optimized searches.", "",
              "Limitations: Yahoo futures proxy differs from Investing.com data; shared test windows across split ratios are not identical;",
              "Spearman ranks can change by fold; model optimization may overfit a single validation block.", ""]
    (out / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")
    charts(out)


def charts(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if (out / "sa_search_history.csv").exists():
        h = pd.read_csv(out / "sa_search_history.csv")
        h = h[h.search == "sa_lr_neurons"]
        if len(h):
            fig, ax = plt.subplots(2, 1, figsize=(9, 6))
            ax[0].plot(h.step, h.candidate_mse, label="candidate")
            ax[0].plot(h.step, h.best_mse, label="best so far")
            ax[0].legend(); ax[0].set_ylabel("Validation MSE scaled")
            ax[1].plot(h.step, h.temperature); ax[1].set_ylabel("Temperature"); ax[1].set_xlabel("SA step")
            fig.tight_layout(); fig.savefig(out / "sa_convergence.png"); plt.close(fig)
    if (out / "optuna_trials.csv").exists() and (out / "sa_search_history.csv").exists():
        t = pd.read_csv(out / "optuna_trials.csv")
        s = pd.read_csv(out / "sa_search_history.csv")
        s = s[s.search == "sa_lr_neurons"]
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(s.evaluations, s.best_mse, label="SA")
        ax.plot(np.arange(1,len(t)+1), t.validation_mse.cummin(), label="TPE")
        ax.set(xlabel="Model evaluations", ylabel="Best validation MSE scaled"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "sa_vs_optuna.png"); plt.close(fig)
    files = list(out.glob("predictions_final_*.csv"))
    if files:
        for file in files:
            p = pd.read_csv(file, parse_dates=["Date"])
            fig, ax = plt.subplots(figsize=(9,4))
            ax.plot(p.Date, p.actual, label="actual"); ax.plot(p.Date, p.predicted, label="predicted")
            ax.legend(); fig.tight_layout(); fig.savefig(out / f"{file.stem}.png"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7,4))
        pd.read_csv(files[0]).error.hist(ax=ax, bins=30)
        ax.set(xlabel="Prediction error price", ylabel="Count")
        fig.tight_layout(); fig.savefig(out / "error_distribution.png"); plt.close(fig)
    if (out / "seed_summary.csv").exists():
        s = pd.read_csv(out / "seed_summary.csv")
        for ratio, p in s.groupby("ratio"):
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(p.model_type, p.mean_mse_price2, yerr=p.std_mse_price2)
            ax.tick_params(axis="x", rotation=30); ax.set_ylabel("MSE price² mean ± SD")
            fig.tight_layout(); fig.savefig(out / f"seed_variation_{ratio}.png"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8,4))
        for name, p in s.groupby("model_type"):
            ax.plot(p.ratio, p.mean_mse_price2, marker="o", label=name)
        ax.legend(); ax.set(xlabel="Train fraction", ylabel="Mean validation MSE price² across seeds")
        fig.tight_layout(); fig.savefig(out / "mse_by_split.png"); plt.close(fig)
    if (out / "experiment_results.csv").exists():
        final = pd.read_csv(out / "experiment_results.csv")
        final = final[final.phase == "final"]
        if len(final):
            fig, ax = plt.subplots(figsize=(8,4))
            for name, p in final.groupby("model_type"):
                ax.plot(p.ratio, p.mse_price2, marker="o", label=name)
            ax.legend(fontsize=7); ax.set(xlabel="Train fraction", ylabel="Test MSE price² (different periods)")
            fig.tight_layout(); fig.savefig(out / "test_mse_by_split.png"); plt.close(fig)


def run(phase, mode="QUICK_TEST", allow_large=False):
    cfg = load_config(mode)
    out = output_dir(cfg)
    raw = load_data(cfg, out)
    manifest(cfg, raw, out)
    actions = {"phase1": phase1, "phase2": phase2, "phase3": phase3,
               "phase4": phase4, "phase5": phase5}
    if phase == "report":
        report(out)
    elif phase == "phase3":
        phase3(raw, cfg, out, allow_large)
        report(out)
    else:
        actions[phase](raw, cfg, out)
        if (out / "experiment_results.csv").exists():
            report(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["phase1", "phase2", "phase3", "phase4", "phase5", "report"])
    parser.add_argument("--mode", choices=["QUICK_TEST", "FULL_RESEARCH"], default="QUICK_TEST")
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args()
    run(args.phase, args.mode, args.allow_large)
