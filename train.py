"""One-step-ahead educational regression on simulated daily data."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "Energy_Consumption_kWh"
FEATURES = ["Temperature_C", "Electricity_Price_€/kWh", "month", "weekday", "lag_1"]

def prepare(frame):
    required = {"Date", TARGET, "Temperature_C", "Electricity_Price_€/kWh"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required-set(frame.columns))}")
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame = frame.sort_values("Date").reset_index(drop=True)
    if frame["Date"].duplicated().any():
        raise ValueError("Daily dates must be unique.")
    if not frame["Date"].diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("Daily sequence must be continuous; review missing days before making lag features.")
    for name in (TARGET, "Temperature_C", "Electricity_Price_€/kWh"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if frame[TARGET].isna().any() or not np.isfinite(frame[TARGET]).all():
        raise ValueError("Targets must be finite.")
    frame["month"] = frame["Date"].dt.month
    frame["weekday"] = frame["Date"].dt.dayofweek
    frame["lag_1"] = frame[TARGET].shift(1)
    return frame.iloc[1:].reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="simulated_energy_data.csv")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    frame = prepare(pd.read_csv(args.data))
    if len(frame) < 20:
        parser.error("At least 20 daily observations are required.")
    cutoff = int(len(frame)*0.8)
    train, test = frame.iloc[:cutoff], frame.iloc[cutoff:]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1),
    }
    result = {"data": "simulated", "split": "chronological_80_20", "train_end": str(train["Date"].max().date()), "test_start": str(test["Date"].min().date()), "train_rows": len(train), "test_rows": len(test), "models": {}}
    predictions = test[["Date", TARGET]].copy()
    for name, estimator in models.items():
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("regressor", estimator)])
        pipeline.fit(train[FEATURES], train[TARGET])
        prediction = pipeline.predict(test[FEATURES])
        result["models"][name] = {"mae": float(mean_absolute_error(test[TARGET], prediction)), "rmse": float(mean_squared_error(test[TARGET], prediction)**0.5), "r2": float(r2_score(test[TARGET], prediction))}
        predictions[name] = prediction
        joblib.dump({"pipeline": pipeline, "features": FEATURES, "target": TARGET}, output / f"{name}.joblib")
    result["limitation"] = "Simulated data; no operational accuracy claim. Rolling one-step evaluation uses observed previous-day demand and contemporaneous temperature/price, not a multi-day forecast. Hyperparameters fixed before this holdout; do not repeatedly tune on it."
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions.to_csv(output / "predictions.csv", index=False)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
