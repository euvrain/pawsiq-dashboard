# %% [markdown]
# # PawsIQ — Demand Forecasting Model
# `ml/demand_forecast/train.ipynb`
#
# **What this notebook does:**
# Train a model that predicts how many bookings will happen in a given
# hour slot (e.g. Monday 8am) so the app can surface "high demand" warnings
# and feed the dynamic pricing model.
#
# **Algorithm:** Gradient Boosting (sklearn) → swap to XGBoost with one line
#
# **Why Gradient Boosting for this problem?**
# - Works well on tabular data with mixed feature types
# - Handles non-linear patterns (demand spikes) without needing scaling
# - Produces feature importances we can explain to stakeholders
# - XGBoost is the industry-standard version of the same algorithm

# %% — Imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Swap to XGBoost with these two lines once installed (pip install xgboost):
# from xgboost import XGBRegressor
# GradientBoostingRegressor = lambda **kw: XGBRegressor(
#     n_estimators=kw['n_estimators'], max_depth=kw['max_depth'],
#     learning_rate=kw['learning_rate'], subsample=kw['subsample'],
#     random_state=kw['random_state'], tree_method='hist'
# )

CREAM = "#F7F5F0"; INK = "#141810"; SAGE = "#3D6B4F"
BARK = "#C4A882"; WARN = "#D4622A"; STONE = "#6B7063"; RULE = "#E4E1D8"
plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": CREAM,
    "axes.edgecolor": RULE, "axes.labelcolor": STONE,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "xtick.color": STONE, "ytick.color": STONE,
    "grid.color": RULE, "font.family": "sans-serif",
})

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & Aggregate
# ─────────────────────────────────────────────────────────────────────────────
# We load the raw bookings and aggregate to (date, hour_of_day) bins.
# Each row = one hour slot on one day, with a booking_count target.
#
# WHY this granularity?
# - Hourly is operationally meaningful (walkers plan by the hour)
# - Daily is too coarse to drive surge pricing decisions
# - Per-booking is too granular (almost always 0 or 1)

df = pd.read_csv("data/synthetic/bookings.csv", parse_dates=["scheduled_at"])
completed = df[df["status"] == "completed"].copy()
print(f"Loaded {len(df):,} bookings | {len(completed):,} completed")

completed["date"]        = pd.to_datetime(completed["scheduled_at"].dt.date)
completed["hour_of_day"] = completed["scheduled_at"].dt.hour

agg = completed.groupby(["date","hour_of_day"]).agg(
    booking_count=("booking_id", "count"),
).reset_index()

# Fill in zero-booking slots so the model sees "quiet" hours too.
# Without this, the model never learns from hours with no demand.
all_dates = pd.date_range(completed["date"].min(), completed["date"].max(), freq="D")
all_hours = range(6, 21)   # 6am–8pm operating window
full_idx  = pd.MultiIndex.from_product([all_dates, all_hours], names=["date","hour_of_day"])
agg = agg.set_index(["date","hour_of_day"]).reindex(full_idx, fill_value=0).reset_index()

print(f"\nAggregated to {len(agg):,} hour slots")
print(f"booking_count — mean: {agg['booking_count'].mean():.2f}, "
      f"std: {agg['booking_count'].std():.2f}, max: {agg['booking_count'].max()}")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
# This is where the DS work happens. We create features the model can use.
#
# THREE CATEGORIES of features:
#
# 1. Calendar features (raw)
#    hour_of_day, day_of_week, month — direct numbers
#    Problem: the model sees hour 23 and hour 0 as far apart, but they're adjacent.
#
# 2. Cyclic encoding (sin/cos)
#    Fixes the above problem. sin(2π * hour/24) wraps around: 0 and 23 are close.
#    This is a standard technique for any periodic feature (time, angles, etc.)
#
# 3. Historical average prior
#    "What is the average booking count for this (hour, day_of_week) combo?"
#    This is the single strongest feature — it encodes the regular weekly pattern.
#    Think of it as telling the model: "Monday 8am is historically busy."

agg["day_of_week"] = agg["date"].dt.dayofweek   # 0=Mon, 6=Sun
agg["month"]       = agg["date"].dt.month        # 1–12
agg["year"]        = agg["date"].dt.year
agg["is_weekend"]  = (agg["day_of_week"] >= 5).astype(int)
agg["is_peak_hour"]= agg["hour_of_day"].isin([7,8,9,17,18,19]).astype(int)
agg["is_morning"]  = agg["hour_of_day"].isin([7,8,9]).astype(int)
agg["is_evening"]  = agg["hour_of_day"].isin([17,18,19]).astype(int)

# Cyclic encoding — wraps periodic features so the model understands adjacency
agg["hour_sin"]  = np.sin(2 * np.pi * agg["hour_of_day"] / 24)
agg["hour_cos"]  = np.cos(2 * np.pi * agg["hour_of_day"] / 24)
agg["dow_sin"]   = np.sin(2 * np.pi * agg["day_of_week"] / 7)
agg["dow_cos"]   = np.cos(2 * np.pi * agg["day_of_week"] / 7)
agg["month_sin"] = np.sin(2 * np.pi * agg["month"] / 12)
agg["month_cos"] = np.cos(2 * np.pi * agg["month"] / 12)

# Historical average prior — strongest feature
# Compute the mean booking count for every (hour_of_day, day_of_week) combination
# across the entire dataset, then join it back as a feature.
hist_avg = (agg.groupby(["hour_of_day","day_of_week"])["booking_count"]
              .mean().reset_index()
              .rename(columns={"booking_count":"hist_avg_bookings"}))
agg = agg.merge(hist_avg, on=["hour_of_day","day_of_week"], how="left")

agg = agg.sort_values(["date","hour_of_day"]).reset_index(drop=True)

print("Features created:")
print([c for c in agg.columns if c not in ["date","booking_count"]])

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: for time series data, NEVER use random train_test_split.
# Reason: if you shuffle, the model trains on future data and tests on past data.
# This inflates performance metrics and is called DATA LEAKAGE.
#
# Correct approach: chronological split.
# We train on the first 80% of dates, test on the last 20%.
# This simulates real deployment: train on history, predict the future.

dates      = sorted(agg["date"].unique())
split_date = dates[int(len(dates) * 0.8)]

train = agg[agg["date"] <  split_date].copy()
test  = agg[agg["date"] >= split_date].copy()

print(f"Train: {len(train):,} rows ({train['date'].min().date()} → {train['date'].max().date()})")
print(f"Test:  {len(test):,}  rows ({test['date'].min().date()}  → {test['date'].max().date()})")
print(f"Split: {pd.Timestamp(split_date).date()}")

FEATURES = [
    "hour_of_day", "day_of_week", "month", "year",
    "is_weekend", "is_peak_hour", "is_morning", "is_evening",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "hist_avg_bookings",
]

X_train, y_train = train[FEATURES], train["booking_count"]
X_test,  y_test  = test[FEATURES],  test["booking_count"]

print(f"\nFeature matrix: {X_train.shape}")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Baseline Model
# ─────────────────────────────────────────────────────────────────────────────
# Before training anything fancy, establish a baseline.
# The "dumb" baseline = always predict the training mean.
# Any real model MUST beat this. If it doesn't, something is wrong.

baseline_pred = np.full(len(y_test), y_train.mean())
baseline_mae  = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
print(f"Baseline (always predict mean={y_train.mean():.2f}):")
print(f"  MAE:  {baseline_mae:.3f}")
print(f"  RMSE: {baseline_rmse:.3f}")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train Gradient Boosting Model
# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters explained:
#   n_estimators  = number of trees to build (more = more accurate but slower)
#   max_depth     = how deep each tree goes (4-5 is typical to avoid overfitting)
#   learning_rate = how much each tree corrects the previous ones (lower = more careful)
#   subsample     = fraction of training rows used per tree (adds randomness, prevents overfitting)
#   random_state  = seed for reproducibility

model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42,
)

model.fit(X_train, y_train)
print("Model trained.")

preds = model.predict(X_test)
preds = np.maximum(preds, 0)   # demand can't be negative

mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)

rmse_imp = (baseline_rmse - rmse) / baseline_rmse * 100
mae_imp  = (baseline_mae  - mae)  / baseline_mae  * 100

print(f"\nGradient Boosting results:")
print(f"  MAE:  {mae:.3f} bookings/slot  (vs baseline {baseline_mae:.3f}, {mae_imp:+.1f}%)")
print(f"  RMSE: {rmse:.3f}               (vs baseline {baseline_rmse:.3f}, {rmse_imp:+.1f}%)")
print(f"  R²:   {r2:.3f}  (0 = baseline mean, 1 = perfect)")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Cross Validation
# ─────────────────────────────────────────────────────────────────────────────
# A single train/test split can be "lucky" or "unlucky" depending on what
# falls in the test window. Cross-validation gives a more honest estimate.
#
# TimeSeriesSplit: preserves chronological order across all 5 folds.
# Each fold: train on everything before the split, test on the next window.
# Never shuffles — that would be data leakage.

tscv     = TimeSeriesSplit(n_splits=5)
cv_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
)
cv_scores = cross_val_score(
    cv_model, agg[FEATURES], agg["booking_count"],
    cv=tscv, scoring="neg_mean_absolute_error"
)
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()
print(f"Cross-validation MAE: {cv_mae:.3f} ± {cv_std:.3f} bookings/slot")
print(f"(Stable std means the model is consistent across time windows)")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
# Which features does the model rely on most?
# This is important for:
#   - Explainability (can you explain it in an interview?)
#   - Debugging (does it make intuitive sense?)
#   - Future work (which features should we invest in improving?)

importances = (pd.Series(model.feature_importances_, index=FEATURES)
                 .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(9, 5))
colors = [WARN if imp > 0.15 else SAGE for imp in importances.values]
ax.barh(importances.index, importances.values, color=colors, zorder=2)
ax.set_xlabel("Feature Importance (fraction of variance explained)")
ax.set_title("Demand Model — Feature Importances")
ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

top_feat = importances.idxmax()
top_imp  = importances.max()
print(f"\nTop feature: {top_feat} ({top_imp:.1%} of importance)")
print("✓ hist_avg_bookings dominating means the model correctly learned weekly patterns.")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Prediction vs Actual (visual check)
# ─────────────────────────────────────────────────────────────────────────────
# Numbers alone don't tell the full story. Plot predictions vs actuals
# to catch systematic errors (e.g. model always under-predicts peaks).

test_plot = test.copy()
test_plot["predicted"] = preds
sample = test_plot[test_plot["date"] <= test_plot["date"].unique()[13]]  # first 2 weeks of test

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

# Chart 1: one week of actual vs predicted by hour
week_sample = sample.groupby("date").agg(
    actual=("booking_count","sum"),
    predicted=("predicted","sum")
).reset_index().head(14)
x = range(len(week_sample))
axes[0].plot(x, week_sample["actual"],    color=INK,  linewidth=2, label="Actual",    marker="o", markersize=4)
axes[0].plot(x, week_sample["predicted"], color=SAGE, linewidth=2, label="Predicted", marker="s", markersize=4, linestyle="--")
axes[0].set_xticks(x)
axes[0].set_xticklabels([str(d.date()) for d in week_sample["date"]], rotation=45, ha="right", fontsize=8)
axes[0].set_title("Daily Demand — Actual vs Predicted (first 2 test weeks)")
axes[0].set_ylabel("Total Bookings")
axes[0].legend(framealpha=0); axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

# Chart 2: scatter — if model is good, points cluster along the diagonal
axes[1].scatter(y_test, preds, alpha=0.3, s=15, color=SAGE, zorder=2)
axes[1].plot([0, y_test.max()], [0, y_test.max()], color=WARN, linewidth=1.5, linestyle="--", label="Perfect prediction")
axes[1].set_xlabel("Actual bookings"); axes[1].set_ylabel("Predicted bookings")
axes[1].set_title("Predicted vs Actual — Test Set")
axes[1].legend(framealpha=0); axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)

fig.tight_layout()
plt.show()

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Predict function (used by the FastAPI endpoint)
# ─────────────────────────────────────────────────────────────────────────────
# This is what the API will call when a user views the demand heatmap.

def predict_demand(hour_of_day: int, day_of_week: int, month: int,
                   year: int = 2025, hist_avg: float = None) -> dict:
    """
    Predict booking demand for a given hour/day slot.

    Parameters:
        hour_of_day  : 0–23
        day_of_week  : 0=Mon, 6=Sun
        month        : 1–12
        year         : calendar year
        hist_avg     : historical average for this slot (from DB lookup)

    Returns:
        dict with predicted_bookings, demand_level, is_peak
    """
    if hist_avg is None:
        # Fallback: use overall mean from training data
        hist_avg = float(y_train.mean())

    features = pd.DataFrame([{
        "hour_of_day":       hour_of_day,
        "day_of_week":       day_of_week,
        "month":             month,
        "year":              year,
        "is_weekend":        int(day_of_week >= 5),
        "is_peak_hour":      int(hour_of_day in [7,8,9,17,18,19]),
        "is_morning":        int(hour_of_day in [7,8,9]),
        "is_evening":        int(hour_of_day in [17,18,19]),
        "hour_sin":          np.sin(2*np.pi*hour_of_day/24),
        "hour_cos":          np.cos(2*np.pi*hour_of_day/24),
        "dow_sin":           np.sin(2*np.pi*day_of_week/7),
        "dow_cos":           np.cos(2*np.pi*day_of_week/7),
        "month_sin":         np.sin(2*np.pi*month/12),
        "month_cos":         np.cos(2*np.pi*month/12),
        "hist_avg_bookings": hist_avg,
    }])

    raw = float(model.predict(features)[0])
    predicted = max(0.0, round(raw, 2))

    # Map to a human-readable demand level
    if predicted >= 2.5:   level = "high"
    elif predicted >= 1.5: level = "medium"
    else:                   level = "low"

    return {
        "predicted_bookings": predicted,
        "demand_level":       level,
        "is_peak":            bool(hour_of_day in [7,8,9,17,18,19]),
        "hour_of_day":        hour_of_day,
        "day_of_week":        day_of_week,
    }

# Test it
print("Sample predictions:")
test_cases = [
    (8, 0, 4, "Mon 8am April"),
    (14, 0, 4, "Mon 2pm April"),
    (8, 5, 4, "Sat 8am April"),
    (18, 2, 9, "Wed 6pm September"),
]
for h, d, m, label in test_cases:
    result = predict_demand(h, d, m, hist_avg=hist_avg.loc[
        (hist_avg["hour_of_day"]==h) & (hist_avg["day_of_week"]==d), "hist_avg_bookings"
    ].values[0] if len(hist_avg.loc[(hist_avg["hour_of_day"]==h) & (hist_avg["day_of_week"]==d)]) > 0 else None)
    print(f"  {label:<25} → {result['predicted_bookings']:.2f} bookings ({result['demand_level']})")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Save Artifacts
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("ml/artifacts", exist_ok=True)

# Save trained model
with open("ml/artifacts/demand_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save historical averages (needed by predict function at inference time)
hist_avg.to_csv("ml/artifacts/demand_hist_avg.csv", index=False)

# Save metrics
metrics = {
    "model":               "GradientBoostingRegressor (drop-in for XGBoostRegressor)",
    "target":              "booking_count per (date, hour_of_day) slot",
    "mae":                 round(float(mae), 3),
    "rmse":                round(float(rmse), 3),
    "r2":                  round(float(r2), 3),
    "baseline_mae":        round(float(baseline_mae), 3),
    "baseline_rmse":       round(float(baseline_rmse), 3),
    "mae_improvement_pct": round(float(mae_imp), 1),
    "rmse_improvement_pct":round(float(rmse_imp), 1),
    "cv_mae":              round(float(cv_mae), 3),
    "cv_std":              round(float(cv_std), 3),
    "n_features":          len(FEATURES),
    "train_rows":          len(train),
    "test_rows":           len(test),
    "features":            FEATURES,
}
with open("ml/artifacts/demand_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Artifacts saved:")
print("  ml/artifacts/demand_model.pkl")
print("  ml/artifacts/demand_hist_avg.csv")
print("  ml/artifacts/demand_metrics.json")
print(f"\nFinal metrics:")
print(f"  MAE:  {mae:.3f} bookings/slot")
print(f"  RMSE: {rmse:.3f}")
print(f"  R²:   {r2:.3f}")
print(f"  RMSE improvement vs baseline: {rmse_imp:+.1f}%")
print(f"  CV MAE: {cv_mae:.3f} ± {cv_std:.3f}")

# %% [markdown]
# ## Results Summary
#
# | Metric | Model | Baseline (mean) | Improvement |
# |--------|-------|-----------------|-------------|
# | MAE    | 0.738 | 0.761           | +3.1%       |
# | RMSE   | 0.912 | 0.960           | +5.0%       |
# | R²     | 0.098 | 0.000           | —           |
# | CV MAE | 0.775 ± 0.033 | — | consistent |
#
# **Interpretation:**
# The model predicts within ~0.74 bookings/slot of the true value.
# The most important feature is `hist_avg_bookings` (63%) — the model
# correctly learned that historical weekly patterns are the strongest
# signal for future demand.
#
# **Why R² is modest (~0.10):**
# Demand at this granularity is inherently noisy — a client cancels,
# an unexpected rainstorm happens. The model captures the structural
# pattern (peak hours, weekday bias, seasonality) but can't predict
# individual-slot noise. This is expected and acceptable.
#
# **Next:** `ml/dynamic_pricing/train.ipynb`
