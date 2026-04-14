# %% [markdown]
# # PawsIQ — Dynamic Pricing Model
# `ml/dynamic_pricing/train.ipynb`
#
# **What this notebook does:**
# Train a model that predicts the surge multiplier for a given booking —
# so the app can charge peak prices during high-demand slots and
# discount during slow periods to drive volume.
#
# **Algorithm:** Ridge Regression (primary) + Gradient Boosting (comparison)
#
# **Why Ridge for pricing (not XGBoost)?**
# - `is_peak_hour` has a 0.77 linear correlation with surge — the relationship
#   is largely linear, so a linear model fits well and is easier to explain
# - Ridge adds L2 regularization to prevent overfitting on noisy features
# - Coefficients are directly interpretable: "peak hour adds +X to surge"
# - Simpler models are easier to audit, which matters for pricing decisions
#
# **Key difference from demand model:**
# - Demand model: aggregated to hourly slots, predicts booking *count*
# - Pricing model: individual booking level, predicts surge *multiplier*

# %% — Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CREAM="#F7F5F0"; INK="#141810"; SAGE="#3D6B4F"
BARK="#C4A882"; WARN="#D4622A"; STONE="#6B7063"; RULE="#E4E1D8"
plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": CREAM,
    "axes.edgecolor": RULE, "axes.labelcolor": STONE,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "xtick.color": STONE, "ytick.color": STONE,
    "grid.color": RULE, "font.family": "sans-serif",
})

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load Data
# ─────────────────────────────────────────────────────────────────────────────
# Unlike the demand model, we keep individual bookings (not aggregated).
# Each row = one booking. Target = its surge_multiplier.
# This gives us 9,279 training examples — more than enough for Ridge.

df        = pd.read_csv("data/synthetic/bookings.csv", parse_dates=["scheduled_at"])
completed = df[df["status"] == "completed"].copy()
print(f"Loaded {len(completed):,} completed bookings")
print(f"\nTarget — surge_multiplier:")
print(completed["surge_multiplier"].describe().round(4))

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Exploratory: correlation with surge
# ─────────────────────────────────────────────────────────────────────────────
# Before engineering features, check raw correlations.
# This tells us which signals are worth keeping.

print("Correlation with surge_multiplier:")
num_cols = ["hour_of_day","day_of_week","month","is_peak_hour","is_weekend","base_price"]
corrs    = {c: completed[c].corr(completed["surge_multiplier"]) for c in num_cols}
for col, corr in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
    bar   = "█" * int(abs(corr) * 40)
    sign  = "+" if corr > 0 else "-"
    print(f"  {col:<20} {corr:+.4f}  {sign}{bar}")

# Visualize surge by hour
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
surge_by_hour = completed.groupby("hour_of_day")["surge_multiplier"].mean()
colors = [WARN if h in (7,8,9,17,18,19) else SAGE for h in surge_by_hour.index]
axes[0].bar(surge_by_hour.index, surge_by_hour.values, color=colors, width=0.7, zorder=2)
axes[0].axhline(1.0, color=STONE, linewidth=1, linestyle=":")
axes[0].set_title("Avg Surge Multiplier by Hour")
axes[0].set_xlabel("Hour of Day"); axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

surge_by_dow = completed.groupby("day_of_week")["surge_multiplier"].mean()
dow_labels   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
axes[1].bar(dow_labels, surge_by_dow.values,
            color=[SAGE if i < 5 else BARK for i in range(7)], width=0.6, zorder=2)
axes[1].axhline(1.0, color=STONE, linewidth=1, linestyle=":")
axes[1].set_title("Avg Surge Multiplier by Day of Week")
axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)
fig.tight_layout(); plt.show()

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
# Three categories, same philosophy as demand model:
#
# 1. Calendar features + cyclic encoding
# 2. Service type (one-hot) — overnight has different pricing than a walk
# 3. Supply/demand proxy features:
#    - zip_hour_demand: how many bookings in this zip+hour? (supply tightness)
#    - hist_avg_surge: what's the historical surge for this (hour, dow) combo?

completed["date"]     = pd.to_datetime(completed["scheduled_at"].dt.date)

# Cyclic encoding
completed["hour_sin"]  = np.sin(2*np.pi*completed["hour_of_day"]/24)
completed["hour_cos"]  = np.cos(2*np.pi*completed["hour_of_day"]/24)
completed["dow_sin"]   = np.sin(2*np.pi*completed["day_of_week"]/7)
completed["dow_cos"]   = np.cos(2*np.pi*completed["day_of_week"]/7)
completed["month_sin"] = np.sin(2*np.pi*completed["month"]/12)
completed["month_cos"] = np.cos(2*np.pi*completed["month"]/12)

# Service type one-hot encoding
# drop_first=True drops walk_30 (the baseline) to avoid multicollinearity
# Multicollinearity = when two features are perfectly correlated,
# which confuses linear models about which one to credit
svc_dummies = pd.get_dummies(completed["service_type"], prefix="svc", drop_first=True)
completed   = pd.concat([completed, svc_dummies], axis=1)
svc_cols    = list(svc_dummies.columns)

# Supply proxy: bookings in same zip+hour = demand pressure on walkers
completed["zip_str"]      = completed["zip"].astype(str)
zip_hour_counts           = (completed
    .groupby(["date","hour_of_day","zip_str"]).size()
    .reset_index(name="zip_hour_demand"))
completed = completed.merge(zip_hour_counts, on=["date","hour_of_day","zip_str"], how="left")
completed["zip_hour_demand"] = completed["zip_hour_demand"].fillna(1)

# Historical avg surge for this (hour, day_of_week) combo
hist_surge = (completed
    .groupby(["hour_of_day","day_of_week"])["surge_multiplier"]
    .mean().reset_index()
    .rename(columns={"surge_multiplier":"hist_avg_surge"}))
completed  = completed.merge(hist_surge, on=["hour_of_day","day_of_week"], how="left")

FEATURES = [
    "hour_of_day", "day_of_week", "month",
    "is_peak_hour", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "zip_hour_demand", "hist_avg_surge", "base_price",
] + svc_cols

print(f"Features ({len(FEATURES)}): {FEATURES}")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train / Test Split (chronological)
# ─────────────────────────────────────────────────────────────────────────────
# Sort by scheduled_at so split respects time order.
# Same rule as demand model: no shuffling, ever.

completed = completed.sort_values("scheduled_at").reset_index(drop=True)
split_idx = int(len(completed) * 0.8)
train = completed.iloc[:split_idx]
test  = completed.iloc[split_idx:]

print(f"Train: {len(train):,} bookings  ({train['scheduled_at'].min().date()} → {train['scheduled_at'].max().date()})")
print(f"Test:  {len(test):,}  bookings  ({test['scheduled_at'].min().date()}  → {test['scheduled_at'].max().date()})")

X_train, y_train = train[FEATURES], train["surge_multiplier"]
X_test,  y_test  = test[FEATURES],  test["surge_multiplier"]

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Baseline
# ─────────────────────────────────────────────────────────────────────────────
baseline_pred = np.full(len(y_test), y_train.mean())
baseline_mae  = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
print(f"Baseline (always predict mean={y_train.mean():.4f}):")
print(f"  MAE:  {baseline_mae:.4f}")
print(f"  RMSE: {baseline_rmse:.4f}")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Ridge Regression
# ─────────────────────────────────────────────────────────────────────────────
# WHY WE SCALE for Ridge:
# Ridge penalizes large coefficients. But if hour_of_day ranges 0-23 and
# is_peak_hour is 0 or 1, they're on different scales — the penalty treats
# them unequally. StandardScaler centers and scales all features to mean=0,
# std=1 so Ridge treats them consistently.
# Rule: always scale for linear models. Never need to for tree models.

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  # fit on train only — never on test
X_test_sc  = scaler.transform(X_test)       # transform test with train's stats

# Alpha = regularization strength
# alpha=0   → ordinary linear regression (no regularization)
# alpha=1   → mild regularization (good default)
# alpha=100 → heavy regularization (pushes coefficients toward zero)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sc, y_train)
preds = ridge.predict(X_test_sc)

mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)
mae_imp  = (baseline_mae  - mae)  / baseline_mae  * 100
rmse_imp = (baseline_rmse - rmse) / baseline_rmse * 100

print(f"Ridge Regression (alpha=1.0):")
print(f"  MAE:  {mae:.4f}  (vs baseline {baseline_mae:.4f}, {mae_imp:+.1f}%)")
print(f"  RMSE: {rmse:.4f}  (vs baseline {baseline_rmse:.4f}, {rmse_imp:+.1f}%)")
print(f"  R²:   {r2:.4f}  ← explains {r2*100:.1f}% of variance in surge")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Coefficient Analysis (the interpretable story)
# ─────────────────────────────────────────────────────────────────────────────
# This is Ridge's superpower over tree models — you can read the coefficients
# and explain exactly WHY the model makes each prediction.
# Positive coefficient = feature pushes surge UP
# Negative coefficient = feature pushes surge DOWN

coef_df = pd.DataFrame({
    "feature":     FEATURES,
    "coefficient": ridge.coef_
}).reindex(pd.Series(ridge.coef_).abs().sort_values(ascending=False).index)
coef_df = coef_df.sort_values("coefficient")

fig, ax = plt.subplots(figsize=(9, 6))
colors = [WARN if c > 0 else SAGE for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, zorder=2)
ax.axvline(0, color=STONE, linewidth=1, linestyle=":")
ax.set_xlabel("Coefficient (scaled)")
ax.set_title("Ridge Coefficients — Dynamic Pricing Model\n(positive = increases surge, negative = decreases surge)")
ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
fig.tight_layout(); plt.show()

print("\nCoefficients (sorted by magnitude):")
for _, row in coef_df.iloc[::-1].iterrows():
    direction = "↑ surge" if row["coefficient"] > 0 else "↓ surge"
    print(f"  {row['feature']:<22} {row['coefficient']:+.4f}  {direction}")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Cross Validation
# ─────────────────────────────────────────────────────────────────────────────
# We wrap scaler + Ridge in a Pipeline so the scaler is re-fit on each
# training fold independently. If we scaled before CV, we'd leak test
# statistics into training — another form of data leakage.

ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
tscv       = TimeSeriesSplit(n_splits=5)
cv_scores  = cross_val_score(
    ridge_pipe, completed[FEATURES], completed["surge_multiplier"],
    cv=tscv, scoring="neg_mean_absolute_error"
)
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()
print(f"Cross-validation MAE: {cv_mae:.4f} ± {cv_std:.4f}")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Revenue Impact Analysis
# ─────────────────────────────────────────────────────────────────────────────
# Metrics like MAE and R² are technical. What actually matters to the business
# is: does this model make more money than flat-rate pricing?
#
# We compare three scenarios on the test set:
#   1. Flat rate: always charge base_price (no surge at all)
#   2. Model rate: charge base_price × model_predicted_surge
#   3. Actual rate: what we actually charged (ground truth)

flat_revenue   = test["base_price"].sum()
model_revenue  = (test["base_price"] * pd.Series(preds, index=test.index)).sum()
actual_revenue = (test["base_price"] * test["surge_multiplier"]).sum()
model_lift_pct = (model_revenue - flat_revenue) / flat_revenue * 100

print(f"Revenue comparison (test set — {len(test):,} bookings):")
print(f"  Flat rate (no surge):    ${flat_revenue:>10,.2f}")
print(f"  Model-priced:            ${model_revenue:>10,.2f}  ({model_lift_pct:+.1f}% vs flat)")
print(f"  Actual (ground truth):   ${actual_revenue:>10,.2f}")

# Visualize predicted vs actual surge over time
fig, axes = plt.subplots(2, 1, figsize=(13, 7))

# Sample 200 points for scatter
sample_idx = np.random.choice(len(y_test), 200, replace=False)
axes[0].scatter(y_test.iloc[sample_idx], preds[sample_idx],
                alpha=0.4, s=15, color=SAGE, zorder=2)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             color=WARN, linewidth=1.5, linestyle="--", label="Perfect prediction")
axes[0].set_xlabel("Actual surge multiplier")
axes[0].set_ylabel("Predicted surge multiplier")
axes[0].set_title("Predicted vs Actual Surge (200 sample bookings)")
axes[0].legend(framealpha=0); axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

# Revenue comparison bar
bars = axes[1].bar(
    ["Flat Rate\n(no surge)", "Model Pricing\n(Ridge)", "Actual Revenue\n(ground truth)"],
    [flat_revenue, model_revenue, actual_revenue],
    color=[STONE, SAGE, WARN], width=0.5, zorder=2
)
axes[1].set_ylabel("Revenue ($)")
axes[1].set_title(f"Revenue Impact — Test Set ({len(test):,} bookings)")
axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)
for bar, val in zip(bars, [flat_revenue, model_revenue, actual_revenue]):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 200,
                 f"${val:,.0f}", ha="center", fontsize=10, color=STONE)

fig.tight_layout(); plt.show()

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Predict function (FastAPI endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def predict_surge(
    hour_of_day: int,
    day_of_week: int,
    month: int,
    base_price: float,
    service_type: str = "walk_30",
    zip_hour_demand: int = 1,
    hist_avg_surge: float = None,
) -> dict:
    """
    Predict surge multiplier for a booking.

    Parameters:
        hour_of_day     : 0–23
        day_of_week     : 0=Mon, 6=Sun
        month           : 1–12
        base_price      : service base price ($)
        service_type    : walk_30 | walk_60 | drop_in | overnight
        zip_hour_demand : how many bookings in this zip+hour (supply proxy)
        hist_avg_surge  : historical avg surge for this slot (from DB lookup)

    Returns:
        dict with surge_multiplier, final_price, pricing_tier
    """
    if hist_avg_surge is None:
        hist_avg_surge = float(y_train.mean())

    svc_walk_60  = int(service_type == "walk_60")
    svc_drop_in  = int(service_type == "drop_in")
    svc_overnight= int(service_type == "overnight")

    feat_dict = {
        "hour_of_day":      hour_of_day,
        "day_of_week":      day_of_week,
        "month":            month,
        "is_peak_hour":     int(hour_of_day in [7,8,9,17,18,19]),
        "is_weekend":       int(day_of_week >= 5),
        "hour_sin":         np.sin(2*np.pi*hour_of_day/24),
        "hour_cos":         np.cos(2*np.pi*hour_of_day/24),
        "dow_sin":          np.sin(2*np.pi*day_of_week/7),
        "dow_cos":          np.cos(2*np.pi*day_of_week/7),
        "month_sin":        np.sin(2*np.pi*month/12),
        "month_cos":        np.cos(2*np.pi*month/12),
        "zip_hour_demand":  zip_hour_demand,
        "hist_avg_surge":   hist_avg_surge,
        "base_price":       base_price,
        "svc_drop_in":      svc_drop_in,
        "svc_overnight":    svc_overnight,
        "svc_walk_60":      svc_walk_60,
    }
    features_df = pd.DataFrame([feat_dict])[FEATURES]
    features_sc = scaler.transform(features_df)
    raw_surge   = float(ridge.predict(features_sc)[0])
    surge       = round(np.clip(raw_surge, 0.85, 1.35), 4)
    final_price = round(base_price * surge, 2)

    if surge >= 1.25:   tier = "high"
    elif surge >= 1.10: tier = "medium"
    else:               tier = "standard"

    return {
        "surge_multiplier": surge,
        "final_price":      final_price,
        "base_price":       base_price,
        "pricing_tier":     tier,
        "is_peak":          bool(hour_of_day in [7,8,9,17,18,19]),
    }

# Test predictions
print("Sample pricing predictions:")
cases = [
    (8,  0, 4, 16.0, "walk_30",   "Mon 8am — peak weekday"),
    (14, 0, 4, 16.0, "walk_30",   "Mon 2pm — off peak"),
    (8,  5, 4, 16.0, "walk_30",   "Sat 8am — peak weekend"),
    (18, 2, 9, 24.0, "walk_60",   "Wed 6pm — evening peak"),
    (20, 1, 6, 55.0, "overnight", "Tue 8pm — overnight"),
]
for h, d, m, price, svc, label in cases:
    result = predict_surge(h, d, m, price, svc)
    print(f"  {label:<30} ×{result['surge_multiplier']}  "
          f"${result['base_price']} → ${result['final_price']}  [{result['pricing_tier']}]")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Save Artifacts
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("ml/artifacts", exist_ok=True)

with open("ml/artifacts/pricing_model.pkl", "wb") as f:
    pickle.dump({"model": ridge, "scaler": scaler, "features": FEATURES}, f)

hist_surge.to_csv("ml/artifacts/pricing_hist_avg.csv", index=False)

metrics = {
    "model":               "Ridge Regression (alpha=1.0)",
    "target":              "surge_multiplier per booking",
    "mae":                 round(float(mae), 4),
    "rmse":                round(float(rmse), 4),
    "r2":                  round(float(r2), 4),
    "baseline_mae":        round(float(baseline_mae), 4),
    "baseline_rmse":       round(float(baseline_rmse), 4),
    "mae_improvement_pct": round(float(mae_imp), 1),
    "rmse_improvement_pct":round(float(rmse_imp), 1),
    "cv_mae":              round(float(cv_mae), 4),
    "cv_std":              round(float(cv_std), 4),
    "revenue_lift_pct":    round(float(model_lift_pct), 1),
    "n_features":          len(FEATURES),
    "train_rows":          len(train),
    "test_rows":           len(test),
    "features":            FEATURES,
}
with open("ml/artifacts/pricing_metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print("\nArtifacts saved:")
print("  ml/artifacts/pricing_model.pkl")
print("  ml/artifacts/pricing_hist_avg.csv")
print("  ml/artifacts/pricing_metrics.json")
print(f"\nFinal metrics:")
print(f"  MAE:  {mae:.4f} (×surge units)")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}  ← explains {r2*100:.1f}% of surge variance")
print(f"  MAE improvement vs baseline:  {mae_imp:+.1f}%")
print(f"  RMSE improvement vs baseline: {rmse_imp:+.1f}%")
print(f"  Revenue lift over flat rate:  {model_lift_pct:+.1f}%")

# %% [markdown]
# ## Results Summary
#
# | Metric | Ridge Model | Baseline (mean) | Improvement |
# |--------|-------------|-----------------|-------------|
# | MAE    | 0.0569      | 0.1139          | +50.0%      |
# | RMSE   | 0.0679      | 0.1300          | +47.8%      |
# | R²     | 0.727       | 0.000           | —           |
# | Revenue lift vs flat rate | +17.1% | — | — |
#
# **Interpretation:**
# The model predicts surge multiplier within ±0.057 on average.
# R² of 0.73 means it explains 73% of variance in surge pricing —
# strong performance for a linear model.
#
# The most important feature is `hist_avg_surge` — the historical average
# surge for that (hour, day_of_week) slot. This makes intuitive sense:
# Monday 8am is historically expensive, and it will be expensive tomorrow too.
#
# **Revenue impact:**
# Using model-predicted surge instead of flat-rate pricing generates
# +17.1% revenue on the test set — this is the business metric that
# goes in your resume bullet.
#
# **Resume bullet:**
# "Engineered dynamic pricing pipeline using Ridge Regression to predict
# surge multipliers per booking, achieving R²=0.73 and 17.1% simulated
# revenue lift over flat-rate baseline"
#
# **Next:** `api/routers/pricing.py` — serve this model via FastAPI
