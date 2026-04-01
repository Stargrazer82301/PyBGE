"""
pybge.correlate
---------------
Trains a regression model relating weather features to energy usage, using
the data produced by ``pybge.run()``.

Training window  : all data EXCEPT the most recent ``predict_days`` days.
Prediction window: the most recent ``predict_days`` days.

Outputs saved to ``data_dir``:
  - ``BGE_Regressor.hkl``                   -- hickle model bundle
  - ``BGE_Correlate_Relation.png``           -- scatter: temperature vs usage
  - ``BGE_Correlate_Prediction.png``         -- scatter: predicted vs actual
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.gaussian_process.kernels
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import quantile_forest
import hickle

from .utils import c_to_f

plt.ioff()
sns.set(context="talk")
sns.set_style("darkgrid", {"font.sans-serif": "DejaVu Sans"})

# Weather feature columns fed to the model (always in Celsius internally)
_COLS_TRAIN = [
    "temp_p24",
    "tmax_p24",
    "tmin_p24",
    "prcp_p24",
    "rhum_p24",
    "pres_p24",
    "wdir_p24",
    "wspd_p24",
    "cldc_p24",
]

_VALID_METHODS = [
    "GPR",
    "linear",
    "BRR",
    "ARDR",
    "Random-Forest",
    "Quantile-Forest",
    "Neural-Net",
]


# ---------------------------------------------------------------------------
# Helpers (also re-exported for use by forecast.py)
# ---------------------------------------------------------------------------

def pred_descale(
    features: pd.DataFrame,
    model,
    features_scaler,
    target_scaler,
) -> np.ndarray:
    """Return de-scaled predictions from a fitted model."""
    return target_scaler.inverse_transform(
        model.predict(features_scaler.transform(features)).reshape(-1, 1)
    ).ravel()


def _build_model(method: str, n_features: int):
    """Construct an unfitted sklearn-compatible model."""
    if method == "linear":
        return sklearn.linear_model.LinearRegression()
    elif method == "BRR":
        return sklearn.linear_model.BayesianRidge(
            tol=1e-6, fit_intercept=True, compute_score=True
        )
    elif method == "ARDR":
        return sklearn.linear_model.ARDRegression(
            fit_intercept=False, compute_score=True
        )
    elif method == "Random-Forest":
        return sklearn.ensemble.RandomForestRegressor(n_estimators=500)
    elif method == "Quantile-Forest":
        return quantile_forest.RandomForestQuantileRegressor(
            n_estimators=20,
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=2,
        )
    elif method == "Neural-Net":
        return sklearn.neural_network.MLPRegressor(
            max_iter=1000, alpha=0.00001, hidden_layer_sizes=(2000,)
        )
    elif method == "GPR":
        kernel = sklearn.gaussian_process.kernels.Matern(
            length_scale=0.1 * np.ones(n_features),
            length_scale_bounds=(0.01, 100.0),
            nu=1.5,
        ) + sklearn.gaussian_process.kernels.WhiteKernel(
            noise_level=1.0, noise_level_bounds=(1e-3, 1000.0)
        )
        return sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=kernel, alpha=0.1, n_restarts_optimizer=10
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose from: {_VALID_METHODS}"
        )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def correlate(
    data_dir: str,
    *,
    predict_days: int = 30,
    method: str = "GPR",
    plot_unit: str = "auto",
    dollars_per_kwhr: Optional[float] = None,
    dollars_per_therm: Optional[float] = None,
    freedom_units: bool = False,
) -> None:
    """Train a weather-to-energy-usage regression model and save outputs.

    Parameters
    ----------
    data_dir:
        Directory containing ``BGE_Collated.csv.gz`` (produced by
        ``pybge.run()``).  All outputs are also written here.
    predict_days:
        Number of most-recent days to use as the *prediction* (current)
        window.  All earlier data forms the *training* (previous) window.
        Defaults to 30.
    method:
        Regression method.  One of: ``"GPR"`` (default), ``"linear"``,
        ``"BRR"``, ``"ARDR"``, ``"Random-Forest"``, ``"Quantile-Forest"``,
        ``"Neural-Net"``.
    plot_unit:
        ``"dollars"`` -- plot and model in USD (requires ``dollars_per_kwhr``
        and ``dollars_per_therm`` to be provided).
        ``"ekwh"``    -- plot and model in effective kWh.
        ``"auto"``    -- use ``"dollars"`` if rates are provided, otherwise
        ``"ekwh"``.  This is the default.
    dollars_per_kwhr:
        Electricity cost per kWh.  Required when ``plot_unit="dollars"``.
    dollars_per_therm:
        Gas cost per therm.  Required when ``plot_unit="dollars"``.
    freedom_units:
        If ``True``, temperature axes on all plots use Fahrenheit.
        The model is still trained on Celsius values internally; only the
        display is converted.  Defaults to ``False``.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from: {_VALID_METHODS}")

    # Resolve plotting unit
    use_dollars = (dollars_per_kwhr is not None) and (dollars_per_therm is not None)
    if plot_unit == "auto":
        plot_unit = "dollars" if use_dollars else "ekwh"
    if plot_unit == "dollars" and not use_dollars:
        raise ValueError(
            "plot_unit='dollars' requires both dollars_per_kwhr and "
            "dollars_per_therm to be provided."
        )

    temp_label = "Day Average Temperature (F)" if freedom_units else "Day Average Temperature (C)"

    # ------------------------------------------------------------------ #
    # Load and prepare data
    # ------------------------------------------------------------------ #
    collated_path = os.path.join(data_dir, "BGE_Collated.csv.gz")
    if not os.path.exists(collated_path):
        raise FileNotFoundError(
            f"BGE_Collated.csv.gz not found in {data_dir!r}. "
            "Run pybge.run() first."
        )

    thermo_hourly = pd.read_csv(collated_path)
    thermo_hourly["date"] = pd.to_datetime(thermo_hourly["date"])
    thermo_hourly = thermo_hourly.drop_duplicates(subset=["date"], keep="first")

    # Work at daily (6am) granularity, only where weather data is valid
    thermo_daily = thermo_hourly.loc[
        (thermo_hourly["date_hour"] == 6) & (~np.isnan(thermo_hourly["temp_p24"]))
    ].copy()

    # Optionally recompute ekwh with supplied rates
    if use_dollars:
        ekwh_per_therm = dollars_per_therm / dollars_per_kwhr
        thermo_daily["usage_ekwh_p24"] = (
            thermo_daily["usage_kwh_p24"] + ekwh_per_therm * thermo_daily["usage_therm_p24"]
        )

    col_target_base = "usage_ekwh_p24"
    col_target_unit_base = "ekWh"

    # ------------------------------------------------------------------ #
    # Split into training and prediction windows
    # ------------------------------------------------------------------ #
    thermo_daily = thermo_daily.reset_index(drop=True)
    required_cols = _COLS_TRAIN + [col_target_base]
    thermo_daily = thermo_daily.dropna(subset=required_cols)
    thermo_daily = thermo_daily[thermo_daily[col_target_base] > 0].reset_index(drop=True)

    cutoff_date = thermo_daily["date"].max() - pd.Timedelta(days=predict_days)
    thermo_prev = thermo_daily[thermo_daily["date"] <= cutoff_date].copy()
    thermo_curr = thermo_daily[thermo_daily["date"] > cutoff_date].copy()

    if len(thermo_prev) == 0:
        raise ValueError(
            f"No training data before cutoff ({cutoff_date.date()}). "
            "Reduce predict_days or collect more data."
        )

    # ------------------------------------------------------------------ #
    # Optionally convert target to dollars
    # ------------------------------------------------------------------ #
    if plot_unit == "dollars":
        thermo_prev["usage_usd_p24"] = thermo_prev[col_target_base] * dollars_per_kwhr
        thermo_curr["usage_usd_p24"] = thermo_curr[col_target_base] * dollars_per_kwhr
        col_target = ["usage_usd_p24"]
        col_target_unit = "USD"
    else:
        col_target = [col_target_base]
        col_target_unit = col_target_unit_base

    # ------------------------------------------------------------------ #
    # Build plot-facing temperature series (convert if freedom_units)
    # ------------------------------------------------------------------ #
    temp_prev = c_to_f(thermo_prev["temp_p24"]) if freedom_units else thermo_prev["temp_p24"]
    temp_curr = c_to_f(thermo_curr["temp_p24"]) if freedom_units else thermo_curr["temp_p24"]

    # ------------------------------------------------------------------ #
    # Save split CSVs
    # ------------------------------------------------------------------ #
    thermo_prev.to_csv(os.path.join(data_dir, "BGE_Correlate_Prev.csv"))
    thermo_curr.to_csv(os.path.join(data_dir, "BGE_Correlate_Curr.csv"))

    # ------------------------------------------------------------------ #
    # Relation plot (temperature vs usage)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    ax.scatter(
        temp_prev,
        thermo_prev[col_target],
        s=25,
        label="Training data",
        c=np.arange(len(thermo_prev)),
        cmap=cmocean.tools.crop_by_percent(cmocean.cm.algae, 45, which="both"),
    )
    ax.scatter(
        temp_curr,
        thermo_curr[col_target],
        s=25,
        label=f"Recent {predict_days} days",
        c=np.arange(len(thermo_curr)),
        cmap=cmocean.tools.crop_by_percent(cmocean.cm.dense, 45, which="both"),
    )
    ax.set_ylabel(f"Energy Use ({col_target_unit})", fontsize=15)
    ax.set_xlabel(temp_label, fontsize=15)
    lgd = ax.legend(loc="best", prop={"family": "sans", "size": 13}, borderpad=0.4)
    lgd.get_frame().set_color("white")
    relation_path = os.path.join(data_dir, "BGE_Correlate_Relation.png")
    fig.savefig(relation_path, dpi=175)
    plt.close(fig)
    print(f"Saved relation plot to {relation_path}")

    # ------------------------------------------------------------------ #
    # Fit model  (always on Celsius features)
    # ------------------------------------------------------------------ #
    features_scaler = sklearn.preprocessing.StandardScaler()
    features_scaler.fit(thermo_prev[_COLS_TRAIN])
    target_scaler = sklearn.preprocessing.StandardScaler()
    target_scaler.fit(thermo_prev[col_target])

    model = _build_model(method, n_features=len(_COLS_TRAIN))
    model_curr = copy.deepcopy(model)

    model.fit(
        features_scaler.transform(thermo_prev[_COLS_TRAIN]),
        target_scaler.transform(thermo_prev[col_target]),
    )
    if len(thermo_curr) > 0:
        model_curr.fit(
            features_scaler.transform(thermo_curr[_COLS_TRAIN]),
            target_scaler.transform(thermo_curr[col_target]),
        )

    # ------------------------------------------------------------------ #
    # Predictions and metrics
    # ------------------------------------------------------------------ #
    thermo_prev["predict_prev"] = pred_descale(
        thermo_prev[_COLS_TRAIN], model, features_scaler, target_scaler
    )
    if len(thermo_curr) > 0:
        thermo_curr["predict_curr"] = pred_descale(
            thermo_curr[_COLS_TRAIN], model, features_scaler, target_scaler
        )

    truth_prev = thermo_prev[col_target].values[:, 0]
    model_r2   = sklearn.metrics.r2_score(truth_prev, thermo_prev["predict_prev"])
    model_rmse = sklearn.metrics.root_mean_squared_error(truth_prev, thermo_prev["predict_prev"])
    model_mad  = sklearn.metrics.median_absolute_error(truth_prev, thermo_prev["predict_prev"])

    cv = sklearn.model_selection.ShuffleSplit(n_splits=20, test_size=0.1)
    Xtr = features_scaler.transform(thermo_prev[_COLS_TRAIN])
    ytr = target_scaler.transform(thermo_prev[col_target])[:, 0]
    cv_r2   = sklearn.model_selection.cross_val_score(model, Xtr, ytr, cv=cv, scoring="r2", n_jobs=-1)
    cv_rmse = sklearn.model_selection.cross_val_score(
        model, Xtr, ytr, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    ) * -1.0
    cv_mad  = sklearn.model_selection.cross_val_score(
        model, Xtr, ytr, cv=cv, scoring="neg_median_absolute_error", n_jobs=-1
    ) * -1.0

    print(f"R2  : full={model_r2:.3f}  CV={np.mean(cv_r2):.3f} +/- {np.std(cv_r2):.3f}")
    print(f"RMSE: full={model_rmse:.3f}  CV={np.mean(cv_rmse):.3f} +/- {np.std(cv_rmse):.3f}")
    print(f"MAD : full={model_mad:.3f}  CV={np.mean(cv_mad):.3f} +/- {np.std(cv_mad):.3f}")

    thermo_prev["resid_prev"] = thermo_prev["predict_prev"] - truth_prev
    error_prev = np.std(thermo_prev["resid_prev"][truth_prev > 0])

    offset_curr_abs = float("nan")
    offset_curr_percent = float("nan")
    if len(thermo_curr) > 0:
        truth_curr = thermo_curr[col_target].values[:, 0]
        thermo_curr["resid_curr"] = thermo_curr["predict_curr"] - truth_curr
        offset_curr_abs = float(np.median(thermo_curr["resid_curr"][truth_curr > 0]))
        offset_curr_percent = float(
            100.0 * np.sum(thermo_curr["resid_curr"]) / np.sum(thermo_curr["predict_curr"])
        )

    # ------------------------------------------------------------------ #
    # Prediction plot (predicted vs actual) -- no temperature axis here
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    ax.scatter(
        thermo_prev["predict_prev"],
        thermo_prev[col_target],
        s=25,
        label="Training data",
        c=np.arange(len(thermo_prev)),
        cmap=cmocean.tools.crop_by_percent(cmocean.cm.algae, 45, which="both"),
    )
    if len(thermo_curr) > 0:
        ax.scatter(
            thermo_curr["predict_curr"],
            thermo_curr[col_target],
            s=25,
            label=f"Recent {predict_days} days",
            c=np.arange(len(thermo_curr)),
            cmap=cmocean.tools.crop_by_percent(cmocean.cm.dense, 45, which="both"),
        )
    all_vals = np.concatenate([
        thermo_prev[col_target].values.flatten(),
        thermo_prev["predict_prev"].values,
    ])
    if len(thermo_curr) > 0:
        all_vals = np.concatenate([
            all_vals,
            thermo_curr[col_target].values.flatten(),
            thermo_curr["predict_curr"].values,
        ])
    ax_lim = int(np.ceil(1.1 * np.nanmax(all_vals)))
    ax.plot([0, ax_lim], [0, ax_lim], lw=1, ls=":", c="gray", label="1:1")
    ax.set_xlim(0, ax_lim)
    ax.set_ylim(0, ax_lim)
    ax.set_xlabel(f"Predicted Energy Use ({col_target_unit})", fontsize=15)
    ax.set_ylabel(f"Actual Energy Use ({col_target_unit})", fontsize=15)
    ax.text(0.02, 0.95, f"Training accuracy: +/- {error_prev:.2f} {col_target_unit}",
            fontsize=12, transform=ax.transAxes)
    if not np.isnan(offset_curr_abs):
        ax.text(0.02, 0.90,
                f"Recent change: {offset_curr_abs:.2f} {col_target_unit} ({offset_curr_percent:.0f}%)",
                fontsize=12, transform=ax.transAxes)
    ax.text(0.02, 0.85, f"Method: {method}", fontsize=12, transform=ax.transAxes)
    lgd = ax.legend(loc="lower right", prop={"family": "sans", "size": 13}, borderpad=0.4)
    lgd.get_frame().set_color("white")
    pred_path = os.path.join(data_dir, "BGE_Correlate_Prediction.png")
    fig.savefig(pred_path, dpi=175)
    plt.close(fig)
    print(f"Saved prediction plot to {pred_path}")

    # ------------------------------------------------------------------ #
    # Save model bundle
    # ------------------------------------------------------------------ #
    regr_dict = {
        "method":            method,
        "cols_train":        _COLS_TRAIN,
        "features_scaler":   features_scaler,
        "target_scaler":     target_scaler,
        "model_prev":        model,
        "model_curr":        model_curr,
        "col_target":        col_target,
        "col_target_unit":   col_target_unit,
        "dollars_per_kwhr":  dollars_per_kwhr,
        "dollars_per_therm": dollars_per_therm,
        "plot_unit":         plot_unit,
        "freedom_units":     freedom_units,
    }
    hkl_path = os.path.join(data_dir, "BGE_Regressor.hkl")
    hickle.dump(regr_dict, hkl_path)
    print(f"Saved model bundle to {hkl_path}")
    print("Correlate complete.  And furthermore, Carthage must be destroyed.")
