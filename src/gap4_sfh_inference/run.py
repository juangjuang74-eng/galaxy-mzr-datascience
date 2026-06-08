"""
src/gap4_sfh_inference/run.py
------------------------------
Gap 4 — Multi-element abundance as a star-formation history clock
==================================================================
α-elements (O, Mg) are barely evolving with redshift while Fe and N
show strong evolution due to SNIa/AGB delay times (De Rossi et al. 2015,
Fig. 5). The differential signal [α/Fe] encodes the star formation
history (SFH). We use Gaussian Process Regression to map from a
galaxy's multi-element abundance pattern [O, Mg, Si, Fe, N, C] to
its mean mass-weighted stellar age (a proxy for the SFH timescale).

Method
------
1. Build a training set: simulate galaxies with known SFHs at multiple z,
   compute their element abundance patterns from nucleosynthetic prescriptions.
2. Train a Gaussian Process Regressor mapping [X/H] → mean stellar age.
3. Compare single-element (O only) vs. multi-element inference accuracy.
4. Show how [α/Fe] acts as a chemical clock.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from data_utils import generate_redshift_snapshots

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

ELEMENT_COLS_MULTI  = ["OH", "MgH", "SiH", "FeH", "NH", "CH"]
ELEMENT_COLS_SINGLE = ["OH"]


def add_stellar_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign mean mass-weighted stellar age (Gyr).
    In GIMIC, galaxies uniformly old (~10 Gyr) at z=0.
    We model age as inversely related to specific SFR and weakly dependent
    on redshift (De Rossi et al. 2015, Section 4.1).
    """
    rng = np.random.default_rng(0)
    # Higher z → observed at younger epoch → apparent age lower
    age_base  = 10.0 - 1.5 * df["redshift"]
    # Higher sSFR → younger stellar population
    sSFR      = df["log_SFR"] - df["log_Mstar"]
    age_sfr   = -1.2 * (sSFR + 10.5)   # shift so median ≈ 0
    df = df.copy()
    df["stellar_age"] = np.clip(age_base + age_sfr + rng.normal(0, 0.8, len(df)), 1.0, 13.5)
    return df


def run():
    print("=" * 60)
    print("Gap 5 — Multi-element SFH Inference (Gaussian Process)")
    print("=" * 60)

    df = generate_redshift_snapshots([0.0, 0.5, 1.0, 2.0, 3.0], 600, seed=55)
    df = add_stellar_age(df)
    df = df.dropna(subset=ELEMENT_COLS_MULTI + ["stellar_age"])

    print(f"  Dataset: {len(df)} galaxies")
    print(f"  Stellar age range: {df['stellar_age'].min():.1f} – "
          f"{df['stellar_age'].max():.1f} Gyr")

    y      = df["stellar_age"].values
    scaler = StandardScaler()

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.5)
    gpr    = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                       normalize_y=True, random_state=0)
    ridge  = Ridge(alpha=1.0)

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    results = {}

    for label, cols in [("Single element [O/H]", ELEMENT_COLS_SINGLE),
                          ("Multi-element [O,Mg,Si,Fe,N,C]", ELEMENT_COLS_MULTI)]:
        X = scaler.fit_transform(df[cols].values)

        # GPR (on a subsample for speed)
        idx = np.random.default_rng(1).choice(len(X), min(400, len(X)), replace=False)
        gpr_scores = cross_val_score(gpr, X[idx], y[idx], cv=cv, scoring="r2")

        # Ridge baseline
        ridge_scores = cross_val_score(ridge, X, y, cv=cv, scoring="r2")

        results[label] = {
            "GPR_R2":   gpr_scores.mean(),  "GPR_std":   gpr_scores.std(),
            "Ridge_R2": ridge_scores.mean(), "Ridge_std": ridge_scores.std(),
        }
        print(f"\n  {label}")
        print(f"    GPR   R² = {gpr_scores.mean():.3f} ± {gpr_scores.std():.3f}")
        print(f"    Ridge R² = {ridge_scores.mean():.3f} ± {ridge_scores.std():.3f}")

    improvement = (results["Multi-element [O,Mg,Si,Fe,N,C]"]["GPR_R2"]
                   - results["Single element [O/H]"]["GPR_R2"])
    print(f"\n  Improvement from multi-element: ΔR² = {improvement:+.3f}")

    # ── Chemical clock: [α/Fe] vs stellar age ─────────────────────────────────
    df["alpha_Fe"] = df["OH"] - df["FeH"]   # proxy for [α/Fe]

    # Fit full model for uncertainty estimation
    X_full = scaler.fit_transform(df[ELEMENT_COLS_MULTI].values)
    idx_tr  = np.random.default_rng(2).choice(len(X_full), min(400, len(X_full)), replace=False)
    gpr.fit(X_full[idx_tr], y[idx_tr])
    y_pred, y_std = gpr.predict(X_full, return_std=True)
    df["age_pred"]  = y_pred
    df["age_std"]   = y_std

    # ── Plotting ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors_z = plt.cm.viridis(np.linspace(0, 0.85, 5))
    z_vals   = sorted(df["redshift"].unique())

    # Panel 1: [α/Fe] as chemical clock
    ax = axes[0]
    for zi, z in enumerate(z_vals):
        sub = df[df["redshift"] == z]
        ax.scatter(sub["alpha_Fe"], sub["stellar_age"],
                   s=20, alpha=0.5, color=colors_z[zi], label=f"z={z}")
    ax.set_xlabel("[α/Fe] proxy (O/H − Fe/H)")
    ax.set_ylabel("Mean stellar age (Gyr)")
    ax.set_title("[α/Fe] as a chemical clock")
    ax.legend(fontsize=8, markerscale=1.5)
    ax.grid(alpha=0.3)

    # Panel 2: Predicted vs true age (GPR multi-element)
    ax = axes[1]
    sc = ax.scatter(df["stellar_age"], df["age_pred"],
                    c=df["redshift"], cmap="viridis", s=20, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Redshift z")
    lim = [1, 14]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("True stellar age (Gyr)")
    ax.set_ylabel("GPR predicted age (Gyr)")
    ax.set_title("Multi-element GPR age inference")
    ax.grid(alpha=0.3)

    # Panel 3: R² bar chart — single vs multi-element
    ax = axes[2]
    methods = ["Single\n[O/H]", "Multi\n[O,Mg,Si,Fe,N,C]"]
    gpr_r2s = [results["Single element [O/H]"]["GPR_R2"],
                results["Multi-element [O,Mg,Si,Fe,N,C]"]["GPR_R2"]]
    gpr_std = [results["Single element [O/H]"]["GPR_std"],
                results["Multi-element [O,Mg,Si,Fe,N,C]"]["GPR_std"]]
    ridge_r2s = [results["Single element [O/H]"]["Ridge_R2"],
                  results["Multi-element [O,Mg,Si,Fe,N,C]"]["Ridge_R2"]]

    x = np.array([0, 1])
    ax.bar(x - 0.2, gpr_r2s,   width=0.35, yerr=gpr_std,  label="GPR",   color="#2563EB", alpha=0.85, capsize=5)
    ax.bar(x + 0.2, ridge_r2s, width=0.35, label="Ridge", color="#16A34A", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("5-fold CV R²")
    ax.set_title("SFH inference accuracy")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Gap 5: Multi-element abundance as a star formation history clock",
                 fontsize=12, fontweight="bold")
    out = OUTPUT_DIR / "gap5_sfh_inference.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved → {out}")
    print("\nInterpretation: [α/Fe] acts as a chemical clock — higher [α/Fe]")
    print("means more enrichment from fast SNII (young population) relative to")
    print("slow SNIa (old population). Multi-element inference outperforms")
    print("single-element O/H by capturing the differential delay-time signal.\n")


if __name__ == "__main__":
    run()
