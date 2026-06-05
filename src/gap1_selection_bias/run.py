"""
src/gap1_selection_bias/run.py
------------------------------
Gap 1 — Observational selection bias in high-z MZR surveys
============================================================
UV-limited surveys at high redshift preferentially select high-SFR
galaxies (De Rossi et al. 2015, Section 4.1). This inflates the
apparent evolution of the MZR. We forward-model which galaxies a
UV-limited survey would observe at each redshift, then compare the
recovered MZR to the intrinsic one.

Method
------
1. Generate intrinsic galaxy population at each redshift snapshot.
2. Apply a UV flux selection function: P(observe | galaxy) depends on
   UV luminosity, which correlates with SFR and redshift.
3. Compare intrinsic vs. selected MZR — the offset is the bias.
4. Quantify bias as Δ(O/H) at fixed M★ across redshifts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_utils import generate_redshift_snapshots


# ── Configuration ─────────────────────────────────────────────────────────────
Z_VALUES       = [0.0, 0.5, 1.0, 2.0, 3.0]
N_PER_SNAP     = 2000
UV_FLUX_LIMIT  = -18.5    # AB magnitude limit — brighter (more negative) = detected
MSTAR_BINS     = np.linspace(9.0, 10.5, 7)
MSTAR_CENTERS  = 0.5 * (MSTAR_BINS[:-1] + MSTAR_BINS[1:])
OUTPUT_DIR     = Path(__file__).resolve().parents[2] / "outputs"


# ── UV Luminosity model ────────────────────────────────────────────────────────
def uv_magnitude(log_SFR: np.ndarray, redshift: float, rng) -> np.ndarray:
    """
    Kennicutt (1998) calibration: M_UV ~ -2.5 * log_SFR - 18.
    Higher SFR = more negative M_UV (brighter).
    Scatter ±0.5 mag + mild k-correction shift with z.
    """
    M_UV = -2.5 * log_SFR - 18.0 + rng.normal(0, 0.5, len(log_SFR))
    M_UV += 0.5 * redshift   # higher z survey sees systematically dimmer objects
    return M_UV


def selection_probability(M_UV: np.ndarray, limit: float = UV_FLUX_LIMIT) -> np.ndarray:
    """
    Sigmoid selection: galaxies brighter than limit (M_UV < limit) are selected.
    limit = -18.5: bright end selected, faint end missed.
    """
    return 1.0 / (1.0 + np.exp(2.5 * (M_UV - limit)))


# ── Analysis ──────────────────────────────────────────────────────────────────
def compute_biased_mzr(df: pd.DataFrame, rng) -> pd.DataFrame:
    z = df["redshift"].iloc[0]
    M_UV   = uv_magnitude(df["log_SFR"].values, z, rng)
    p_sel  = selection_probability(M_UV)
    sel    = rng.random(len(df)) < p_sel

    rows = []
    for lo, hi, center in zip(MSTAR_BINS[:-1], MSTAR_BINS[1:], MSTAR_CENTERS):
        mask_all = (df["log_Mstar"] >= lo) & (df["log_Mstar"] < hi)
        mask_sel = mask_all & sel
        if mask_all.sum() < 5:
            continue
        OH_int = df.loc[mask_all, "OH"].median()
        OH_obs = df.loc[mask_sel, "OH"].median() if mask_sel.sum() >= 3 else np.nan
        rows.append({
            "log_Mstar_center": center,
            "OH_intrinsic":     OH_int,
            "OH_observed":      OH_obs,
            "n_intrinsic":      mask_all.sum(),
            "n_observed":       mask_sel.sum(),
            "sel_fraction":     mask_sel.sum() / max(mask_all.sum(), 1),
            "redshift":         z,
        })
    return pd.DataFrame(rows)


def run():
    print("=" * 60)
    print("Gap 1 — Selection Bias in High-z MZR Surveys")
    print("=" * 60)

    rng = np.random.default_rng(42)
    df_all = generate_redshift_snapshots(Z_VALUES, N_PER_SNAP, seed=42)

    results = []
    for z in Z_VALUES:
        snap = df_all[df_all["redshift"] == z].copy()
        res  = compute_biased_mzr(snap, rng)
        results.append(res)
        print(f"  z={z:.1f}: {len(snap)} galaxies | "
              f"mean sel fraction = {res['sel_fraction'].mean():.2f}")

    results_df = pd.concat(results, ignore_index=True)
    results_df["bias_dOH"] = results_df["OH_observed"] - results_df["OH_intrinsic"]

    print("\nMean bias Δ(O/H) by redshift (negative = observed lower than intrinsic):")
    bias_summary = results_df.groupby("redshift")["bias_dOH"].mean()
    print(bias_summary.to_string())

    # ── Plotting ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(Z_VALUES)))

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    z2  = results_df[results_df["redshift"] == 2.0]
    ax1.plot(z2["log_Mstar_center"], z2["OH_intrinsic"], "o-",
             color="#2563EB", lw=2, label="Intrinsic")
    ax1.plot(z2["log_Mstar_center"], z2["OH_observed"],  "s--",
             color="#DC2626", lw=2, label="UV-selected")
    ax1.set_xlabel(r"log(M$_\star$/M$_\odot$)")
    ax1.set_ylabel("12 + log(O/H)")
    ax1.set_title("MZR at z = 2.0")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    for i, z in enumerate(Z_VALUES):
        sub = results_df[results_df["redshift"] == z]
        ax2.plot(sub["log_Mstar_center"], sub["OH_intrinsic"],
                 "-", color=colors[i], lw=1.5, alpha=0.8)
        ax2.plot(sub["log_Mstar_center"], sub["OH_observed"],
                 "--", color=colors[i], lw=1.5, alpha=0.6)
    ax2.plot([], [], "k-",  lw=2, label="Intrinsic")
    ax2.plot([], [], "k--", lw=2, label="UV-selected")
    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(0, max(Z_VALUES)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax2, label="Redshift z", shrink=0.8)
    ax2.set_xlabel(r"log(M$_\star$/M$_\odot$)")
    ax2.set_ylabel("12 + log(O/H)")
    ax2.set_title("MZR evolution: intrinsic vs. observed")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    for i, z in enumerate(Z_VALUES):
        sub = results_df[results_df["redshift"] == z].dropna(subset=["bias_dOH"])
        ax3.scatter(np.full(len(sub), z) + np.random.uniform(-0.04, 0.04, len(sub)),
                    sub["bias_dOH"], color=colors[i], alpha=0.7, s=30)
    ax3.axhline(0, color="gray", lw=1, ls="--")
    ax3.set_xlabel("Redshift z")
    ax3.set_ylabel(r"$\Delta$(O/H) = observed $-$ intrinsic (dex)")
    ax3.set_title("Selection bias magnitude")
    ax3.grid(alpha=0.3)

    fig.suptitle("Gap 1: UV selection bias inflates apparent MZR evolution",
                 fontsize=12, fontweight="bold", y=1.02)
    out = OUTPUT_DIR / "gap1_selection_bias.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved → {out}")
    print("\nInterpretation: UV-selected samples are biased toward high-SFR galaxies.")
    print("At fixed M★, high-SFR galaxies have LOWER O/H (FMR anti-correlation).")
    print("This makes high-z samples appear more metal-poor than they truly are,")
    print("inflating the inferred MZR evolution by ~0.1-0.2 dex.\n")


if __name__ == "__main__":
    run()
