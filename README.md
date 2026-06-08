# Galaxy Metallicity Scaling Relations — Data Science Repository

Addressing research gaps from **De Rossi et al. (2015)** *"The evolution of galaxy metallicity scaling relations in cosmological hydrodynamical simulations"* using modern data science methods.

## Research Gaps Addressed

| Gap | Topic | Method |
|-----|-------|--------|
| [1](#gap-1) | Observational selection bias in high-z MZR | Forward selection bias modeling |
| [2](#gap-2) | AGN feedback metallicity correction | Residual regression (GIMIC vs EAGLE) |
| [3](#gap-3) | Satellite vs central metallicity offset | Dimensionality reduction (UMAP/PCA) |
| [4](#gap-4) | Multi-element abundance as SFH clock | Sequence-based SFH inference |
| [5](#gap-5) | MZR scatter drivers | Gradient Boosting feature importance |

---

## Quick Start

```bash
pip install -r requirements.txt

# Run all s
python src/run.py

# Run individual gaps
python src/gap1_selection_bias/run.py
python src/gap2_agn_correction/run.py
python src/gap3_satellite_offset/run.py
python src/gap4_sfh_inference/run.py
python src/gap5_mzr_scatter/run.py
```

---

## Gap 1: Selection Bias in High-z MZR

**UV-limited surveys at high redshift preferentially select high-SFR galaxies.** This makes the MZR appear to evolve more strongly than it truly does. We forward-model which galaxies a UV-limited survey would detect at each redshift, then compare recovered vs. intrinsic MZR.

**Module:** `src/gap1_selection_bias/`

---

## Gap 2: AGN Feedback Metallicity Correction

**GIMIC overpredicts O/H at log M★ > 10.5** because it lacks AGN feedback. We train a residual regression model on the difference between GIMIC and EAGLE outputs (which include AGN) as a function of M★, SFR, and environment — producing an empirical correction function.

**Module:** `src/gap2_agn_correction/`

---

## Gap 3: Satellite vs Central Metallicity Offset

**Satellites are 0.1–0.2 dex more metal-rich than centrals at fixed M★.** We apply PCA and UMAP to halo properties (Vcirc, gas fraction, ram-pressure proxies, infall time) to find the dominant axis driving this offset.

**Module:** `src/gap3_satellite_offset/`

---

## Gap 4: Multi-Element Abundance as SFH Clock

**α-elements (O, Mg) barely evolve with redshift** while Fe and N show strong evolution due to SNIa/AGB delay times. We use this differential signal to infer star formation histories from multi-element abundance patterns using a Gaussian Process model.

**Module:** `src/gap4_sfh_inference/`

---

## Gap 5: Drivers of MZR Scatter

**At fixed M★, galaxies with deeper potential wells (higher Vcirc) are more metal-poor.** We train a Gradient Boosting model predicting O/H from {M★, Vcirc, fg, R₅₀, SFR} and use SHAP values to rank the physical drivers of scatter.

**Module:** `src/gap5_mzr_scatter/`

---

## Data

All modules use synthetic data that mirrors the statistics of GIMIC/SDSS. To plug in real data:

- **SDSS MPA-JHU catalog**: https://www.mpa-garching.mpg.de/SDSS/
- **EAGLE public data**: https://icc.dur.ac.uk/Eagle/
- Replace `generate_synthetic_*()` calls in each module with your loader.

---

## Improvements & Future Work

- Add `pyproject.toml` or `environment.yml` for better reproducibility.
- Include unit tests for data generators.
- Consider interactive dashboards (Streamlit/Plotly).
- Add support for real catalog loaders (`astropy`).

---

## Citation

*The evolution of galaxy metallicity scaling relations in cosmological hydrodynamical simulations.*  
De Rossi et al. (2015) — arXiv:1506.02772

---

**License**: MIT
