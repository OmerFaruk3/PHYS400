# PHYS400 — Information Gain in Gravitational Wave Parameter Estimation

**METU Physics Senior Project (2025–2026)**  
**Student:** Ömer Faruk Soydan

## Project Summary

This project investigates how much information is gained during Bayesian parameter estimation of gravitational wave (GW) events, using KL divergence (D_KL) between prior and posterior distributions from LIGO/Virgo observations.

We compare multiple KL divergence estimation methods:
- **Gaussian approximation** (analytical)
- **kNN-based** non-parametric estimator
- **KDE-based** estimator
- **Binning-based** estimator
- **MINE** (Mutual Information Neural Estimator)

Analysis covers GW150914 and a catalog of ~30 GWTC-3 events across 15 binary black hole parameters.

## Repository Structure

```
PHYS400/
├── code/
│   ├── exploration/       # Early-stage scripts & notebooks
│   ├── knn_analysis/      # Main kNN KL divergence analysis
│   └── group_kld/         # Group KLD hybrid estimator (catalog analysis)
├── plots/
│   ├── prior_posterior/   # Prior vs posterior histograms
│   ├── gaussian_kl/       # Gaussian KL comparison plots
│   ├── group_kld/         # Per-event group KLD plots (GWTC-3 catalog)
│   └── fisher/            # Fisher information / eigenvalue plots
├── results/
│   ├── gaussian_kl/       # CSV & JSON results (Gaussian KL)
│   └── group_kld/         # CSV & JSON results (catalog KLD)
├── reports/               # LaTeX source, PDFs (proposal, interim, final, poster)
└── docs/                  # Analysis notes, project summaries, talk outline
```

## Key Files

| File | Description |
|------|-------------|
| `code/knn_analysis/gw_knn_kl_divergence.py` | Main kNN KL divergence estimator |
| `code/knn_analysis/gw150914_knn_information.py` | GW150914 detailed analysis |
| `code/group_kld/gw_grup_kld_hibrit.py` | Hybrid group KLD estimator |
| `code/group_kld/gw_kld_hibrit_oto_katalog.py` | Automated catalog runner |
| `results/group_kld/oto_master_ozet.csv` | Summary results for all events |
| `reports/phys400_final_report.tex` | Final report LaTeX source |

## Data

GW posterior samples (`.h5` files from IGWN/GWTC) are **not included** in this repository due to file size. Download from:
- [GWTC-2.1](https://zenodo.org/record/6513631)
- [GWTC-3](https://zenodo.org/record/8177023)

Place downloaded `.h5` files in `code/knn_analysis/data/` or `code/group_kld/data/`.

## Requirements

```bash
pip install numpy scipy bilby pesummary h5py matplotlib seaborn scikit-learn
```

## References

- Shannon (1948) — *A Mathematical Theory of Communication*
- Abbott et al. (2023) — *GWTC-3: Compact Binary Coalescences Observed by LIGO and Virgo* (PhysRevX.13.011048)
- Veitch & Vecchio (2010) — Bayesian coherent analysis of GW signals
