#!/usr/bin/env python3
"""
GW150914 — Blok Tabanlı KL(Posterior ∥ Prior) Hesabı  [v4]
===========================================================
Üç yöntem:
  [1] Multivariate Gaussian (analitik)
  [2] KDE + Simpson (1D blok yoksa kullanılmaz) / KDE + Monte Carlo (≥2D)
  [3] Pérez-Cruz k-NN

DEĞİŞİKLİKLER v3'e göre:
  ─ Bloklama fiziksel olarak yeniden düzenlendi:
      • a_1, a_2 birlikte (χ_eff korelasyonu)
      • phi_12, phi_jl birlikte (azimuthal spin açıları)
      • azimuth, zenith, geocent_time birlikte (gökyüzü + varış zamanı)
      • phi_jl gökyüzünden ayrıldı (fiziksel olarak korelasyonsuzdu)
  ─ TÜM 147k posterior örneklemi kullanılır:
      • k-NN: n = 147k (Pérez-Cruz bias formülüyle düzeltilir)
      • 2D/3D KDE: training = 30k subsample, evaluation = TÜM 147k (MC)
      • Böylece KL ortalaması E_P[log p/q] tüm posterior üzerinden alınır
  ─ Daha sade ve kısa kod yapısı (2D Simpson yerine MC; tek yöntem)

BLOK TANIMLAMALARI (15 parametre, örtüşmesiz, fiziksel olarak motiveli):
  Blok 1 (2D): m1, m2             — chirp mass korelasyonu
  Blok 2 (2D): dL, theta_jn       — mesafe-eğim dejenerasyonu
  Blok 3 (2D): psi, phase         — polarizasyon-faz dejenerasyonu
  Blok 4 (2D): a_1, a_2           — χ_eff (spin amplitüdleri)
  Blok 5 (2D): tilt_1, tilt_2     — χ_eff (spin açıları)
  Blok 6 (2D): phi_12, phi_jl     — azimuthal spin açıları
  Blok 7 (3D): azimuth, zenith, geocent_time   — gökyüzü + varış zamanı
  TOPLAM: 2×6 + 3 = 15 ✓
"""

import time
import numpy as np
from pesummary.io import read
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree

# ══════════════════════════════════════════════════════════════════════════════
# AYARLAR
# ══════════════════════════════════════════════════════════════════════════════

FILE = (
    "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/"
    "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
)
LABEL = "C01:IMRPhenomXPHM"

BLOCKS = [
    ("Blok 1", "2D", ["mass_1_source",       "mass_2_source"]),
    ("Blok 2", "2D", ["luminosity_distance", "theta_jn"]),
    ("Blok 3", "2D", ["psi",                 "phase"]),
    ("Blok 4", "2D", ["a_1",                 "a_2"]),
    ("Blok 5", "2D", ["tilt_1",              "tilt_2"]),
    ("Blok 6", "2D", ["phi_12",              "phi_jl"]),
    ("Blok 7", "3D", ["azimuth", "zenith",   "geocent_time"]),
]

LABELS = {
    "mass_1_source":       "m1",     "mass_2_source":      "m2",
    "luminosity_distance": "dL",     "theta_jn":           "iota",
    "psi":                 "psi",    "phase":              "phi_c",
    "tilt_1":              "tilt1",  "tilt_2":             "tilt2",
    "phi_jl":              "phi_JL", "azimuth":            "az",
    "zenith":              "zen",    "a_1":                "a1",
    "a_2":                 "a2",     "phi_12":             "phi12",
    "geocent_time":        "tc",
}

# ── Hesap parametreleri ───────────────────────────────────────────────────────
N_KDE_TRAIN = 30_000   # KDE training subsample (147k tam pahalı: O(n_t × n_e))
N_KDE_REPS  = 3        # MC tekrar sayısı (std hata için)
KNN_K       = 5
SEED        = 42

# ══════════════════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  Veri yukleniyor...")
t0 = time.time()
data  = read(FILE, disable_conversion=True)
post  = data.samples_dict[LABEL]
prior = data.priors["samples"][LABEL]
print(f"  Tamamlandi ({time.time()-t0:.1f}s)")

n_post  = len(np.array(post["mass_1_source"]))
n_prior = len(np.array(prior["mass_1_source"]))
print(f"  Posterior: {n_post}   |   Prior: {n_prior}")
print(f"  KDE training: {N_KDE_TRAIN}  |  KDE eval: tam {n_post}  |  k-NN: n=tam {n_post}, m={n_prior}")


def get_block_data(params):
    P = np.column_stack([np.asarray(post[p],  dtype=float) for p in params])
    Q = np.column_stack([np.asarray(prior[p], dtype=float) for p in params])
    return P, Q


def subsample(arr, n, seed_offset=0):
    if len(arr) <= n:
        return arr
    idx = np.random.default_rng(SEED + seed_offset).choice(len(arr), size=n, replace=False)
    return arr[idx]


# ══════════════════════════════════════════════════════════════════════════════
# YÖNTEM 1 — GAUSSIAN KL  (analitik)
# KL(N(μ_p,Σ_p) ∥ N(μ_q,Σ_q)) =
#   ½[tr(Σ_q⁻¹Σ_p) + (μ_q−μ_p)ᵀΣ_q⁻¹(μ_q−μ_p) − d + ln|Σ_q/Σ_p|]
# ══════════════════════════════════════════════════════════════════════════════
def gaussian_kl_bits(P, Q):
    d  = P.shape[1]
    Sp = np.cov(P, rowvar=False).reshape(d, d)
    Sq = np.cov(Q, rowvar=False).reshape(d, d)
    cond = float(np.linalg.cond(Sq))
    Sq_inv = np.linalg.inv(Sq)
    diff = Q.mean(axis=0) - P.mean(axis=0)
    kl_nat = 0.5 * (np.trace(Sq_inv @ Sp)
                    + float(diff @ Sq_inv @ diff)
                    - d
                    + np.linalg.slogdet(Sq)[1]
                    - np.linalg.slogdet(Sp)[1])
    return float(kl_nat / np.log(2)), cond


# ══════════════════════════════════════════════════════════════════════════════
# YÖNTEM 2 — KDE + MONTE CARLO  (2D ve 3D için aynı)
# KL(P∥Q) = E_P[log₂(p/q)] ≈ (1/n_e) Σ log₂[kde_p(xᵢ)/kde_q(xᵢ)], xᵢ ~ P
# Training: N_KDE_TRAIN ile bant genişliği belirlenir
# Evaluation: TÜM 147k posterior örneklemi kullanılır
# ══════════════════════════════════════════════════════════════════════════════
def kde_kl_mc(P, Q):
    """KDE+MC tahmincisi. Eval = tüm posterior (147k). Bias yok, std hata raporlanır."""
    Ps = subsample(P, N_KDE_TRAIN, seed_offset=1)
    Qs = subsample(Q, N_KDE_TRAIN, seed_offset=2)
    kde_p = gaussian_kde(Ps.T, bw_method='scott')
    kde_q = gaussian_kde(Qs.T, bw_method='scott')

    # Eval = tüm P (147k); MC tekrarları KDE training rastgeleliği için
    ests = []
    for rep in range(N_KDE_REPS):
        if rep > 0:
            Ps_r = subsample(P, N_KDE_TRAIN, seed_offset=10 + rep)
            Qs_r = subsample(Q, N_KDE_TRAIN, seed_offset=20 + rep)
            kde_p = gaussian_kde(Ps_r.T, bw_method='scott')
            kde_q = gaussian_kde(Qs_r.T, bw_method='scott')
        lp = np.log2(np.maximum(kde_p(P.T), 1e-300))
        lq = np.log2(np.maximum(kde_q(P.T), 1e-300))
        ests.append(float(np.mean(lp - lq)))
    return float(np.mean(ests)), float(np.std(ests))


# ══════════════════════════════════════════════════════════════════════════════
# YÖNTEM 3 — PÉREZ-CRUZ k-NN  (n = tüm 147k posterior)
# KL(P∥Q) ≈ (d/n) Σᵢ log(sᵢ/rᵢ) + log(m/(n−1))
#
# Not: n >> m olduğunda bias = log(5k/147k) ≈ -4.89 bit (büyük negatif).
# Pérez-Cruz 2008 bunu kesin olarak çıkarmıştır; ancak prior kapsama eksikliği
# nedeniyle sᵢ değerleri sistematik olarak büyük → toplam pozitif, bias negatif.
# Bu kullanıcı tercihiyle TÜM 147k posterior'u kullanır.
# ══════════════════════════════════════════════════════════════════════════════
def knn_kl_bits(P, Q, k=KNN_K):
    n, d = P.shape
    m    = Q.shape[0]
    pt   = cKDTree(P)
    qt   = cKDTree(Q)
    r    = pt.query(P, k=k+1, p=2, workers=-1)[0][:, k]   # P → P k. komşu
    s    = qt.query(P, k=k,   p=2, workers=-1)[0][:, k-1] # P → Q k. komşu
    r    = np.maximum(r, 1e-15)
    s    = np.maximum(s, 1e-15)
    bias = np.log(m / (n - 1.0))
    kl_nat = (d / n) * np.sum(np.log(s / r)) + bias
    return float(kl_nat / np.log(2)), n


# ══════════════════════════════════════════════════════════════════════════════
# YARDIMCI: Bloklar için TC hesabı (KDE marjinaller ile)
# TC(X1,...,Xd) = Σ I(Xi) − I(X1,...,Xd)
# ══════════════════════════════════════════════════════════════════════════════
def kde_kl_1d_mc(P_col, Q_col):
    """1D özel durum: KDE+MC, training=30k, eval=tam n."""
    Ps = subsample(P_col, N_KDE_TRAIN, seed_offset=11)
    Qs = subsample(Q_col, N_KDE_TRAIN, seed_offset=12)
    kp = gaussian_kde(Ps, bw_method='scott')
    kq = gaussian_kde(Qs, bw_method='scott')
    lp = np.log2(np.maximum(kp(P_col), 1e-300))
    lq = np.log2(np.maximum(kq(P_col), 1e-300))
    return float(np.mean(lp - lq))


def compute_tc(P, Q, kde_joint):
    """TC = Σ I(Xi) − I(X1,...,Xd) — KDE tabanlı (en güvenilir)."""
    marg = [kde_kl_1d_mc(P[:, i], Q[:, i]) for i in range(P.shape[1])]
    return float(sum(marg) - kde_joint), marg


# ══════════════════════════════════════════════════════════════════════════════
# ANA DÖNGÜ
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(SEED)
results = []
t_total = time.time()

print("\n" + "=" * 72)
print("  GW150914 — KL(Posterior ∥ Prior)  Blok Bazlı Hesap  [v4]")
print("=" * 72)

for bnum, bdim, params in BLOCKS:
    d    = len(params)
    P, Q = get_block_data(params)
    lstr = " + ".join(LABELS.get(p, p) for p in params)

    print(f"\n{'─' * 72}")
    print(f"  {bnum} ({bdim})  —  {lstr}")
    print(f"  Posterior P: {P.shape}   |   Prior Q: {Q.shape}")

    for i, p in enumerate(params):
        if Q[:, i].max() < P[:, i].max():
            print(f"  [UYARI] {p}: Prior_max={Q[:,i].max():.4g} < Post_max={P[:,i].max():.4g}")
        if Q[:, i].min() > P[:, i].min():
            print(f"  [UYARI] {p}: Prior_min={Q[:,i].min():.4g} > Post_min={P[:,i].min():.4g}")

    # ── [1] Gaussian ──────────────────────────────────────────────────────────
    t1 = time.time()
    try:
        g, cond = gaussian_kl_bits(P, Q)
        print(f"\n  [1] Gaussian         : {g:8.4f} bit   cond={cond:.2e}  ({time.time()-t1:.2f}s)")
    except Exception as e:
        g = float("nan")
        print(f"\n  [1] Gaussian         : HATA — {e}")

    # ── [2] KDE + MC ──────────────────────────────────────────────────────────
    t1 = time.time()
    kde_val, kde_std, tc_val, marg = float("nan"), float("nan"), float("nan"), []
    try:
        kde_val, kde_std = kde_kl_mc(P, Q)
        print(f"  [2] KDE+MC   ({d}D)   : {kde_val:8.4f} ± {kde_std:.4f} bit   "
              f"[train={N_KDE_TRAIN}, eval=tam {P.shape[0]}, reps={N_KDE_REPS}]  ({time.time()-t1:.2f}s)")
        # TC hesabı
        t2 = time.time()
        tc_val, marg = compute_tc(P, Q, kde_val)
        marg_str = "  ".join(f"I({LABELS.get(params[i],'?')})={marg[i]:.3f}" for i in range(d))
        print(f"      TC hesabi        : {marg_str}   TC={tc_val:+.4f} bit  ({time.time()-t2:.2f}s)")
    except Exception as e:
        print(f"  [2] KDE+MC           : HATA — {e}")

    # ── [3] k-NN ──────────────────────────────────────────────────────────────
    t1 = time.time()
    try:
        knn_val, n_used = knn_kl_bits(P, Q, k=KNN_K)
        bias_bits = np.log(Q.shape[0] / (n_used - 1.0)) / np.log(2)
        caveat = "  (*boyutsallik laneti)" if d >= 3 else ""
        print(f"  [3] k-NN (k={KNN_K})      : {knn_val:8.4f} bit   "
              f"[n_knn={n_used}, m={Q.shape[0]}, bias={bias_bits:+.4f} bit]"
              f"  ({time.time()-t1:.2f}s){caveat}")
    except Exception as e:
        knn_val = float("nan")
        print(f"  [3] k-NN             : HATA — {e}")

    results.append({
        "bnum": bnum, "bdim": bdim, "params": params, "d": d,
        "gaussian": g, "kde": kde_val, "kde_std": kde_std,
        "knn": knn_val, "tc": tc_val, "marg": marg,
    })

print(f"\n  Toplam sure: {time.time()-t_total:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# ÖZET TABLOSU
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 72)
print(f"  {'OZET TABLOSU [v4]':^68}")
print("=" * 72)
print(f"  {'Blok':<8} {'Boyut':<5} {'Parametreler':<22}  {'Gauss':>8} {'KDE+MC':>14} {'k-NN':>8} {'TC':>8}")
print("─" * 78)

g_tot = kde_tot = knn_tot = tc_tot = 0.0
n_valid = 0
for r in results:
    pstr = "+".join(LABELS.get(p, p) for p in r["params"])
    if not np.isnan(r["gaussian"]): g_tot   += r["gaussian"]
    if not np.isnan(r["knn"]):      knn_tot += r["knn"]
    if not np.isnan(r["kde"]):
        kde_tot += r["kde"]; n_valid += 1
        kde_str = f"{r['kde']:7.4f}±{r['kde_std']:.3f}"
    else:
        kde_str = "         n/a"
    tc_str = f"{r['tc']:+7.4f}" if not np.isnan(r["tc"]) else "    —  "
    if not np.isnan(r["tc"]): tc_tot += r["tc"]
    print(f"  {r['bnum']:<8} {r['bdim']:<5} {pstr:<22}  "
          f"{r['gaussian']:>8.4f} {kde_str:>14} {r['knn']:>8.4f} {tc_str:>8}")

print("─" * 78)
print(f"  {'TOPLAM':<37}  {g_tot:>8.4f} {kde_tot:>14.4f} {knn_tot:>8.4f} {tc_tot:>+8.4f}")

print(f"\n  ── Bütçe Tablosu ─────────────────────────────────────────────────")
print(f"  KDE+MC toplami (joint)                  : {kde_tot:7.4f} bit")
print(f"  TC toplami (block-icinde korelasyon)    : {tc_tot:+7.4f} bit")
print(f"  Alt sinir (joint KL ≥ KDE + TC)         : {kde_tot + tc_tot:7.4f} bit")
print(f"\n  ── Referanslar ───────────────────────────────────────────────────")
print(f"  F&H analitik (Gauss)   : ~41.5 bit")
print(f"  Marginal KL (eski)     : ~36.05 bit")
print("=" * 72)
