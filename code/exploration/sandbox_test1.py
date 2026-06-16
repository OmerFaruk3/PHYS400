#!/usr/bin/env python3
"""
GW150914 — Blok Tabanlı KL(Posterior ∥ Prior) Hesabı
=====================================================
Üç yöntem:
  [1] Multivariate Gaussian  (analitik)
  [2] KDE + Simpson          (1D/2D)  |  KDE + Monte Carlo (4D)
  [3] Pérez-Cruz k-NN (k=5)

Blok Tanımları (toplam 15 parametre, örtüşmesiz):
  Blok 1 (2D): mass_1_source, mass_2_source
  Blok 2 (1D): luminosity_distance       ← azimuth Blok 5'e taşındı
  Blok 3 (2D): psi, phase
  Blok 4 (2D): tilt_1, tilt_2
  Blok 5 (4D): phi_jl, theta_jn, azimuth, zenith
  Blok 6 (1D): a_1
  Blok 7 (1D): a_2
  Blok 8 (1D): phi_12
  Blok 9 (1D): geocent_time

Önceden hesaplanan TC değerleri (Blok 1-4):
  Blok 1: TC = 0.55 bit
  Blok 2: TC = 0 bit (1D → TC tanımsız, sıfır alınır)
  Blok 3: TC = 0.27 bit
  Blok 4: TC = 0.19 bit
"""

import numpy as np
from pesummary.io import read
from scipy.stats import gaussian_kde
from scipy.integrate import simpson
from scipy.spatial import cKDTree

# ══════════════════════════════════════════════════════════════════════════════
# AYARLAR
# ══════════════════════════════════════════════════════════════════════════════

FILE = (
    "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/"
    "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
)
LABEL = "C01:IMRPhenomXPHM"

# Blok tanımları: (isim, boyut etiketi, parametre listesi)
BLOCKS = [
    ("Blok 1", "2D", ["mass_1_source",   "mass_2_source"]),
    ("Blok 2", "1D", ["luminosity_distance"]),
    ("Blok 3", "2D", ["psi",             "phase"]),
    ("Blok 4", "2D", ["tilt_1",          "tilt_2"]),
    ("Blok 5", "4D", ["phi_jl",          "theta_jn",  "azimuth", "zenith"]),
    ("Blok 6", "1D", ["a_1"]),
    ("Blok 7", "1D", ["a_2"]),
    ("Blok 8", "1D", ["phi_12"]),
    ("Blok 9", "1D", ["geocent_time"]),
]

# Görüntü etiketleri
LABELS = {
    "mass_1_source":      "m1",
    "mass_2_source":      "m2",
    "luminosity_distance":"dL",
    "psi":                "psi",
    "phase":              "phi_c",
    "tilt_1":             "tilt1",
    "tilt_2":             "tilt2",
    "phi_jl":             "phi_JL",
    "theta_jn":           "iota",
    "azimuth":            "az",
    "zenith":             "zen",
    "a_1":                "a1",
    "a_2":                "a2",
    "phi_12":             "phi12",
    "geocent_time":       "tc",
}

# Önceden hesaplanan TC değerleri (Blok 1-4 için 2D KDE+Simpson sonuçları)
# Blok 2 artık 1D olduğu için TC = 0
TC_KNOWN = {
    "Blok 1": 0.55,
    "Blok 2": 0.00,   # 1D → TC = 0
    "Blok 3": 0.27,
    "Blok 4": 0.19,
}

KDE_GRID    = 400      # Simpson ızgara çözünürlüğü (1D ve 2D için)
KNN_K       = 5        # k-NN komşu sayısı
MC_N        = 30_000   # Monte Carlo örnek sayısı (4D KDE için)
MC_REPS     = 5        # Monte Carlo tekrar sayısı (varyans tahmini)
SEED        = 42


# ══════════════════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("  Veri yukleniyor...")
print("=" * 72)
data  = read(FILE, disable_conversion=True)
post  = data.samples_dict[LABEL]
prior = data.priors["samples"][LABEL]

n_post  = len(np.array(post["mass_1_source"]))
n_prior = len(np.array(prior["mass_1_source"]))
print(f"  Posterior: {n_post} ornek   |   Prior: {n_prior} ornek")


def get_block_data(params):
    """Parametre listesine gore (n, d) boyutlu P ve Q matrislerini dondur."""
    if len(params) == 1:
        P = np.array(post[params[0]])[:, None]   # (n, 1)
        Q = np.array(prior[params[0]])[:, None]  # (m, 1)
    else:
        P = np.column_stack([np.array(post[p])  for p in params])
        Q = np.column_stack([np.array(prior[p]) for p in params])
    return P, Q


# ══════════════════════════════════════════════════════════════════════════════
# YÖNTEM 1 — MULTİVARİATE GAUSSIAN KL
# ══════════════════════════════════════════════════════════════════════════════
# KL(N(μ_p, Σ_p) ∥ N(μ_q, Σ_q))
#   = ½ [tr(Σ_q⁻¹ Σ_p) + (μ_q−μ_p)ᵀ Σ_q⁻¹ (μ_q−μ_p) − d + ln|Σ_q|/|Σ_p|]
# Referans: Flanagan & Hughes 1998, Eq. 6.8 (yüksek SNR limiti)

def gaussian_kl_bits(P, Q):
    """
    Dondurur
    --------
    kl   : KL divergence [bit]
    cond : prior kovaryan matrisinin kondisyon sayısı (ill-conditioning uyarisi)
    """
    d      = P.shape[1]
    mu_p   = P.mean(axis=0)
    mu_q   = Q.mean(axis=0)
    Sp     = np.cov(P, rowvar=False).reshape(d, d)  # (d,d) — d=1 icin scalar → reshape
    Sq     = np.cov(Q, rowvar=False).reshape(d, d)
    cond   = float(np.linalg.cond(Sq))
    Sq_inv = np.linalg.inv(Sq)
    diff   = mu_q - mu_p
    term1  = np.trace(Sq_inv @ Sp)
    term2  = float(diff @ Sq_inv @ diff)
    _, ldp = np.linalg.slogdet(Sp)
    _, ldq = np.linalg.slogdet(Sq)
    kl_nat = 0.5 * (term1 + term2 - d + ldq - ldp)
    return float(kl_nat / np.log(2)), cond


# ══════════════════════════════════════════════════════════════════════════════
# YÖNTEM 2 — KDE + SİMPSON (1D / 2D)
# ══════════════════════════════════════════════════════════════════════════════
# KL(P∥Q) = ∫ p(x) log₂[p(x)/q(x)] dx
# Entegrasyon aralığı: posterior'un tam veri aralığı
# Not: q(x) ≈ 0 olan bölgeler integrand'ı patlatabileceğinden
#      maske ile korunur (mask: p > ε ve q > ε)

def kde_kl_simpson_1d(P, Q, grid_points=KDE_GRID):
    """1D: KDE yoğunlukları + Simpson entegrasyonu."""
    p_arr = P[:, 0];  q_arr = Q[:, 0]
    x     = np.linspace(p_arr.min(), p_arr.max(), grid_points)
    pv    = gaussian_kde(p_arr, bw_method='scott')(x)
    qv    = gaussian_kde(q_arr, bw_method='scott')(x)
    mask  = (pv > 1e-300) & (qv > 1e-300)
    intg  = np.zeros_like(pv)
    intg[mask] = pv[mask] * np.log2(pv[mask] / qv[mask])
    return float(simpson(intg, x=x))


def kde_kl_simpson_2d(P, Q, grid_points=KDE_GRID):
    """
    2D: KDE yoğunlukları + çift Simpson entegrasyonu.
    KL = ∫∫ p(x,y) log₂[p(x,y)/q(x,y)] dx dy
    İç entegral: y sabitken x üzerinden,  dış: y üzerinden
    """
    kde_p = gaussian_kde(P.T, bw_method='scott')
    kde_q = gaussian_kde(Q.T, bw_method='scott')
    x = np.linspace(P[:, 0].min(), P[:, 0].max(), grid_points)
    y = np.linspace(P[:, 1].min(), P[:, 1].max(), grid_points)
    X, Y  = np.meshgrid(x, y)                              # her biri (grid, grid)
    pos   = np.vstack([X.ravel(), Y.ravel()])               # (2, grid²)
    pv    = kde_p(pos).reshape(grid_points, grid_points)    # (grid, grid)
    qv    = kde_q(pos).reshape(grid_points, grid_points)
    mask  = (pv > 1e-300) & (qv > 1e-300)
    intg  = np.zeros_like(pv)
    intg[mask] = pv[mask] * np.log2(pv[mask] / qv[mask])
    # Simpson: once x (axis=1) sonra y (axis=0)
    return float(simpson(simpson(intg, x=x, axis=1), x=y))


# ══════════════════════════════════════════════════════════════════════════════
# YÖNTEM 2b — KDE + MONTE CARLO (4D ve üzeri)
# ══════════════════════════════════════════════════════════════════════════════
# 4D için Simpson ızgarası 400⁴ ≈ 25.6 milyar nokta gerektirir → imkansız.
# Monte Carlo tahmincisi:
#   KL(P∥Q) = E_P[log₂(p(x)/q(x))] ≈ (1/n) Σᵢ log₂[kde_p(xᵢ)/kde_q(xᵢ)]
#                                       xᵢ ~ P  (posterior örneklerinden)
# Bu beklenti değeri önyargısız bir tahminci; varyans MC tekrarıyla ölçülür.
# Referans: Hershey & Olsen (2007), ICASSP — KDE tabanlı KL MC tahmini

def kde_kl_monte_carlo(P, Q, n_mc=MC_N, n_rep=MC_REPS, seed=SEED):
    """
    Dondurur
    --------
    mean_bits : KL [bit], n_rep tekrar ortalaması
    std_bits  : Tekrarlar arası standart sapma [bit]
    """
    rng   = np.random.default_rng(seed)
    n     = P.shape[0]
    kde_p = gaussian_kde(P.T, bw_method='scott')
    kde_q = gaussian_kde(Q.T, bw_method='scott')
    ests  = []
    for _ in range(n_rep):
        idx  = rng.choice(n, size=min(n_mc, n), replace=False)
        Ps   = P[idx]
        lp   = np.log2(np.maximum(kde_p(Ps.T), 1e-300))
        lq   = np.log2(np.maximum(kde_q(Ps.T), 1e-300))
        ests.append(float(np.mean(lp - lq)))
    return float(np.mean(ests)), float(np.std(ests))


# ══════════════════════════════════════════════════════════════════════════════
# YÖNTEM 3 — PÉREZ-CRUZ k-NN
# ══════════════════════════════════════════════════════════════════════════════
# Pérez-Cruz (2008, NeurIPS), Eq. 14:
#   KL(P∥Q) ≈ (d/n) Σᵢ log(sᵢ/rᵢ) + log(m/(n−1))
# rᵢ : P içindeki k. komşu mesafesi (kendisi hariç)
# sᵢ : Q içindeki k. komşu mesafesi
# Bias düzeltmesi: log(m/(n-1)) terimi asimptotik önyargıyı dengeler
#
# UYARI: d=4 için n ~ 10¹² örnek gerekir (boyutsallık laneti);
#        mevcut n ≈ 147k ile bu tahmin güvenilmez

def knn_kl_bits(P, Q, k=KNN_K):
    n, d = P.shape
    m    = Q.shape[0]
    pt   = cKDTree(P)
    qt   = cKDTree(Q)
    # r: P'den P'ye k. komşu (k+1 sorgula, 0. kolon kendisi)
    r    = pt.query(P, k=k+1, p=2, workers=-1)[0][:, k]
    # s: P'den Q'ya k. komşu
    s    = qt.query(P, k=k,   p=2, workers=-1)[0][:, k-1]
    r    = np.maximum(r, 1e-15)
    s    = np.maximum(s, 1e-15)
    kl_nat = (d / n) * np.sum(np.log(s / r)) + np.log(m / (n - 1.0))
    return float(kl_nat / np.log(2))


# ══════════════════════════════════════════════════════════════════════════════
# ANA DÖNGÜ
# ══════════════════════════════════════════════════════════════════════════════

np.random.seed(SEED)
results = []

print("\n" + "=" * 72)
print("  GW150914 — KL(Posterior ∥ Prior)  Blok Bazlı Hesap")
print("=" * 72)

for bnum, bdim, params in BLOCKS:
    d    = len(params)
    P, Q = get_block_data(params)
    n, m = P.shape[0], Q.shape[0]
    lstr = " + ".join(LABELS.get(p, p) for p in params)

    print(f"\n{'─' * 72}")
    print(f"  {bnum} ({bdim})  —  {lstr}")
    print(f"  Posterior P: {P.shape}   |   Prior Q: {Q.shape}")

    # Prior kapsam kontrolü (her parametre için)
    for i, p in enumerate(params):
        pmin, pmax = P[:, i].min(), P[:, i].max()
        qmin, qmax = Q[:, i].min(), Q[:, i].max()
        if qmax < pmax:
            print(f"  [UYARI] {p:<24} Prior max={qmax:.4g} < Post max={pmax:.4g}")
        if qmin > pmin:
            print(f"  [UYARI] {p:<24} Prior min={qmin:.4g} > Post min={pmin:.4g}")

    # ── [1] Gaussian ─────────────────────────────────────────────────────────
    try:
        g, cond = gaussian_kl_bits(P, Q)
        print(f"\n  [1] Gaussian KL          : {g:8.4f} bit   (cond(Sq) = {cond:.2e})")
    except Exception as e:
        g, cond = float("nan"), float("nan")
        print(f"\n  [1] Gaussian KL          : HATA — {e}")

    # ── [2] KDE ──────────────────────────────────────────────────────────────
    kde_std = None
    try:
        if d == 1:
            kde_val     = kde_kl_simpson_1d(P, Q)
            kde_label   = f"KDE+Simpson (1D, grid={KDE_GRID})"
            kde_str     = f"{kde_val:8.4f} bit"
        elif d == 2:
            kde_val     = kde_kl_simpson_2d(P, Q)
            kde_label   = f"KDE+Simpson (2D, grid={KDE_GRID})"
            kde_str     = f"{kde_val:8.4f} bit"
        else:
            # 4D: MC entegrasyonu
            kde_val, kde_std = kde_kl_monte_carlo(P, Q)
            kde_label   = f"KDE+MC ({d}D, n={MC_N}, reps={MC_REPS})"
            kde_str     = f"{kde_val:8.4f} ± {kde_std:.4f} bit"
        print(f"  [2] {kde_label:<36}: {kde_str}")
    except Exception as e:
        kde_val = float("nan")
        print(f"  [2] KDE                  : HATA — {e}")

    # ── [3] k-NN ─────────────────────────────────────────────────────────────
    try:
        knn_val = knn_kl_bits(P, Q, k=KNN_K)
        caveat  = "  [!] d=4: boyutsallik laneti — dikkatli yorumla" if d >= 4 else ""
        print(f"  [3] k-NN (k={KNN_K})             : {knn_val:8.4f} bit{caveat}")
    except Exception as e:
        knn_val = float("nan")
        print(f"  [3] k-NN                 : HATA — {e}")

    results.append({
        "bnum": bnum, "bdim": bdim, "params": params, "d": d,
        "gaussian": g, "kde": kde_val, "kde_std": kde_std, "knn": knn_val,
    })


# ══════════════════════════════════════════════════════════════════════════════
# ÖZET TABLOSU
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 72)
print(f"  {'OZET TABLOSU':^68}")
print("=" * 72)
hdr = f"  {'Blok':<8} {'Boyut':<5} {'Parametreler':<24} {'Gaussian':>9} {'KDE':>9} {'k-NN':>9}"
print(hdr)
print("─" * 72)

g_tot = kde_tot = knn_tot = 0.0
for r in results:
    pstr  = "+".join(LABELS.get(p, p) for p in r["params"])
    g_tot   += r["gaussian"]
    kde_tot += r["kde"]
    knn_tot += r["knn"]
    kde_disp = f"{r['kde']:9.4f}" + (f"±{r['kde_std']:.3f}" if r["kde_std"] else "      ")
    print(f"  {r['bnum']:<8} {r['bdim']:<5} {pstr:<24} "
          f"{r['gaussian']:>9.4f} {kde_disp} {r['knn']:>9.4f}")

print("─" * 72)
print(f"  {'TOPLAM (marginal toplamı)':<38} {g_tot:>9.4f} {kde_tot:>9.4f}       {knn_tot:>9.4f}")

# TC katkılarını dahil et
tc_known_sum = sum(TC_KNOWN.values())
print(f"\n  Önceden hesaplanan TC (Blok 1-4)            : +{tc_known_sum:.2f} bit")
print(f"  KDE alt sınır tahmini (marginal + TC_1..4)  : ~{kde_tot + tc_known_sum:.2f} bit")

print(f"\n  {'─'*50}")
print(f"  Referanslar:")
print(f"    F&H analitik (Gauss varsayımı)           : ~41.5 bit")
print(f"    Marginal KL toplamı (önceki)             : ~36.05 bit")
print(f"    TC alt sınırı (Blok 1-4, bilinen)        :  {tc_known_sum:.2f} bit")
print("=" * 72)