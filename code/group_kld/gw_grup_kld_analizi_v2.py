"""
GW — ≤5D grup KLD(posterior||prior) analizi  [kde / knn]  — v2 (analitik prior)
==================================================================================

v2'deki fark (prior sorun düzeltmesi)
--------------------------------------
Orijinal kodda prior örnekleri doğrudan HDF5 dosyasından alınıyordu. Bu iki
parametre için ciddi destek (support) uyumsuzluğu yaratıyordu:

  1. luminosity_distance:  HDF5 prior örnekleri 681–10000 Mpc arasındayken
     gerçek analitik prior PowerLaw(alpha=2, min=10, max=10000) Mpc.
     Posterior 166 Mpc'ye kadar iniyor → HDF5 örnekleri o bölgeyi temsil
     etmiyor → KDE yoğunluğu ~0 → p/q → ∞ → KLD şişiyor.

  2. mass_1_source / mass_2_source: HDF5 prior örnekleri ~8.51 M☉'de
     kesiyor, posterior 10.78 M☉'ya ulaşıyor. Gerçek prior,
     chirp_mass (detector) ~ Uniform[5.63, 11.42] + mass_ratio ~ Uniform[0.05, 1]
     + d_L ~ PowerLaw(α=2, 10-10000 Mpc) kombinasyonundan geliyor;
     bu prior kaynakçerçevesinde 10.8 M☉'ya kadar destek veriyor ama
     HDF5 örneklemesi bu ucu yeterince örneklememiş.

Çözüm: HDF5'ten okunan prior örneklerini bu üç parametre için analitik
prior'dan üretilen örneklerle DEĞİŞTİR. Diğer tüm parametreler HDF5'ten
olduğu gibi kullanılır.

Analitik prior kaynakları (HDF5 priors/analytic grubundan okundu):
  luminosity_distance : PowerLaw(alpha=2, minimum=10, maximum=10000) [Mpc]
  chirp_mass (det.)   : UniformInComponentsChirpMass(min=5.628, max=11.423)
  mass_ratio          : UniformInComponentsMassRatio(min=0.05, max=1.0)
  Redshift / kosm.    : Planck15 — d_L → z dönüşümü
  Kütle dönüşümü      : m_source = m_detector / (1 + z)
"""

import os
import sys
import glob
import json

import numpy as np
import h5py
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kde_estimators_reference import calc_kde_kld
from knn_estimators_reference import calc_knn_kld

NATS_TO_BITS = 1.0 / np.log(2.0)
RANDOM_STATE = 42
N_ANALYTIC_PRIOR = 30000   # üretilecek analitik prior örneksi sayısı

# GW150914 için 15 standart CBC parametresi
PARAMS_15 = [
    "mass_1_source", "mass_2_source", "a_1", "a_2", "tilt_1", "tilt_2",
    "phi_12", "phi_jl", "luminosity_distance", "theta_jn", "psi",
    "azimuth", "zenith", "geocent_time", "phase",
]

METHODS = ["kde-scott", "kde-silverman", "knn-k1"]

# ========== İSTEDİĞİN DOSYAYI BURAYA YAZ ==========
file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191103_012549_PEDataRelease_mixed_cosmo.h5"
# file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"


# ==========================================================================
# ANALİTİK PRIOR ÜRETİCİLER
# ==========================================================================

def sample_powerlaw(alpha, d_min, d_max, n, rng):
    """
    PowerLaw(alpha) CDF tersinden örnekle.
    p(x) ∝ x^alpha,  x ∈ [d_min, d_max]
    CDF: F(x) = (x^(a+1) - d_min^(a+1)) / (d_max^(a+1) - d_min^(a+1))
    Ters: x = (u*(d_max^(a+1) - d_min^(a+1)) + d_min^(a+1))^(1/(a+1))
    """
    a1 = alpha + 1.0
    lo = d_min ** a1
    hi = d_max ** a1
    u = rng.uniform(0, 1, n)
    return (u * (hi - lo) + lo) ** (1.0 / a1)


def dL_to_z_planck15(dL_Mpc, z_grid=None, dL_grid=None):
    """
    d_L (Mpc) → redshift, Planck15 kozmolojisi.
    Tablo interpolasyonu (astropy olmadan da çalışır, ama astropy varsa daha doğru).
    """
    try:
        from astropy.cosmology import Planck15
        from astropy import units as u
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z_arr = np.zeros(len(dL_Mpc))
            for i, d in enumerate(dL_Mpc):
                from astropy.cosmology import z_at_value
                z_arr[i] = float(z_at_value(Planck15.luminosity_distance, d * u.Mpc,
                                             zmin=1e-4, zmax=20.0))
        return z_arr
    except Exception:
        # Fallback: Hubble yaklaşımı (düşük z için yeterli)
        H0 = 67.74  # km/s/Mpc (Planck15)
        c  = 2.998e5  # km/s
        return dL_Mpc * H0 / c  # z ≈ H0*dL/c (düşük z yaklaşımı)


def build_dL_z_table(d_min=10.0, d_max=10000.0, n_grid=500):
    """d_L ↔ z arama tablosu yap (tekrar hesaplamayı önler)."""
    try:
        from astropy.cosmology import Planck15, z_at_value
        from astropy import units as u
        import warnings
        dL_arr = np.linspace(d_min, d_max, n_grid)
        z_arr  = np.zeros(n_grid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, d in enumerate(dL_arr):
                z_arr[i] = float(z_at_value(Planck15.luminosity_distance,
                                             d * u.Mpc, zmin=1e-4, zmax=20.0))
        return dL_arr, z_arr
    except Exception:
        H0 = 67.74; c = 2.998e5
        dL_arr = np.linspace(d_min, d_max, n_grid)
        return dL_arr, dL_arr * H0 / c


def chirp_mass_to_components(Mc, q):
    """
    Chirp kütle + kütle oranından bileşen kütlelerini hesapla.
    m1 ≥ m2,  q = m2/m1 ≤ 1
    m2 = Mc * (1+q)^(1/5) * q^(2/5)
    m1 = m2 / q
    """
    m2 = Mc * (1.0 + q) ** (1.0/5.0) * q ** (2.0/5.0)
    m1 = m2 / q
    return m1, m2


def generate_analytic_prior(n=30000, rng=None,
                             dL_min=10.0, dL_max=10000.0,
                             Mc_min=5.628, Mc_max=11.423,
                             q_min=0.05, q_max=1.0,
                             m_min_constraint=1.0):
    """
    luminosity_distance, mass_1_source, mass_2_source için
    analitik prior'dan örnekler üret.

    Döndürür: dict {parametre: array(n_geçerli)}
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    print(f"\n[ANALİTİK PRIOR] {n} örnek üretiliyor...")

    # 1. d_L ~ PowerLaw(alpha=2, min=10, max=10000)
    dL = sample_powerlaw(alpha=2, d_min=dL_min, d_max=dL_max, n=n, rng=rng)
    print(f"  d_L örneklendi: [{dL.min():.1f}, {dL.max():.1f}] Mpc")

    # 2. d_L → z (Planck15 tablosu)
    print("  d_L → z dönüşümü hesaplanıyor (Planck15)...")
    dL_grid, z_grid = build_dL_z_table(dL_min, dL_max, n_grid=800)
    z = np.interp(dL, dL_grid, z_grid)
    print(f"  z aralığı: [{z.min():.4f}, {z.max():.4f}]")

    # 3. Chirp kütle (detector frame) ~ Uniform[Mc_min, Mc_max]
    Mc_det = rng.uniform(Mc_min, Mc_max, n)

    # 4. Kütle oranı ~ Uniform[q_min, q_max]
    q = rng.uniform(q_min, q_max, n)

    # 5. Bileşen kütleleri (detector frame)
    m1_det, m2_det = chirp_mass_to_components(Mc_det, q)

    # 6. Kaynak çerçeveye dönüştür
    m1_src = m1_det / (1.0 + z)
    m2_src = m2_det / (1.0 + z)

    # 7. Kısıt: m1 ≥ m_min, m2 ≥ m_min  (Constraint prior)
    mask = (m1_src >= m_min_constraint) & (m2_src >= m_min_constraint)
    n_valid = mask.sum()
    print(f"  Kısıt sonrası geçerli örnek: {n_valid}/{n} "
          f"({100*n_valid/n:.1f}%)")

    result = {
        "luminosity_distance": dL[mask],
        "mass_1_source":       m1_src[mask],
        "mass_2_source":       m2_src[mask],
    }

    print(f"  mass_2_source aralığı: [{result['mass_2_source'].min():.3f}, "
          f"{result['mass_2_source'].max():.3f}] M☉")
    print(f"  luminosity_distance aralığı: [{result['luminosity_distance'].min():.1f}, "
          f"{result['luminosity_distance'].max():.1f}] Mpc")

    return result


# ==========================================================================
# HDF5 VERİ OKUMA
# ==========================================================================

def load_post_prior(path, params):
    """Posterior + prior örneklerini yükle; analitik prior ile problematik
    parametreleri değiştir."""
    with h5py.File(path, "r") as f:
        chosen = None
        for key in f.keys():
            g = f[key]
            if not isinstance(g, h5py.Group):
                continue
            if "posterior_samples" in g and "priors" in g and "samples" in g["priors"]:
                psamp = g["priors"]["samples"]
                have = all(p in psamp for p in params) and \
                    all(psamp[p].shape and psamp[p].shape[0] > 50
                        for p in params if p in psamp)
                if have:
                    chosen = key
                    break
        if chosen is None:
            raise ValueError("Tüm parametreler için posterior+prior içeren grup bulunamadı.")

        g = f[chosen]
        post_tbl = g["posterior_samples"][()]
        post  = {p: np.asarray(post_tbl[p], dtype=float) for p in params}
        prior = {p: np.asarray(g["priors"]["samples"][p][()], dtype=float)
                 for p in params}

        # HDF5'ten analitik prior tanımlarını oku (raporlama için)
        analytic_info = {}
        if "priors" in g and "analytic" in g["priors"]:
            an = g["priors"]["analytic"]
            for k in ["luminosity_distance", "chirp_mass", "mass_ratio"]:
                if k in an:
                    analytic_info[k] = an[k][()][0].decode()

    return post, prior, chosen, analytic_info


# ==========================================================================
# DESTEK (SUPPORT) TANISI
# ==========================================================================

def support_diagnosis(post, prior, params):
    """Her parametre için posterior'un prior desteği dışına taşan % hesapla."""
    print("\n" + "=" * 78)
    print("AŞAMA 2 — Prior/posterior DESTEK (support) tanısı")
    print("=" * 78)
    print(f"{'Parametre':<25} {'post[min,max]':>22} {'prior[min,max]':>20}  %dışarı")
    print("-" * 78)
    warnings = []
    for p in params:
        pst = post[p]; pri = prior[p]
        lo, hi = pri.min(), pri.max()
        frac = np.mean((pst < lo) | (pst > hi)) * 100
        flag = "  <-- DİKKAT" if frac > 1.0 else ""
        print(f"{p:<25} [{pst.min():>10.3g},{pst.max():>10.3g}]"
              f"   [{lo:>10.3g},{hi:>10.3g}]  {frac:>6.2f}%{flag}")
        if frac > 1.0:
            warnings.append((p, frac))
    print("-" * 78)
    if warnings:
        print("UYARI: Aşağıdaki parametrelerde prior desteği HÂLÂ yetersiz:")
        for p, f in warnings:
            print(f"  - {p}: %{f:.2f} dışarıda")
    else:
        print("✓ Tüm parametreler prior desteği içinde (<%1).")
    return warnings


# ==========================================================================
# KLD HESABI
# ==========================================================================

def kld_one(method, P, Q):
    if method == "kde-scott":
        return calc_kde_kld(P, Q, bandwidth=None)
    if method == "kde-silverman":
        return calc_kde_kld(P, Q, bandwidth="silverman")
    if method == "knn-k1":
        return calc_knn_kld(P, Q, k=1)
    raise ValueError(method)


# ==========================================================================
# GRUPLAMA YARDIMCISI
# ==========================================================================

def avg_between_group_corr(groups, abscorr):
    vals = [abscorr[a, b]
            for i in range(len(groups)) for j in range(i + 1, len(groups))
            for a in groups[i] for b in groups[j]]
    return float(np.mean(vals)) if vals else 0.0


# ==========================================================================
# ANA AKIŞ
# ==========================================================================

def main(path=None):
    here = os.path.dirname(os.path.abspath(__file__))

    if path is None:
        path = file_name

    print("=" * 78)
    print("GW — ≤5D grup KLD(posterior||prior) analizi  [kde / knn]  v2-analitik")
    print("=" * 78)
    print(f"Dosya: {path}\n")

    post, prior, grp, analytic_info = load_post_prior(path, PARAMS_15)

    print(f"Analiz grubu : {grp}")
    n_post_raw = len(post[PARAMS_15[0]])
    n_prior_raw = len(prior[PARAMS_15[0]])
    print(f"Posterior    : {n_post_raw} örnek | Prior (HDF5): {n_prior_raw} örnek")

    if analytic_info:
        print("\n[HDF5'ten okunan analitik prior tanımları]")
        for k, v in analytic_info.items():
            print(f"  {k}: {v}")

    # ---- ANALİTİK PRIOR ÜRETİMİ ----
    rng = np.random.default_rng(RANDOM_STATE)
    analytic_samples = generate_analytic_prior(
        n=N_ANALYTIC_PRIOR, rng=rng,
        dL_min=10.0, dL_max=10000.0,
        Mc_min=5.628459258092691, Mc_max=11.422992347700943,
        q_min=0.05, q_max=1.0,
        m_min_constraint=1.0,
    )

    # Analitik örnekleri prior dict'e yaz
    # Ortak boyut: analitik örnekler sayısı (n_analytic_valid)
    n_analytic_valid = len(analytic_samples["luminosity_distance"])
    for param in ["luminosity_distance", "mass_1_source", "mass_2_source"]:
        old_min = prior[param].min()
        old_max = prior[param].max()
        prior[param] = analytic_samples[param]
        new_min = prior[param].min()
        new_max = prior[param].max()
        print(f"\n[PRIOR DEĞİŞTİRİLDİ] {param}")
        print(f"  HDF5  : [{old_min:.4g}, {old_max:.4g}]  (n={n_prior_raw})")
        print(f"  Analitik: [{new_min:.4g}, {new_max:.4g}]  "
              f"(n={len(prior[param])})")

    # HDF5 kaynaklı diğer parametreleri analitik boyuta yeniden örnekle
    hdf5_params = [p for p in PARAMS_15
                   if p not in {"luminosity_distance", "mass_1_source", "mass_2_source"}]
    for param in hdf5_params:
        idx = rng.choice(len(prior[param]), size=n_analytic_valid, replace=True)
        prior[param] = prior[param][idx]
    print(f"\nTüm prior parametreleri {n_analytic_valid} örneğe eşitlendi.")

    # ---- MATRİSLERE ÇEVIR ----
    P_raw = np.column_stack([post[p]  for p in PARAMS_15])
    Q_raw = np.column_stack([prior[p] for p in PARAMS_15])

    # ---- TEKİLLEŞTİRME ----
    from collections import Counter
    n0, npar = P_raw.shape
    uniq_counts = [len(np.unique(P_raw[:, j])) for j in range(npar)]
    cnt = Counter(c for c in uniq_counts if c < 0.99 * n0)
    if cnt:
        modal, freq = cnt.most_common(1)[0]
        if freq >= 2:
            key_cols = [j for j in range(npar) if uniq_counts[j] == modal]
            _, keep = np.unique(P_raw[:, key_cols], axis=0, return_index=True)
            keep = np.sort(keep)
            if len(keep) < n0:
                print(f"\nTekilleştirme: {n0} -> {len(keep)} benzersiz örnek "
                      f"(kopya sütunlar: "
                      f"{', '.join(PARAMS_15[j] for j in key_cols)})")
                P_raw = P_raw[keep]

    print(f"\nPosterior (tekilleştirilmiş): {P_raw.shape[0]} örnek")
    print(f"Prior (analitik):             {Q_raw.shape[0]} örnek")

    # ---- DESTEK TANISI (analitik prior sonrası) ----
    post_diag  = {p: P_raw[:, i] for i, p in enumerate(PARAMS_15)}
    prior_diag = {p: Q_raw[:, i] for i, p in enumerate(PARAMS_15)}
    remaining_warnings = support_diagnosis(post_diag, prior_diag, PARAMS_15)

    # ---- STANDARDİZASYON ----
    pool = np.vstack([P_raw, Q_raw])
    mu, sd = pool.mean(axis=0), pool.std(axis=0)
    sd[sd == 0] = 1.0
    P = (P_raw - mu) / sd
    Q = (Q_raw - mu) / sd
    P = P + rng.normal(0.0, 1e-10, size=P.shape)
    Q = Q + rng.normal(0.0, 1e-10, size=Q.shape)

    # ---- GRUPLAR (elle ayarlanmış, fiziksel) ----
    # PARAMS_15 indeksleri:
    # 0=mass_1_source, 1=mass_2_source, 2=a_1, 3=a_2, 4=tilt_1, 5=tilt_2,
    # 6=phi_12, 7=phi_jl, 8=luminosity_distance, 9=theta_jn, 10=psi,
    # 11=azimuth, 12=zenith, 13=geocent_time, 14=phase
    groups = [
        [0, 1],               # G1: kütleler
        [2, 3, 4, 5],         # G2: spinler
        [6, 7, 14],           # G3: açısal fazlar
        [8, 9, 10, 11, 12, 13],  # G4: mesafe + yön + zaman
    ]

    rho, _ = spearmanr(P)
    abscorr = np.abs(np.atleast_2d(rho))
    np.fill_diagonal(abscorr, 0.0)
    between = avg_between_group_corr(groups, abscorr)

    group_names = [[PARAMS_15[i] for i in g] for g in groups]
    print(f"\n{'='*78}")
    print("GRUPLAR")
    print(f"{'='*78}")
    print(f"Gruplar arası ort. |Spearman| = {between:.3f}")
    for gi, gn in enumerate(group_names, 1):
        print(f"  G{gi}: {', '.join(gn)}")

    # ---- KLD HESABI ----
    results = {m: [] for m in METHODS}
    for gi, g in enumerate(groups, 1):
        Pg, Qg = P[:, g], Q[:, g]
        for m in METHODS:
            try:
                val = float(kld_one(m, Pg, Qg))
            except Exception as e:
                val = float("nan")
                print(f"  (G{gi} {m} hata: {e})")
            results[m].append(val)

    # ---- TABLO ----
    print("\n" + "-" * 78)
    hdr = f"{'Yöntem':<15}" + \
          "".join(f"{'G'+str(i):>10}" for i in range(1, len(groups)+1)) + \
          f"{'TOPLAM':>12}"
    print(hdr)
    print("-" * 78)
    totals = {}
    for m in METHODS:
        vals = results[m]
        tot = float(np.nansum(vals))
        totals[m] = tot
        row = f"{m:<15}" + "".join(f"{v:>10.3f}" for v in vals) + f"{tot:>12.3f}"
        print(row)
    print("-" * 78)
    bit_str = "   ".join(f"{m}={totals[m]*NATS_TO_BITS:.2f}" for m in METHODS)
    print(f"(nats)  |  bit:  {bit_str}")

    mean_tot = np.mean([totals[m] for m in METHODS])
    print(f"\nYöntem-ortalaması toplam KLD ≈ {mean_tot:.3f} nats "
          f"({mean_tot*NATS_TO_BITS:.2f} bit)")

    # ---- KARŞILAŞTIRMA: eski vs yeni ----
    print("\n" + "=" * 78)
    print("ÖZET — HDF5 prior vs Analitik prior karşılaştırması")
    print("=" * 78)
    print("  Eski (HDF5 prior, destek sorunu ile)  : ~32.01 bit  [önceki çalıştırma]")
    print(f"  Yeni (analitik prior, düzeltilmiş)   : "
          f"~{mean_tot*NATS_TO_BITS:.2f} bit")
    print("  Gauss (Laplace) çapraz-kontrol        : ~28.91 bit")
    print()
    if remaining_warnings:
        print("  ⚠ Hâlâ destek uyarısı olan parametre(ler):")
        for p, f in remaining_warnings:
            print(f"    {p}: %{f:.2f} dışarıda")
    else:
        print("  ✓ Tüm parametreler artık prior desteği içinde.")

    # ---- JSON ----
    out = {
        "file": os.path.basename(path),
        "version": "v2-analytic-prior",
        "analysis_group": grp,
        "n_posterior": int(P.shape[0]),
        "n_prior_analytic": int(Q.shape[0]),
        "parameters": PARAMS_15,
        "groups": group_names,
        "avg_between_group_abs_corr": between,
        "methods": METHODS,
        "kld_per_group_nats": {m: results[m] for m in METHODS},
        "kld_total_nats": totals,
        "kld_total_bits": {m: totals[m] * NATS_TO_BITS for m in METHODS},
        "analytic_prior_used": {
            "luminosity_distance": "PowerLaw(alpha=2, min=10, max=10000) [Mpc]",
            "mass_1_source": "derived: Mc~Uniform[5.628,11.423], q~Uniform[0.05,1], Planck15",
            "mass_2_source": "derived: same as mass_1_source",
        },
    }
    out_json = os.path.join(here, "results_grup_kld_v2.json")
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nJSON kaydedildi: {out_json}")

    # ---- GRAFİK ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ng = len(groups)
        x = np.arange(ng + 1)
        labels = [f"G{i}\n({','.join(n[:2] for n in gn)}...)"
                  if len(gn) > 2 else f"G{i}\n({','.join(gn)})"
                  for i, gn in enumerate(group_names, 1)] + ["TOPLAM"]
        labels = [f"G{i}" for i in range(1, ng+1)] + ["TOPLAM"]

        w = 0.75 / len(METHODS)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Sol: grup bazında KLD (bit)
        ax = axes[0]
        for k, m in enumerate(METHODS):
            heights = [v * NATS_TO_BITS for v in results[m]] + \
                      [totals[m] * NATS_TO_BITS]
            ax.bar(x + k * w, heights, w, label=m, alpha=0.85)
        ax.set_xticks(x + w)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("KLD(posterior || prior)  [bit]")
        ax.set_title("Grup bazında KLD — ANALİTİK PRIOR (v2)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Sağ: parametre bazında destek kontrolü
        ax2 = axes[1]
        params_show = ["mass_1_source", "mass_2_source", "luminosity_distance",
                       "theta_jn", "zenith"]
        colors = []
        bars = []
        for p in params_show:
            pst = post_diag[p]
            pri = prior_diag[p]
            frac = np.mean((pst < pri.min()) | (pst > pri.max())) * 100
            bars.append(frac)
            colors.append("green" if frac < 1 else "red")
        ax2.barh(params_show, bars, color=colors, alpha=0.8)
        ax2.axvline(1, color="k", linestyle="--", linewidth=0.8, label="1% eşik")
        ax2.set_xlabel("Posterior dışarıda (%)")
        ax2.set_title("Destek uyumu (analitik prior sonrası)")
        ax2.legend(fontsize=8)
        ax2.grid(axis="x", alpha=0.3)

        fig.suptitle(f"{os.path.basename(path)}", fontsize=10, y=1.01)
        fig.tight_layout()
        out_png = os.path.join(here, "grup_kld_v2_analitik.png")
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        print(f"Grafik kaydedildi: {out_png}")
    except Exception as e:
        print(f"(Grafik atlandı: {e})")

    return out


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if path_arg is None and file_name is not None:
        print(f"\n>>> DOSYA: {file_name}\n")
        main(file_name)
    else:
        main(path_arg)
