"""
GW — 15 parametreyi ≤5D gruplara bölüp KLD(posterior||prior) hesabı [kde / knn]
=================================================================================

Amaç
----
Makaledeki (Álvarez Chaves et al., 2024, Entropy 26(5):387) mantıkla: yüksek
boyutta (15D) bilgi-teorik kestirim güvenilmezdir. Bu yüzden 15 parametreyi
≤5D gruplara böleriz ve her grupta, yakınsadığını bildiğimiz GÜVENİLİR
estimatörlerle KLD(posterior || prior) hesaplarız. İki bağımsız yöntem
(KDE, kNN) karşılaştırılır; uyuşmaları sonuca güven verir.

Prior örnekleri
---------------
HDF5 dosyasındaki prior örnekleri (priors/samples) iki parametre için
support uyumsuzluğu yaratıyordu:
  - luminosity_distance : dosyadaki örnekler 681+ Mpc, gerçek prior 10+ Mpc
  - mass_1/2_source     : source-frame örnekleri prior sınırını kapsamamış
Bu yüzden tüm 15 parametre için prior örnekleri HDF5'teki analytic
tanımlardan (priors/analytic) üretilir:
  - Uniform  -> np.uniform
  - Sine     -> theta = arccos(cos(lo) - u*(cos(lo)-cos(hi)))
  - PowerLaw -> ters-CDF
  - mass_1/2 -> chirp_mass + mass_ratio + d_L->z (Planck15) -> source frame

Neden 4 grup ve korelasyona göre?
----------------------------------
Grup KLD'lerinin TOPLAMI, gruplar birbirinden BAGIMSIZ ise ortak (joint) 15D
KLD'ye esittir. Parametreleri korelasyona gore oyle gruplarız ki gruplar ARASI
bagimlılık en aza iner -> toplam, joint KLD'ye iyi bir yaklaşım olur.

Çıktı
-----
- Konsolda yöntem x grup tablosu + toplamlar
- results_grup_kld.json
- grup_kld_karsilastirma.png

Kullanım
--------
    python gw_grup_kld_analizi.py [posterior_dosyasi.h5]
    (argümansız: file_name değişkenini düzenle)

Bağımlılıklar: numpy, scipy, h5py, astropy, matplotlib
"""

import os
import sys
import glob
import json
import re

import numpy as np
import h5py
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kde_estimators_reference import calc_kde_kld
from knn_estimators_reference import calc_knn_kld

NATS_TO_BITS = 1.0 / np.log(2.0)
RANDOM_STATE = 42
N_PRIOR      = 30000   # üretilecek analitik prior örnek sayısı

PARAMS_15 = [
    "mass_1_source", "mass_2_source", "a_1", "a_2", "tilt_1", "tilt_2",
    "phi_12", "phi_jl", "luminosity_distance", "theta_jn", "psi",
    "azimuth", "zenith", "geocent_time", "phase",
]

METHODS = ["kde-scott", "kde-silverman", "knn-k1"]

# ========== İSTEDİĞİN DOSYAYI BURAYA YAZ ==========
# file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191103_012549_PEDataRelease_mixed_cosmo.h5"
# file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191105_143521_PEDataRelease_mixed_cosmo.h5"


# ==========================================================================
# ANALİTİK PRIOR ÜRETİCİLER
# ==========================================================================

def sample_uniform(lo, hi, n, rng):
    return rng.uniform(lo, hi, n)

def sample_sine(lo, hi, n, rng):
    """Sine: p(theta) prop sin(theta) -> CDF tersi."""
    u = rng.uniform(0, 1, n)
    return np.arccos(np.cos(lo) - u * (np.cos(lo) - np.cos(hi)))

def sample_powerlaw(alpha, lo, hi, n, rng):
    """PowerLaw(alpha): p(x) prop x^alpha -> CDF tersi."""
    a1 = alpha + 1.0
    u  = rng.uniform(0, 1, n)
    return (u * (hi**a1 - lo**a1) + lo**a1) ** (1.0 / a1)

def build_dL_z_table(d_min=10.0, d_max=10000.0, n_grid=1000):
    """d_L [Mpc] -> z arama tablosu, Planck15 kozmolojisi."""
    try:
        from astropy.cosmology import Planck15, z_at_value
        from astropy import units as u
        import warnings
        dL_arr = np.linspace(d_min, d_max, n_grid)
        z_arr  = np.zeros(n_grid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, d in enumerate(dL_arr):
                z_arr[i] = float(z_at_value(
                    Planck15.luminosity_distance, d * u.Mpc, zmin=1e-4, zmax=20.0))
        return dL_arr, z_arr
    except Exception:
        H0 = 67.74; c = 2.998e5
        dL_arr = np.linspace(d_min, d_max, n_grid)
        return dL_arr, dL_arr * H0 / c

def parse_analytic(s):
    """Bilby analytic string'inden tip + parametre dict çıkar."""
    m = re.match(r'(\w+)\((.+)\)', s)
    if not m:
        return None, {}
    kind = m.group(1)
    kv   = {}
    for part in m.group(2).split(','):
        part = part.strip()
        kvm  = re.match(r'(\w+)\s*=\s*(.+)', part)
        if kvm:
            k, v = kvm.group(1).strip(), kvm.group(2).strip()
            try:    kv[k] = float(v)
            except: kv[k] = v.strip("'\"")
    return kind, kv

def generate_analytic_prior(analytic_strings, n, rng,
                             Mc_min, Mc_max, q_min=0.05, q_max=1.0,
                             m_min_constraint=1.0):
    """
    HDF5 priors/analytic tanımlarından tüm 15 parametre için örnek üret.
    mass_1/2_source dogrudan analytic'te yok; chirp_mass + mass_ratio + d_L->z
    zinciriyle türetilir.
    """
    print(f"\n[ANALİTİK PRIOR] {n} örnek üretiliyor...")

    # 1. luminosity_distance (PowerLaw)
    kind, kv = parse_analytic(analytic_strings["luminosity_distance"])
    dL = sample_powerlaw(kv["alpha"], kv["minimum"], kv["maximum"], n, rng)

    # 2. d_L -> z (Planck15)
    print("  d_L -> z tablosu hesaplanıyor (Planck15)...")
    dL_grid, z_grid = build_dL_z_table(kv["minimum"], kv["maximum"], n_grid=1000)
    z = np.interp(dL, dL_grid, z_grid)

    # 3. Kütleler: chirp_mass (det) x mass_ratio -> source frame
    Mc_det = sample_uniform(Mc_min, Mc_max, n, rng)
    q      = sample_uniform(q_min,  q_max,  n, rng)
    m2_det = Mc_det * (1 + q)**(1/5) * q**(2/5)
    m1_det = m2_det / q
    m1_src = m1_det / (1 + z)
    m2_src = m2_det / (1 + z)

    # Kısıt: her iki kütle >= m_min
    mask = (m1_src >= m_min_constraint) & (m2_src >= m_min_constraint)
    n_ok = mask.sum()
    print(f"  Kütle kısıtı sonrası: {n_ok}/{n} geçerli")

    samples = {
        "luminosity_distance": dL[mask],
        "mass_1_source"      : m1_src[mask],
        "mass_2_source"      : m2_src[mask],
    }

    # 4. Diğer 12 parametre: analytic string'e göre
    OTHER_PARAMS = {
        "a_1": "a_1", "a_2": "a_2",
        "phi_12": "phi_12", "phi_jl": "phi_jl",
        "psi": "psi", "azimuth": "azimuth",
        "geocent_time": "geocent_time", "phase": "phase",
        "tilt_1": "tilt_1", "tilt_2": "tilt_2",
        "theta_jn": "theta_jn", "zenith": "zenith",
    }
    for param, key in OTHER_PARAMS.items():
        kind, kv = parse_analytic(analytic_strings[key])
        lo, hi   = kv["minimum"], kv["maximum"]
        if kind == "Sine":
            arr = sample_sine(lo, hi, n, rng)[mask]
        else:  # Uniform ve diğerleri
            arr = sample_uniform(lo, hi, n, rng)[mask]
        samples[param] = arr

    n_final = len(samples["mass_1_source"])
    print(f"  Final prior örnekleri: {n_final}")
    print(f"  mass_2_source  : [{samples['mass_2_source'].min():.3f}, "
          f"{samples['mass_2_source'].max():.3f}] M_sun")
    print(f"  lum. distance  : [{samples['luminosity_distance'].min():.1f}, "
          f"{samples['luminosity_distance'].max():.1f}] Mpc")
    return samples


# ==========================================================================
# VERİ OKUMA
# ==========================================================================

def load_post_prior(path, params):
    """Posterior HDF5'ten yükle; analytic prior tanımlarını oku."""
    with h5py.File(path, "r") as f:
        chosen = None
        for key in f.keys():
            g = f[key]
            if not isinstance(g, h5py.Group):
                continue
            if ("posterior_samples" in g and "priors" in g
                    and "samples" in g["priors"]):
                psamp = g["priors"]["samples"]
                if all(p in psamp for p in params):
                    chosen = key
                    break
        if chosen is None:
            raise ValueError("Posterior + prior içeren grup bulunamadı.")

        g        = f[chosen]
        post_tbl = g["posterior_samples"][()]
        post     = {p: np.asarray(post_tbl[p], dtype=float) for p in params}

        an_raw = g["priors"]["analytic"]
        analytic_strings = {k: an_raw[k][()][0].decode() for k in an_raw.keys()}

        _, kv_Mc = parse_analytic(analytic_strings["chirp_mass"])
        Mc_min, Mc_max = kv_Mc["minimum"], kv_Mc["maximum"]
        _, kv_q  = parse_analytic(analytic_strings["mass_ratio"])
        q_min, q_max   = kv_q["minimum"],  kv_q["maximum"]

    return post, analytic_strings, chosen, Mc_min, Mc_max, q_min, q_max


# ==========================================================================
# DESTEK TANISI
# ==========================================================================

def support_diagnosis(post, prior, params):
    print("\n" + "=" * 78)
    print("AŞAMA 2 — Prior/posterior DESTEK (support) tanısı")
    print("=" * 78)
    print(f"{'Parametre':<25} {'post[min,max]':>22} {'prior[min,max]':>20}  %dışarı")
    print("-" * 78)
    warnings_list = []
    for p in params:
        pst = post[p]; pri = prior[p]
        lo, hi = pri.min(), pri.max()
        frac = np.mean((pst < lo) | (pst > hi)) * 100
        flag = "  <-- DIKKAT" if frac > 1.0 else ""
        print(f"{p:<25} [{pst.min():>10.3g},{pst.max():>10.3g}]"
              f"   [{lo:>10.3g},{hi:>10.3g}]  {frac:>6.2f}%{flag}")
        if frac > 1.0:
            warnings_list.append((p, frac))
    print("-" * 78)
    if warnings_list:
        print("UYARI: Hâlâ destek uyumsuzluğu olan parametreler:")
        for p, f in warnings_list:
            print(f"  {p}: %{f:.2f} dışarıda")
    else:
        print("Tum parametreler prior destegi icinde (<%1).")
    return warnings_list


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

def avg_between_group_corr(groups, abscorr):
    vals = [abscorr[a, b]
            for i in range(len(groups)) for j in range(i+1, len(groups))
            for a in groups[i] for b in groups[j]]
    return float(np.mean(vals)) if vals else 0.0


# ==========================================================================
# ANA AKIŞ
# ==========================================================================

def main(path=None):
    here = os.path.dirname(os.path.abspath(__file__))

    if path is None:
        if file_name is not None:
            path = file_name
        else:
            cands = []
            for ext in ("*.h5", "*.hdf5"):
                cands += glob.glob(os.path.join(here, "..", "data", ext))
            if not cands:
                print("HATA: dosya yolu verin."); return
            path = cands[0]

    print("=" * 78)
    print("GW — <=5D grup KLD(posterior||prior) analizi  [kde / knn]")
    print("=" * 78)
    print(f"Dosya: {os.path.basename(path)}\n")

    (post, analytic_strings, grp,
     Mc_min, Mc_max, q_min, q_max) = load_post_prior(path, PARAMS_15)

    n_post_raw = len(post[PARAMS_15[0]])
    print(f"Analiz grubu : {grp}")
    print(f"Posterior    : {n_post_raw} örnek")

    rng = np.random.default_rng(RANDOM_STATE)
    prior_samples = generate_analytic_prior(
        analytic_strings, N_PRIOR, rng,
        Mc_min=Mc_min, Mc_max=Mc_max,
        q_min=q_min, q_max=q_max,
        m_min_constraint=1.0,
    )
    n_prior = len(prior_samples["mass_1_source"])
    print(f"Prior        : {n_prior} örnek (analitik)")

    P_raw = np.column_stack([post[p]          for p in PARAMS_15])
    Q_raw = np.column_stack([prior_samples[p] for p in PARAMS_15])

    # Tekilleştirme
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
                print(f"Tekilleştirme: {n0} -> {len(keep)} benzersiz örnek "
                      f"(sütunlar: {', '.join(PARAMS_15[j] for j in key_cols)})")
                P_raw = P_raw[keep]

    # Destek tanısı
    post_diag  = {p: P_raw[:, i] for i, p in enumerate(PARAMS_15)}
    prior_diag = {p: Q_raw[:, i] for i, p in enumerate(PARAMS_15)}
    support_diagnosis(post_diag, prior_diag, PARAMS_15)

    # Standardizasyon
    pool = np.vstack([P_raw, Q_raw])
    mu, sd = pool.mean(axis=0), pool.std(axis=0)
    sd[sd == 0] = 1.0
    P = (P_raw - mu) / sd
    Q = (Q_raw - mu) / sd
    P = P + rng.normal(0.0, 1e-10, size=P.shape)
    Q = Q + rng.normal(0.0, 1e-10, size=Q.shape)

    # ======== GRUPLAR ========
    # 0=mass_1_source, 1=mass_2_source, 2=a_1, 3=a_2, 4=tilt_1, 5=tilt_2,
    # 6=phi_12, 7=phi_jl, 8=luminosity_distance, 9=theta_jn, 10=psi,
    # 11=azimuth, 12=zenith, 13=geocent_time, 14=phase
    groups = [
        [0, 1],                    # G1: kütleler
        [2, 3, 4, 5],              # G2: spin büyüklükleri + tiltler
        [6, 7, 14],                # G3: açısal fazlar
        [8, 9, 10, 11, 12, 13],   # G4: mesafe + yön + zaman
    ]

    rho, _ = spearmanr(P)
    abscorr = np.abs(np.atleast_2d(rho))
    np.fill_diagonal(abscorr, 0.0)
    between = avg_between_group_corr(groups, abscorr)

    group_names = [[PARAMS_15[i] for i in g] for g in groups]
    print(f"\nGruplar arası ort. |Spearman| = {between:.3f}")
    for gi, gn in enumerate(group_names, 1):
        print(f"  G{gi} ({len(gn)}D): {', '.join(gn)}")

    # KLD hesabı
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

    # Tablo
    print("\n" + "-" * 78)
    hdr = f"{'Yöntem':<15}" + \
          "".join(f"{'G'+str(i):>10}" for i in range(1, len(groups)+1)) + \
          f"{'TOPLAM':>12}"
    print(hdr); print("-" * 78)
    totals = {}
    for m in METHODS:
        vals = results[m]
        tot  = float(np.nansum(vals))
        totals[m] = tot
        row = f"{m:<15}" + "".join(f"{v:>10.3f}" for v in vals) + f"{tot:>12.3f}"
        print(row)
    print("-" * 78)
    print("(nats)  |  bit:  " +
          "   ".join(f"{m}={totals[m]*NATS_TO_BITS:.2f}" for m in METHODS))

    mean_tot = float(np.mean(list(totals.values())))
    print(f"\nYöntem-ortalaması toplam KLD = {mean_tot:.3f} nats "
          f"({mean_tot*NATS_TO_BITS:.2f} bit)")

    # 1D marjinal alt sınır
    kld_1d = []
    for i in range(len(PARAMS_15)):
        Pi = P[:, i:i+1]; Qi = Q[:, i:i+1]
        try:   kld_1d.append(float(calc_knn_kld(Pi, Qi, k=1)))
        except: kld_1d.append(0.0)
    sum_1d = sum(kld_1d)
    print(f"Marjinal (1D) toplam [knn-k1] = {sum_1d*NATS_TO_BITS:.2f} bit  (alt sınır)")
    diff = totals["knn-k1"] - sum_1d
    sign = "pozitif (beklenen)" if diff >= 0 else "NEGATIF (tahminci gurultusu)"
    print(f"Grup-toplam - marjinal = {diff*NATS_TO_BITS:.2f} bit  [{sign}]")

    # JSON
    out = {
        "file": os.path.basename(path),
        "analysis_group": grp,
        "n_posterior": int(P.shape[0]),
        "n_prior_analytic": n_prior,
        "prior_source": "analytic (HDF5 priors/analytic)",
        "parameters": PARAMS_15,
        "groups": group_names,
        "avg_between_group_abs_corr": between,
        "methods": METHODS,
        "kld_per_group_nats": {m: results[m] for m in METHODS},
        "kld_total_nats": totals,
        "kld_total_bits": {m: totals[m] * NATS_TO_BITS for m in METHODS},
        "mean_total_bits": mean_tot * NATS_TO_BITS,
        "marginal_1d_total_bits": sum_1d * NATS_TO_BITS,
        "marginal_kld_1d_nats": {p: kld_1d[i] for i, p in enumerate(PARAMS_15)},
    }
    out_json = os.path.join(here, "results_grup_kld.json")
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nJSON kaydedildi: {out_json}")

    # Grafik
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ng = len(groups)
        x  = np.arange(ng + 1)
        w  = 0.75 / len(METHODS)
        labels = [f"G{i}" for i in range(1, ng+1)] + ["TOPLAM"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for k, m in enumerate(METHODS):
            heights = [v * NATS_TO_BITS for v in results[m]] + \
                      [totals[m] * NATS_TO_BITS]
            ax.bar(x + k*w, heights, w, label=m, alpha=0.85)
        ax.set_xticks(x + w)
        ax.set_xticklabels(labels)
        ax.set_ylabel("KLD (bit)")
        ax.set_title("Grup bazında KLD — analitik prior")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

        ax2 = axes[1]
        vals_bit = [v * NATS_TO_BITS for v in kld_1d]
        colors   = ["#e74c3c" if v > 3 else "#3498db" for v in vals_bit]
        ax2.barh(PARAMS_15, vals_bit, color=colors, alpha=0.85)
        ax2.set_xlabel("KLD 1D (bit)")
        ax2.set_title("Parametre bazında marjinal KLD")
        ax2.grid(axis="x", alpha=0.3)

        fig.suptitle(os.path.basename(path), fontsize=9)
        fig.tight_layout()
        out_png = os.path.join(here, "grup_kld_karsilastirma.png")
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        print(f"Grafik kaydedildi: {out_png}")
    except Exception as e:
        print(f"(Grafik atlandi: {e})")

    return out


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if path_arg is None and file_name is not None:
        print(f"\n>>> DOSYA: {file_name}\n")
        main(file_name)
    else:
        main(path_arg)