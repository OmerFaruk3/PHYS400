"""
GW — 15 parametre, ≤5D grup KLD(posterior||prior)  [HİBRİT PRIOR SÜRÜMÜ]
=========================================================================

gw_grup_kld_final.py ile AYNI yöntem; TEK fark PRIOR kaynağı:

  HİBRİT PRIOR
  ------------
  Her parametre için posterior, HDF5'teki ORİJİNAL prior örneklerinin
  (priors/samples) aralığını ne kadar aşıyor bakılır:
    - Aşım > %THRESH ise  -> o parametre için ANALİTİK prior üretilir
      (priors/analytic tanımından; support sorununu kapatır).
    - Aşım küçükse        -> o parametre ORİJİNAL prior örnekleriyle kalır.
  Böylece yalnızca gerçekten gereken parametreler (tipik olarak
  luminosity_distance, mass_*_source) analitik priora çevrilir; diğerleri
  dosyadaki gerçek prior örneklerini kullanır.

  NOT: GW priorları parametreler arası BAĞIMSIZdır (çarpım priorı), bu yüzden
  sütunları farklı kaynaklardan kurmak geçerlidir. Orijinal sütunlar
  N_PRIOR'a bootstrap ile yeniden örneklenir (uzunluk tutarlılığı için).

Marjinal (1D) alt sınır KDE ile hesaplanır (kNN-k1 1D'de tekrarlı değerlerde
şişer; KDE ince-grid 'gold' ile uyuşur).

Çıktı: results_grup_kld_hibrit_<EVENT>.json , grup_kld_hibrit_<EVENT>.png
Bağımlılıklar: numpy, scipy, h5py, astropy, matplotlib
"""

import os, sys, glob, json, re
from collections import Counter

import numpy as np
import h5py
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from scipy.special import digamma

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kde_estimators_reference import calc_kde_kld
from knn_estimators_reference import calc_knn_kld

NATS_TO_BITS = 1.0 / np.log(2.0)
RANDOM_STATE = 42
N_PRIOR      = 30000
MAX_POST     = None       # büyük event'lerde hız için ör. 15000 yap
TC_K         = 3
TC_NMAX      = 8000
THRESH_PCT   = 0.5        # posteriorun orijinal prior dışına taşma eşiği (%)

PARAMS_15 = [
    "mass_1_source", "mass_2_source", "a_1", "a_2", "tilt_1", "tilt_2",
    "phi_12", "phi_jl", "luminosity_distance", "theta_jn", "psi",
    "azimuth", "zenith", "geocent_time", "phase",
]
METHODS = ["kde-scott", "kde-silverman", "knn-k1"]

GROUPS = [
    [0, 1],            # G1 (2D): kütleler
    [2, 3, 4, 5],      # G2 (4D): spin büyüklükleri + tiltler
    [6, 7, 14],        # G3 (3D): açısal fazlar
    [8, 9, 10],        # G4 (3D): mesafe + eğim + polarizasyon
    [11, 12, 13],      # G5 (3D): gökyüzü konumu + zaman
]

# ========== İSTEDİĞİN DOSYAYI BURAYA YAZ ==========
file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
# file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191103_012549_PEDataRelease_mixed_cosmo.h5"
# file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191105_143521_PEDataRelease_mixed_cosmo.h5"


# ==========================================================================
# ANALİTİK ÖRNEKLEYİCİLER (final ile aynı)
# ==========================================================================
def sample_uniform(lo, hi, n, rng): return rng.uniform(lo, hi, n) # (phase, psi, azimuth) ya da spin büyüklükleri (a_1, a_2) 

def sample_sine(lo, hi, n, rng):     #tilt_1, tilt_2 (spin eğim açıları) ve zenith, theta_jn
    u = rng.uniform(0, 1, n)
    return np.arccos(np.cos(lo) - u * (np.cos(lo) - np.cos(hi)))

def sample_powerlaw(alpha, lo, hi, n, rng):  #luminosity_distance (d_L) için power-law prior: p(d_L) ~ d_L^alpha
    a1 = alpha + 1.0
    u  = rng.uniform(0, 1, n)
    return (u * (hi**a1 - lo**a1) + lo**a1) ** (1.0 / a1)

def build_dL_z_table(d_min=10.0, d_max=10000.0, n_grid=1000):
    """VEKTÖRLEŞTİRİLMİŞ d_L->z (Planck15)."""
    try:
        from astropy.cosmology import Planck15
        from astropy import units as u
        z_arr  = np.linspace(1e-4, 20.0, 8000)
        dL_z   = Planck15.luminosity_distance(z_arr).to(u.Mpc).value
        dL_arr = np.linspace(d_min, d_max, n_grid)
        return dL_arr, np.interp(dL_arr, dL_z, z_arr)
    except Exception:
        H0 = 67.74; c = 2.998e5
        dL_arr = np.linspace(d_min, d_max, n_grid)
        return dL_arr, dL_arr * H0 / c

def parse_analytic(s):
    m = re.match(r'(\w+)\((.+)\)', s)
    if not m: return None, {}
    kind, kv = m.group(1), {}
    for part in m.group(2).split(','):
        kvm = re.match(r'(\w+)\s*=\s*(.+)', part.strip())
        if kvm:
            k, v = kvm.group(1).strip(), kvm.group(2).strip()
            try:    kv[k] = float(v)
            except: kv[k] = v.strip("'\"")
    return kind, kv

def analytic_one(param, analytic_strings, n, rng):
    """Tek parametre için analitik prior örneği (mass_* HARİÇ)."""
    kind, kv = parse_analytic(analytic_strings[param])
    lo, hi = kv["minimum"], kv["maximum"]
    if kind == "Sine":     return sample_sine(lo, hi, n, rng)
    if kind == "PowerLaw": return sample_powerlaw(kv["alpha"], lo, hi, n, rng)
    return sample_uniform(lo, hi, n, rng)

def analytic_masses(analytic_strings, n, rng, Mc_min, Mc_max, q_min, q_max,
                    m_min=1.0):
    """chirp_mass x mass_ratio x d_L->z -> (m1_src, m2_src), uzunluk >= n hedefli."""
    over = int(n * 1.5) + 100
    _, kv = parse_analytic(analytic_strings["luminosity_distance"])
    dL = sample_powerlaw(kv["alpha"], kv["minimum"], kv["maximum"], over, rng)
    dL_grid, z_grid = build_dL_z_table(kv["minimum"], kv["maximum"], 1000)
    z = np.interp(dL, dL_grid, z_grid)
    Mc = sample_uniform(Mc_min, Mc_max, over, rng)
    q  = sample_uniform(q_min,  q_max,  over, rng)
    m2 = Mc * (1 + q)**(1/5) * q**(2/5)
    m1 = m2 / q
    m1s, m2s = m1 / (1 + z), m2 / (1 + z)
    ok = (m1s >= m_min) & (m2s >= m_min)
    return m1s[ok][:n], m2s[ok][:n]


# ==========================================================================
# VERİ OKUMA (orijinal prior örnekleri + analytic tanımlar)
# ==========================================================================
def load_post_orig_analytic(path, params):
    with h5py.File(path, "r") as f:
        chosen = None
        for key in f.keys():
            g = f[key]
            if not isinstance(g, h5py.Group): continue
            if ("posterior_samples" in g and "priors" in g
                    and "samples" in g["priors"]):
                if all(p in g["priors"]["samples"] for p in params):
                    chosen = key; break
        if chosen is None:
            raise ValueError("Posterior + prior içeren grup bulunamadı.")
        g = f[chosen]
        post_tbl = g["posterior_samples"][()]
        post   = {p: np.asarray(post_tbl[p], dtype=float) for p in params}
        q_orig = {p: np.asarray(g["priors"]["samples"][p][()], dtype=float)
                  for p in params}
        an_raw = g["priors"]["analytic"]
        analytic_strings = {k: an_raw[k][()][0].decode() for k in an_raw.keys()}
        _, kv_Mc = parse_analytic(analytic_strings["chirp_mass"])
        _, kv_q  = parse_analytic(analytic_strings["mass_ratio"])
    return (post, q_orig, analytic_strings, chosen,
            kv_Mc["minimum"], kv_Mc["maximum"], kv_q["minimum"], kv_q["maximum"])


# ==========================================================================
# HİBRİT PRIOR KURUCU
# ==========================================================================
def build_hybrid_prior(post, q_orig, analytic_strings, N, rng,
                       Mc_min, Mc_max, q_min, q_max, thresh_pct=THRESH_PCT):
    """Aşan parametreler için analitik, diğerleri için orijinal (bootstrap) prior."""
    print("\n" + "=" * 78)
    print(f"HİBRİT PRIOR — posterior, ORİJİNAL prior dışına > %{thresh_pct} taşıyor mu?")
    print("=" * 78)
    print(f"{'Parametre':<22}{'orijinal prior[min,max]':>26}{'%dışarı':>9}  kaynak")
    print("-" * 78)
    flags = {}
    for p in PARAMS_15:
        lo, hi = q_orig[p].min(), q_orig[p].max()
        frac = np.mean((post[p] < lo) | (post[p] > hi)) * 100
        flags[p] = frac > thresh_pct
        print(f"{p:<22}[{lo:>11.4g},{hi:>11.4g}]{frac:>8.2f}%  "
              f"{'ANALİTİK' if flags[p] else 'orijinal'}")
    # kütle eşleşmesi: biri aşıyorsa ikisi de analitik (aynı zincirden türer)
    if flags["mass_1_source"] or flags["mass_2_source"]:
        flags["mass_1_source"] = flags["mass_2_source"] = True
    print("-" * 78)

    Q = {}
    if flags["mass_1_source"]:
        m1, m2 = analytic_masses(analytic_strings, N, rng,
                                 Mc_min, Mc_max, q_min, q_max)
        Q["mass_1_source"], Q["mass_2_source"] = m1, m2
    for p in PARAMS_15:
        if p in ("mass_1_source", "mass_2_source") and flags["mass_1_source"]:
            continue
        if flags[p]:
            Q[p] = analytic_one(p, analytic_strings, N, rng)
        else:
            Q[p] = rng.choice(q_orig[p], N, replace=True)  # bootstrap
    # eşit uzunluğa hizala (kütle zinciri biraz kısa kalabilir)
    nmin = min(len(v) for v in Q.values())
    for p in PARAMS_15:
        if len(Q[p]) > nmin:
            Q[p] = Q[p][:nmin]
    used = [p for p in PARAMS_15 if flags[p]]
    print(f"ANALİTİK kullanılan: {', '.join(used) if used else '(hiçbiri)'}")
    print(f"Orijinal kullanılan: {len([p for p in PARAMS_15 if not flags[p]])} parametre")
    print(f"Hibrit prior örnek sayısı: {nmin}")
    return Q, flags


# ==========================================================================
# KLD / TC / tanı (final ile aynı)
# ==========================================================================
def kld_one(method, P, Q):
    if method == "kde-scott":     return calc_kde_kld(P, Q, bandwidth=None)
    if method == "kde-silverman": return calc_kde_kld(P, Q, bandwidth="silverman")
    if method == "knn-k1":        return calc_knn_kld(P, Q, k=1)
    raise ValueError(method)

def avg_between_group_corr(groups, abscorr):
    vals = [abscorr[a, b]
            for i in range(len(groups)) for j in range(i + 1, len(groups))
            for a in groups[i] for b in groups[j]]
    return float(np.mean(vals)) if vals else 0.0

def estimate_tc_groups(X, groups, k=3, nmax=8000, seed=0):
    rng = np.random.default_rng(seed)
    if X.shape[0] > nmax:
        X = X[rng.choice(X.shape[0], nmax, replace=False)]
    N, m = X.shape[0], len(groups)
    eps = cKDTree(X).query(X, k=k + 1, p=np.inf)[0][:, k]
    eps = np.maximum(eps, 1e-12)
    tc = digamma(k) + (m - 1) * digamma(N)
    for g in groups:
        tree_g = cKDTree(X[:, g])
        try:
            counts = tree_g.query_ball_point(X[:, g], eps - 1e-12, p=np.inf,
                                             return_length=True)
        except TypeError:
            counts = np.array([len(tree_g.query_ball_point(
                X[i, g], eps[i] - 1e-12, p=np.inf)) for i in range(N)])
        tc -= np.mean(digamma(np.maximum(counts, 1)))
    return float(tc)

def support_diagnosis(post, prior, params):
    print("\n" + "=" * 78)
    print("DESTEK TANISI (kullanılan hibrit prior ile)")
    print("=" * 78)
    print(f"{'Parametre':<22}{'post[min,max]':>24}{'prior[min,max]':>22}{'%dışarı':>9}")
    print("-" * 78)
    for p in params:
        pst, pri = post[p], prior[p]
        lo, hi = pri.min(), pri.max()
        frac = np.mean((pst < lo) | (pst > hi)) * 100
        flag = "  <-- DİKKAT" if frac > 1.0 else ""
        print(f"{p:<22}[{pst.min():>10.3g},{pst.max():>10.3g}]"
              f"  [{lo:>9.3g},{hi:>9.3g}]{frac:>8.2f}%{flag}")
    print("-" * 78)


# ==========================================================================
# ANA AKIŞ
# ==========================================================================
def main(path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    if path is None: path = file_name
    if path is None:
        cands = []
        for ext in ("*.h5", "*.hdf5"):
            cands += glob.glob(os.path.join(here, "..", "data", ext))
        if not cands: print("HATA: dosya yolu verin."); return
        path = cands[0]

    event = (re.search(r"(GW\d{6})", os.path.basename(path)) or [None])
    event = event.group(1) if event else "event"

    print("=" * 78)
    print(f"GW — ≤5D grup KLD [HİBRİT PRIOR]  [{event}]")
    print("=" * 78)
    print(f"Dosya: {os.path.basename(path)}\n")

    (post, q_orig, analytic_strings, grp,
     Mc_min, Mc_max, q_min, q_max) = load_post_orig_analytic(path, PARAMS_15)
    print(f"Analiz grubu : {grp}")
    print(f"Posterior    : {len(post[PARAMS_15[0]])} örnek")

    rng = np.random.default_rng(RANDOM_STATE)
    prior_samples, flags = build_hybrid_prior(
        post, q_orig, analytic_strings, N_PRIOR, rng,
        Mc_min, Mc_max, q_min, q_max)
    n_prior = len(prior_samples[PARAMS_15[0]])

    P_raw = np.column_stack([post[p]          for p in PARAMS_15])
    Q_raw = np.column_stack([prior_samples[p] for p in PARAMS_15])

    # Tekilleştirme
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
                print(f"\nTekilleştirme: {n0} -> {len(keep)} benzersiz örnek")
                P_raw = P_raw[keep]

    if MAX_POST is not None and P_raw.shape[0] > MAX_POST:
        P_raw = P_raw[rng.choice(P_raw.shape[0], MAX_POST, replace=False)]
        print(f"Posterior alt-örnekleme: {MAX_POST}")

    support_diagnosis({p: P_raw[:, i] for i, p in enumerate(PARAMS_15)},
                      {p: Q_raw[:, i] for i, p in enumerate(PARAMS_15)}, PARAMS_15)

    pool = np.vstack([P_raw, Q_raw])
    mu, sd = pool.mean(axis=0), pool.std(axis=0); sd[sd == 0] = 1.0
    P = (P_raw - mu) / sd + rng.normal(0.0, 1e-10, size=P_raw.shape)
    Q = (Q_raw - mu) / sd + rng.normal(0.0, 1e-10, size=Q_raw.shape)

    rho, _ = spearmanr(P)
    abscorr = np.abs(np.atleast_2d(rho)); np.fill_diagonal(abscorr, 0.0)
    between = avg_between_group_corr(GROUPS, abscorr)
    group_names = [[PARAMS_15[i] for i in g] for g in GROUPS]
    print(f"\nGruplar arası ort. |Spearman| = {between:.3f}")
    for gi, gn in enumerate(group_names, 1):
        print(f"  G{gi} ({len(gn)}D): {', '.join(gn)}")

    results = {m: [] for m in METHODS}
    for gi, g in enumerate(GROUPS, 1):
        for m in METHODS:
            try:    results[m].append(float(kld_one(m, P[:, g], Q[:, g])))
            except Exception as e:
                results[m].append(float("nan")); print(f"  (G{gi} {m} hata: {e})")

    print("\n" + "-" * 78)
    print(f"{'Yöntem':<15}" + "".join(f"{'G'+str(i):>10}" for i in range(1, len(GROUPS)+1))
          + f"{'TOPLAM':>12}"); print("-" * 78)
    totals = {}
    for m in METHODS:
        tot = float(np.nansum(results[m])); totals[m] = tot
        print(f"{m:<15}" + "".join(f"{v:>10.3f}" for v in results[m]) + f"{tot:>12.3f}")
    print("-" * 78)
    print("(nats)  |  bit: " + "   ".join(f"{m}={totals[m]*NATS_TO_BITS:.2f}" for m in METHODS))
    mean_tot = float(np.mean(list(totals.values())))
    print(f"Yöntem-ortalaması grup-toplam KLD = {mean_tot:.3f} nats ({mean_tot*NATS_TO_BITS:.2f} bit)")

    print("\n" + "=" * 78)
    print("GRUPLAR-ARASI TC DÜZELTMESİ")
    print("=" * 78)
    tc_P = estimate_tc_groups(P, GROUPS, k=TC_K, nmax=TC_NMAX, seed=RANDOM_STATE)
    tc_Q = estimate_tc_groups(Q, GROUPS, k=TC_K, nmax=TC_NMAX, seed=RANDOM_STATE + 1)
    correction = tc_P - tc_Q
    print(f"TC(post)={tc_P:+.3f}  TC(prior)={tc_Q:+.3f}  Düzeltme={correction:+.3f} nats "
          f"({correction*NATS_TO_BITS:+.2f} bit)")
    joint_est = {m: totals[m] + correction for m in METHODS}
    joint_mean = mean_tot + correction
    print(">>> JOINT 15D KLD tahmini (grup-toplam + TC):  "
          + "   ".join(f"{m}={joint_est[m]*NATS_TO_BITS:.2f}bit" for m in METHODS))
    print(f"    yöntem-ort. = {joint_mean*NATS_TO_BITS:.2f} bit")

    # 1D marjinal alt sınır (KDE — 1D'de güvenilir)
    kld_1d = []
    for i in range(len(PARAMS_15)):
        try:    kld_1d.append(float(calc_kde_kld(P[:, i:i+1], Q[:, i:i+1], bandwidth=None)))
        except: kld_1d.append(0.0)
    sum_1d = sum(kld_1d)
    print(f"\nHiyerarşi (bit):  marjinal(1D,KDE)={sum_1d*NATS_TO_BITS:.2f}  ≤  "
          f"grup-toplam={mean_tot*NATS_TO_BITS:.2f}  ≈  joint(+TC)={joint_mean*NATS_TO_BITS:.2f}")
    if sum_1d > mean_tot + 1e-9:
        print("  UYARI: marjinal > grup-toplam -> tahminci tutarsızlığı (beklenmedik).")

    out = {
        "event": event, "file": os.path.basename(path), "analysis_group": grp,
        "prior_source": "HYBRID (analytic for overflow params, original otherwise)",
        "hybrid_analytic_params": [p for p in PARAMS_15 if flags[p]],
        "hybrid_original_params": [p for p in PARAMS_15 if not flags[p]],
        "overflow_threshold_pct": THRESH_PCT,
        "n_posterior": int(P.shape[0]), "n_prior": int(n_prior),
        "parameters": PARAMS_15, "groups": group_names,
        "avg_between_group_abs_corr": between, "methods": METHODS,
        "kld_per_group_nats": {m: results[m] for m in METHODS},
        "kld_group_total_nats": totals,
        "kld_group_total_bits": {m: totals[m] * NATS_TO_BITS for m in METHODS},
        "tc_posterior_nats": tc_P, "tc_prior_nats": tc_Q,
        "tc_correction_nats": correction,
        "joint_kld_estimate_bits": {m: joint_est[m] * NATS_TO_BITS for m in METHODS},
        "joint_kld_estimate_mean_bits": joint_mean * NATS_TO_BITS,
        "group_total_mean_bits": mean_tot * NATS_TO_BITS,
        "marginal_1d_total_bits": sum_1d * NATS_TO_BITS,
        "marginal_1d_method": "kde-scott",
        "marginal_kld_1d_nats": {p: kld_1d[i] for i, p in enumerate(PARAMS_15)},
    }
    out_json = os.path.join(here, f"results_grup_kld_hibrit_{event}.json")
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nJSON kaydedildi: {out_json}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ng = len(GROUPS); x = np.arange(ng + 1); w = 0.75 / len(METHODS)
        labels = [f"G{i}" for i in range(1, ng + 1)] + ["Total"]
        
        # İlk grafik: Group KLD
        fig1, ax = plt.subplots(figsize=(10, 5))
        for k, m in enumerate(METHODS):
            h = [v * NATS_TO_BITS for v in results[m]] + [totals[m] * NATS_TO_BITS]
            ax.bar(x + k * w, h, w, label=m, alpha=0.85)
        ax.set_xticks(x + w); ax.set_xticklabels(labels, fontsize=12); ax.set_ylabel("KLD (bit)", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_title(f"{event} — Group KLD with Hybrid Prior"); ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig1.tight_layout()
        out_png1 = os.path.join(here, f"grup_kld_hibrit_groups_{event}.png")
        fig1.savefig(out_png1, dpi=130, bbox_inches="tight")
        print(f"Grafik 1 kaydedildi: {out_png1}")
        
        # İkinci grafik: 1D Marginal KLD
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        vb = [v * NATS_TO_BITS for v in kld_1d]
        cols = ["#e74c3c" if flags[p] else "#3498db" for p in PARAMS_15]
        ax2.barh(PARAMS_15, vb, color=cols, alpha=0.85)
        ax2.set_xlabel("KLD 1D (bit)  [red = Analytic prior, blue = Original prior]", fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.set_title("Marginal (1D) KLD"); ax2.grid(axis="x", alpha=0.3)
        fig2.tight_layout()
        out_png2 = os.path.join(here, f"grup_kld_hibrit_marginal_{event}.png")
        fig2.savefig(out_png2, dpi=130, bbox_inches="tight")
        print(f"Grafik 2 kaydedildi: {out_png2}")
    except Exception as e:
        print(f"(Grafik atlandı: {e})")
    return out


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg if arg is not None else file_name)
