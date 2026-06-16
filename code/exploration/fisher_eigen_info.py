#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fisher_eigen_info.py
====================
Prior-bağımsız BİLGİ SIRALAMASI: GW posterior'undan Fisher özdeğerlerini çıkarıp
her PARAMETRE-KOMBİNASYONU (özyön) için bilgi içeriğini

        I_k = 1/2 * log2(1 + lambda_k)   [bit]

ile hesaplar. Burada lambda_k, prior ile "beyazlatılmış" (whitened) Fisher
matrisinin özdeğeridir = o yöndeki etkin SNR^2.

Neden bu, marjinal KLD sıralamasındaki "mesafe artefaktını" çözer?
  - Marjinal KLD her parametreyi tek başına ölçer; geniş prior'lı parametreler
    (luminosity_distance) yapay olarak yüksek bilgi gösterir.
  - Fisher özdeğerleri, prior kovaryansına göre normalize edilmiş ÖZYÖNLERİ
    sıralar. En büyük özdeğer doğrudan chirp-kütlesi yönüne düşer; mesafe-eğim
    dejenere bloğu küçük özdeğerlerde toplanır. Birim/prior etkisi otomatik gider.

Matematik (Gauss yaklaşımı):
  Sigma_post^-1 = F + Sigma_prior^-1   (Laplace)  =>  F = Sigma_post^-1 - Sigma_prior^-1
  Genelleştirilmiş özproblem:  Sigma_post v = mu * Sigma_prior v
      mu_k  = posterior/prior varyans oranı (özyönde)
      lambda_k = 1/mu_k - 1  = prior-beyazlatılmış Fisher özdeğeri (= SNR_k^2)
      I_k   = 1/2 log2(1+lambda_k) = -1/2 log2(mu_k)
  Toplam:  sum I_k = 1/2 log2(det Sigma_prior / det Sigma_post)   (hacim/Occam terimi)

Çapraz kontroller:
  (A) sum I_k  ==  doğrudan slogdet hacim terimi
  (B) sum lambda_k  ~  rho^2 (toplam optimal SNR^2)  -- "equipartition" varsayımının testi
  (C) equipartition kestirimi  I_equi = 1/2 N log2(1+rho^2/N)  vs gerçek sum I_k

Kullanım:
  python3 fisher_eigen_info.py <h5_dosyasi>            # tek olay
  python3 fisher_eigen_info.py                         # GW150914 (varsayılan)

Not: Loader önce pesummary dener; yoksa doğrudan h5py'ye düşer (test edilebilirlik).
"""

import os
import re
import sys
import json
import numpy as np
from scipy.linalg import eigh

# ─────────────────────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FILE = (
    "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/"
    "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
)

# 15 fiziksel parametre. (isim_posterior, isim_prior, donusum)
#   donusum: None | "log" | "cos"  -> Gauss yaklaşımını iyileştirmek için.
# chirp_mass + mass_ratio, m1/m2'den çok daha Gauss'tur; baskın özyön burada yaşar.
PARAM_SPEC = [
    ("chirp_mass",          "chirp_mass",          "log"),  # baskın yön burada
    ("mass_ratio",          "mass_ratio",          None),
    ("a_1",                 "a_1",                 None),
    ("a_2",                 "a_2",                 None),
    ("cos_tilt_1",          "cos_tilt_1",          None),
    ("cos_tilt_2",          "cos_tilt_2",          None),
    ("phi_12",              "phi_12",              None),
    ("phi_jl",              "phi_jl",              None),
    ("luminosity_distance", "luminosity_distance", "log"),  # mesafe: log daha Gauss
    ("cos_theta_jn",        "cos_theta_jn",        None),
    ("psi",                 "psi",                 None),
    ("azimuth",             "azimuth",             None),
    ("zenith",              "zenith",              None),
    ("geocent_time",        "geocent_time",        None),
    ("phase",               "phase",               None),
]

# İnsan-okur etiketler
PRETTY = {
    "chirp_mass": "Mc", "mass_ratio": "q", "a_1": "a1", "a_2": "a2",
    "cos_tilt_1": "cosθ1", "cos_tilt_2": "cosθ2", "phi_12": "φ12", "phi_jl": "φjl",
    "luminosity_distance": "dL", "cos_theta_jn": "cosθJN", "psi": "ψ",
    "azimuth": "az", "zenith": "zen", "geocent_time": "tc", "phase": "φ",
}


# ─────────────────────────────────────────────────────────────────────────────
# VERİ YÜKLEME (pesummary -> h5py fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _transform(arr, how):
    if how == "log":
        return np.log(np.asarray(arr, float))
    if how == "cos":
        return np.cos(np.asarray(arr, float))
    return np.asarray(arr, float)


def load_event(path):
    """posterior dict, prior dict, label, snr(array) döndürür."""
    try:
        from pesummary.io import read
        f = read(path, disable_conversion=True)
        label = f.labels[0]
        post = f.samples_dict[label]
        prior = f.priors["samples"][label]
        post = {k: np.asarray(post[k]) for k in post.keys()}
        prior = {k: np.asarray(prior[k]) for k in prior.keys()}
        snr = post.get("network_optimal_snr", post.get("network_matched_filter_snr"))
        return post, prior, label, np.asarray(snr) if snr is not None else None
    except Exception as e:
        print(f"[bilgi] pesummary kullanılamadı ({type(e).__name__}); h5py fallback.")
        import h5py
        f = h5py.File(path, "r")
        # IMRPhenomXPHM tercih; yoksa ilk uygun grup
        label = None
        for k in f.keys():
            if hasattr(f[k], "keys") and "posterior_samples" in f[k]:
                label = k
                if "IMRPhenomXPHM" in k:
                    break
        g = f[label]
        ps = g["posterior_samples"][()]
        post = {n: np.asarray(ps[n]) for n in ps.dtype.names}
        prs = g["priors/samples"]
        prior = {n: np.asarray(prs[n][()]) for n in prs.keys()}
        snr = post.get("network_optimal_snr", post.get("network_matched_filter_snr"))
        return post, prior, label, (np.asarray(snr) if snr is not None else None)


def build_matrix(samples, which):
    """PARAM_SPEC'e göre (N, d) matris kur. which='post' veya 'prior'."""
    cols, names, missing = [], [], []
    for p_post, p_prior, how in PARAM_SPEC:
        key = p_post if which == "post" else p_prior
        if key in samples:
            cols.append(_transform(samples[key], how))
            names.append(p_post)
        else:
            missing.append(key)
    if missing:
        print(f"[uyarı] {which}: bulunamayan parametreler atlandı -> {missing}")
    n = min(len(c) for c in cols)
    M = np.column_stack([c[:n] for c in cols])
    return M, names


def dedup_rows(M, decimals=10):
    """GWTC posterior'u her örneği ~2x kopyalar (iç paramlar birebir aynı).
    Aynı satırları tekilleştir (memory: 147634 -> ~71747)."""
    _, idx = np.unique(np.round(M, decimals), axis=0, return_index=True)
    return M[np.sort(idx)]


# ─────────────────────────────────────────────────────────────────────────────
# FISHER ÖZDEĞER ANALİZİ
# ─────────────────────────────────────────────────────────────────────────────
def fisher_eigen_analysis(P, Q):
    """
    P: posterior (Np, d), Q: prior (Nq, d)
    Genelleştirilmiş özproblem Sigma_post v = mu Sigma_prior v üzerinden
    lambda_k, I_k, ortalama-kayması (delta_k) ve yön yorumlarını döndürür.

    Tam Gauss KLD'nin özyön bazında ayrışımı (nats):
        KLD = 1/2 sum_k [ mu_k + delta_k^2 - 1 - ln mu_k ]
              \____ölçülebilirlik____/   \__sürpriz__/
        I_k  (ölçülebilirlik, bit) = -1/2 log2(mu_k) = 1/2 log2(1+lambda_k)
        KLD_k(tam, bit)            = 1/2 (mu_k + delta_k^2 - 1 - ln mu_k)/ln2
    """
    d = P.shape[1]
    mp = np.mean(P, axis=0)
    mq = np.mean(Q, axis=0)
    Sp = np.cov(P, rowvar=False)
    Sq = np.cov(Q, rowvar=False)

    # Sayısal güven: prior kovaryansa minik jitter (pozitif-tanımlılık)
    Sq = Sq + 1e-12 * np.eye(d) * np.trace(Sq) / d

    # eigh(a, b): a v = w b v,  V^T b V = I   ->  w = mu (varyans oranları)
    mu, V = eigh(Sp, Sq)                 # mu artan sırada
    mu = np.clip(mu, 1e-15, None)
    lam = 1.0 / mu - 1.0                 # prior-beyazlatılmış Fisher özdeğeri = SNR_k^2
    I_k = -0.5 * np.log2(mu)             # ölçülebilirlik biti = 1/2 log2(1+lam)

    # Ortalama kayması, genelleştirilmiş özbazda (prior-sigma birimi):
    #   V^-1 = V^T Sq  =>  delta = V^T Sq (mp - mq)
    delta = V.T @ (Sq @ (mp - mq))
    kld_k = 0.5 * (mu + delta**2 - 1.0 - np.log(mu)) / np.log(2)   # tam KLD katkısı, bit

    # Büyükten küçüğe (ölçülebilirlik) sırala
    order = np.argsort(I_k)[::-1]
    mu, lam, I_k, delta, kld_k, V = (mu[order], lam[order], I_k[order],
                                     delta[order], kld_k[order], V[:, order])

    # Yön yorumu: bileşenleri prior-sigma birimine çevir (boyutsuz yük)
    sig_q = np.sqrt(np.diag(Sq))
    loadings = V * sig_q[:, None]                      # (d, d)
    loadings = loadings / (np.linalg.norm(loadings, axis=0, keepdims=True) + 1e-30)

    return dict(mu=mu, lam=lam, I_k=I_k, delta=delta, kld_k=kld_k,
                V=V, loadings=loadings, Sp=Sp, Sq=Sq, sig_q=sig_q)


def describe_direction(loadings_col, names, topn=3):
    """Bir özyönü en baskın topn parametresiyle özetle."""
    w = loadings_col**2
    idx = np.argsort(w)[::-1][:topn]
    parts = []
    for i in idx:
        sgn = "+" if loadings_col[i] >= 0 else "−"
        parts.append(f"{sgn}{abs(loadings_col[i]):.2f}·{PRETTY.get(names[i], names[i])}")
    return "  ".join(parts)


# İç (intrinsic) parametre alt-kümesi — chirp-kütle baskınlığı burada en net görünür.
INTRINSIC = ["chirp_mass", "mass_ratio", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]


def run_subset(post, prior, spec_names, snr, title):
    """Belirli parametre alt-kümesi için tam analiz + tablo + çapraz kontrol."""
    global PARAM_SPEC
    full_spec = PARAM_SPEC
    PARAM_SPEC = [s for s in full_spec if s[0] in spec_names]
    try:
        P, names = build_matrix(post, "post")
        Q, _ = build_matrix(prior, "prior")
    finally:
        PARAM_SPEC = full_spec
    d = P.shape[1]
    P = dedup_rows(P)

    res = fisher_eigen_analysis(P, Q)
    lam, I_k, kld_k, delta, loadings = (res["lam"], res["I_k"], res["kld_k"],
                                        res["delta"], res["loadings"])

    print("\n" + "=" * 84)
    print(f"  {title}   (d={d}, N_post={len(P)}, N_prior={len(Q)})")
    print("=" * 84)
    print(f"{'#':>2} {'lambda_k':>10} {'SNR_k':>6} {'I_k[bit]':>8} {'KLD_k[bit]':>10} "
          f"{'%I':>4}  baskın kombinasyon")
    print("-" * 84)
    tot_I = float(np.sum(np.clip(I_k, 0, None)))
    rows = []
    for k in range(d):
        snr_k = np.sqrt(max(lam[k], 0.0))
        pct = 100 * max(I_k[k], 0) / tot_I if tot_I > 0 else 0
        desc = describe_direction(loadings[:, k], names)
        print(f"{k+1:>2} {lam[k]:>10.2f} {snr_k:>6.2f} {I_k[k]:>8.3f} "
              f"{kld_k[k]:>10.3f} {pct:>3.0f}  {desc}")
        rows.append(dict(rank=k + 1, lambda_k=float(lam[k]), snr_k=float(snr_k),
                         I_k_bits=float(I_k[k]), kld_k_bits=float(kld_k[k]),
                         pct=float(pct), direction=desc))

    # Çapraz kontroller
    _, ld_p = np.linalg.slogdet(res["Sp"])
    _, ld_q = np.linalg.slogdet(res["Sq"])
    I_volume = 0.5 * (ld_q - ld_p) / np.log(2)
    I_sum = float(np.sum(I_k))
    KLD_full = float(np.sum(kld_k))
    sum_lam = float(np.sum(np.clip(lam, 0, None)))
    sum_d2 = float(np.sum(delta**2))
    print("-" * 84)
    print(f"(A) Σ I_k = {I_sum:8.3f} bit   ≟  ½log2(detSq/detSp) = {I_volume:8.3f} bit"
          f"   (fark {abs(I_sum-I_volume):.1e})")
    print(f"    → ÖLÇÜLEBİLİRLİK bilgisi (prior-bağımsız sıralama): {I_sum:.2f} bit")
    print(f"(D) Σ KLD_k = {KLD_full:8.3f} bit   = tam Gauss KLD(post||prior)"
          f"   (ölçülebilirlik + ortalama-kayması/sürpriz)")
    if snr is not None:
        rho = float(np.median(snr))
        print(f"(B) Σλ_k = {sum_lam:9.1f}   |   Σδ_k² = {sum_d2:8.1f}   |   "
              f"ρ² = {rho**2:7.1f}  (ρ={rho:.2f})")
        N = d
        I_equi = 0.5 * N * np.log2(1 + rho**2 / N)
        print(f"(C) equipartition I=½N log2(1+ρ²/N) = {I_equi:6.2f} bit"
              f"  vs gerçek Σ KLD_k = {KLD_full:6.2f} bit")
    return dict(d=int(d), rows=rows, I_measurability=I_sum, KLD_full=KLD_full,
                sum_lambda=sum_lam, sum_delta2=sum_d2, names=names,
                lam=lam.tolist(), I_k=I_k.tolist(), kld_k=kld_k.tolist())


def main(path):
    m = re.search(r"(GW\d{6})", os.path.basename(path))
    event = m.group(1) if m else "Event"
    print("=" * 84)
    print(f"  FISHER ÖZDEĞER BİLGİ SIRALAMASI — {event}")
    print("=" * 84)
    print("  I_k  = ½log2(1+λ_k)  : ölçülebilirlik (prior-bağımsız) — 'mesafe artefaktı' yok")
    print("  KLD_k                : tam bilgi katkısı (ölçülebilirlik + ortalama-kayması)")

    post, prior, label, snr = load_event(path)
    print(f"\nlabel = {label}")

    full = run_subset(post, prior, [s[0] for s in PARAM_SPEC], snr,
                      "TAM 15-PARAMETRE")
    intr = run_subset(post, prior, INTRINSIC, snr,
                      "İÇ (INTRINSIC) ALT-UZAY — chirp-kütle baskınlığı")

    # Kaydet
    here = os.path.dirname(os.path.abspath(__file__))
    out = dict(event=event, label=str(label), full_15D=full, intrinsic=intr,
               rho=(float(np.median(snr)) if snr is not None else None))
    out_json = os.path.join(here, f"fisher_eigen_{event}.json")
    with open(out_json, "w") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"\n[kaydedildi] {out_json}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        for r, lab, c in [(full, "15D", "steelblue"), (intr, "intrinsic", "crimson")]:
            kk = np.arange(1, r["d"] + 1)
            ax[0].plot(kk, np.clip(r["I_k"], 0, None), "o-", color=c, label=lab)
            ax[1].semilogy(kk, np.clip(r["lam"], 1e-3, None), "o-", color=c, label=lab)
        ax[0].set_xlabel("özyön k"); ax[0].set_ylabel("I_k [bit]")
        ax[0].set_title(f"{event} — yön başına ölçülebilirlik"); ax[0].legend()
        ax[1].set_xlabel("özyön k"); ax[1].set_ylabel("λ_k = SNR_k²")
        ax[1].set_title("özdeğer spektrumu (log)"); ax[1].legend()
        for a in ax: a.grid(alpha=0.3)
        fig.tight_layout()
        pth = os.path.join(here, f"fisher_eigen_{event}.png")
        fig.savefig(pth, dpi=140)
        print(f"[kaydedildi] {pth}")
    except Exception as e:
        print(f"[bilgi] plot atlandı: {e}")
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    main(path)
