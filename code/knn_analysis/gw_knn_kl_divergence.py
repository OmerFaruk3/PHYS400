"""
=============================================================================
GW olaylarindan elde edilen TOPLAM BILGI (bit) — k-NN KL Iraksamasi
=============================================================================
Amac
----
Bir GW olayinin posterior dagiliminin prior'a gore KL iraksamasini hesaplamak:
    I = D_KL(posterior || prior)          [bit]
Bu, gozlemden parametreler hakkinda kazanilan toplam bilgi miktaridir
(information gain). GW150914 icin Flanagan & Hughes (1998) analitik tahmini
~41.5 bit, Gaussian ortak (joint) yaklasimi ~41.24 bit.

Yontem (makaledeki estimator)
-----------------------------
Alvarez Chaves et al. (2024, Entropy 26(5):387) toolbox'undaki k-NN KL
tahmincisi `calc_knn_kld` ile BIREBIR ayni estimator: Wang, Kulkarni & Verdu
(2009) = Perez-Cruz (2008, NeurIPS, Denklem 14):

    D_KL(P||Q) = (d/n) * sum_i log( nu_k(i) / rho_k(i) ) + log( m / (n-1) )

    rho_k(i) : x_i'nin P (posterior) icindeki k. komsuya uzakligi (kendisi haric)
    nu_k(i)  : x_i'nin Q (prior)     icindeki k. komsuya uzakligi
    d = boyut (15), n = posterior orneklem sayisi, m = prior orneklem sayisi
    p-norm = 2 (Oklid) — makale KLD icin p=2 ve k=1 onerir.

Makale dogal logaritma (nats) kullanir; biz sonucu bit'e ceviririz
(bit = nat / ln2). Bu, log2 ile hesaplamakla matematiksel olarak ozdestir.

Proje konvansiyonlari (mevcut Codes/ ile tutarli)
-------------------------------------------------
- Analiz etiketi : C01:IMRPhenomXPHM
- 15 F&H parametresi: mass_1_source, mass_2_source temel (turetilmis
  chirp_mass/mass_ratio DEGIL); gokyuzu icin azimuth/zenith (prior dosyasinda
  ra/dec yerine bunlar saklanir; ikisi sabit kuresel rotasyon => izometrik =>
  KL'yi degistirmez). Hem P hem Q ayni koordinatlarda.
- Standardizasyon : z-skor, PRIOR ortalama/std referans (KL affine-degismez).
- Tum posterior orneklemleri kullanilir (~147k).

Kullanim
--------
    python gw_knn_kl_divergence.py [olay_h5_dosyasi]
"""

import os
import sys
import time
import numpy as np
from scipy.spatial import cKDTree

LN2 = np.log(2.0)

# 15 Flanagan & Hughes (1998) parametresi — hem posterior hem prior'da mevcut
PARAMS_15 = [
    "mass_1_source", "mass_2_source", "a_1", "a_2", "tilt_1", "tilt_2",
    "phi_12", "phi_jl", "luminosity_distance", "azimuth", "zenith",
    "theta_jn", "psi", "geocent_time", "phase",
]

DEFAULT_FILE = ("/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/"
                "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5")
LABEL = "C01:IMRPhenomXPHM"


# -------------------------------------------------------------------------
# Makaledeki estimator (Wang+2009 / Perez-Cruz 2008) — bit doner
# -------------------------------------------------------------------------
def knn_kld_bits(P, Q, k=1, workers=-1):
    """D_KL(P||Q) [bit].  P: posterior (n,d), Q: prior (m,d).

    Makalenin calc_knn_kld'si ile ayni formul; hiz icin cKDTree + workers,
    sonuc bit'e cevrilir. Negatif kucuk degerler tahmin gurultusudur (0'a
    kirpilmaz; ham deger raporlanir).
    """
    P = np.atleast_2d(P)
    Q = np.atleast_2d(Q)
    n, d = P.shape
    m = Q.shape[0]

    Ptree = cKDTree(P)
    Qtree = cKDTree(Q)

    # rho: P icindeki k. komsu (kendisi haric -> k+1 sorgu, son sutun)
    rho = Ptree.query(P, k=k + 1, p=2, workers=workers)[0][:, k]
    # nu: Q icindeki k. komsu (kendisi yok -> k sorgu)
    nu = Qtree.query(P, k=k, p=2, workers=workers)[0]
    nu = nu if k == 1 else nu[:, k - 1]

    # log(0) guvenligi
    rho = np.maximum(rho, 1e-15)
    nu = np.maximum(nu, 1e-15)

    kld_nats = (d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1.0))
    return kld_nats / LN2  # -> bit


# -------------------------------------------------------------------------
# Veri yukleme (h5py, pesummary gerekmez)
# -------------------------------------------------------------------------
def load_event(path, label=LABEL, params=PARAMS_15):
    """Sadece gereken parametreleri oku (tum 137 alani okumak cok yavas)."""
    import h5py
    with h5py.File(path, "r") as f:
        if label not in f:
            label = [k for k in f.keys() if k.startswith("C01")][0]
        g = f[label]
        ps = g["posterior_samples"]
        avail_post = set(ps.dtype.names)
        prg = g["priors"]["samples"]
        avail_pri = set(prg.keys())
        use = [p for p in params if p in avail_post and p in avail_pri]
        post = {p: np.asarray(ps[p], dtype=float) for p in use}
        pri = {p: np.asarray(prg[p], dtype=float) for p in use}
    return post, pri, label


def build_matrices(post, pri, params):
    """Prior referansli z-skor ile P (posterior) ve Q (prior) matrisleri."""
    Pc, Qc = [], []
    for p in params:
        pv = post[p]
        qv = pri[p]
        mu, sd = np.mean(qv), np.std(qv)
        sd = sd if sd > 1e-15 else 1.0
        Pc.append((pv - mu) / sd)
        Qc.append((qv - mu) / sd)
    return np.column_stack(Pc), np.column_stack(Qc)


def dedup_resampling_copies(P):
    """Yeniden-orneklemeden gelen kopya satirlari ele.

    GWTC-2.1 'mixed_cosmo' dosyasinda her ornek bir kez kopyalanmistir (redshift'ten
    BAGIMSIZ ic parametreler -spin, egim, acilar- ikizler arasinda BIREBIR aynidir;
    yalnizca dL/gokyuzu/zaman yeniden cizilir). Bu kopyalar k-NN'in iid varsayimini
    bozar: k=1'de en yakin komsu kendi 'ikizi' olur, rho~0 -> KL devasa sisar.
    Cozum: ikizler arasinda ayni kalan sutun blogunu (en cok sutunun PAYLASTIGI,
    N'den kucuk benzersiz-deger sayisi) tespit edip ona gore tekillestirmek.
    """
    n, d = P.shape
    uniq_counts = [len(np.unique(P[:, j])) for j in range(d)]
    from collections import Counter
    cnt = Counter(c for c in uniq_counts if c < 0.95 * n)
    if not cnt:
        return np.arange(n)
    modal, freq = cnt.most_common(1)[0]
    if freq < 2:
        return np.arange(n)
    key_cols = [j for j in range(d) if uniq_counts[j] == modal]
    _, keep = np.unique(P[:, key_cols], axis=0, return_index=True)
    return np.sort(keep)


def main(path=None):
    path = path or DEFAULT_FILE
    event = os.path.basename(path).split("_PEDataRelease")[0]

    print("=" * 64)
    print("GW olayindan toplam bilgi:  I = D_KL(posterior || prior)  [bit]")
    print("Estimator: Wang+2009 / Perez-Cruz 2008 (makale calc_knn_kld)")
    print("=" * 64)
    print(f"Olay   : {event}")

    post, pri, label = load_event(path)
    print(f"Etiket : {label}")

    params = [p for p in PARAMS_15 if p in post and p in pri]
    if len(params) != len(PARAMS_15):
        eksik = set(PARAMS_15) - set(params)
        print(f"UYARI: eksik parametreler atlandi: {eksik}")
    print(f"Boyut  : {len(params)} parametre")

    P, Q = build_matrices(post, pri, params)
    n_raw = P.shape[0]
    keep = dedup_resampling_copies(P)
    if len(keep) < n_raw:
        print(f"Tekillestirme: {n_raw:,} -> {len(keep):,} GERCEK bagimsiz ornek "
              f"(yeniden-orneklenmis kopyalar elendi)")
        P = P[keep]
    print(f"Posterior (P): {P.shape[0]:,} ornek | Prior (Q): {Q.shape[0]:,} ornek")
    print(f"Bias terimi log2(m/(n-1)) = {np.log2(Q.shape[0]/(P.shape[0]-1)):.3f} bit\n")

    # Makale k=1 onerir; kararliligi gormek icin birkac k
    results = {}
    for k in (1,2,3,4,5,6,7,8,9,10,20,50,100):
        t0 = time.perf_counter()
        val = knn_kld_bits(P, Q, k=k)
        dt = time.perf_counter() - t0
        results[k] = val
        flag = "  <- onerilen (makale)" if k == 1 else ""
        print(f"  k={k:>2}:  I = {val:8.2f} bit   ({dt:4.1f} s){flag}")

    print("\n" + "-" * 64)
    print(f"  SONUC (k=1):  I = {results[1]:.2f} bit")
    print(f"  Referans   :  F&H 1998 ~41.5 bit | Gaussian joint ~41.24 bit")
    print("-" * 64)

    # Sonuclari kaydet
    import json
    out = {
        "event": event, "label": label, "dimension": len(params),
        "parameters": params,
        "n_posterior": int(P.shape[0]), "n_prior": int(Q.shape[0]),
        "KL_posterior_given_prior_bits": {str(k): float(v) for k, v in results.items()},
        "reference_bits": {"Flanagan_Hughes_1998": 41.5, "Gaussian_joint": 41.24},
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"kl_knn_{event}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nKaydedildi: {out_path}")
    return results


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
