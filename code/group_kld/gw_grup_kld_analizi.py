"""
GW — 15 parametreyi ≤5D gruplara bölüp KLD(posterior||prior) hesabı (bin/kde/knn)
=================================================================================

Amaç
----
Makaledeki (Álvarez Chaves et al., 2024, Entropy 26(5):387) mantıkla: yüksek
boyutta (15D) bilgi-teorik kestirim güvenilmezdir. Bu yüzden 15 parametreyi
≤5D gruplara böleriz ve her grupta, yakınsadığını bildiğimiz GÜVENİLİR
estimatörlerle KLD(posterior || prior) hesaplarız. Üç bağımsız yöntem
(binning, KDE, kNN) karşılaştırılır; uyuşmaları sonuca güven verir.

Neden 5+5+5 ve korelasyona göre?
--------------------------------
Grup KLD'lerinin TOPLAMI, gruplar birbirinden BAĞIMSIZ ise ortak (joint) 15D
KLD'ye eşittir. Parametreleri korelasyona göre öyle gruplarız ki gruplar ARASI
bağımlılık en aza iner -> toplam, joint KLD'ye iyi bir yaklaşım olur. (Grup İÇİ
korelasyonlar zaten ≤5D estimatör tarafından yakalanır.)

Çıktı
-----
- Konsolda yöntem x grup tablosu + toplamlar
- results_grup_kld.json
- grup_kld_karsilastirma.png  (yöntemlerin grup bazında karşılaştırması)

Kullanım
--------
    python gw_grup_kld_analizi.py [posterior_dosyasi.h5]
    (argümansız: ../data/ içindeki ilk .h5/.hdf5 dosyası kullanılır)

Bağımlılıklar: numpy, scipy, h5py, matplotlib
"""

import os
import sys
import glob
import json

import numpy as np
import h5py
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bin_estimators_reference import estimate_ideal_bins, calc_bin_kld
from kde_estimators_reference import calc_kde_kld
from knn_estimators_reference import calc_knn_kld

NATS_TO_BITS = 1.0 / np.log(2.0)
MAX_POST = None    # posterior alt-örnekleme (hız); prior zaten ~5000
RANDOM_STATE = None

# GW150914 için 15 standart CBC parametresi (kullanıcının mevcut çalışmasıyla aynı)
PARAMS_15 = [
    "mass_1_source", "mass_2_source", "a_1", "a_2", "tilt_1", "tilt_2",
    "phi_12", "phi_jl", "luminosity_distance", "theta_jn", "psi",
    "azimuth", "zenith", "geocent_time", "phase",
]

# Üç yöntem ve hiper-parametreleri (makaledeki gibi birden çok bin/bw kuralı)
# "bin-scott", "bin-fd",  # Binning yöntemleri yorum satırına çevrildi
METHODS = ["kde-scott", "kde-silverman", "knn-k1"]

# ========== İSTEDİĞİN DOSYAYI BURAYA YAZ ==========
# file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
# file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191103_012549_PEDataRelease_mixed_cosmo.h5"
file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191105_143521_PEDataRelease_mixed_cosmo.h5"

print(f"[DEBUG] Seçilen dosya: {file_name}")  # Debug için


# --------------------------------------------------------------------------
# 1) GWTC PEDataRelease (pesummary) HDF5'ten posterior + prior oku
# --------------------------------------------------------------------------
def load_post_prior(path, params):
    """Bir analiz grubu seç (hem posterior_samples hem dolu priors/samples olan)
    ve istenen parametreler için (posterior, prior) sözlüklerini döndür."""
    with h5py.File(path, "r") as f:
        # Posterior tablosu + prior örnekleri olan analiz grubunu bul
        chosen = None
        for key in f.keys():
            g = f[key]
            if not isinstance(g, h5py.Group):
                continue
            if "posterior_samples" in g and "priors" in g and "samples" in g["priors"]:
                psamp = g["priors"]["samples"]
                have = all(p in psamp for p in params) and \
                    all(psamp[p].shape and psamp[p].shape[0] > 50 for p in params if p in psamp)
                if "posterior_samples" in g and have:
                    chosen = key
                    break
        if chosen is None:
            raise ValueError("Tüm 15 parametre için posterior+prior içeren grup bulunamadı.")

        g = f[chosen]
        post_tbl = g["posterior_samples"][()]
        post = {p: np.asarray(post_tbl[p], dtype=float) for p in params}
        prior = {p: np.asarray(g["priors"]["samples"][p][()], dtype=float) for p in params}
    return post, prior, chosen


# --------------------------------------------------------------------------
# 2) Korelasyona göre gruplama (gruplar arası bağımlılığı en aza indir)
#    (mevcut knn script'iyle aynı açgözlü mantık)
# --------------------------------------------------------------------------
def group_by_independence(abscorr, sizes):
    n = abscorr.shape[0]
    remaining = list(range(n))
    groups = []
    for size in sizes:
        if not remaining:
            break
        size = min(size, len(remaining))
        sub = np.array(remaining)
        totals = abscorr[np.ix_(sub, sub)].sum(axis=1)
        seed = int(sub[int(np.argmax(totals))])
        group = [seed]
        remaining.remove(seed)
        while len(group) < size and remaining:
            best, best_val = None, -1.0
            for r in remaining:
                val = float(np.mean([abscorr[r, gg] for gg in group]))
                if val > best_val:
                    best_val, best = val, r
            group.append(int(best))
            remaining.remove(int(best))
        groups.append(group)
    if remaining and groups:
        groups[-1].extend(int(r) for r in remaining)
    return groups


def avg_between_group_corr(groups, abscorr):
    vals = [abscorr[a, b]
            for i in range(len(groups)) for j in range(i + 1, len(groups))
            for a in groups[i] for b in groups[j]]
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------
# 3) Tek grupta tek yöntemle KLD(p||q)
# --------------------------------------------------------------------------
def kld_one(method, P, Q):
    """P: posterior (n,d), Q: prior (m,d) — standardize edilmiş. nats döndürür."""
    if method.startswith("bin-"):
        rule = method.split("-")[1]
        nbins = estimate_ideal_bins(Q, counts=False)   # kenarlar prior desteğinden
        return calc_bin_kld(P, Q, nbins[rule])
    if method == "kde-scott":
        return calc_kde_kld(P, Q, bandwidth=None)        # scipy varsayılan = scott
    if method == "kde-silverman":
        return calc_kde_kld(P, Q, bandwidth="silverman")
    if method == "knn-k1":
        return calc_knn_kld(P, Q, k=1)
    raise ValueError(method)


# --------------------------------------------------------------------------
# Ana akış
# --------------------------------------------------------------------------
def main(path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Eğer path doğrudan geçilmişse onu kullan (en yüksek öncelik)
    # 2. Değilse, file_name global değişkenini kontrol et
    # 3. İkisi de tanımlı değilse, ../data/ klasöründe ara
    if path is None:
        if file_name is not None:
            path = file_name
        else:
            cands = []
            for ext in ("*.h5", "*.hdf5"):
                cands += glob.glob(os.path.join(here, "..", "data", ext))
            if not cands:
                print("HATA: ../data/ içine GW posterior .h5 dosyasını koyun "
                      "ya da yolu argüman verin.")
                return
            path = cands[0]

    print("=" * 78)
    print("GW — ≤5D grup KLD(posterior||prior) analizi  [bin / kde / knn]")
    print("=" * 78)
    print(f"[DEBUG] Kullanılacak dosya: {path}")
    print()

    post, prior, grp = load_post_prior(path, PARAMS_15)
    print(f"Dosya        : {os.path.basename(path)}")
    print(f"Analiz grubu : {grp}")

    # (N,15) matrislere çevir
    P_raw = np.column_stack([post[p] for p in PARAMS_15])
    Q_raw = np.column_stack([prior[p] for p in PARAMS_15])
    print(f"Posterior    : {P_raw.shape[0]} örnek | Prior: {Q_raw.shape[0]} örnek")

    # --- Tekilleştirme (GWTC yeniden-örnekleme artefaktını gider) ---
    # GWTC-2.1/3 veri yayınlarında örnekler kopyalanır: iç (intrinsic) parametreler
    # birebir aynı, mesafe/zaman yeniden çizilir. Bu, bir grubun alt-uzayında
    # birebir aynı noktalar üretir -> kNN uzaklığı rho=0 (log -inf / şişme). Aynı
    # benzersiz-değer sayısını PAYLAŞAN en kalabalık sütun bloğu kopya bloğudur;
    # bu sütunlara göre tekilleştirip her benzersiz örnekten birini tutarız.
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
                      f"(kopya sütunlar: {', '.join(PARAMS_15[j] for j in key_cols)})")
                P_raw = P_raw[keep]

    # Posterior alt-örnekleme (hız)
    rng = np.random.default_rng(RANDOM_STATE)
    if MAX_POST is not None and P_raw.shape[0] > MAX_POST:
        idx = rng.choice(P_raw.shape[0], MAX_POST, replace=False)
        P_raw = P_raw[idx]
        print(f"Posterior alt-örnekleme: {MAX_POST} örnek")

    # Ortak (özdeş) standardizasyon: KLD bu doğrusal dönüşüme göre DEĞİŞMEZ;
    # yalnızca estimatörlerin sayısal kararlılığını artırır (ör. geocent_time
    # ~1e9 ölçeği). Aynı dönüşüm hem P hem Q'ya uygulanır.
    pool = np.vstack([P_raw, Q_raw])
    mu, sd = pool.mean(axis=0), pool.std(axis=0)
    sd[sd == 0] = 1.0
    P = (P_raw - mu) / sd
    Q = (Q_raw - mu) / sd

    # Çok küçük jitter: alt-uzaylardaki birebir aynı noktaları kır (knn log(0) önlemi)
    P = P + rng.normal(0.0, 1e-10, size=P.shape)
    Q = Q + rng.normal(0.0, 1e-10, size=Q.shape)

    # ======== OTOMATIK GRUPLAMA (YORUM SATIRI) ========
    # rho, _ = spearmanr(P)
    # abscorr = np.abs(np.atleast_2d(rho))
    # np.fill_diagonal(abscorr, 0.0)
    # groups = group_by_independence(abscorr, [5, 5, 5])
    # between = avg_between_group_corr(groups, abscorr)

    # ======== EL İLE AYARLANAN GRUPLAR ========
    # İndeksleri PARAMS_15 listesine göre belirtin
    # PARAMS_15 = [
    #     "mass_1_source",        # 0
    #     "mass_2_source",        # 1
    #     "a_1",                  # 2
    #     "a_2",                  # 3
    #     "tilt_1",               # 4
    #     "tilt_2",               # 5
    #     "phi_12",               # 6
    #     "phi_jl",               # 7
    #     "luminosity_distance",  # 8
    #     "theta_jn",             # 9
    #     "psi",                  # 10
    #     "azimuth",              # 11
    #     "zenith",               # 12
    #     "geocent_time",         # 13
    #     "phase",                # 14
    # ]
    groups = [
        [0, 1],                            # Grup 1: mass_1_source, mass_2_source
        [2, 3, 4, 5],                      # Grup 2: a_1, a_2, tilt_1, tilt_2
        [6, 7, 14],                        # Grup 3: phi_12, phi_jl, phase
        [8, 9, 10, 11, 12, 13],           # Grup 4: luminosity_distance, theta_jn, psi, azimuth, zenith, geocent_time
    ]

    # groups = [
    #     [2, 3, 4, 5], 
    #      [0, 1, 8, 9, 11], 
    #      [6], 
    #      [7], 
    #      [10], 
    #      [12], 
    #      [13], 
    #      [14]]          
    

    # Gruplar arası korelasyon hesapla (opsiyonel, raporlama için)
    rho, _ = spearmanr(P)
    abscorr = np.abs(np.atleast_2d(rho))
    np.fill_diagonal(abscorr, 0.0)
    between = avg_between_group_corr(groups, abscorr)

    group_names = [[PARAMS_15[i] for i in g] for g in groups]
    print(f"\nGruplama (5+5+5), gruplar arası ort. |korelasyon| = {between:.3f} "
          f"(düşük = bağımsız, toplam ≈ joint KLD):")
    for gi, gn in enumerate(group_names, 1):
        print(f"  G{gi}: {', '.join(gn)}")

    # Her grup x her yöntem için KLD
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

    # ---- Tablo ----
    print("\n" + "-" * 78)
    hdr = f"{'Yöntem':<15}" + "".join(f"{'G'+str(i):>11}" for i in range(1, len(groups)+1)) \
          + f"{'TOPLAM':>12}"
    print(hdr); print("-" * 78)
    totals = {}
    for m in METHODS:
        vals = results[m]
        tot = float(np.nansum(vals))
        totals[m] = tot
        row = f"{m:<15}" + "".join(f"{v:>11.3f}" for v in vals) + f"{tot:>12.3f}"
        print(row)
    print("-" * 78)
    print("(birim: nats)  |  Toplam (bit):  " +
          "   ".join(f"{m}={totals[m]*NATS_TO_BITS:.2f}" for m in METHODS))
    # Yalnızca KDE ve kNN metodlarının ortalamasını hesapla
    active_methods = [m for m in METHODS if m in totals]
    mean_tot = np.mean([totals[m] for m in active_methods])
    print(f"\nYöntem-ortalaması (KDE + kNN) toplam KLD ≈ {mean_tot:.3f} nats "
          f"({mean_tot*NATS_TO_BITS:.2f} bit)")

    # ---- JSON kaydet ----
    out = {
        "file": os.path.basename(path),
        "analysis_group": grp,
        "n_posterior": int(P.shape[0]),
        "n_prior": int(Q.shape[0]),
        "parameters": PARAMS_15,
        "groups": group_names,
        "avg_between_group_abs_corr": between,
        "methods": METHODS,
        "kld_per_group_nats": {m: results[m] for m in METHODS},
        "kld_total_nats": totals,
        "kld_total_bits": {m: totals[m] * NATS_TO_BITS for m in METHODS},
    }
    out_json = os.path.join(here, "results_grup_kld.json")
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nJSON kaydedildi: {out_json}")

    # ---- Karşılaştırma grafiği ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ng = len(groups)
        x = np.arange(ng + 1)            # gruplar + TOPLAM
        labels = [f"G{i}" for i in range(1, ng + 1)] + ["TOPLAM"]
        w = 0.8 / len(METHODS)
        fig, ax = plt.subplots(figsize=(11, 6))
        for k, m in enumerate(METHODS):
            heights = results[m] + [totals[m]]
            ax.bar(x + k * w, heights, w, label=m)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(labels)
        ax.set_ylabel("KLD(posterior || prior)  [nats]")
        ax.set_title(f"{os.path.basename(path)} — grup bazında KLD, üç yöntem")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out_png = os.path.join(here, "grup_kld_karsilastirma.png")
        fig.savefig(out_png, dpi=130)
        print(f"Grafik kaydedildi: {out_png}")
    except Exception as e:
        print(f"(Grafik atlandı: {e})")

    return out


if __name__ == "__main__":
    # file_name global değişkeni tanımlanmışsa onu kullan, argüman verme
    if file_name is not None:
        print(f"\n>>> KULLANILACAK DOSYA: {file_name}\n")
        main(file_name)
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else None)
