"""
GW150914 — kNN tabanlı bilgi-teorik analiz
==========================================

Amaç
----
GW150914 olayının posterior örnek dağılımından (parametre kestirimi sonuçları),
makaledeki (Álvarez Chaves et al., 2024, Entropy 26(5):387) kNN tahmin edicilerini
kullanarak "toplam bilgi miktarını" hesaplamak.

Adımlar
-------
1. Posterior örnek dosyasını esnek biçimde oku (GWTC-1 / PESummary / bilby HDF5
   veya CSV). Tüm sayısal parametreleri al.
2. 15 standart CBC parametresini seç (varsa). Her parametreyi z-skoruna
   standardize et (kNN uzaklıkları ölçek-bağımlı olmasın diye).
3. Spearman korelasyon matrisini hesapla.
4. Parametreleri, GRUPLAR ARASI korelasyon (bağımlılık) EN AZ olacak şekilde
   gruplara ayır: korele parametreler aynı grupta toplanır, gruplar birbirinden
   olabildiğince bağımsız olur. Birincil bölme: 10D + 5D. İkincil: 5+5+5 (≤3 grup).
5. kNN ile hesapla (makaledeki estimatörler, knn_estimators_reference.py):
     - Her grubun ortak (joint) entropisi  H(grup)        [Kozachenko-Leonenko, k=1]
     - Tüm parametrelerin ortak entropisi   H(tüm)         [k=1]
     - Toplam korelasyon (multi-information) C = Σ H(grup) − H(tüm)
       => Gruplar arasında PAYLAŞILAN / fazlalık (redundant) bilgi. Ölçek-bağımsız.
     - Grup çiftleri arası karşılıklı bilgi I(grup_i; grup_j)  [Kraskov KSG, k=15]
6. Sonuçları nats ve bit cinsinden raporla, JSON + korelasyon ısı haritası kaydet.

"Toplam bilgi miktarı" yorumu
-----------------------------
- H(tüm): dağılımın toplam diferansiyel entropisi = parametrelerdeki toplam
  belirsizlik/bilgi içeriği (standardize edilmiş birimlerde).
- C (toplam korelasyon): parametre grupları arasındaki toplam istatistiksel
  bağımlılık = "paylaşılan bilgi". Gruplar tam bağımsız olsa C≈0 olurdu.
  C ölçek/birim seçiminden BAĞIMSIZDIR (per-koordinat standardizasyona göre
  değişmez), bu yüzden asıl "sağlam" bilgi ölçüsü budur.

Kullanım
--------
    python gw150914_knn_information.py /yol/GW150914_posterior.hdf5
    python gw150914_knn_information.py            # varsayilan: data/ icindeki ilk dosya

Bağımlılıklar: numpy, scipy, pandas, matplotlib, h5py
"""

import os
import sys
import json
import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# kNN estimatörleri (aynı klasördeki referans dosyadan)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knn_estimators_reference import (
    calc_knn_entropy,
    calc_knn_mutual_information,
)

NATS_TO_BITS = 1.0 / np.log(2.0)
MAX_SAMPLES = 20_000   # kNN için fazlasıyla yeterli; MI (query_ball_point) hızı için sınır
RANDOM_STATE = 42

# GW150914 için tercih edilen 15 standart CBC parametresi (öncelik sırası).
# Her giriş: (kanonik_ad, [dosyalarda gecebilecek alternatif adlar])
# Standart 15 bağımsız CBC parametresi. (mass_1/mass_2, chirp_mass/mass_ratio ile
# neredeyse birebir bağımlı olduğundan kanonik sete DAHİL EDİLMEZ; yalnızca kütle
# parametresi hiç yoksa yedek olarak kullanılırlar.)
CANONICAL_PARAMS = [
    ("chirp_mass",          ["chirp_mass", "chirp_mass_source", "mc", "mchirp",
                             "m1_detector_frame_msun", "mass_1"]),
    ("mass_ratio",          ["mass_ratio", "q", "symmetric_mass_ratio",
                             "m2_detector_frame_msun", "mass_2"]),
    ("a_1",                 ["a_1", "spin1", "a1", "spin_1a"]),
    ("a_2",                 ["a_2", "spin2", "a2", "spin_2a"]),
    ("tilt_1",              ["tilt_1", "tilt1", "costilt1", "cos_tilt_1"]),
    ("tilt_2",              ["tilt_2", "tilt2", "costilt2", "cos_tilt_2"]),
    ("phi_12",              ["phi_12", "phi12"]),
    ("phi_jl",              ["phi_jl", "phijl"]),
    ("luminosity_distance", ["luminosity_distance", "luminosity_distance_mpc", "distance", "dist"]),
    ("theta_jn",            ["theta_jn", "costheta_jn", "iota", "cos_theta_jn", "inclination"]),
    ("ra",                  ["ra", "right_ascension", "rightascension"]),
    ("dec",                 ["dec", "declination"]),
    ("psi",                 ["psi", "polarization", "polarisation"]),
    ("phase",               ["phase", "coa_phase", "phi_ref"]),
    ("geocent_time",        ["geocent_time", "time", "geocenter_time", "tc"]),
]


# --------------------------------------------------------------------------
# 1) Esnek posterior yükleyici
# --------------------------------------------------------------------------
def _find_structured_datasets(h5obj, path="", found=None):
    """HDF5 ağacını gez; yapısal (compound) veya 1B-dizi grubu posterior tablolarını bul."""
    import h5py
    if found is None:
        found = []
    for key in h5obj.keys():
        item = h5obj[key]
        cur = f"{path}/{key}"
        if isinstance(item, h5py.Dataset):
            if item.dtype.names is not None and item.shape and item.shape[0] > 50:
                found.append((cur, "compound", item))
        elif isinstance(item, h5py.Group):
            # "grup içinde eşit uzunlukta 1B diziler" deseni (bilby/pesummary)
            arrs = [k for k in item.keys()
                    if isinstance(item[k], h5py.Dataset)
                    and item[k].ndim == 1 and item[k].shape and item[k].shape[0] > 50]
            lengths = {item[k].shape[0] for k in arrs}
            if len(arrs) >= 3 and len(lengths) == 1:
                found.append((cur, "group1d", item))
            _find_structured_datasets(item, cur, found)
    return found


def load_posterior(path):
    """Posterior örnek dosyasını oku → (DataFrame, kaynak_açıklaması)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".dat", ".txt"):
        sep = "," if ext == ".csv" else r"\s+"
        df = pd.read_csv(path, sep=sep, engine="python")
        return df, f"CSV/metin tablosu ({df.shape[0]} örnek, {df.shape[1]} sütun)"

    import h5py
    with h5py.File(path, "r") as f:
        cands = _find_structured_datasets(f)
        if not cands:
            raise ValueError("HDF5 içinde posterior tablosu bulunamadı.")
        # En çok satıra (örnek) sahip adayı seç; eşitse en çok sütunlu
        def score(c):
            _, kind, item = c
            if kind == "compound":
                return (item.shape[0], len(item.dtype.names))
            keys = [k for k in item.keys() if isinstance(item[k], h5py.Dataset) and item[k].ndim == 1]
            return (item[keys[0]].shape[0], len(keys))
        cands.sort(key=score, reverse=True)
        cpath, kind, item = cands[0]
        if kind == "compound":
            arr = item[()]
            df = pd.DataFrame({name: np.asarray(arr[name], dtype=float)
                               for name in arr.dtype.names
                               if np.issubdtype(np.asarray(arr[name]).dtype, np.number)})
        else:
            keys = [k for k in item.keys()
                    if isinstance(item[k], h5py.Dataset) and item[k].ndim == 1]
            df = pd.DataFrame()
            for k in keys:
                col = np.asarray(item[k][()])
                if np.issubdtype(col.dtype, np.number):
                    df[k] = col.astype(float)
        return df, f"HDF5 yolu '{cpath}' ({df.shape[0]} örnek, {df.shape[1]} sayısal sütun)"


def select_parameters(df, n_target=15):
    """Kanonik adlara göre en fazla n_target parametre seç (sabit/NaN sütunları at)."""
    cols_lower = {c.lower(): c for c in df.columns}
    chosen, used = [], set()
    for canon, alts in CANONICAL_PARAMS:
        for a in alts:
            if a.lower() in cols_lower:
                real = cols_lower[a.lower()]
                if real in used:
                    continue
                col = df[real].to_numpy(dtype=float)
                if np.all(np.isfinite(col)) and np.nanstd(col) > 0:
                    chosen.append((canon, real))
                    used.add(real)
                    break
        if len(chosen) >= n_target:
            break
    # Yeterli kanonik parametre yoksa, kalan sayısal sütunlarla tamamla
    if len(chosen) < n_target:
        for c in df.columns:
            if c in used:
                continue
            col = df[c].to_numpy(dtype=float)
            if np.all(np.isfinite(col)) and np.nanstd(col) > 0:
                chosen.append((c, c))
                used.add(c)
            if len(chosen) >= n_target:
                break
    names = [canon for canon, _ in chosen]
    data = np.column_stack([df[real].to_numpy(dtype=float) for _, real in chosen])
    return names, data


# --------------------------------------------------------------------------
# 2) Korelasyona göre gruplama (gruplar arası bağımlılığı en aza indir)
# --------------------------------------------------------------------------
def group_by_independence(names, abscorr, sizes):
    """Korele parametreleri aynı grupta topla; gruplar birbirinden bağımsız olsun.

    Açgözlü (greedy) büyütme: her grup için, kalanlar içinde diğer kalanlarla
    toplam korelasyonu en yüksek 'merkez' parametreyle başla, sonra mevcut grup
    üyelerine ortalama |korelasyon|u en yüksek parametreleri ekle.
    """
    n = len(names)
    remaining = list(range(n))
    groups = []
    for size in sizes:
        if not remaining:
            break
        size = min(size, len(remaining))
        # merkez: kalanlar içinde en 'merkezi' (yüksek toplam korelasyon)
        sub = np.array(remaining)
        totals = abscorr[np.ix_(sub, sub)].sum(axis=1)
        seed = sub[int(np.argmax(totals))]
        group = [int(seed)]
        remaining.remove(int(seed))
        while len(group) < size and remaining:
            best, best_val = None, -1.0
            for r in remaining:
                val = np.mean([abscorr[r, g] for g in group])
                if val > best_val:
                    best_val, best = val, r
            group.append(int(best))
            remaining.remove(int(best))
        groups.append(group)
    # kalan varsa son gruba ekle
    if remaining and groups:
        groups[-1].extend(int(r) for r in remaining)
    return groups


def avg_between_group_corr(groups, abscorr):
    """Gruplar arası ortalama |korelasyon| (düşük = gruplar bağımsız)."""
    vals = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for a in groups[i]:
                for b in groups[j]:
                    vals.append(abscorr[a, b])
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------
# 3) Bilgi-teorik hesaplamalar (kNN)
# --------------------------------------------------------------------------
def analyze_partition(Z, names, groups, label):
    """Bir bölme için H(grup), H(tüm), toplam korelasyon ve çiftli MI hesapla."""
    H_full = calc_knn_entropy(Z, k=1)                       # tüm parametreler
    group_names = [[names[i] for i in g] for g in groups]
    H_groups = [calc_knn_entropy(Z[:, g], k=1) for g in groups]
    total_corr = float(np.sum(H_groups) - H_full)          # multi-information

    pairwise_mi = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            mi = calc_knn_mutual_information(Z[:, groups[i]], Z[:, groups[j]], k=15)
            pairwise_mi[f"I(G{i+1};G{j+1})"] = float(mi)

    return {
        "label": label,
        "groups": group_names,
        "group_dims": [len(g) for g in groups],
        "H_full_nats": float(H_full),
        "H_groups_nats": [float(h) for h in H_groups],
        "total_correlation_nats": total_corr,
        "total_correlation_bits": total_corr * NATS_TO_BITS,
        "pairwise_mutual_information_nats": pairwise_mi,
    }


# --------------------------------------------------------------------------
# Ana akış
# --------------------------------------------------------------------------
def main(path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    if path is None:
        cands = []
        for ext in ("*.hdf5", "*.h5", "*.csv", "*.dat"):
            cands += glob.glob(os.path.join(here, "data", ext))
        if not cands:
            print("HATA: data/ klasörüne GW150914 posterior dosyasını koyun "
                  "veya yolunu argüman verin.\n  python gw150914_knn_information.py <dosya>")
            return
        path = cands[0]

    print("=" * 70)
    print("GW150914 — kNN tabanlı bilgi-teorik analiz")
    print("=" * 70)
    df, src = load_posterior(path)
    print(f"Veri kaynağı : {os.path.basename(path)}")
    print(f"Okundu       : {src}")

    names, data = select_parameters(df, n_target=15)
    n_samples, n_params = data.shape
    print(f"Seçilen parametreler ({n_params}): {', '.join(names)}")

    # --- Tekilleştirme (yeniden-örnekleme artefaktını gider) ---
    # Bazı veri yayınlarında örnekler kopyalanır (ör. GWTC-2.1: iç parametreler
    # aynı, mesafe/ra yeniden çizilir). Bu, alt-uzaylarda bire bir aynı noktalar
    # üretip kNN uzaklığını 0 (log -inf) yapar. Çoğunlukla yinelenen değer taşıyan
    # sütunlara göre tekilleştirip her benzersiz örnekten birini tutarız.
    # Kopyalanmayı tanımlayan sütun bloğunu bul: aynı (N'den küçük) benzersiz-değer
    # sayısını PAYLAŞAN en kalabalık sütun grubu. Bu sütunlar kopyalar arası
    # birebir aynıdır; tekilleştirme anahtarı olarak bunları kullanırız.
    uniq_counts = [len(np.unique(data[:, j])) for j in range(n_params)]
    from collections import Counter
    cnt = Counter(c for c in uniq_counts if c < 0.99 * n_samples)
    key_cols = np.array([], dtype=int)
    if cnt:
        modal_count, freq = cnt.most_common(1)[0]
        if freq >= 2:  # en az 2 sütun aynı benzersiz sayıyı paylaşıyorsa = kopya bloğu
            key_cols = np.array([j for j in range(n_params) if uniq_counts[j] == modal_count])
    if len(key_cols) > 0:
        _, keep_idx = np.unique(data[:, key_cols], axis=0, return_index=True)
        keep_idx = np.sort(keep_idx)
        if len(keep_idx) < n_samples:
            print(f"Tekilleştirme: {n_samples} → {len(keep_idx)} benzersiz örnek "
                  f"(yinelenen sütunlar: {', '.join(names[j] for j in key_cols)})")
            data = data[keep_idx]
            n_samples = len(keep_idx)

    # Alt-örnekleme (hız/bellek)
    if n_samples > MAX_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n_samples, MAX_SAMPLES, replace=False)
        data = data[idx]
        print(f"Alt-örnekleme: {n_samples} → {MAX_SAMPLES} örnek")
        n_samples = MAX_SAMPLES

    # Standardizasyon (z-skor): kNN uzaklıkları ölçek-bağımsız olsun
    Z = (data - data.mean(axis=0)) / data.std(axis=0)

    # Çok küçük jitter: tekilleştirme sonrası kalan nadir alt-uzay bağlarını kırar
    # (Kraskov+2004 önerisi). Büyüklük ~1e-10 olduğundan entropi/MI'ye etkisi ihmal
    # edilebilir, yalnızca log(0) tekilliğini engeller.
    rng_j = np.random.default_rng(RANDOM_STATE)
    Z = Z + rng_j.normal(0.0, 1e-10, size=Z.shape)

    # Spearman korelasyon (monoton ilişkiler için sağlam)
    rho, _ = spearmanr(Z)
    rho = np.atleast_2d(rho)
    abscorr = np.abs(rho)
    np.fill_diagonal(abscorr, 0.0)

    # Bölme boyutlarını belirle (birincil: 10+5; param sayısı azsa uyarlanır)
    if n_params >= 15:
        primary_sizes = [10, 5]
        secondary_sizes = [5, 5, 5]
    else:
        g1 = max(1, n_params - 5)
        primary_sizes = [g1, n_params - g1]
        third = n_params // 3
        secondary_sizes = [n_params - 2 * third, third, third]

    results = {"file": os.path.basename(path), "n_samples": int(n_samples),
               "n_params": int(n_params), "parameters": names, "partitions": []}

    for sizes in [primary_sizes, secondary_sizes]:
        groups = group_by_independence(names, abscorr, sizes)
        label = "+".join(str(len(g)) for g in groups) + "D"
        res = analyze_partition(Z, names, groups, label)
        res["avg_between_group_abs_corr"] = avg_between_group_corr(groups, abscorr)
        results["partitions"].append(res)

        print("\n" + "-" * 70)
        print(f"BÖLME: {label}  (gruplar arası ort. |korelasyon| = "
              f"{res['avg_between_group_abs_corr']:.3f}; düşük = bağımsız)")
        for gi, (gn, hg) in enumerate(zip(res["groups"], res["H_groups_nats"]), 1):
            print(f"  G{gi} ({len(gn)}D)  H={hg:7.3f} nats  | {', '.join(gn)}")
        print(f"  H(tüm {n_params}D)          = {res['H_full_nats']:.3f} nats "
              f"({res['H_full_nats']*NATS_TO_BITS:.3f} bit)")
        print(f"  Toplam korelasyon C       = {res['total_correlation_nats']:.3f} nats "
              f"({res['total_correlation_bits']:.3f} bit)   [gruplar arası paylaşılan bilgi]")
        for k, v in res["pairwise_mutual_information_nats"].items():
            print(f"  {k} = {v:.3f} nats ({v*NATS_TO_BITS:.3f} bit)")

    # Kaydet: JSON sonuç + korelasyon ısı haritası
    out_json = os.path.join(here, "results_gw150914_knn.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nSonuçlar kaydedildi: {out_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(abscorr + np.eye(n_params), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(n_params)); ax.set_xticklabels(names, rotation=90, fontsize=8)
        ax.set_yticks(range(n_params)); ax.set_yticklabels(names, fontsize=8)
        ax.set_title("GW150914 parametreleri — |Spearman korelasyon|")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        out_png = os.path.join(here, "gw150914_correlation.png")
        fig.savefig(out_png, dpi=130)
        print(f"Isı haritası kaydedildi: {out_png}")
    except Exception as e:
        print(f"(Grafik atlandı: {e})")

    return results


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
