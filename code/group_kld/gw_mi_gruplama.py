"""
GW150914 — Karşılıklı Bilgi (MI) ile parametre gruplama (≤5'li gruplar)
=======================================================================

Amaç
----
15 parametreyi, KLD pipeline'ı (gw_grup_kld_analizi.py) için ≤5'li gruplara
ayırmak. Mantık: BİRBİRİYLE BİLGİ PAYLAŞAN (yüksek MI) parametreler AYNI grupta
toplanır; böylece GRUPLAR ARASI bağımlılık en aza iner ve "grup KLD'lerinin
toplamı ≈ ortak (joint) KLD" yaklaşımı sağlam olur.

Bu dosya, kullanıcının orijinal dendrogram script'inin GÖZDEN GEÇİRİLMİŞ ve
DÜZELTİLMİŞ halidir. Yapılan başlıca düzeltmeler/iyileştirmeler kod içinde
[DÜZELTME]/[İYİLEŞTİRME] etiketleriyle işaretlidir.

Bağımlılıklar: numpy, pandas, h5py, matplotlib, seaborn, scikit-learn, scipy
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")            # [İYİLEŞTİRME] başsız (notebook'suz) çalışma
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_selection import mutual_info_regression
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 0. AYARLAR
# ==========================================
FILE_NAME = os.path.join(HERE, "..", "data",
                         "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5")


# IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5
# IGWN-GWTC3p0-v2-GW191103_012549_PEDataRelease_mixed_cosmo.h5

PARAM_NAMES = [
    "mass_1_source", "mass_2_source",
    "a_1", "a_2", "tilt_1", "tilt_2",
    "phi_12", "phi_jl", "phase",
    "luminosity_distance", "theta_jn", "psi", "azimuth", "zenith", "geocent_time",
]
MAX_SAMPLES = 70_000      # MI (kNN) için yeterli; isterseniz artırın
N_NEIGHBORS = 3           # Kraskov MI komşu sayısı (3–5 tipik)
MAX_GROUP_SIZE = 5        # [İYİLEŞTİRME] KLD pipeline'ı için sert üst sınır
RANDOM_STATE = 42


# ==========================================
# 1. VERİ + TEKİLLEŞTİRME
# ==========================================
def load_clean_dataframe():
    print(f"HDF5 okunuyor: {os.path.basename(FILE_NAME)}")
    with h5py.File(FILE_NAME, "r") as f:
        target = next((k for k in f.keys() if "IMRPhenom" in k), None)
        if target is None:
            raise KeyError(f"'IMRPhenom' grubu yok. Mevcut: {list(f.keys())}")
        print(f"Veri grubu: {target}")
        samples = f[target]["posterior_samples"]
        # [DÜZELTME] h5py veri kümesini açıkça float diziye çevir ([:] ile)
        data = {p: np.asarray(samples[p][:], dtype=float) for p in PARAM_NAMES}
    df = pd.DataFrame(data)
    print(f"Ham veri: {len(df)} satır")

    # --- GWTC yeniden-örnekleme artefaktı temizliği ---
    # [DÜZELTME] df_clean HER DURUMDA tanımlı olmalı; orijinalde bazı dallarda
    # tanımsız kalıp NameError veriyordu. Önce güvenli varsayılan atanır.
    df_clean = df.copy()
    P = df.values
    n0, npar = P.shape
    uniq = [len(np.unique(P[:, j])) for j in range(npar)]
    cnt = Counter(c for c in uniq if c < 0.99 * n0)
    if cnt:
        modal, freq = cnt.most_common(1)[0]
        if freq >= 2:
            key = [j for j in range(npar) if uniq[j] == modal]
            _, keep = np.unique(P[:, key], axis=0, return_index=True)
            keep = np.sort(keep)
            if len(keep) < n0:
                print(f"Tekilleştirme: {n0} -> {len(keep)} benzersiz örnek "
                      f"(kopya sütunlar: {', '.join(PARAM_NAMES[j] for j in key)})")
                df_clean = df.iloc[keep].copy()
    else:
        print("Belirgin GWTC kopya bloğu yok; tekilleştirme atlandı.")

    # Alt-örnekleme (MI hızı için)
    if len(df_clean) > MAX_SAMPLES:
        df_clean = df_clean.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)
    else:
        df_clean = df_clean.sample(frac=1, random_state=RANDOM_STATE)
    print(f"MI analizi için örnek sayısı: {len(df_clean)}")
    return df_clean


# ==========================================
# 2. SİMETRİK MI MATRİSİ
# ==========================================
def mi_matrix(df):
    """15x15 pairwise MI [nats]. [İYİLEŞTİRME] her çift iki yönde hesaplanıp
    ortalanır (mutual_info_regression simetrik değildir: I(X;Y) tahmini, X mi Y mi
    'hedef' seçildiğine göre biraz değişir). Ortalama daha kararlıdır."""
    n = len(PARAM_NAMES)
    M = np.zeros((n, n))
    X = df[PARAM_NAMES].to_numpy(dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            mi_ij = mutual_info_regression(
                X[:, [i]], X[:, j], n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE)[0]
            mi_ji = mutual_info_regression(
                X[:, [j]], X[:, i], n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE)[0]
            mi = max(0.0, 0.5 * (mi_ij + mi_ji))   # MI>=0; tahmin gürültüsünü kırp
            M[i, j] = M[j, i] = mi
    np.fill_diagonal(M, 0.0)
    return M


# ==========================================
# 3. MI -> UZAKLIK
# ==========================================
def mi_to_distance(M):
    """[İYİLEŞTİRME] MI'yi 'bilgi-korelasyonu' uzaklığına çevir:
        rho_info = sqrt(1 - exp(-2 I))   (Gauss ilişkisi I = -0.5 ln(1-rho^2))
        d = 1 - rho_info  ∈ [0, 1]
    Bu, ölçekten bağımsız ve [0,1]'e sınırlı; doğrusal korelasyona indirgenebilir
    bir benzerliktir. (Orijinaldeki exp(-MI) de çalışır ama normalize değildir;
    MI=0 -> d=1 (tam bağımsız), MI büyük -> d->0 (tam bağımlı).)"""
    rho = np.sqrt(1.0 - np.exp(-2.0 * M))
    D = 1.0 - rho
    D = 0.5 * (D + D.T)          # [DÜZELTME] squareform için tam simetri garantisi
    np.fill_diagonal(D, 0.0)
    return D


# ==========================================
# 4. ≤5'Lİ GRUPLARA KÜMELE
# ==========================================
def clusters_max_size(Z, n_items, max_size):
    """[İYİLEŞTİRME] Dendrogramda sabit bir eşik, grup boyutunu GARANTİ ETMEZ.
    KLD pipeline'ı ≤5 boyut ister. Bu yüzden küme sayısını artırarak EN KÜÇÜK
    küme sayısını buluruz ki tüm kümeler ≤ max_size olsun (hiyerarşiyi korur)."""
    for k in range(1, n_items + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        sizes = np.bincount(labels)[1:]
        if sizes.max() <= max_size:
            return labels, k
    return fcluster(Z, t=n_items, criterion="maxclust"), n_items


# ==========================================
# ANA AKIŞ
# ==========================================
def main():
    df = load_clean_dataframe()
    print("-" * 60)

    print("15x15 simetrik MI matrisi hesaplanıyor (kNN)...")
    M = mi_matrix(df)
    D = mi_to_distance(M)
    Z = linkage(squareform(D, checks=False), method="average")

    # --- Gruplar (≤5) ---
    labels, k = clusters_max_size(Z, len(PARAM_NAMES), MAX_GROUP_SIZE)
    groups = {}
    for p, c in zip(PARAM_NAMES, labels):
        groups.setdefault(int(c), []).append(p)
    groups = [sorted_g for _, sorted_g in sorted(groups.items())]

    print(f"\n--- ≤{MAX_GROUP_SIZE} boyutlu gruplar ({k} küme) ---")
    for gi, g in enumerate(groups, 1):
        print(f"  G{gi} ({len(g)}D): {', '.join(g)}")

    # KLD script'ine yapıştırmak için indeks gösterimi
    idx_groups = [[PARAM_NAMES.index(p) for p in g] for g in groups]
    print("\ngw_grup_kld_analizi.py için (indeks) gruplar:")
    print(f"  GROUPS = {idx_groups}")

    # --- Dendrogram ---
    plt.figure(figsize=(14, 7))
    plt.title("GW150914 (IMRPhenomXPHM): MI Dendrogramı (bilgi-korelasyonu uzaklığı)")
    dendrogram(Z, labels=PARAM_NAMES, leaf_rotation=45, leaf_font_size=10)
    plt.ylabel(r"Uzaklık $1-\sqrt{1-e^{-2I}}$")
    plt.tight_layout()
    p1 = os.path.join(HERE, "gw150914_mi_dendrogram.png")
    plt.savefig(p1, dpi=200); plt.close()
    print(f"\nKaydedildi: {os.path.basename(p1)}")

    # --- Isı haritası (MI, bit) ---
    plt.figure(figsize=(11, 9))
    sns.heatmap(M / np.log(2), xticklabels=PARAM_NAMES, yticklabels=PARAM_NAMES,
                cmap="magma", annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar_kws={"label": "MI [bit]"})
    plt.title("Karşılıklı Bilgi (MI) Matrisi [bit] — tekilleştirilmiş")
    plt.tight_layout()
    p2 = os.path.join(HERE, "gw150914_mi_heatmap.png")
    plt.savefig(p2, dpi=200); plt.close()
    print(f"Kaydedildi: {os.path.basename(p2)}")

    return groups, M


if __name__ == "__main__":
    main()
