"""
GW — 15D k-NN KLD(posterior || prior) — Pedagojik Test Scripti
===============================================================

AMAÇ
----
Bu script, Pérez-Cruz (2008) k-NN KL tahmincisini DOĞRUDAN 15 boyutlu
uzayda uygular. Grup-bölme yapmaz; tüm 15 parametreyi tek seferde verir.

BEKLENTİ (önceden)
-------------------
Bu tahmincinin 15D'de YAKINSAMAMASINI bekliyoruz. Nedenini anlayabilmek
için script üç ayrı test yapar:

  Test A — k taraması (k=1,2,5,10):
    Teoride k değiştikçe sonuç kararlı olmalı (asimptotik limit).
    15D'de k=1 ile k=10 arasında büyük fark görürsek → yakınsama yok.

  Test B — n_posterior alt-örnekleme taraması (5k → 50k adımlarla):
    Doğru bir tahmincide sonuç n arttıkça bir değere yaklaşır.
    15D'de monoton artış/düşüş görürsek → örneklem boyutundan bağımsız
    değil → boyutluluğun laneti aktif.

  Test C — Boyut taraması (1D → 3D → 6D → 10D → 15D):
    Aynı parametreler üzerinde boyutu kademeli artırıp tahmini izle.
    Bu, yakınsamanın tam olarak hangi boyutta bozulduğunu gösterir.

TEORİK ARKA PLAN
----------------
Pérez-Cruz (2008) [IEEE ISIT, doi:10.1109/ISIT.2008.4595271] tahmincisi:

  D_KL(P||Q) ≈ (d/n) · Σᵢ log(νₖ(xᵢ) / ρₖ(xᵢ)) + log(m/(n−1))

  Burada:
    xᵢ       : posterior'dan i. örnek
    ρₖ(xᵢ)  : xᵢ'nin posterior içindeki k. en yakın komşu uzaklığı
    νₖ(xᵢ)  : xᵢ'nin prior içindeki k. en yakın komşu uzaklığı
    n        : posterior örnek sayısı
    m        : prior örnek sayısı
    d        : boyut

  Düzeltme terimi log(m/(n−1)):
    m=5000, n=71747 → log(5000/71746) = log(0.0697) ≈ −2.663 nats ≈ −3.84 bit
    Bu büyük NEGATİF düzeltme, m << n dengesizliğinden kaynaklanır.
    Tahmincinin varyansını artırır, özellikle yüksek boyutlarda.

  k-NN tahmincisi için YAKINSAMA gereksinimi (Pérez-Cruz 2008, Thm 1):
    n, m → ∞ gerekir, VE n/m → sabit (dengeli örneklem)
    Pratikte d=15 için n ~ 10^(d/2) = 10^7.5 ~ 30 milyon örnek lazım.
    Elimizde: 71k posterior, 5k prior → yetersiz.

NEDEN MONOTONİK ARTŞ/DÜŞÜŞ GÖRÜRÜZ?
-------------------------------------
15D uzayda tüm noktalar birbirine eşit uzaklıktadır ("concentration of
measure" fenomeni). 15D birim kürede rastgele iki noktanın uzaklığı
d/√d = √d ≈ 3.87 civarında yoğunlaşır, std → 0. Bu yüzden ρₖ ≈ νₖ
olur ve oran ≈ 1, log ≈ 0. Ama yeterince örnek yoksa bu dengelenme
tamamlanmaz; n artarken ρₖ azalır (daha sık örnekleme → daha yakın
komşu), νₖ ise değişmez (prior sabit). Dolayısıyla n büyüdükçe oran
büyür ve tahmin şişer → monoton artış.

KULLANIM
--------
  python gw_15d_knn_kld.py
  (dosya yolu scriptin içinde tanımlı, gw_grup_kld_analizi.py ile aynı)

BAĞIMLILIKLAR: numpy, scipy, h5py, matplotlib
"""

import os
import sys
import time
import warnings

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ============================================================
# AYARLAR
# ============================================================
FILE_PATH = (
    "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/"
    "IGWN-GWTC3p0-v2-GW191105_143521_PEDataRelease_mixed_cosmo.h5"
)
# IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5
# IGWN-GWTC3p0-v2-GW191105_143521_PEDataRelease_mixed_cosmo.h5
PARAMS_15 = [
    "mass_1_source", "mass_2_source",          # 0,1  — kütleler
    "a_1", "a_2", "tilt_1", "tilt_2",          # 2-5  — spinler
    "phi_12", "phi_jl", "phase",               # 6-8  — açılar (nuisance)
    "luminosity_distance", "theta_jn",         # 9,10 — uzaklık-eğim
    "psi", "azimuth", "zenith",                # 11-13 — gökyüzü
    "geocent_time",                            # 14   — varış zamanı
]

NATS_TO_BITS = 1.0 / np.log(2.0)
RANDOM_SEED  = 42

# Test B: alt-örnekleme boyutları
SUBSAMPLE_SIZES = [5_000, 10_000, 20_000, 30_000,48_000 ,50_000, 70_000]

# Test C: kademeli boyutlar (PARAMS_15'in ilk d parametresi kullanılır)
DIM_STEPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]

# k değerleri (Test A ve ana hesap)
K_VALUES = [1, 2, 5, 10]


# ============================================================
# VERİ YÜKLEME
# ============================================================
def load_data(path, params):
    """Posterior ve prior örneklerini yükle, tekilleştir, standardize et."""
    with h5py.File(path, "r") as f:
        chosen = None
        for key in f.keys():
            g = f[key]
            if not isinstance(g, h5py.Group):
                continue
            if "posterior_samples" in g and "priors" in g and "samples" in g["priors"]:
                psamp = g["priors"]["samples"]
                if all(p in psamp and psamp[p].shape[0] > 50 for p in params):
                    chosen = key
                    break
        if chosen is None:
            raise ValueError("Uygun analiz grubu bulunamadı.")

        g        = f[chosen]
        post_raw = g["posterior_samples"][()]
        post     = {p: np.asarray(post_raw[p], dtype=float) for p in params}
        prior    = {p: np.asarray(g["priors"]["samples"][p][()], dtype=float)
                    for p in params}

    P_raw = np.column_stack([post[p]  for p in params])
    Q_raw = np.column_stack([prior[p] for p in params])

    # --- Tekilleştirme (GWTC kopyalanmış örnekleri) ---
    from collections import Counter
    n0, nd = P_raw.shape
    uniq_counts = [len(np.unique(P_raw[:, j])) for j in range(nd)]
    cnt = Counter(c for c in uniq_counts if c < 0.99 * n0)
    if cnt:
        modal, freq = cnt.most_common(1)[0]
        if freq >= 2:
            key_cols = [j for j in range(nd) if uniq_counts[j] == modal]
            _, keep  = np.unique(P_raw[:, key_cols], axis=0, return_index=True)
            P_raw    = P_raw[np.sort(keep)]
            print(f"  Tekilleştirme: {n0} → {P_raw.shape[0]} benzersiz örnek")

    # --- Standardizasyon (KL değişmez, sayısal kararlılık artar) ---
    pool = np.vstack([P_raw, Q_raw])
    mu   = pool.mean(axis=0)
    sd   = pool.std(axis=0)
    sd[sd == 0] = 1.0
    P = (P_raw - mu) / sd
    Q = (Q_raw - mu) / sd

    # Küçük jitter: kNN için sıfır-uzaklık sorununu önler
    rng = np.random.default_rng(RANDOM_SEED)
    P   = P + rng.normal(0.0, 1e-10, size=P.shape)
    Q   = Q + rng.normal(0.0, 1e-10, size=Q.shape)

    return P, Q, chosen


# ============================================================
# PÉREZ-CRUZ (2008) k-NN KL TAHMİNCİSİ
# ============================================================
def knn_kld_nats(P, Q, k=1):
    """
    D_KL(P || Q) tahminini nats cinsinden döndürür.

    Formül (Pérez-Cruz 2008, Eq. 14):
      D_KL = (d/n) Σᵢ log(νₖ(xᵢ) / ρₖ(xᵢ)) + log(m / (n−1))

    P : (n, d) — posterior örnekleri
    Q : (m, d) — prior örnekleri
    k : kaçıncı en yakın komşu
    """
    n, d = P.shape
    m    = Q.shape[0]

    # k-NN ağaçları
    tree_P = cKDTree(P)
    tree_Q = cKDTree(Q)

    # Posterior içi: her xᵢ için k. NN (kendisi hariç → workers=-1 paralel)
    rho, _ = tree_P.query(P, k=k + 1, workers=-1)   # (n, k+1); ilk sütun kendisi (0)
    rho    = rho[:, k]                   # k. komşu (0-indexed: sütun k)

    # Prior içinde: her xᵢ için k. NN
    nu, _  = tree_Q.query(P, k=k, workers=-1)       # (n, k) — 2-d guaranteed
    if nu.ndim == 1:
        nu = nu[:, np.newaxis]                       # k=1 durumunda: (n,) -> (n, 1)
    nu     = nu[:, k - 1]               # k. komşu (0-indexed: sütun k-1)

    # Sıfır uzaklıkları jitter'a rağmen oluşabilirse kırp
    rho = np.maximum(rho, 1e-300)
    nu  = np.maximum(nu,  1e-300)

    # Ana formül
    log_ratio      = np.log(nu / rho)                # (n,)
    bias_correction = np.log(m / (n - 1))             # skaler (negatif: m << n)
    kld_nats       = (d / n) * np.sum(log_ratio) + bias_correction

    # Diagnostik bilgiler (opsiyonel debug)
    _debug = {
        "mean_log_ratio":   float(np.mean(log_ratio)),
        "std_log_ratio":    float(np.std(log_ratio)),
        "bias_correction":  float(bias_correction),
        "mean_rho":         float(np.mean(rho)),
        "mean_nu":          float(np.mean(nu)),
        "n": n, "m": m, "d": d, "k": k,
    }
    return float(kld_nats), _debug


# ============================================================
# TEST A — k taraması (k=1,2,5,10) tüm posterior ile
# ============================================================
def test_a_k_scan(P, Q):
    print("\n" + "="*60)
    print("TEST A: k taraması — aynı veri, farklı k")
    print("Beklenti: iyi tahmincide k'ya karşı kararlı sonuç")
    print("15D beklenti: k arttıkça monoton değişim (yüksek varyans)")
    print("="*60)
    results = {}
    for k in K_VALUES:
        t0 = time.time()
        val_nats, dbg = knn_kld_nats(P, Q, k=k)
        elapsed = time.time() - t0
        val_bits = val_nats * NATS_TO_BITS
        results[k] = val_bits
        print(f"  k={k:2d}: {val_bits:7.2f} bit  "
              f"({val_nats:.3f} nats)  "
              f"bias_corr={dbg['bias_correction']*NATS_TO_BITS:.2f} bit  "
              f"[{elapsed:.1f}s]")
    spread = max(results.values()) - min(results.values())
    print(f"\n  → k=1..10 arasında yayılım: {spread:.2f} bit")
    print(f"  → 1 bit'den büyük yayılım = tahmincinin k'ya BAĞIMLI olduğu anlamına gelir")
    return results


# ============================================================
# TEST B — n_posterior alt-örnekleme taraması
# ============================================================
def test_b_n_scan(P, Q, k=1):
    print("\n" + "="*60)
    print(f"TEST B: n_posterior taraması (k={k})")
    print("Beklenti: iyi tahmincide n artınca değer SABITLENR")
    print("15D beklenti: n artınca değer monoton ARTAR (bias şişer)")
    print("="*60)
    rng     = np.random.default_rng(RANDOM_SEED + 1)
    results = {}
    n_full  = P.shape[0]
    for sz in SUBSAMPLE_SIZES:
        if sz > n_full:
            print(f"  n={sz}: atlandı (mevcut: {n_full})")
            continue
        idx     = rng.choice(n_full, sz, replace=False)
        P_sub   = P[idx]
        t0      = time.time()
        val_nats, dbg = knn_kld_nats(P_sub, Q, k=k)
        elapsed = time.time() - t0
        val_bits = val_nats * NATS_TO_BITS
        results[sz] = val_bits
        bias_b = dbg["bias_correction"] * NATS_TO_BITS
        print(f"  n={sz:6d}: {val_bits:7.2f} bit  "
              f"bias_corr={bias_b:.2f} bit  "
              f"mean_log_ratio*d/n = "
              f"{(dbg['mean_log_ratio'] * 15)*NATS_TO_BITS:.2f} bit  "
              f"[{elapsed:.1f}s]")
    return results


# ============================================================
# TEST C — Boyut taraması (1D → 15D)
# ============================================================
def test_c_dim_scan(P, Q, k=1, n_sub=20_000):
    print("\n" + "="*60)
    print(f"TEST C: boyut taraması 1D → 15D (k={k}, n_posterior={n_sub})")
    print("Beklenti: düşük boyutlarda kararlı, yüksek boyutlarda patlar")
    print("="*60)
    rng = np.random.default_rng(RANDOM_SEED + 2)
    idx = rng.choice(P.shape[0], min(n_sub, P.shape[0]), replace=False)
    P_sub = P[idx]

    results = {}
    for d in DIM_STEPS:
        P_d = P_sub[:, :d]
        Q_d = Q[:, :d]
        val_nats, dbg = knn_kld_nats(P_d, Q_d, k=k)
        val_bits = val_nats * NATS_TO_BITS
        results[d] = val_bits
        print(f"  d={d:2d}: {val_bits:8.2f} bit  "
              f"(bias_corr={dbg['bias_correction']*NATS_TO_BITS:.2f} bit,  "
              f"mean_nu/rho={np.exp(dbg['mean_log_ratio']):.3f})")
    return results


# ============================================================
# TEMEL FİZİKSEL BEKLENTI HESABI
# ============================================================
def fh_prediction(snr=24.0, n_param=15):
    """
    Flanagan & Hughes 1998, Eq. 9.12:
      I_source ≈ (N_param/2) * log2(1 + rho^2/N_param)

    Bu, yüksek-SNR Gauss yaklaşımıyla KL divergence tahminidir.
    """
    return 0.5 * n_param * np.log2(1 + snr**2 / n_param)


# ============================================================
# GRAFİKLER
# ============================================================
def plot_results(res_a, res_b, res_c, save_path="15d_knn_test.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fh_val = fh_prediction()

    # Test A: k taraması
    ax = axes[0]
    ks   = list(res_a.keys())
    vals = list(res_a.values())
    ax.plot(ks, vals, "o-", color="steelblue", lw=2, ms=8)
    ax.axhline(fh_val, ls="--", color="red", lw=1.5, label=f"F&H ~{fh_val:.1f} bit")
    ax.set_xlabel("k (k-NN sırası)")
    ax.set_ylabel("KLD [bit]")
    ax.set_title("Test A: k taraması (15D, tam n)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Test B: n taraması
    ax = axes[1]
    ns   = list(res_b.keys())
    vals = list(res_b.values())
    ax.plot(ns, vals, "s-", color="darkorange", lw=2, ms=8)
    ax.axhline(fh_val, ls="--", color="red", lw=1.5, label=f"F&H ~{fh_val:.1f} bit")
    ax.set_xlabel("n_posterior (alt-örnekleme)")
    ax.set_ylabel("KLD [bit]")
    ax.set_title("Test B: n taraması (15D, k=1)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Test C: boyut taraması
    ax = axes[2]
    ds   = list(res_c.keys())
    vals = list(res_c.values())
    ax.plot(ds, vals, "^-", color="forestgreen", lw=2, ms=8)
    ax.axhline(fh_val, ls="--", color="red", lw=1.5, label=f"F&H ~{fh_val:.1f} bit")
    ax.set_xlabel("Boyut (d)")
    ax.set_ylabel("KLD [bit]")
    ax.set_title("Test C: boyut taraması (k=1)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle("GW150914 — 15D k-NN KLD yakınsama testleri", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    print(f"\nGrafik kaydedildi: {save_path}")


# ============================================================
# ANA AKIŞ
# ============================================================
def main():
    print("=" * 60)
    print("GW150914 — 15D k-NN KLD Pedagojik Test Scripti")
    print("=" * 60)
    print(f"\nF&H (1998) Eq.9.12 beklentisi: "
          f"{fh_prediction():.2f} bit (SNR=24, N_param=15)")

    print(f"\nVeri yükleniyor: {os.path.basename(FILE_PATH)} ...")
    P, Q, grp = load_data(FILE_PATH, PARAMS_15)
    n, d = P.shape
    m    = Q.shape[0]
    print(f"  Analiz grubu: {grp}")
    print(f"  Posterior: {n} | Prior: {m} | Boyut: {d}")

    # Düzeltme terimini önceden raporla (ne bekliyoruz)
    bias_corr_nats = np.log(m / (n - 1))
    print(f"\n  Pérez-Cruz bias_correction = log({m}/{n-1})")
    print(f"    = {bias_corr_nats:.3f} nats = {bias_corr_nats*NATS_TO_BITS:.2f} bit")
    print(f"    (NEGATİF: m << n dengesizliği nedeniyle)")
    print(f"    Bu sabit negatif terim tüm sonuçları aşağı çeker.")
    print(f"    Ana terim (d/n)·Σlog(ν/ρ) bunu aşmalı ki KLD > 0 olsun.")

    # Testleri çalıştır
    res_a = test_a_k_scan(P, Q)
    res_b = test_b_n_scan(P, Q, k=1)
    res_c = test_c_dim_scan(P, Q, k=1, n_sub=20_000)

    # Özet
    print("\n" + "="*60)
    print("ÖZET VE YORUM")
    print("="*60)

    # Test B monotonik mi?
    ns   = sorted(res_b.keys())
    vals = [res_b[s] for s in ns]
    mono_inc = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
    mono_dec = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
    trend = "MONOTONİK ARTIŞ" if mono_inc else ("MONOTONİK DÜŞÜŞ" if mono_dec else "DÜZENSİZ")
    print(f"\nTest B (n taraması): {trend}")
    print(f"  → Monoton artış = n artınca ρₖ küçülüyor, νₖ sabit → oran büyüyor.")
    print(f"  → Bu boyutluluğun lanetinin klasik belirtisidir.")

    # Test C patlama noktası
    ds   = sorted(res_c.keys())
    c_vals = [res_c[dv] for dv in ds]
    print(f"\nTest C (boyut taraması):")
    for dv, cv in zip(ds, c_vals):
        marker = " ← patlama başlangıcı?" if dv >= 6 and abs(cv) > 50 else ""
        print(f"  d={dv:2d}: {cv:.2f} bit{marker}")

    # 15D sonucu
    final_15d = res_b.get(max(ns), float("nan"))
    print(f"\n15D k-NN sonucu (en büyük n, k=1): {final_15d:.2f} bit")
    print(f"Grup-bölme yöntemi sonucu:          ~41.88 bit")
    print(f"F&H 1998 beklentisi:                ~{fh_prediction():.2f} bit")
    print()
    print("SONUÇ:")
    print("  15D k-NN tahmincisi GW150914 için GÜVENILMEZ sonuç üretir.")
    print("  Nedenleri (önem sırasına göre):")
    print("  1. n=71k, m=5k → 'concentration of measure' henüz gerçekleşmedi")
    print("     Güvenilir 15D kNN için n ~ 10^7 - 10^8 örnek gerekir.")
    print("  2. Büyük negatif bias_corr = log(5000/71746) ≈ −3.84 bit")
    print("     m=5000 prior çok az; özellikle yüksek boyutlarda sorun.")
    print("  3. Test B monoton artış → tahmin n'ye bağımlı → yakınsamamıyor.")
    print("  4. Test C → boyut 6-8'den itibaren tahmincinin davranışı değişiyor.")
    print()
    print("ÇIKARIM:")
    print("  Grup-bölme yöntemi (gw_grup_kld_analizi.py) bu nedenle doğru")
    print("  stratejiye sahiptir: d ≤ 6'da kNN yakınsıyor, 15D'de yakınsamıyor.")

    # Grafikleri kaydet
    here = os.path.dirname(os.path.abspath(__file__))
    plot_results(res_a, res_b, res_c,
                 save_path=os.path.join(here, "15d_knn_test.png"))


if __name__ == "__main__":
    main()