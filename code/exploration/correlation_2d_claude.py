"""
Gaussian KL Varsayımı Doğrulama Scripti
=========================================
Yüksek korelasyonlu m1_source – m2_source parametreleri için
KL(Posterior || Prior) değerini iki farklı yöntemle hesaplar:

  1) Analitik Çok Değişkenli (Multivariate) Gaussian KL
  2) Wang et al. (2009) Genel-k k-NN KL Tahmincisi
     (Referans: blakeaw/Python-knn-entropy)

İki sonuç arasındaki fark, "Gaussian varsayımının" ne kadar
geçerli olduğunu nicel olarak gösterir.
"""

import numpy as np
from pesummary.io import read
from scipy.spatial import cKDTree
from scipy.special import digamma

# ── 1. VERİ YÜKLEME ──────────────────────────────────────────────────────
FILE = (
    "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/"
    "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
)
LABEL = "C01:IMRPhenomXPHM"
PARAM1, PARAM2 = "mass_1_source", "mass_2_source"

print("Veri yükleniyor...")
data   = read(FILE, disable_conversion=True)
post   = data.samples_dict[LABEL]
prior  = data.priors["samples"][LABEL]

P = np.column_stack([np.array(post[PARAM1]),  np.array(post[PARAM2])])   # (N, 2)
Q = np.column_stack([np.array(prior[PARAM1]), np.array(prior[PARAM2])])  # (M, 2)

n, d = P.shape
m    = Q.shape[0]

print(f"Posterior  P : {P.shape}  |  Prior Q : {Q.shape}")
print(f"P Korelasyon (Pearson): {np.corrcoef(P[:, 0], P[:, 1])[0,1]:.4f}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# YÖNTEM 1 – ANALİTİK ÇOK DEĞİŞKENLİ GAUSSIAN KL
# ═══════════════════════════════════════════════════════════════════════════
def gaussian_kl_bits(P, Q):
    """
    KL(P_gauss || Q_gauss) formülü (nats → bits):

        KL = 0.5 * [tr(Σ_q⁻¹ Σ_p) + (μ_q - μ_p)ᵀ Σ_q⁻¹ (μ_q - μ_p)
                    - d + ln|Σ_q| - ln|Σ_p|]
    """
    d     = P.shape[1]
    mu_p  = np.mean(P, axis=0)
    mu_q  = np.mean(Q, axis=0)
    Sp    = np.cov(P, rowvar=False)
    Sq    = np.cov(Q, rowvar=False)
    Sq_inv = np.linalg.inv(Sq)

    diff  = mu_q - mu_p
    term1 = np.trace(Sq_inv @ Sp)
    term2 = diff @ Sq_inv @ diff
    term3 = -d
    _, ldp = np.linalg.slogdet(Sp)
    _, ldq = np.linalg.slogdet(Sq)
    term4 = ldq - ldp

    kl_nat = 0.5 * (term1 + term2 + term3 + term4)
    return kl_nat / np.log(2)           # nats → bits


# ═══════════════════════════════════════════════════════════════════════════
# YÖNTEM 2 – WANG ET AL. (2009) GENEL-k k-NN KL TAHMİNCİSİ
# ═══════════════════════════════════════════════════════════════════════════
#
# Referans: Wang, Kulkarni & Verdú (2009) — IEEE Trans. Inf. Theory
#           "Divergence Estimation for Multidimensional Densities
#            Via k-Nearest-Neighbor Distances"
#
# Formül (Theorem 4):
#
#   KL(P||Q) ≈ (d/n) Σ_i log( ν_k(i) / ρ_k(i) )
#              + log( m / (n-1) )
#              + (digamma(n) - digamma(k)) * ... (bias correction)
#
# Burada:
#   ρ_k(i) = x_i'nin P içindeki k. en yakın komşuya uzaklığı (kendisi hariç)
#   ν_k(i) = x_i'nin Q içindeki k. en yakın komşuya uzaklığı
#
# Bias düzeltmesi için Pérez-Cruz (2008) yaklaşımı kullanılır:
#   log(m/(n-1)) terimi asimptotik sapmanın büyük bölümünü karşılar.
#   digamma düzeltmesi ek ince ayar sağlar (özellikle küçük k için önemli).
# ═══════════════════════════════════════════════════════════════════════════
def knn_kl_bits(P, Q, k=5):
    """
    Wang et al. (2009) genel-k k-NN KL tahmincisi.

    Parametreler
    ------------
    P : ndarray (n, d) – Posterior örnekleri
    Q : ndarray (m, d) – Prior örnekleri
    k : int            – Komşu sayısı (önerilen: 3–10)

    Döndürür
    --------
    kl_bits : float – KL(P||Q) bit cinsinden
    """
    n, d = P.shape
    m    = Q.shape[0]

    # KD-Tree'leri oluştur
    p_tree = cKDTree(P)
    q_tree = cKDTree(Q)

    # ρ_k(i): P içinde k. en yakın komşu mesafesi (k+1 sorgula; 1. kendisi)
    # np.atleast_2d: k=1 durumunda scipy (n,) döndürür, biz (n,1) isteriz
    rho, _ = p_tree.query(P, k=k + 1, p=2, workers=-1)
    rho    = np.atleast_2d(rho)[:, k]     # k. başka komşu (0-indexed: kolon k)

    # ν_k(i): Q içinde k. en yakın komşu mesafesi
    nu, _  = q_tree.query(P, k=k, p=2, workers=-1)
    nu     = np.atleast_2d(nu)[:, k - 1]  # k. komşu (0-indexed: kolon k-1)

    # Sayısal sıfır koruması
    rho = np.maximum(rho, 1e-15)
    nu  = np.maximum(nu,  1e-15)

    # Wang et al. formülü (log₂ cinsinden direkt hesap)
    kl_bits = (
        (d / n) * np.sum(np.log2(nu / rho))        # ana terim
        + np.log2(m / (n - 1.0))                    # yoğunluk oranı düzeltmesi
        # Digamma bias düzeltmesi (ψ(k) terimi k>1 için etkilidir):
        + (digamma(n) - digamma(k)) / np.log(2)     # ince ayar (Wang eq. 12)
    )

    return kl_bits


# ── 3. HESAPLA VE KARŞILAŞTIR ────────────────────────────────────────────
print("=" * 60)
print(" GW150914 | m1_source – m2_source (2D) | KL(Post || Prior)")
print("=" * 60)

# Gaussian
g_bits = gaussian_kl_bits(P, Q)
print(f"\n[1] Analitik Gaussian KL          : {g_bits:.4f} bit")
print(f"    (Gauss varsayımı altında kapalı form çözüm)")

# k-NN – birden fazla k değeri için (kararlılık testi)
print(f"\n[2] Wang et al. k-NN KL (çeşitli k):")
knn_results = {}
for k in [1, 3, 5, 10, 20]:
    bits = knn_kl_bits(P, Q, k=k)
    knn_results[k] = bits
    print(f"    k={k:2d}  →  {bits:.4f} bit")

# k=5 referans değeri al
knn_ref = knn_results[5]

# Fark ve yorum
delta      = knn_ref - g_bits
delta_pct  = 100 * abs(delta) / knn_ref

print(f"\n{'─'*60}")
print(f"  k-NN referans (k=5)               : {knn_ref:.4f} bit")
print(f"  Gaussian KL                        : {g_bits:.4f} bit")
print(f"  Fark (Gauss − k-NN)               : {delta:+.4f} bit")
print(f"  Göreceli sapma                     : {delta_pct:.2f}%")
print(f"{'─'*60}")

# Yorum
print("\n[YORUM]")
if delta_pct < 5:
    print("  ✔ Gaussian varsayımı m1–m2 için oldukça sağlıklı.")
    print("  ✔ 2D joint KL hesabında Gaussian yaklaşımı kullanılabilir.")
elif delta_pct < 15:
    print("  ⚠ Gaussian varsayımı kabul edilebilir, ancak ~%{:.0f} sapma var.".format(delta_pct))
    print("  ⚠ Non-Gaussian kuyruk etkileri mevcut olabilir.")
else:
    print("  ✘ Gaussian varsayımı yetersiz! Sapma %{:.0f} düzeyinde.".format(delta_pct))
    print("  ✘ 15D joint Gaussian KL sonuçları şüpheyle değerlendirilmeli.")

# Ek diagnostik: korelasyon matrisi ve marjinal kontrol
print(f"\n[DIAGNOSTİK]")
print(f"  P (posterior) kovaryans matrisi:")
Sp = np.cov(P, rowvar=False)
print(f"    [[{Sp[0,0]:.4f}, {Sp[0,1]:.4f}],")
print(f"     [{Sp[1,0]:.4f}, {Sp[1,1]:.4f}]]")
print(f"  Q (prior) kovaryans matrisi:")
Sq = np.cov(Q, rowvar=False)
print(f"    [[{Sq[0,0]:.4f}, {Sq[0,1]:.4f}],")
print(f"     [{Sq[1,0]:.4f}, {Sq[1,1]:.4f}]]")
print(f"\n  Posterior Pearson korelasyonu (m1,m2): "
      f"{np.corrcoef(P[:,0],P[:,1])[0,1]:.4f}")
print(f"  Prior Pearson korelasyonu (m1,m2):     "
      f"{np.corrcoef(Q[:,0],Q[:,1])[0,1]:.4f}")
print()