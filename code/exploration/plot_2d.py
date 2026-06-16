"""
Gaussian KL Varsayımı Doğrulama Scripti  (DÜZELTİLMİŞ)
=========================================================
m1_source – m2_source parametreleri için KL(Posterior || Prior)
dört farklı yöntemle hesaplanır ve karşılaştırılır.

Düzeltilen Hatalar:
  [1] Wang et al. formülündeki digamma() düzeltmesi KALDIRILDI.
      Bu terim entropi tahmincisine (Kozachenko-Leonenko) aittir;
      KL divergence tahmincisinde yoktur.
  [2] KDE grid sınırı artık Q yerine P'ye (posterior) göre tanımlanıyor.
      KL = ∫ p(x) log(p/q) dx → entegrand yalnızca p>0 bölgede sıfırdan
      farklıdır → grid P'nin desteğini kapsayacak şekilde kurulmalıdır.

Yöntemler:
  1) Analitik Multivariate Gaussian KL  [parametrik, varsayım var]
  2) 2D KDE + Simpson İntegrasyonu      [non-parametrik, 2D referans]
  3) Pérez-Cruz (2008) k-NN, k=1       [non-parametrik]
  4) Wang et al. (2009) k-NN, k=1,3,5  [non-parametrik, genel-k]
"""

import numpy as np
from pesummary.io import read
from scipy.stats import gaussian_kde
from scipy.integrate import simpson
from scipy.spatial import cKDTree

# ── 1. VERİ YÜKLEME ──────────────────────────────────────────────────────
FILE = (
    "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/"
    "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
)
LABEL  = "C01:IMRPhenomXPHM"
PARAM1 = "mass_1_source"
PARAM2 = "mass_2_source"

print("Veri yükleniyor...")
data  = read(FILE, disable_conversion=True)
post  = data.samples_dict[LABEL]
prior = data.priors["samples"][LABEL]

P = np.column_stack([np.array(post[PARAM1]),  np.array(post[PARAM2])])
Q = np.column_stack([np.array(prior[PARAM1]), np.array(prior[PARAM2])])

n, d = P.shape
m    = Q.shape[0]

print(f"Posterior  P : {P.shape}  |  Prior Q : {Q.shape}")
print(f"P araligi m1 : [{P[:,0].min():.1f}, {P[:,0].max():.1f}]  "
      f"| Q araligi m1 : [{Q[:,0].min():.1f}, {Q[:,0].max():.1f}]")
print(f"P araligi m2 : [{P[:,1].min():.1f}, {P[:,1].max():.1f}]  "
      f"| Q araligi m2 : [{Q[:,1].min():.1f}, {Q[:,1].max():.1f}]")
print(f"Posterior Pearson korelasyonu (m1,m2): {np.corrcoef(P[:,0],P[:,1])[0,1]:.4f}")
print(f"Prior Pearson korelasyonu (m1,m2):     {np.corrcoef(Q[:,0],Q[:,1])[0,1]:.4f}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# YÖNTEM 1 - ANALITIK MULTIVARIATE GAUSSIAN KL
# ═══════════════════════════════════════════════════════════════════════════
#
# KL(N_p || N_q) = 0.5 * [tr(Sigma_q^-1 Sigma_p)
#                         + (mu_q - mu_p)^T Sigma_q^-1 (mu_q - mu_p)
#                         - d + ln|Sigma_q| - ln|Sigma_p|]
#
# DIKKAT: Bu formul her iki dagilimi da Gaussian varsayar.
# Prior tipik olarak duz (uniform/power-law) oldugundan bu varsayim
# buyuk ihtimalle bozulmaktadir.
# ═══════════════════════════════════════════════════════════════════════════
def gaussian_kl_bits(P, Q):
    d      = P.shape[1]
    mu_p   = np.mean(P, axis=0)
    mu_q   = np.mean(Q, axis=0)
    Sp     = np.cov(P, rowvar=False)
    Sq     = np.cov(Q, rowvar=False)
    Sq_inv = np.linalg.inv(Sq)
    diff   = mu_q - mu_p
    term1  = np.trace(Sq_inv @ Sp)
    term2  = diff @ Sq_inv @ diff
    _, ldp = np.linalg.slogdet(Sp)
    _, ldq = np.linalg.slogdet(Sq)
    kl_nat = 0.5 * (term1 + term2 - d + ldq - ldp)
    return kl_nat / np.log(2)


# ═══════════════════════════════════════════════════════════════════════════
# YÖNTEM 2 - 2D KDE + SIMPSON ENTEGRASYONU  (DUZELTILMIS GRID)
# ═══════════════════════════════════════════════════════════════════════════
#
# KL = integral p(x) log2(p(x)/q(x)) dx
#
# HATA DURUMU (ESKI KOD): Grid Q'nun (prior) araligina gore tanimlaniyordu.
#   Prior m1 araligi ~[5, 100] Msun -> 250 grid noktasi ~0.38 Msun cozunurluk
#   Posterior m1 araligi ~[25, 50] Msun -> o bolgede sadece ~66 grid noktasi
#   -> Underestimation (dusuk tahmin)
#
# DUZELTILMIS: Grid P'nin (posterior) araligina gore tanimlanıyor.
#   Posterior bölgesinde tum 300 nokta kullaniliyor -> daha iyi cozunurluk.
#   Posterior disina tasan q(x) bolgeleri zaten entegranda p(x)~0 oldugundan
#   KL'ye katki yapmaz.
# ═══════════════════════════════════════════════════════════════════════════
def kde_kl_2d_bits(P, Q, grid_points=300):
    kde_p = gaussian_kde(P.T, bw_method='scott')
    kde_q = gaussian_kde(Q.T, bw_method='scott')

    # Grid: P'nin 5*sigma araligini kapsayacak sekilde
    sigma_p = np.std(P, axis=0)
    x_min = P[:, 0].mean() - 5 * sigma_p[0]
    x_max = P[:, 0].mean() + 5 * sigma_p[0]
    y_min = P[:, 1].mean() - 5 * sigma_p[1]
    y_max = P[:, 1].mean() + 5 * sigma_p[1]

    # Guvenlik kontrolu: Q bu araligi kapsayamali
    if not (Q[:, 0].min() <= x_min and Q[:, 0].max() >= x_max):
        print("  UYARI: Prior, posterior'un m1 araligini tam kapsamiyor!")
    if not (Q[:, 1].min() <= y_min and Q[:, 1].max() >= y_max):
        print("  UYARI: Prior, posterior'un m2 araligini tam kapsamiyor!")

    x_grid = np.linspace(x_min, x_max, grid_points)
    y_grid = np.linspace(y_min, y_max, grid_points)
    X, Y   = np.meshgrid(x_grid, y_grid)

    positions = np.vstack([X.ravel(), Y.ravel()])   # (2, grid_points^2)
    p_val = kde_p(positions).reshape(grid_points, grid_points)
    q_val = kde_q(positions).reshape(grid_points, grid_points)

    mask      = (p_val > 1e-300) & (q_val > 1e-300)
    integrand = np.zeros_like(p_val)
    integrand[mask] = p_val[mask] * np.log2(p_val[mask] / q_val[mask])

    int_x  = simpson(integrand, x=x_grid, axis=1)
    kl_bit = simpson(int_x,     x=y_grid)
    return kl_bit


# ═══════════════════════════════════════════════════════════════════════════
# YÖNTEM 3 - PEREZ-CRUZ (2008) k-NN KL, k=1
# ═══════════════════════════════════════════════════════════════════════════
#
# Perez-Cruz (2008), NeurIPS - Eq. (14):
#
#   KL(P||Q) = (d/n) * sum_i log(s_i / r_i)  +  log(m / (n-1))
#
# r_i : x_i'nin P icindeki en yakin komsuna (kendisi haric) uzakligi  (k=1)
# s_i : x_i'nin Q icindeki en yakin komsuna uzakligi                  (k=1)
# ═══════════════════════════════════════════════════════════════════════════
def perez_cruz_kl_bits(P, Q):
    n, d   = P.shape
    m      = Q.shape[0]
    p_tree = cKDTree(P)
    q_tree = cKDTree(Q)

    r_all, _ = p_tree.query(P, k=2, p=2)
    r = r_all[:, 1]               # kendisi haric 1. komsu
    s, _ = q_tree.query(P, k=1, p=2)   # Q'daki 1. komsu; skaler dizi (n,)

    r = np.maximum(r, 1e-15)
    s = np.maximum(s, 1e-15)

    kl_nat = (d / n) * np.sum(np.log(s / r)) + np.log(m / (n - 1.0))
    return kl_nat / np.log(2)


# ═══════════════════════════════════════════════════════════════════════════
# YÖNTEM 4 - WANG ET AL. (2009) k-NN KL, GENEL k  (DUZELTILMIS)
# ═══════════════════════════════════════════════════════════════════════════
#
# Wang, Kulkarni & Verdu (2009), IEEE Trans. Inf. Theory - Theorem 1:
#
#   KL(P||Q) = (d/n) * sum_i log(nu_k(i) / rho_k(i))  +  log(m / (n-1))
#
# rho_k(i) : x_i'nin P icindeki k. komsuna uzakligi (kendisi haric)
# nu_k(i)  : x_i'nin Q icindeki k. komsuna uzakligi
#
# HATA DURUMU (ESKI KOD): + (digamma(n) - digamma(k)) / ln2 eklenmisti.
#   Bu terim Kozachenko-Leonenko ENTROPI tahmincisine aittir.
#   KL divergence formulunde digamma terimleri analitik olarak iptal olur.
#   n=147634 icin bu yanlis duzeltme k=1'de +18 bit, k=5'te +15 bit
#   sahte katki yapiyordu.
#
# DUZELTILMIS: Digamma terimi kaldirildi.
# ═══════════════════════════════════════════════════════════════════════════
def wang_kl_bits(P, Q, k=5):
    n, d   = P.shape
    m      = Q.shape[0]
    p_tree = cKDTree(P)
    q_tree = cKDTree(Q)

    # rho_k(i): k+1 sorgu (ilki kendisi), k. sutunu al
    rho_all, _ = p_tree.query(P, k=k + 1, p=2, workers=-1)
    rho = np.atleast_2d(rho_all)[:, k]

    # nu_k(i): k sorgu, (k-1). sutunu al (0-indexed)
    nu_all, _  = q_tree.query(P, k=k,     p=2, workers=-1)
    nu  = np.atleast_2d(nu_all)[:, k - 1]

    rho = np.maximum(rho, 1e-15)
    nu  = np.maximum(nu,  1e-15)

    # Wang et al. formulu - digamma terimi YOK (bkz. Theorem 1)
    kl_nat = (d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1.0))
    return kl_nat / np.log(2)


# ── CALISTIR VE KARSILASTIR ──────────────────────────────────────────────
print("=" * 62)
print("  GW150914 | m1_source - m2_source (2D) | KL(Post || Prior)")
print("=" * 62)

g_bits = gaussian_kl_bits(P, Q)
print(f"\n[1] Analitik Gaussian KL             : {g_bits:.4f} bit")
print(f"    DIKKAT: Prior Gaussian degilse bu deger guvenilir DEGILDIR.")

print(f"\n[2] KDE + Simpson (grid=300) hesaplaniyor...")
kde_bits = kde_kl_2d_bits(P, Q, grid_points=300)
print(f"    Sonuc                              : {kde_bits:.4f} bit")
print(f"    (referans yontem - 2D'de en guvenilir)")

pc_bits = perez_cruz_kl_bits(P, Q)
print(f"\n[3] Perez-Cruz k-NN (k=1)            : {pc_bits:.4f} bit")

print(f"\n[4] Wang et al. k-NN (cesitli k):")
wang_results = {}
for k in [1,2,3,4,5,6,7,8,9,10]:
    wb = wang_kl_bits(P, Q, k=k)
    wang_results[k] = wb
    print(f"    k={k:2d}  ->  {wb:.4f} bit")

# ── OZET ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*62}")
print("  OZET VE YORUMLAMA")
print(f"{'─'*62}")

non_param_vals = [kde_bits, pc_bits] + list(wang_results.values())
np_mean = np.mean(non_param_vals)
np_std  = np.std(non_param_vals)

print(f"  Non-parametrik yontemler (KDE + k-NN):")
print(f"    Ortalama : {np_mean:.4f} bit")
print(f"    Std      : {np_std:.4f} bit  (tutarlilik gostergesi)")

delta     = g_bits - np_mean
delta_pct = 100 * abs(delta) / np_mean
sign      = "ABARTMAKTADIR" if delta > 0 else "KUCUMSEMEKTEDIR"
print(f"\n  Gaussian KL                          : {g_bits:.4f} bit")
print(f"  Non-parametrik ortalama              : {np_mean:.4f} bit")
print(f"  Fark (Gauss - Non-param)             : {delta:+.4f} bit ({delta_pct:.1f}%)")
print(f"\n  YORUM: Gaussian, gercek KL'yi {delta_pct:.1f}% {sign}.")
print(f"""
  Sebep: Gaussian KL hem p(x) hem q(x)'i Gaussian varsayarak
  kovaryans matrislerini karsilastirir. Prior (Q) ise tipik olarak
  duz (flat) ya da power-law dagilimlıdır; Gaussian fit priorun
  gercek seklini yanlis temsil eder -> KL yanlis cikar.

  15D Gaussian KL (~41.24 bit) bu nedenle sistematik hata iceriyor
  olabilir. Non-parametrik yontemlerin mutabik kaldigi deger gerceğe
  daha yakindir.
""")

# ── DIAGNOSTIK ───────────────────────────────────────────────────────────
Sp = np.cov(P, rowvar=False)
Sq = np.cov(Q, rowvar=False)
print(f"[DIAGNOSTIK]")
print(f"  Posterior Sigma_p : [[{Sp[0,0]:.2f}, {Sp[0,1]:.2f}], [{Sp[1,0]:.2f}, {Sp[1,1]:.2f}]]")
print(f"  Prior     Sigma_q : [[{Sq[0,0]:.2f}, {Sq[0,1]:.2f}], [{Sq[1,0]:.2f}, {Sq[1,1]:.2f}]]")
print(f"  Prior m1  aralik  : [{Q[:,0].min():.1f}, {Q[:,0].max():.1f}] Msun")
print(f"  Prior m2  aralik  : [{Q[:,1].min():.1f}, {Q[:,1].max():.1f}] Msun")
print(f"  Post. m1  aralik  : [{P[:,0].min():.1f}, {P[:,0].max():.1f}] Msun")
print(f"  Post. m2  aralik  : [{P[:,1].min():.1f}, {P[:,1].max():.1f}] Msun")