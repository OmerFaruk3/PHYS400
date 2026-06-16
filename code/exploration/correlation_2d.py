
import numpy as np
from pesummary.io import read
from scipy.stats import gaussian_kde
from scipy.integrate import simpson
from scipy.spatial import cKDTree

FILE = (
    "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/"
    "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
)
LABEL  = "C01:IMRPhenomXPHM"

# parameters_15D = [
#     ("mass_1_source",      r"$m_1\ [M_\odot]$"),
#     ("mass_2_source",      r"$m_2\ [M_\odot]$"),
#     ("a_1",                r"$a_1$ (spin mag.)"),
#     ("a_2",                r"$a_2$ (spin mag.)"),
#     ("tilt_1",             r"tilt$_1$ [rad]"),
#     ("tilt_2",             r"tilt$_2$ [rad]"),
#     ("phi_12",             r"$\phi_{12}$ [rad]"),
#     ("phi_jl",             r"$\phi_{JL}$ [rad]"),
#     ("luminosity_distance",r"$d_L$ [Mpc]"),
#     ("theta_jn",           r"$\theta_{JN}$ [rad]"),
#     ("psi",                r"$\psi$ [rad]"),
#     ("azimuth",            r"azimuth [rad]"),
#     ("zenith",             r"zenith [rad]"),
#     ("geocent_time",       r"$t_c$ [s]"),
#     ("phase",              r"phase [rad]"),
# ]


# PARAM1 = "theta_jn"   
# PARAM2 = "luminosity_distance"   

PARAM1 = "mass_1_source"   
PARAM2 = "mass_2_source"   


KDE_GRID   = 300           # KDE izgara cozunurlugu (her axis icin)
WANG_K_MAX = 10            # maks k , Wang k-NN  


# ── 1. VERİ YÜKLEME ──────────────────────────────────────────────────────
data  = read(FILE, disable_conversion=True)
post  = data.samples_dict[LABEL]
prior = data.priors["samples"][LABEL]

P = np.column_stack([np.array(post[PARAM1]),  np.array(post[PARAM2])])
Q = np.column_stack([np.array(prior[PARAM1]), np.array(prior[PARAM2])])

n, d = P.shape
m    = Q.shape[0]

# ── 2. VERİ  ──────────────────────────────────────────────────────
print(f"\nParametreler  : ({PARAM1},  {PARAM2})")
print(f"Posterior  P  : {P.shape}  |  Prior Q : {Q.shape}")
print(f"  {PARAM1:<28} P: [{P[:,0].min():.4g}, {P[:,0].max():.4g}]"
      f"  |  Q: [{Q[:,0].min():.4g}, {Q[:,0].max():.4g}]")
print(f"  {PARAM2:<28} P: [{P[:,1].min():.4g}, {P[:,1].max():.4g}]"
      f"  |  Q: [{Q[:,1].min():.4g}, {Q[:,1].max():.4g}]")


# YÖNTEM 1 — ANALİTİK MULTİVARİATE GAUSSIAN KL ─────────────────────────────────────────────────

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
# YÖNTEM 2 — 2D KDE 
# ═══════════════════════════════════════════════════════════════════════════
# Grid P'nin (posterior) 5*sigma araligina gore tanimlıyoruz.
# KL = integral p(x) log2(p/q) dx formulunde entegrand yalnizca
# p(x) > 0 bölgesinde hesaplama mantıklı

def kde_kl_2d_bits(P, Q, grid_points=300):
    kde_p = gaussian_kde(P.T, bw_method='scott')
    kde_q = gaussian_kde(Q.T, bw_method='scott')


    x_min = P[:, 0].min()
    x_max = P[:, 0].max()
    y_min = P[:, 1].min()
    y_max = P[:, 1].max()

    # Tüm değerleri kapsamak için 5 sigma yerine doğrudan veri setinin sınırlarını alıyoruz
    # sigma_p = np.std(P, axis=0)
    # x_min = P[:, 0].mean() - 5 * sigma_p[0]
    # x_max = P[:, 0].mean() + 5 * sigma_p[0]
    # y_min = P[:, 1].mean() - 5 * sigma_p[1]
    # y_max = P[:, 1].mean() + 5 * sigma_p[1]

    # Prior aralik kontrolleri
    warn_x = not (Q[:, 0].min() <= x_min and Q[:, 0].max() >= x_max)
    warn_y = not (Q[:, 1].min() <= y_min and Q[:, 1].max() >= y_max)
    if warn_x:
        print(f"  [UYARI] Prior, posterior'un {PARAM1} araligini tam kapsamiyor")
    if warn_y:
        print(f"  [UYARI] Prior, posterior'un {PARAM2} araligini tam kapsamiyor")

    x_grid = np.linspace(x_min, x_max, grid_points) #300 noktali grid
    y_grid = np.linspace(y_min, y_max, grid_points) #300 noktali grid
    X, Y   = np.meshgrid(x_grid, y_grid)  #300x300 grid

    positions = np.vstack([X.ravel(), Y.ravel()])   # X ile Y alt alta ekle ve 2x90000 boyutunda bir koordinat oluştur.
    p_val = kde_p(positions).reshape(grid_points, grid_points)  #yoğunluk hesaplar
    q_val = kde_q(positions).reshape(grid_points, grid_points)

    mask      = (p_val > 1e-300) & (q_val > 1e-300)  #sıfıra bölme hatasından kaçmak için
    integrand = np.zeros_like(p_val)
    integrand[mask] = p_val[mask] * np.log2(p_val[mask] / q_val[mask])

    int_x  = simpson(integrand, x=x_grid, axis=1) # hem y hem x için integral alma (yönü x, sonra y) simpson ile
    return simpson(int_x, x=y_grid)


# ═══════════════════════════════════════════════════════════════════════════
# YÖNTEM 3 — PÉREZ-CRUZ (2008) k-NN, k=1
# ═══════════════════════════════════════════════════════════════════════════
# KL(P||Q) = (d/n) * sum_i log(s_i / r_i)  +  log(m / (n-1))
# r_i: P icindeki en yakin komsu mesafesi (kendisi haric)
# s_i: Q icindeki en yakin komsu mesafesi
def perez_cruz_kl_bits(P, Q):
    n, d   = P.shape         # n: P veri setindeki örnek sayısı.  d=verinin boyutu (bu durumda 2).
    m      = Q.shape[0]     #m: Q veri setindeki örnek sayısı.
    p_tree = cKDTree(P)
    q_tree = cKDTree(Q)
    
    r_all, _ = p_tree.query(P, k=2, p=2) # P içindeki her bir nokta için yine $P$ içindeki en yakın 2 noktayı bulur (k=2), p=2 Euclidean (Öklid) mesafesi
    r = r_all[:, 1]                       #en yakın komşu (0-indexed: kolon 1, çünkü kolon 0 kendisi)   
    
    s, _ = q_tree.query(P, k=1, p=2)       # P içindeki her bir nokta için Q içindeki en yakın komşuyu bulur (k=1)
    r = np.maximum(r, 1e-15)                #sıfıra bölmeden koruma
    s = np.maximum(s, 1e-15)                #sıfıra bölmeden koruma

    kl_nat = (d / n) * np.sum(np.log(s / r)) + np.log(m / (n - 1.0))
    return kl_nat / np.log(2)


# ═══════════════════════════════════════════════════════════════════════════
# YÖNTEM 4 — WANG ET AL. (2009) k-NN, GENEL k
# ═══════════════════════════════════════════════════════════════════════════
# KL(P||Q) = (d/n) * sum_i log(nu_k(i) / rho_k(i))  +  log(m / (n-1))
# rho_k(i): P icindeki k. komsu (kendisi haric)
# nu_k(i) : Q icindeki k. komsu

def wang_kl_bits(P, Q, k=5):
    n, d   = P.shape
    m      = Q.shape[0]
    p_tree = cKDTree(P)
    q_tree = cKDTree(Q)

    rho_all, _ = p_tree.query(P, k=k + 1, p=2, workers=-1) # P içindeki her bir nokta için P içindeki en yakın k+1 komşuyu bulur (k+1 çünkü kendisi de dahil)
    rho = np.atleast_2d(rho_all)[:, k]                      # k. komşu (0-indexed: kolon k)
    nu_all, _  = q_tree.query(P, k=k,     p=2, workers=-1)
    nu  = np.atleast_2d(nu_all)[:, k - 1]

    rho = np.maximum(rho, 1e-15)
    nu  = np.maximum(nu,  1e-15)

    kl_nat = (d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1.0))
    return kl_nat / np.log(2)


# ── 3. HESAPLA ───────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  KL(Posterior || Prior)  |  ({PARAM1}, {PARAM2})")
print(f"{'='*62}")

g_bits = gaussian_kl_bits(P, Q)
print(f"\n[1] Multivariate Gaussian KL         : {g_bits:.4f} bit")


print(f"\n[2] KDE + Simpson (grid={KDE_GRID}) hesaplaniyor...")
kde_bits = kde_kl_2d_bits(P, Q, grid_points=KDE_GRID)
print(f"    Sonuc                          : {kde_bits:.4f} bit")


pc_bits = perez_cruz_kl_bits(P, Q)
print(f"\n[3] Perez-Cruz k-NN (k=1)        : {pc_bits:.4f} bit")

print(f"\n[4] Wang et al. k-NN (k=1..{WANG_K_MAX}):")
wang_results = {}
for k in range(1, WANG_K_MAX + 1):
    wb = wang_kl_bits(P, Q, k=k)
    wang_results[k] = wb
    print(f"    k={k:2d}  ->  {wb:.4f} bit")

# ── 4. ÖZET ──────────────────────────────────────────────────────────────
print(f"\n{'─'*62}")
print("  OZET")
print(f"{'─'*62}")

non_param_vals = [pc_bits] + list(wang_results.values())
np_mean = np.mean(non_param_vals)
np_std  = np.std(non_param_vals)
delta     = g_bits - np_mean
delta_pct = 100 * abs(delta) / np_mean

print(f"  Non-parametrik ortalama (k-NN)     : {np_mean:.4f} bit")
print(f"  Non-parametrik std                 : {np_std:.4f} bit")
print(f"  Multivariate Gaussian KL           : {g_bits:.4f} bit")
print(f"  Fark (Gauss - Non-param)           : {delta:+.4f} bit  ({delta_pct:.1f}%)")

# ── 5. DIAGNOSTİK ────────────────────────────────────────────────────────
Sp = np.cov(P, rowvar=False)
Sq = np.cov(Q, rowvar=False)
print(f"\n{'─'*62}")
print("  DIAGNOSTIK")
print(f"{'─'*62}")
print(f"  Posterior Sigma_p : [[{Sp[0,0]:.4g}, {Sp[0,1]:.4g}],"
      f" [{Sp[1,0]:.4g}, {Sp[1,1]:.4g}]]")
print(f"  Prior     Sigma_q : [[{Sq[0,0]:.4g}, {Sq[0,1]:.4g}],"
      f" [{Sq[1,0]:.4g}, {Sq[1,1]:.4g}]]")
print(f"  Condition(Sq)     : {np.linalg.cond(Sq):.2e}")
print()