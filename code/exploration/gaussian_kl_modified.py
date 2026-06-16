import numpy as np
from pesummary.io import read

# ── 1. VERİ YÜKLEME ─────────────────────────────────────────────────────
file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
print("Veriler okunuyor...")
data = read(file_name, disable_conversion=True)
label = "C01:IMRPhenomXPHM"

posterior_samples = data.samples_dict[label]
prior_samples = data.priors["samples"][label]

parameters_15D = [
    "mass_1_source", "mass_2_source",
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
    "luminosity_distance", "theta_jn", "psi",
    "azimuth", "zenith", "geocent_time", "phase"
]

# ── 2. MATRİSLERİN OLUŞTURULMASI ────────────────────────────────────────
P_cols = []
Q_cols = []

for param in parameters_15D:
    P_cols.append(np.array(posterior_samples[param]))
    Q_cols.append(np.array(prior_samples[param]))

P_matrix = np.column_stack(P_cols)  # Posterior: P (147634, 15)
Q_matrix = np.column_stack(Q_cols)  # Prior: Q (5000, 15)

print(f"Posterior matrisi (P): {P_matrix.shape}")
print(f"Prior matrisi (Q):     {Q_matrix.shape}")

# ── 3. VERİLERİN STANDARTLAŞTIRILMASI (Z-SCORE NORMALİZASYONU) ─────────
# Referans olarak Prior (Q) matrisinin istatistiklerini alıyoruz.
mu_ref = np.mean(Q_matrix, axis=0)
std_ref = np.std(Q_matrix, axis=0)

# Sıfıra bölme hatasını önlemek için (tamamen sabit parametre varsa)
std_ref[std_ref == 0] = 1e-10

# Z-Score Normalizasyonu uygulanıyor: (Değer - Ortalama) / Standart Sapma
Q_scaled = (Q_matrix - mu_ref) / std_ref
P_scaled = (P_matrix - mu_ref) / std_ref

print("\n[BİLGİ] Veriler standardize edildi. Ölçekleme referansı: Prior.")

# ── 4. ANALİTİK ÇOK DEĞİŞKENLİ GAUSS KL HESAPLAMASI ─────────────────────
def gaussian_kl_divergence_bits(P_scaled, Q_scaled, P_orig, Q_orig, params):
    d = P_scaled.shape[1] 

    # 1. Matematiksel hesaplamalar ÖLÇEKLENMİŞ (kararlı) matrislerle yapılıyor
    mu_p = np.mean(P_scaled, axis=0)
    mu_q = np.mean(Q_scaled, axis=0)

    sigma_p = np.cov(P_scaled, rowvar=False)
    sigma_q = np.cov(Q_scaled, rowvar=False)

    inv_sigma_q = np.linalg.inv(sigma_q)

    term1 = np.trace(np.dot(inv_sigma_q, sigma_p))
    diff = mu_q - mu_p
    term2 = np.dot(diff.T, np.dot(inv_sigma_q, diff))
    term3 = -d
    
    sign_p, logdet_p = np.linalg.slogdet(sigma_p)
    sign_q, logdet_q = np.linalg.slogdet(sigma_q)
    term4 = logdet_q - logdet_p

    kl_nat = 0.5 * (term1 + term2 + term3 + term4)
    kl_bit = kl_nat / np.log(2)

    # 2. Kondisyon sayıları (sağlık durumu) yazdırılıyor
    print(f"Prior kovaryans kondisyon sayısı: {np.linalg.cond(sigma_q):.2e}")
    print(f"Posterior kovaryans kondisyon sayısı: {np.linalg.cond(sigma_p):.2e}")

    # 3. İnsan okuyabilirliği için fiziksel limitler ORİJİNAL matrislerden yazdırılıyor
    for i, p in enumerate(params):
        q_col = Q_orig[:, i]
        p_col = P_orig[:, i]
        print(f"{p:<25}: prior [{q_col.min():.2f}, {q_col.max():.2f}]  "
              f"posterior [{p_col.min():.2f}, {p_col.max():.2f}]")

    return kl_bit

# ── 5. HESAPLAMA VE ÇIKTI ────────────────────────────────────────────────
print("\nÇok Değişkenli Gauss (Analitik) hesaplama başlıyor...")

# Fonksiyona hem hesabı yapacağı ölçekli veriyi hem de ekrana basacağı orijinal veriyi gönderiyoruz
bilgi_kazanci = gaussian_kl_divergence_bits(P_scaled, Q_scaled, P_matrix, Q_matrix, parameters_15D)

# Sonuç Ekranı
print("-" * 55)
print(f"Analitik 15D KL Iraksaması (Standartlaştırılmış): {bilgi_kazanci:.4f} bit")
print(f"F&H referans tahmini:                             ~41.5 bit")
print("-" * 55)