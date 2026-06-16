import numpy as np
from pesummary.io import read

# ── 1. VERİ YÜKLEME ─────────────────────────────────────────────────────
# file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
# file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191103_012549_PEDataRelease_mixed_cosmo.h5"
file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191105_143521_PEDataRelease_mixed_cosmo.h5"

data = read(file_name, disable_conversion=True)
label = "C01:IMRPhenomXPHM"

posterior_samples = data.samples_dict[label]

# Sentetik üretim iptal edildi. Doğrudan dosyadaki 5000 örneklemi çekiyoruz.
prior_samples = data.priors["samples"][label]

parameters_15D = [
    "mass_1_source", "mass_2_source",
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
    "luminosity_distance", "theta_jn", "psi",
    "azimuth", "zenith", "geocent_time", "phase"
]

# ── 2. MATRİSLERİN OLUŞTURULMASI ────────────────────────────────────────
P_cols =[]
Q_cols =[]

for param in parameters_15D:
    P_cols.append(np.array(posterior_samples[param]))
    # Doğrudan PESummary'nin kaydettiği 5000 prior örneklemini listeye ekliyoruz
    Q_cols.append(np.array(prior_samples[param]))

P_matrix = np.column_stack(P_cols)  # Posterior: P (147634, 15)
Q_matrix = np.column_stack(Q_cols)  # Prior: Q (5000, 15)

print(f"Posterior matrisi (P): {P_matrix.shape}")
print(f"Prior matrisi (Q):     {Q_matrix.shape}")

# ── 3. ANALİTİK ÇOK DEĞİŞKENLİ GAUSS KL HESAPLAMASI ─────────────────────
def gaussian_kl_divergence_bits(P, Q):
    d = P.shape[1] #Posterior) matrisinin 2. boyutu olan sütun sayısını (parametre sayısını, yani 15'i) verir.

    mu_p = np.mean(P, axis=0) #Prior ve Posterior matrislerinin her bir sütununun(parametresinin) ortalamasını hesaplayıp bir vektör oluşturur. 
    mu_q = np.mean(Q, axis=0) 

    sigma_p = np.cov(P, rowvar=False) #Kovaryans matrislerini ($\Sigma_P, \Sigma_Q$) oluşturur. 
    sigma_q = np.cov(Q, rowvar=False) #parametrelerin hem kendi içindeki varyansını hem de birbirleriyle olan ilişkisini gösterir.

    inv_sigma_q = np.linalg.inv(sigma_q)  #Prior kovaryans matrisinin tersini ($\Sigma_Q^{-1}$) alır

    term1 = np.trace(np.dot(inv_sigma_q, sigma_p)) #Posterior'un varyansının Prior'un varyansına oranını (çok boyutlu uzayda) ölçer
    
    diff = mu_q - mu_p
    term2 = np.dot(diff.T, np.dot(inv_sigma_q, diff)) #Posterior'un ortalamasının Prior'un ortalamasından ne kadar "uzağa" kaydığını ölçer.
    
    term3 = -d #d=15
    
    sign_p, logdet_p = np.linalg.slogdet(sigma_p) #Kovaryans matrislerinin logaritmik determinantını hesaplar. 
    sign_q, logdet_q = np.linalg.slogdet(sigma_q)
    term4 = logdet_q - logdet_p

    kl_nat = 0.5 * (term1 + term2 + term3 + term4)
    kl_bit = kl_nat / np.log(2)

    # # Prior kovaryans matrisinin sağlığını kontrol et
    # print(f"Prior kovaryans kondisyon sayısı: {np.linalg.cond(sigma_q):.2e}")
    # print(f"Posterior kovaryans kondisyon sayısı: {np.linalg.cond(sigma_p):.2e}")

# Prior'un her parametredeki aralığını göster
    for i, p in enumerate(parameters_15D):
        q_col = Q_matrix[:, i]
        p_col = P_matrix[:, i]
        print(f"{p:<25}: prior [{q_col.min():.2f}, {q_col.max():.2f}]  "
            f"posterior [{p_col.min():.2f}, {p_col.max():.2f}]")

    return kl_bit

# ── 4. HESAPLAMA VE ÇIKTI ────────────────────────────────────────────────

bilgi_kazanci = gaussian_kl_divergence_bits(P_matrix, Q_matrix)

print("-" * 55)
print(f"Analitik 15D KL Iraksaması :                {bilgi_kazanci:.4f} bit")
print(f"F&H referans tahmini:                        ~41.5 bit")
print("-" * 55)

