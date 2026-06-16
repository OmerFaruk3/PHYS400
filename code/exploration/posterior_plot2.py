import numpy as np
from pesummary.io import read
from scipy.stats import gaussian_kde
from numpy import trapezoid

# ─── 1. Dosyayı Yükle ───────────────────────────────────────────────────
file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
data = read(file_name, disable_conversion=True)
label = "C01:IMRPhenomXPHM"
samples = data.samples_dict[label]
prior_samples = data.priors["samples"][label]

# ─── 2. F&H'ın 15 BBH Parametresi ──────────────────────────────────────
# Intrinsic (8): kütleler + spinler
# Extrinsic (7): mesafe, yönelim, gökyüzü konumu, zaman, faz
params_15 = [
    # Intrinsic
    "chirp_mass_source",   # M_chirp
    "mass_ratio",          # q = m2/m1
    "a_1",                 # spin magnitude 1
    "a_2",                 # spin magnitude 2
    "tilt_1",              # spin tilt 1
    "tilt_2",              # spin tilt 2
    "phi_12",              # azimuthal angle between spins
    "phi_jl",              # azimuthal angle J-L
    # Extrinsic
    "luminosity_distance", # d_L
    "theta_jn",            # inclination
    "psi",                 # polarization
    "ra",                  # right ascension
    "dec",                 # declination
    "geocent_time",        # merger time
    "phase",               # coalescence phase
]

# ─── 3. Matrix Oluştur: (N_samples, 15) ─────────────────────────────────
# Her satır: bir sample'ın tüm parametreleri
N_sub = 5000  # hesaplama için subsample (147k çok büyük)
idx = np.random.choice(len(np.array(samples[params_15[0]])), N_sub, replace=False)

posterior_matrix = np.column_stack([
    np.array(samples[p])[idx] for p in params_15
])  # shape: (5000, 15)

prior_matrix = np.column_stack([
    np.array(prior_samples[p]) for p in params_15
])  # shape: (N_prior, 15) — prior'dan da subsample al
idx_prior = np.random.choice(len(prior_matrix), N_sub, replace=False)
prior_matrix = prior_matrix[idx_prior]

print(f"Posterior matrix shape: {posterior_matrix.shape}")  # (5000, 15)
print(f"Prior matrix shape:     {prior_matrix.shape}")      # (5000, 15)

# ─── 4. Multivariate Gaussian Fit ───────────────────────────────────────
# Posterior: ortalama ve kovaryans
mu_post  = np.mean(posterior_matrix, axis=0)       # shape: (15,)
cov_post = np.cov(posterior_matrix.T)               # shape: (15, 15)

# Prior: ortalama ve kovaryans
mu_prior = np.mean(prior_matrix, axis=0)
cov_prior = np.cov(prior_matrix.T)

print(f"\nPosterior mean (chirp mass): {mu_post[0]:.2f} M☉")
print(f"Prior mean     (chirp mass): {mu_prior[0]:.2f} M☉")

# ─── 5. Analitik KL Divergence (Gaussian) ───────────────────────────────
def kl_multivariate_gaussian(mu1, cov1, mu2, cov2):
    """
    KL(N1 || N2) — posterior=N1, prior=N2
    Formül: 0.5 * [tr(Σ2⁻¹ Σ1) + (μ2-μ1)ᵀ Σ2⁻¹ (μ2-μ1) - k + ln(det(Σ2)/det(Σ1))]
    Sonuç: nats cinsinden → bits için log2 dönüşümü: / ln(2)
    """
    k = len(mu1)
    cov2_inv = np.linalg.inv(cov2)
    
    # tr(Σ2⁻¹ Σ1)
    trace_term = np.trace(cov2_inv @ cov1)
    
    # (μ2-μ1)ᵀ Σ2⁻¹ (μ2-μ1)
    diff = mu2 - mu1
    quad_term = diff @ cov2_inv @ diff
    
    # ln(det(Σ2)/det(Σ1)) — sayısal kararlılık için log-det kullan
    sign1, logdet1 = np.linalg.slogdet(cov1)
    sign2, logdet2 = np.linalg.slogdet(cov2)
    logdet_term = logdet2 - logdet1
    
    kl_nats = 0.5 * (trace_term + quad_term - k + logdet_term)
    kl_bits = kl_nats / np.log(2)
    return kl_bits

I_joint = kl_multivariate_gaussian(mu_post, cov_post, mu_prior, cov_prior)
print(f"\n=== Sonuç ===")
print(f"Joint KL Divergence (15 parametre): {I_joint:.2f} bits")
print(f"F&H analitik tahmin (I_source):      ~41.5 bits")
print(f"Fark: {abs(I_joint - 41.5):.2f} bits ({abs(I_joint-41.5)/41.5*100:.1f}%)")

# ─── 6. Karşılaştırma: Marjinal Toplamı ─────────────────────────────────
# Her parametre için ayrı ayrı KL hesapla, topla
marginal_sum = 0.0
print("\nParametre başına KL divergence:")
for i, p in enumerate(params_15):
    mu1_i, sig1_i = mu_post[i], np.sqrt(cov_post[i,i])
    mu2_i, sig2_i = mu_prior[i], np.sqrt(cov_prior[i,i])
    # 1D Gaussian KL
    kl_i = (np.log(sig2_i/sig1_i) + (sig1_i**2 + (mu1_i-mu2_i)**2)/(2*sig2_i**2) - 0.5) / np.log(2)
    marginal_sum += kl_i
    print(f"  {p:<25}: {kl_i:.3f} bits")

print(f"\nMarjinal toplamı:   {marginal_sum:.2f} bits")
print(f"Joint KL:           {I_joint:.2f} bits")
print(f"Total Correlation:  {I_joint - marginal_sum:.2f} bits  (korelasyondan gelen katkı)")