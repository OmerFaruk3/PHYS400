import numpy as np
import h5py
from pesummary.io import read

file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"

data = read(file_name, disable_conversion=True)
posterior_samples = data.samples_dict["C01:IMRPhenomXPHM"]

params_15 = [
    "mass_1_source", "mass_2_source",
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
    "luminosity_distance", "theta_jn", "psi",
    "azimuth", "zenith", "geocent_time", "phase"
]

# --- Posterior: 147k × 15 matris ---
post_matrix = np.column_stack([np.array(posterior_samples[p]) for p in params_15])
print(f"Posterior matrix shape: {post_matrix.shape}")  # (147634, 15)

# --- Prior: 5000 × 15 matris (bağımsız marginallerden oluşturuluyor) ---
with h5py.File(file_name, "r") as f:
    prior_matrix = np.column_stack([
        f[f"C01:IMRPhenomXPHM/priors/samples/{p}"][()] for p in params_15
    ])
print(f"Prior matrix shape: {prior_matrix.shape}")  # (5000, 15)


N_bins = 5  # her boyut için

post_hist, edges = np.histogramdd(post_matrix, bins=N_bins, density=True)
prior_hist, _    = np.histogramdd(prior_matrix, bins=edges, density=True)

print(f"Histogram shape: {post_hist.shape}")  # (5, 5, 5, ...) → 5^15 hücre
print(f"Sıfır olmayan posterior bin sayısı: {(post_hist > 0).sum()}")
print(f"Sıfır olmayan prior bin sayısı:     {(prior_hist > 0).sum()}")

# --- KL hesabı ---
# Her hücrenin hacmi
dV = np.prod([e[1] - e[0] for e in edges])

mask = (post_hist > 0) & (prior_hist > 0)
KL_nats = np.sum(post_hist[mask] * np.log(post_hist[mask] / prior_hist[mask])) * dV
KL_bits = KL_nats / np.log(2)

print(f"\nJoint KL (15-dim histogram, N_bins={N_bins}) = {KL_bits:.4f} bit")
print(f"F&H tahmini I_source ≈ 41.50 bit")