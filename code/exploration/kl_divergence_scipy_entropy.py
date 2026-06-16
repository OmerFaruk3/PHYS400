import numpy as np
from pesummary.io import read
from IPython.display import display as print
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy

# GW150914 - SNR = 26
f = read("/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5", disable_conversion=True)

print(f.labels)

label = f.labels[0]
samples = f.samples_dict[label]

# Posterior ve prior
m1_posterior = np.array(samples["mass_1_source"])
prior_samples = f.priors["samples"][label]
m1_prior = np.array(prior_samples["mass_1_source"])

print(f"Posterior N: {len(m1_posterior)}")
print(f"Prior N:     {len(m1_prior)}")
print(f"Prior min/max: {m1_prior.min():.2f} / {m1_prior.max():.2f} M☉")

# KDE
kde_posterior = gaussian_kde(m1_posterior, bw_method='scott')
kde_prior     = gaussian_kde(m1_prior,     bw_method='scott')

# Grid — prior aralığını tam kapsayacak şekilde
grid_min = max(0.1, m1_prior.min() - 1)
grid_max = m1_prior.max() + 5
m1_grid  = np.linspace(grid_min, grid_max, 5000)

p = kde_posterior(m1_grid)
q = kde_prior(m1_grid)

# Normalizasyon kontrolü
dm = m1_grid[1] - m1_grid[0]
print(f"∫p = {np.sum(p)*dm:.6f}   (≈1.0 olmali)")
print(f"∫q = {np.sum(q)*dm:.6f}   (≈1.0 olmali)")

# KL Divergence — scipy.stats.entropy ile
# entropy(p, q) = Σ p * ln(p/q)  →  bölü log(2) ile bit'e çevir
I_m1 = entropy(p * dm, q * dm) / np.log(2)   

print(f"\nI(m1) = {I_m1:.4f} bit")

# Plot
fig, ax = plt.subplots()
ax.plot(m1_grid, p, label='Posterior p(m₁|d)', color='blue')
ax.plot(m1_grid, q, label='Prior π(m₁)',        color='orange', linestyle='--')
ax.set_xlabel('m₁ (M☉)')
ax.set_ylabel('PDF')
ax.set_title(f'GW150914 — m₁ prior vs posterior   [I = {I_m1:.3f} bit]')
ax.legend()
plt.show()

# plt.savefig("prior_vs_posterior_kde.png", dpi=150, bbox_inches='tight')
# print("Kaydedildi: prior_vs_posterior_kde.png")