import numpy as np
from pesummary.io import read
from IPython.display import display as print
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# GW150914 - SNR = 26
f = read("/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5", disable_conversion=True)

print(f.labels)

label = f.labels[0]
samples = f.samples_dict[label]

# Posterior ve prior
m1_posterior = np.array(samples["mass_1_source"])
prior_samples = f.priors["samples"][label]
m1_prior = np.array(prior_samples["mass_1_source"])

print(f"Posterior N (orijinal): {len(m1_posterior)}")
print(f"Prior N:                {len(m1_prior)}")
print(f"Prior min/max:          {m1_prior.min():.2f} / {m1_prior.max():.2f} M☉")

# Subsample — KDE için 5000 yeterli, 147K çok ağır
N_sub = 5000
np.random.seed(42)   # tekrarlanabilirlik için
idx = np.random.choice(len(m1_posterior), size=N_sub, replace=False)
m1_posterior_sub = m1_posterior[idx]

print(f"Posterior N (subsample): {len(m1_posterior_sub)}")

# KDE
kde_posterior = gaussian_kde(m1_posterior_sub, bw_method='scott')
kde_prior     = gaussian_kde(m1_prior,         bw_method='scott')

# Monte Carlo KL Divergence
# D_KL = E_{x~p}[ log2(p(x)/q(x)) ] ≈ (1/N) Σ log2(p(x_i)/q(x_i))
p_vals = kde_posterior(m1_posterior_sub)
q_vals = kde_prior(m1_posterior_sub)

mask = (p_vals > 1e-300) & (q_vals > 1e-300)
I_m1_mc = np.mean(np.log2(p_vals[mask] / q_vals[mask]))

print(f"\nI(m1) Monte Carlo = {I_m1_mc:.4f} bit")

# Plot
grid_min = max(0.1, m1_prior.min() - 1)
grid_max = m1_prior.max() + 5
m1_grid  = np.linspace(grid_min, grid_max, 2000)

fig, ax = plt.subplots()
ax.plot(m1_grid, kde_posterior(m1_grid), label='Posterior p(m₁|d)', color='blue')
ax.plot(m1_grid, kde_prior(m1_grid),     label='Prior π(m₁)',        color='orange', linestyle='--')
ax.set_xlabel('m₁ (M☉)')
ax.set_ylabel('PDF')
ax.set_title(f'GW150914 — m₁   [I = {I_m1_mc:.3f} bit]')
ax.legend()
plt.savefig("prior_vs_posterior_mc.png", dpi=150, bbox_inches='tight')
print("Kaydedildi: prior_vs_posterior_mc.png")