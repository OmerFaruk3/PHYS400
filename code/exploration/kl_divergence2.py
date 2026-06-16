import numpy as np
from pesummary.io import read
from IPython.display import display as print
import matplotlib.pyplot as plt
import h5py
from scipy.stats import gaussian_kde
from numpy import trapezoid   


# GW150914 - SNR = 26
# f = read("Data/IGWN-GWTC3p0-v2-GW200224_222234_PEDataRelease_mixed_cosmo.h5", disable_conversion=True)
f = read("/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5", disable_conversion=True)


print(f.labels)

label = f.labels[0]
samples = f.samples_dict[label]

# Posterior
m2_posterior = np.array(samples["mass_2_source"])

# Prior
prior_samples = f.priors["samples"][label]
m2_prior = np.array(prior_samples["mass_2_source"])

print(f"Posterior N: {len(m2_posterior)}")
print(f"Prior N:     {len(m2_prior)}")
print(f"Prior min/max: {m2_prior.min():.2f} / {m2_prior.max():.2f} M☉")

# KDE
kde_posterior = gaussian_kde(m2_posterior, bw_method='scott')
kde_prior     = gaussian_kde(m2_prior,     bw_method='scott')

# Grid — prior aralığını tam kapsayacak şekilde elle ayarladım.
grid_min = max(0.1, m2_prior.min() - 1)
grid_max = m2_prior.max() + 5
m2_grid  = np.linspace(grid_min, grid_max, 5000)

p = kde_posterior(m2_grid)
q = kde_prior(m2_grid)

# Normalization check
dm = m2_grid[1] - m2_grid[0]
print(f"∫p = {np.sum(p)*dm:.6f} ")  #1'e yakın olmalı, çünkü KDE'ler normalize edilmiş PDF'lerdir.
print(f"∫q = {np.sum(q)*dm:.6f} ")  #1'e yakın olmalı, çünkü KDE'ler normalize edilmiş PDF'lerdir.

# KL Divergence ( bilgi kazancı)
mask = (p > 1e-300) & (q > 1e-300)     #Mask, her iki PDF'nin de pratikte sıfır olmadığı noktaları işaretliyor (True,false)
integrand = np.zeros_like(p)           # Sıfırlarla array doldurduk önce, ardından gerekli kısımları ekleyerek log(0) hatalarından kurtul.
integrand[mask] = p[mask] * np.log2(p[mask] / q[mask]) #Sadece (mask=True) integrandı hesaplıyoruz. 0 olan noktalardan kaçmak için.
I_m2 = trapezoid(integrand, m2_grid)    #Trapezoidal rule ile integral alma, p(x) log(p(x)/q(x)) dx

print(f"I(m2) = {I_m2:.4f} bit")


# Plot
fig, ax = plt.subplots()
ax.plot(m2_grid, p, label='Posterior p(m₂|d)', color='blue')
ax.plot(m2_grid, q, label='Prior π(m₂)',        color='orange', linestyle='--')
ax.set_xlabel('m₂ (M☉)')
ax.set_ylabel('PDF')
ax.set_title(f'GW150914 — m₂ prior vs posterior   [I = {I_m2:.3f} bit]')
ax.legend()
plt.show()