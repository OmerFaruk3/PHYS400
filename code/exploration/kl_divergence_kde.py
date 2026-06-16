import numpy as np
from pesummary.io import read
from IPython.display import display as print
import matplotlib.pyplot as plt
import h5py
from scipy.stats import gaussian_kde
from numpy import trapezoid   


# GW150914 - SNR = 26
f = read("/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5", disable_conversion=True)

print(f.labels)

label = f.labels[0]
samples = f.samples_dict[label]

# Posterior
m1_posterior = np.array(samples["mass_1_source"])

# Prior
prior_samples = f.priors["samples"][label]
m1_prior = np.array(prior_samples["mass_1_source"])

print(f"Posterior N: {len(m1_posterior)}")
print(f"Prior N:     {len(m1_prior)}")
print(f"Prior min/max: {m1_prior.min():.2f} / {m1_prior.max():.2f} M☉")

# KDE
kde_posterior = gaussian_kde(m1_posterior, bw_method='scott')
kde_prior     = gaussian_kde(m1_prior,     bw_method='scott')

# Grid — prior aralığını tam kapsayacak şekilde elle ayarladım.
grid_min = max(0.1, m1_prior.min() - 1)
grid_max = m1_prior.max() + 5
m1_grid  = np.linspace(grid_min, grid_max, 5000)

p = kde_posterior(m1_grid)
q = kde_prior(m1_grid)

# Normalization check
dm = m1_grid[1] - m1_grid[0]
print(f"∫p = {np.sum(p)*dm:.6f} ")  #1'e yakın olmalı, çünkü KDE'ler normalize edilmiş PDF'lerdir.
print(f"∫q = {np.sum(q)*dm:.6f} ")  #1'e yakın olmalı, çünkü KDE'ler normalize edilmiş PDF'lerdir.

# KL Divergence ( bilgi kazancı)
mask = (p > 1e-300) & (q > 1e-300)     #Mask, her iki PDF'nin de pratikte sıfır olmadığı noktaları işaretliyor (True,false)
integrand = np.zeros_like(p)           # Sıfırlarla array doldurduk önce, ardından gerekli kısımları ekleyerek log(0) hatalarından kurtul.
integrand[mask] = p[mask] * np.log2(p[mask] / q[mask]) #Sadece (mask=True) integrandı hesaplıyoruz. 0 olan noktalardan kaçmak için.
I_m1 = trapezoid(integrand, m1_grid)    #Trapezoidal rule ile integral alma, p(x) log(p(x)/q(x)) dx

print(f"I(m1) = {I_m1:.4f} bit")

#PLOt

fig, ax = plt.subplots()

# Label'lar için de LaTeX formatı (r'$...$') kullanmak daha şık bir görüntü sağlar
ax.plot(m1_grid, p, label=r'Posterior $p(m_1|d)$', color='blue')
ax.plot(m1_grid, q, label=r'Prior $\pi(m_1)$', color='orange', linestyle='--')

# ÇÖZÜM BURADA: x ekseni etiketini mathtext (LaTeX) ile yazıyoruz
ax.set_xlabel(r'$m_1 \ (M_\odot)$') 
ax.set_ylabel('PDF')

# Title içinde hem f-string hem de LaTeX kullanmak için 'fr' önekini kullanıyoruz
ax.set_title(fr'GW150914 — $m_1$ prior vs posterior   [I = {I_m1:.3f} bit]')

ax.legend()

# plt.show()'dan önce kaydetmek genellikle daha güvenlidir, aksi takdirde boş görsel kaydedilebilir.
plt.savefig("prior_vs_posterior_kde.png", dpi=150, bbox_inches='tight')
print("Kaydedildi: prior_vs_posterior_kde.png")

plt.show()