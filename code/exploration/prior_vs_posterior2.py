from IPython.display import display as print
import matplotlib.pyplot as plt
import pycbc
import pesummary
from pesummary.io import read
import h5py
import numpy as np
from gwosc.datasets import find_datasets


# file_name = "IGWN-GWTC3p0-v2-GW200224_222234_PEDataRelease_mixed_cosmo.h5"

file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"

data = read(file_name, disable_conversion=True)


prior_samples = data.priors["samples"]["C01:IMRPhenomXPHM"]

samples_dict = data.samples_dict
posterior_samples = samples_dict["C01:IMRPhenomXPHM"]

# Prior örneklerini al (sadece IMRPhenomXPHM'de var)
prior_samples = data.priors["samples"]["C01:IMRPhenomXPHM"]


# Prior histogramı da çiz — posterior ile yan yana karşılaştırma
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(posterior_samples["mass_2_source"], bins=50, density=True, alpha=0.7, label="Posterior")
axes[0].set_xlabel("mass_2_source")
axes[0].set_title("Posterior")
axes[1].hist(prior_samples["mass_2_source"], bins=50, density=True, alpha=0.7, color="orange", label="Prior")
axes[1].set_xlabel("mass_2_source")
axes[1].set_title("Prior")
plt.tight_layout()
plt.savefig("prior_vs_posterior.png", dpi=150, bbox_inches='tight')
print("✓ Kaydedildi: prior_vs_posterior.png")