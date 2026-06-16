from IPython.display import display as print
import matplotlib.pyplot as plt
import pycbc
import pesummary
from pesummary.io import read
import h5py
import numpy as np
from gwosc.datasets import find_datasets

#file_name = "Data/IGWN-GWTC3p0-v2-GW200224_222234_PEDataRelease_mixed_cosmo.h5"

file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"



# data = read(file_name, disable_conversion=True)
# samples_dict = data.samples_dict
# print("Sample keys:")
# print(samples_dict.keys())


# prior_samples = data.priors["samples"]["C01:Mixed"]
# parameters = prior_samples.parameters   
# prior_mass_1_source = prior_samples["mass_1_source"]

# print(prior_samples.parameters)
# # print(len(prior_samples.parameters))

# mass_1_source = prior_samples["mass_1_source"]
# print(mass_1_source.shape)  

# fig1 = prior_samples.plot("mass_1_source", type="hist", kde=True)
# plt.savefig("Prior_mass_1_source_hist.png", dpi=150, bbox_inches='tight')
# print("✓ Plot kaydedildi: Prior_mass_1_source_hist.png")

data = read(file_name, disable_conversion=True)
print("Run labels:")
print(data.labels)

samples_dict = data.samples_dict

print("Sample keys:")
print(samples_dict.keys())

prior_samples = data.priors["samples"]["C01:IMRPhenomXPHM"]

print("Prior parameters:")
print(prior_samples.parameters)
print(f"Prior parametre sayisi: {len(prior_samples.parameters)}")


print(prior_samples["mass_1_source"])
print(f'mass_1_source veri uzunluğu: {len(prior_samples["mass_1_source"])}')

# prior_chirp_mass = prior_samples["chirp_mass_source"]

# parameters = prior_samples.parameters
# print(parameters)
# print(len(parameters))



# print(prior_samples["chirp_mass_source"])
# print(len(prior_samples["chirp_mass_source"]))


fig1 = prior_samples.plot("mass_1_source", type="hist", kde=True)
plt.savefig("Prior_mass_1_source_hist.png", dpi=150, bbox_inches='tight')
print("Plot kaydedildi: Prior_mass_1_source_hist.png")
