from IPython.display import display as print
import matplotlib.pyplot as plt

import pesummary
from pesummary.io import read
import h5py
import numpy as np
from gwosc.datasets import find_datasets

"""""""""""""""""""""
from gwosc.datasets import find_datasets
events = find_datasets(type='event', match="GW")
print("Some available events are:")
print(len(events))
"""""""""""""""""""""""

from pesummary.gw.fetch import fetch_open_samples
#fetch_open_samples("GW200224_222234", unpack=False, read_file=False, delete_on_exit=False, outdir="./", verbose=True)


#h5py ile file açma.

#file_name = "Data/IGWN-GWTC3p0-v2-GW200224_222234_PEDataRelease_mixed_cosmo.h5"

file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"

# with h5py.File(file_name, "r") as f:
#     print("H5 data sets:")
#     print(list(f))

data = read(file_name, disable_conversion=True)
print("Run labels:")
print(data.labels)

samples_dict = data.samples_dict
posterior_samples = samples_dict["C01:IMRPhenomXPHM"]


parameters = posterior_samples.parameters

print("Posterior parameters:")
print(posterior_samples.parameters)
print(f"Posterior parametre sayisi: {len(posterior_samples.parameters)}")

# for parameter in parameters:
#     print(f"The definition of {parameter} is: {parameter.description}")

print(posterior_samples["mass_1_source"])
print(f"Posterior mass_1_source veri uzunluğu: {len(posterior_samples['mass_1_source'])}")


fig1 = posterior_samples.plot("mass_1_source", type="hist", kde=True)
plt.savefig("Posterior_mass_1_source_hist.png", dpi=150, bbox_inches='tight')
print("Plot kaydedildi: Posterior_mass_1_source_hist.png")

# fig2 = posterior_samples.plot(type="corner",
#                              parameters=["mass_1",
#                                          "mass_2",
#                                          "iota",
#                                          "luminosity_distance"])

# plt.savefig("corner_plot.png", dpi=150, bbox_inches='tight')
# print(" Plot kaydedildi: corner_plot.png")


# color_palette = ["#1b9e77", "#d95f02", "#8D7BC5"]

# compared_analyses = ["C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM", "C01:Mixed"]
# fig = samples_dict.plot("chirp_mass",
#                         type="hist",
#                         kde=True,
#                         labels=compared_analyses,
#                         colors=color_palette)
# plt.savefig("chirp_mass_comparison_hist.png", dpi=150, bbox_inches='tight')
# print(" Plot kaydedildi: chirp_mass_comparison_hist.png")

