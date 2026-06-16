import numpy as np
from pesummary.io import read
from IPython.display import display as print
import matplotlib.pyplot as plt
import pycbc
import pesummary
import h5py


# f = read("/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5")
# f = read("/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191103_012549_PEDataRelease_mixed_cosmo.h5")
f = read("/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191105_143521_PEDataRelease_mixed_cosmo.h5")
label = f.labels[0]
samples = f.samples_dict[label]

# Direkt SNR kolonunu çek
snr = samples["network_matched_filter_snr"]
print(snr)  # array döner — posterior sample başına bir SNR değeri

print(f"Median SNR = {np.median(snr):.2f}")
print(f"Mean SNR   = {np.mean(snr):.2f}")