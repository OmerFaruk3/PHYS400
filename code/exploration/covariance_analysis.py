import numpy as np
from pesummary.io import read

# Veri yükle
file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
f = read(file_name, disable_conversion=True)
label = f.labels[0]
samples = f.samples_dict[label]

PARAMS = [
    "mass_1_source", "mass_2_source",
    "a_1", "a_2", "tilt_1", "tilt_2",
    "phi_12", "phi_jl",
    "luminosity_distance", "theta_jn",
    "psi", "azimuth", "zenith",
    "geocent_time", "phase"
]

data = np.array([samples[p] for p in PARAMS]).T  # (N, 15)

# Her çift için kovaryans hesapla
print(f"{'Parametre 1':<25} {'Parametre 2':<25} {'Kovaryans':>15}")
print("-" * 70)

for i in range(len(PARAMS)):
    for j in range(i + 1, len(PARAMS)):
        x = data[:, i]
        y = data[:, j]
        cov = np.cov(x, y)[0, 1]
        print(f"{PARAMS[i]:<25} {PARAMS[j]:<25} {cov:>15.6f}")