import numpy as np
from pesummary.io import read
from scipy.stats import gaussian_kde
from numpy import trapezoid
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. VERİ YÜKLEME ─────────────────────────────────────────────────────
# file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191103_012549_PEDataRelease_mixed_cosmo.h5"
# file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191105_143521_PEDataRelease_mixed_cosmo.h5"

# Event adını dosya adından çıkart (GW150914, GW191103, vs.)
import re
match = re.search(r'(GW\d{6})', file_name)
event_name = match.group(1) if match else "Unknown"

f = read(file_name, disable_conversion=True)
label         = f.labels[0]
samples       = f.samples_dict[label]
prior_samples = f.priors["samples"][label]

# ── 2. 15 PARAMETRE VE GÖSTERİM İSİMLERİ ───────────────────────────────
# Plotlarda düzgün görünmesi için Matplotlib'in LaTeX (matematik) gösterimleri eklendi.
parameters_15D = [
    ("mass_1_source",      r"$m_1\ [M_\odot]$"),
    ("mass_2_source",      r"$m_2\ [M_\odot]$"),
    ("a_1",                r"$a_1$ (spin mag.)"),
    ("a_2",                r"$a_2$ (spin mag.)"),
    ("tilt_1",             r"tilt$_1$ [rad]"),
    ("tilt_2",             r"tilt$_2$ [rad]"),
    ("phi_12",             r"$\phi_{12}$ [rad]"),
    ("phi_jl",             r"$\phi_{JL}$ [rad]"),
    ("luminosity_distance",r"$d_L$ [Mpc]"),
    ("theta_jn",           r"$\theta_{JN}$ [rad]"),
    ("psi",                r"$\psi$ [rad]"),
    ("azimuth",            r"azimuth [rad]"),
    ("zenith",             r"zenith [rad]"),
    ("geocent_time",       r"$t_c$ [s]"),
    ("phase",              r"phase [rad]"),
]

# ── 3. TEK PARAMETRE İÇİN KL HESABI (aynı yöntem, döngüye alındı) ───────
def compute_marginal_kl(param_name, posterior_samples, prior_samples,
                        n_grid=5000, bw='scott'):
    post = np.array(posterior_samples[param_name])
    pri  = np.array(prior_samples[param_name])

    # ──  padding = veri aralığının %5'i ──────────────────────
    combined_min = min(pri.min(), post.min())
    combined_max = max(pri.max(), post.max())
    data_range   = combined_max - combined_min
    padding      = 0.05 * data_range if data_range > 0 else 1e-6

    grid_min = combined_min - padding
    grid_max = combined_max + padding
    # ──────────────────────────────────────────────────────────────────
    grid = np.linspace(grid_min, grid_max, n_grid)
    dm   = grid[1] - grid[0]

    kde_post = gaussian_kde(post, bw_method=bw)
    kde_pri  = gaussian_kde(pri,  bw_method=bw)

    p = kde_post(grid)
    q = kde_pri(grid)
    

    norm_p = np.sum(p) * dm
    norm_q = np.sum(q) * dm

    mask      = (p > 1e-300) & (q > 1e-300)
    integrand = np.zeros_like(p)
    integrand[mask] = p[mask] * np.log2(p[mask] / q[mask])
    kl = trapezoid(integrand, grid)

    return kl, p, q, grid, norm_p, norm_q

# ── 4. TÜM PARAMETRELER İÇİN DÖNGÜ ─────────────────────────────────────
print("=" * 65)
print(f"{'Parametre':<25} {'KL (bit)':>10}  {'∫p':>8}  {'∫q':>8}")
print("=" * 65)

results = {}
toplam  = 0.0

for param, label_str in parameters_15D:
    kl, p, q, grid, norm_p, norm_q = compute_marginal_kl(
        param, samples, prior_samples
    )
    results[param] = {
        "kl": kl, "p": p, "q": q,
        "grid": grid, "label": label_str
    }
    toplam += kl


print("=" * 65)
print(f"{'MARJİNAL TOPLAMI':<25} {toplam:>10.4f}  bit")


# ── 5. PLOT: 5×3 GRID, HER PARAMETRE ────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.5, wspace=0.35)

for idx, (param, label_str) in enumerate(parameters_15D):
    row = idx // 5
    col = idx  % 5
    ax  = fig.add_subplot(gs[row, col])

    r   = results[param]
    ax.plot(r["grid"], r["p"], color='steelblue',  lw=1.5, label='posterior')
    ax.plot(r["grid"], r["q"], color='darkorange',
            lw=1.5, linestyle='--', label='prior')

    ax.set_title(f"{label_str}\nI = {r['kl']:.3f} bit", fontsize=8)
    ax.set_xlabel(label_str, fontsize=7)
    ax.set_ylabel("PDF", fontsize=7)
    ax.tick_params(labelsize=6)

    # Sadece ilk plot için legend
    if idx == 0:
        ax.legend(fontsize=6)

fig.suptitle(
    f"{event_name} — Marginal KL Divergence for 15 Parameters\n"
    f"Total (marginal) = {toplam:.3f} bit ",
    fontsize=12, fontweight='bold'
)
plt.savefig("marginal_kl_15params.png", dpi=150, bbox_inches='tight')
print("\nPlot kaydedildi: marginal_kl_15params.png")