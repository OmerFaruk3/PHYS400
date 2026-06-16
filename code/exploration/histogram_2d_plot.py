"""
m1_source – m2_source 2D Histogram Grafigi
============================================
Posterior ve prior orneklerinin 2D histogram yogunluk haritasini
ve marjinal dagilimlarini yan yana gosterir.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from pesummary.io import read

# ── AYARLAR ──────────────────────────────────────────────────────────────
FILE = (
    "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/"
    "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
)
LABEL  = "C01:IMRPhenomXPHM"
PARAM1 = "mass_1_source"
PARAM2 = "mass_2_source"
BINS   = 60       # histogram kutu sayisi (her eksen)
SAVE   = False     # True ise dosyaya kaydeder
OUT    = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Codes/m1_m2_2d_histogram.png"

# ── VERİ YÜKLEME ─────────────────────────────────────────────────────────
print("Veri yukleniyor...")
data  = read(FILE, disable_conversion=True)
post  = data.samples_dict[LABEL]
prior = data.priors["samples"][LABEL]

m1_post  = np.array(post[PARAM1])
m2_post  = np.array(post[PARAM2])
m1_prior = np.array(prior[PARAM1])
m2_prior = np.array(prior[PARAM2])

print(f"Posterior ornekleri : {len(m1_post):,}")
print(f"Prior ornekleri     : {len(m1_prior):,}")

# ── ORTAK IZGARA SINIRI ───────────────────────────────────────────────────
# Her iki dagilimi da kapsayan ortak eksen siniri
x_min = min(m1_post.min(), m1_prior.min())
x_max = max(m1_post.max(), m1_prior.max())
y_min = min(m2_post.min(), m2_prior.min())
y_max = max(m2_post.max(), m2_prior.max())

x_edges = np.linspace(x_min, x_max, BINS + 1)
y_edges = np.linspace(y_min, y_max, BINS + 1)
x_mid   = 0.5 * (x_edges[:-1] + x_edges[1:])
y_mid   = 0.5 * (y_edges[:-1] + y_edges[1:])

# ── 2D HİSTOGRAMLAR ──────────────────────────────────────────────────────
H_post,  _, _ = np.histogram2d(m1_post,  m2_post,  bins=[x_edges, y_edges])
H_prior, _, _ = np.histogram2d(m1_prior, m2_prior, bins=[x_edges, y_edges])

# Normalize et (yogunluk: toplam alan = 1)
dx = x_edges[1] - x_edges[0]
dy = y_edges[1] - y_edges[0]
H_post_norm  = H_post  / (H_post.sum()  * dx * dy)
H_prior_norm = H_prior / (H_prior.sum() * dx * dy)

# Sifir hucreleri log olcek icin kucuk bir deger yap
eps = 1e-10
H_post_plot  = np.where(H_post_norm  > 0, H_post_norm,  eps)
H_prior_plot = np.where(H_prior_norm > 0, H_prior_norm, eps)

# ── KONTUR SEVİYELERİ (Posterior icin %50 ve %90 CI) ────────────────────
def credible_levels(H, levels=[0.50, 0.90]):
    """Belirtilen credible interval seviyelerine karsili gelen histogram esik degerlerini dondurur."""
    h_flat = np.sort(H.ravel())[::-1]          # buyukten kucuge sirala
    cumsum  = np.cumsum(h_flat) / h_flat.sum()  # kumulatif toplam
    thresholds = []
    for lv in levels:
        idx = np.searchsorted(cumsum, lv)
        thresholds.append(h_flat[min(idx, len(h_flat) - 1)])
    return thresholds

post_levels  = credible_levels(H_post_norm,  [0.50, 0.90])
prior_levels = credible_levels(H_prior_norm, [0.50, 0.90])

# ── ÇİZİM ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
fig.suptitle(
    "GW150914  |  $m_1^{\\rm src}$ – $m_2^{\\rm src}$  2D Histogram\n"
    f"IMRPhenomXPHM  |  Posterior: {len(m1_post):,} örneklem  "
    f"|  Prior: {len(m1_prior):,} örneklem",
    fontsize=12, y=1.01
)

gs = gridspec.GridSpec(
    2, 4,
    width_ratios=[3, 1, 3, 1],
    height_ratios=[1, 3],
    hspace=0.05, wspace=0.05
)

CMAP_POST  = "Blues"
CMAP_PRIOR = "Oranges"

# ════ SOL PANEL: POSTERİOR ════════════════════════════════════════════════
ax_post_main = fig.add_subplot(gs[1, 0])
ax_post_top  = fig.add_subplot(gs[0, 0], sharex=ax_post_main)
ax_post_rgt  = fig.add_subplot(gs[1, 1], sharey=ax_post_main)

# 2D histogram (log olcek)
im_post = ax_post_main.pcolormesh(
    x_edges, y_edges, H_post_plot.T,
    cmap=CMAP_POST, norm=LogNorm(vmin=H_post_norm[H_post_norm > 0].min(),
                                  vmax=H_post_norm.max()),
    rasterized=True
)
# Kontur: %50 ve %90 CI
ax_post_main.contour(
    x_mid, y_mid, H_post_norm.T,
    levels=post_levels[::-1],
    colors=["white", "white"],
    linewidths=[1.8, 0.9],
    linestyles=["solid", "dashed"]
)
# Kitlesel cizgi (m2 <= m1 siniri)
diag = np.linspace(x_min, x_max, 200)
ax_post_main.plot(diag, diag, "k--", lw=0.8, alpha=0.5, label="$m_2 = m_1$")
ax_post_main.set_xlabel(r"$m_1^{\rm src}\ [M_\odot]$", fontsize=11)
ax_post_main.set_ylabel(r"$m_2^{\rm src}\ [M_\odot]$", fontsize=11)
ax_post_main.set_title("Posterior", fontsize=11, pad=4)
ax_post_main.set_xlim(x_min, x_max)
ax_post_main.set_ylim(y_min, y_max)
ax_post_main.legend(fontsize=8, loc="upper left")

# Renk cubugu
cbar_post = plt.colorbar(im_post, ax=ax_post_rgt, pad=0.02, fraction=0.8)
cbar_post.set_label("Olasılik yogunlugu", fontsize=8)

# Marjinal ust (m1)
ax_post_top.fill_between(x_mid, H_post_norm.sum(axis=1) * dy,
                          alpha=0.5, color="steelblue", step="mid")
ax_post_top.step(x_mid, H_post_norm.sum(axis=1) * dy,
                 color="steelblue", lw=1.2, where="mid")
ax_post_top.set_ylabel("p(m₁)", fontsize=9)
ax_post_top.tick_params(labelbottom=False)
ax_post_top.set_xlim(x_min, x_max)

# Marjinal sag (m2)
ax_post_rgt.fill_betweenx(y_mid, H_post_norm.sum(axis=0) * dx,
                           alpha=0.5, color="steelblue", step="mid")
ax_post_rgt.step(H_post_norm.sum(axis=0) * dx, y_mid,
                 color="steelblue", lw=1.2, where="mid")
ax_post_rgt.set_xlabel("p(m₂)", fontsize=9)
ax_post_rgt.tick_params(labelleft=False)
ax_post_rgt.set_ylim(y_min, y_max)

# ════ SAĞ PANEL: PRİOR ════════════════════════════════════════════════════
ax_prior_main = fig.add_subplot(gs[1, 2])
ax_prior_top  = fig.add_subplot(gs[0, 2], sharex=ax_prior_main)
ax_prior_rgt  = fig.add_subplot(gs[1, 3], sharey=ax_prior_main)

im_prior = ax_prior_main.pcolormesh(
    x_edges, y_edges, H_prior_plot.T,
    cmap=CMAP_PRIOR, norm=LogNorm(vmin=H_prior_norm[H_prior_norm > 0].min(),
                                   vmax=H_prior_norm.max()),
    rasterized=True
)
ax_prior_main.contour(
    x_mid, y_mid, H_prior_norm.T,
    levels=prior_levels[::-1],
    colors=["white", "white"],
    linewidths=[1.8, 0.9],
    linestyles=["solid", "dashed"]
)
ax_prior_main.plot(diag, diag, "k--", lw=0.8, alpha=0.5)
ax_prior_main.set_xlabel(r"$m_1^{\rm src}\ [M_\odot]$", fontsize=11)
ax_prior_main.set_ylabel(r"$m_2^{\rm src}\ [M_\odot]$", fontsize=11)
ax_prior_main.set_title("Prior", fontsize=11, pad=4)
ax_prior_main.set_xlim(x_min, x_max)
ax_prior_main.set_ylim(y_min, y_max)

cbar_prior = plt.colorbar(im_prior, ax=ax_prior_rgt, pad=0.02, fraction=0.8)
cbar_prior.set_label("Olasılik yogunlugu", fontsize=8)

# Marjinal ust (m1)
ax_prior_top.fill_between(x_mid, H_prior_norm.sum(axis=1) * dy,
                           alpha=0.5, color="darkorange", step="mid")
ax_prior_top.step(x_mid, H_prior_norm.sum(axis=1) * dy,
                  color="darkorange", lw=1.2, where="mid")
ax_prior_top.set_ylabel("p(m₁)", fontsize=9)
ax_prior_top.tick_params(labelbottom=False)
ax_prior_top.set_xlim(x_min, x_max)

# Marjinal sag (m2)
ax_prior_rgt.fill_betweenx(y_mid, H_prior_norm.sum(axis=0) * dx,
                            alpha=0.5, color="darkorange", step="mid")
ax_prior_rgt.step(H_prior_norm.sum(axis=0) * dx, y_mid,
                  color="darkorange", lw=1.2, where="mid")
ax_prior_rgt.set_xlabel("p(m₂)", fontsize=9)
ax_prior_rgt.tick_params(labelleft=False)
ax_prior_rgt.set_ylim(y_min, y_max)

plt.tight_layout()

# ── KAYDET / GÖSTER ──────────────────────────────────────────────────────
if SAVE:
    plt.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"\nGrafik kaydedildi: {OUT}")
else:
    plt.show()

# ── İSTATİSTİKSEL ÖZET ───────────────────────────────────────────────────
print(f"\n{'─'*55}")
print("  ISTATISTIKSEL OZET")
print(f"{'─'*55}")
print(f"  Posterior  m1 : {m1_post.mean():.2f} ± {m1_post.std():.2f} Msun  "
      f"  [MAP ~{x_mid[H_post_norm.sum(axis=1).argmax()]:.1f}]")
print(f"  Posterior  m2 : {m2_post.mean():.2f} ± {m2_post.std():.2f} Msun  "
      f"  [MAP ~{y_mid[H_post_norm.sum(axis=0).argmax()]:.1f}]")
print(f"  Prior      m1 : {m1_prior.mean():.2f} ± {m1_prior.std():.2f} Msun")
print(f"  Prior      m2 : {m2_prior.mean():.2f} ± {m2_prior.std():.2f} Msun")
print(f"  Posterior Pearson r(m1,m2) : {np.corrcoef(m1_post, m2_post)[0,1]:.4f}")
print(f"  Prior     Pearson r(m1,m2) : {np.corrcoef(m1_prior, m2_prior)[0,1]:.4f}")
print(f"{'─'*55}")
print("  Kontur cizgileri: beyaz duz = %%50 CI, beyaz kesik = %%90 CI")
print(f"{'─'*55}")