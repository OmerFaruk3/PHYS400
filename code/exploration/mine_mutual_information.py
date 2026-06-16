"""
MINE: Mutual Information Neural Estimation
GW150914 — 15 Parametre Joint KL Divergence Hesabı
====================================================

Kaynak: Belghazi et al. (2018). "MINE: Mutual Information Neural Estimation."
        ICML 2018, PMLR 80. arXiv:1801.04062

Temel Matematik (Makale Theorem 1 — Donsker-Varadhan):
    KL(P || Q) = sup_{T: Omega -> R}  E_P[T] - log(E_Q[e^T])
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")

# ─── PARAMETRELER ─────────────────────────────────────────────────────────────
parameters_15D = [
    "chirp_mass_source", "mass_ratio",
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
    "luminosity_distance", "theta_jn", "psi",
    "azimuth", "zenith", "geocent_time", "phase",
]

prior_bounds = {
    "chirp_mass_source":   (10.0,  100.0),
    "mass_ratio":          (0.05,  1.0),
    "a_1":                 (0.0,   1.0),
    "a_2":                 (0.0,   1.0),
    "tilt_1":              (0.0,   np.pi),
    "tilt_2":              (0.0,   np.pi),
    "phi_12":              (0.0,   2*np.pi),
    "phi_jl":              (0.0,   2*np.pi),
    "luminosity_distance": (10.0,  2000.0),
    "theta_jn":            (0.0,   np.pi),
    "psi":                 (0.0,   np.pi),
    "azimuth":             (0.0,   2*np.pi),
    "zenith":              (0.0,   np.pi),
    "geocent_time":        (1126259462.3, 1126259462.5),
    "phase":               (0.0,   2*np.pi),
}

def generate_prior_samples(n, seed=42):
    rng = np.random.default_rng(seed)
    s = {}
    s["chirp_mass_source"]   = rng.uniform(10, 100, n)
    s["mass_ratio"]          = rng.uniform(0.05, 1.0, n)
    s["a_1"]                 = rng.uniform(0, 1, n)
    s["a_2"]                 = rng.uniform(0, 1, n)
    s["tilt_1"]              = np.arccos(1 - 2*rng.uniform(0,1,n))
    s["tilt_2"]              = np.arccos(1 - 2*rng.uniform(0,1,n))
    s["phi_12"]              = rng.uniform(0, 2*np.pi, n)
    s["phi_jl"]              = rng.uniform(0, 2*np.pi, n)
    d_min, d_max = 10.0, 2000.0
    s["luminosity_distance"] = ((d_max**3 - d_min**3)*rng.uniform(0,1,n) + d_min**3)**(1/3)
    s["theta_jn"]            = np.arccos(1 - 2*rng.uniform(0,1,n))
    s["psi"]                 = rng.uniform(0, np.pi, n)
    s["azimuth"]             = rng.uniform(0, 2*np.pi, n)
    s["zenith"]              = np.arccos(1 - 2*rng.uniform(0,1,n))
    s["geocent_time"]        = rng.uniform(1126259462.3, 1126259462.5, n)
    s["phase"]               = rng.uniform(0, 2*np.pi, n)
    return s

# ─── VERİ YÜKLEME ─────────────────────────────────────────────────────────────
print("Veri yükleniyor...")
try:
    from pesummary.io import read
    FILE = ("/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/"
            "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5")
    f = read(FILE, disable_conversion=True)
    label = f.labels[0]
    samples = f.samples_dict[label]
    post_cols = [np.array(samples[p]) for p in parameters_15D]
    DATA_REAL = True
    print(f"  Posterior: {len(post_cols[0])} ornek")
except Exception as e:
    print(f"  Gercek veri yok ({type(e).__name__}), sentetik kullaniliyor.")
    DATA_REAL = False
    rng0 = np.random.default_rng(0)
    post_cols = []
    for p in parameters_15D:
        lo, hi = prior_bounds[p]
        center = lo + 0.35*(hi-lo)
        width  = 0.05*(hi-lo)
        post_cols.append(rng0.normal(center, width, 50000))

N_PRIOR = 200_000
prior_synth = generate_prior_samples(N_PRIOR, seed=7)
print(f"  Prior: {N_PRIOR} ornek")

# ─── STANDARDIZASYON ──────────────────────────────────────────────────────────
P_cols, Q_cols = [], []
for i, param in enumerate(parameters_15D):
    q_val = np.array(prior_synth[param], dtype=np.float32)
    p_val = np.array(post_cols[i], dtype=np.float32)
    ref_mean = float(np.mean(q_val))
    ref_std  = max(float(np.std(q_val)), 1e-12)
    P_cols.append((p_val - ref_mean) / ref_std)
    Q_cols.append((q_val - ref_mean) / ref_std)

P_matrix = np.column_stack(P_cols).astype(np.float32)
Q_matrix = np.column_stack(Q_cols).astype(np.float32)

N_SUB = 50_000
rng_main = np.random.default_rng(42)
P_sub = P_matrix[rng_main.choice(len(P_matrix), N_SUB, replace=False)]
Q_sub = Q_matrix[rng_main.choice(len(Q_matrix), N_SUB, replace=False)]
print(f"  P alt-ornek: {P_sub.shape}, Q alt-ornek: {Q_sub.shape}")

# ─── ISTATISTIK AGI ───────────────────────────────────────────────────────────
class StatisticsNetwork(nn.Module):
    """
    T_theta: R^d -> R
    Donsker-Varadhan temsili icin istatistik fonksiyonu.
    """
    def __init__(self, d_in, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ─── MINE LOSS (BIAS DUZELTMELI) ──────────────────────────────────────────────
def mine_loss_ema(T_p, T_q, ema_log_mean_et, alpha=0.01):
    """
    Makale Bolum 3.2 — EMA ile bias duzeltmeli MINE loss.

    KL(P||Q) ~= E_P[T] - log(E_Q[e^T])

    Gradient bias: mini-batch'te log(E[e^T]) gradyani biased.
    Duzeltme: paydayi EMA ile tahmin et, loss'ta detach et.
    """
    mean_T_p = T_p.mean()

    # log-sum-exp trick: exp overflow'u onle
    t_max = T_q.max().detach()
    log_mean_et = t_max + torch.log(torch.exp(T_q - t_max).mean())

    # EMA guncelle
    new_ema = (1 - alpha) * ema_log_mean_et + alpha * log_mean_et.item()

    # KL tahmini (anlık, izleme icin)
    kl_nats = (mean_T_p - log_mean_et).item()

    # Bias-corrected loss (makale Eq. 12 sonrasi)
    loss = -(mean_T_p - torch.exp(log_mean_et - new_ema).detach() * log_mean_et)

    return loss, new_ema, kl_nats

# ─── EGITIM DONGUSU ───────────────────────────────────────────────────────────
def train_mine(P_data, Q_data, n_epochs=400, batch_size=512,
               lr=3e-4, hidden_dim=256, alpha_ema=0.01, verbose=True):
    d = P_data.shape[1]
    P_t = torch.FloatTensor(P_data).to(device)
    Q_t = torch.FloatTensor(Q_data).to(device)

    net = StatisticsNetwork(d, hidden_dim).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.1)

    kl_history = []
    ema = 0.0

    if verbose:
        print()
        print("=" * 58)
        print("MINE EGITIMI")
        print(f"  d={d}, hidden={hidden_dim}, epochs={n_epochs}, batch={batch_size}")
        print("=" * 58)

    for epoch in range(n_epochs):
        net.train()
        idx_p = torch.randint(0, len(P_t), (batch_size,), device=device)
        idx_q = torch.randint(0, len(Q_t), (batch_size,), device=device)

        T_p = net(P_t[idx_p])
        T_q = net(Q_t[idx_q])

        loss, ema, kl_nats = mine_loss_ema(T_p, T_q, ema, alpha=alpha_ema)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        sch.step()

        kl_history.append(kl_nats)

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            kl_bits = kl_nats / np.log(2)
            recent = np.mean(kl_history[-20:]) / np.log(2) if len(kl_history) >= 20 else kl_bits
            print(f"  Epoch {epoch:>4}/{n_epochs}  "
                  f"KL={kl_bits:7.3f} bit  "
                  f"ort-20={recent:7.3f} bit")

    return np.array(kl_history), net

# ─── ANA HESAPLAMA ────────────────────────────────────────────────────────────
kl_history, T_net = train_mine(
    P_sub, Q_sub,
    n_epochs=500, batch_size=512,
    lr=3e-4, hidden_dim=256,
    alpha_ema=0.01, verbose=True
)

kl_nats_final = np.mean(kl_history[-50:])
kl_bits_final = kl_nats_final / np.log(2)
kl_std_bits   = np.std(kl_history[-50:]) / np.log(2)

print()
print("=" * 58)
print("SONUCLAR")
print("=" * 58)
print(f"  MINE Joint KL:       {kl_bits_final:.3f} +/- {kl_std_bits:.3f} bit")
print(f"  Marjinal toplam:     36.05 bit  (alt sinir)")
print(f"  F&H analitik:        ~41.5 bit")
print(f"  TC tahmini (MINE-marjinal): {kl_bits_final - 36.05:.3f} bit")
print("=" * 58)

# ─── STABILITE TESTI ──────────────────────────────────────────────────────────
print("\nStabilite testi:")
stability = {}
for n_test in [5_000, 10_000, 20_000, 50_000]:
    rng2 = np.random.default_rng(n_test)
    Pt = P_matrix[rng2.choice(len(P_matrix), n_test, replace=False)]
    Qt = Q_matrix[rng2.choice(len(Q_matrix), n_test, replace=False)]
    hist_t, _ = train_mine(Pt, Qt, n_epochs=200, batch_size=256,
                           lr=3e-4, hidden_dim=128, verbose=False)
    m = np.mean(hist_t[-30:]) / np.log(2)
    s = np.std(hist_t[-30:])  / np.log(2)
    stability[n_test] = (m, s)
    print(f"  N={n_test:>6}  KL={m:.3f} +/- {s:.3f} bit")

# ─── PLOT ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Egitim egrisi
ax1 = fig.add_subplot(gs[0, :])
kl_bits_arr = np.array(kl_history) / np.log(2)
ax1.plot(kl_bits_arr, color='steelblue', alpha=0.25, lw=0.8, label='Ham KL')
window = 20
smooth = np.convolve(kl_bits_arr, np.ones(window)/window, mode='valid')
ax1.plot(np.arange(window-1, len(kl_bits_arr)), smooth,
         color='steelblue', lw=2, label=f'{window}-epoch hareketli ort.')
ax1.axhline(41.5,  color='red',    ls='--', lw=1.5, label="F&H ~41.5 bit")
ax1.axhline(36.05, color='orange', ls='--', lw=1.5, label="Marjinal 36.05 bit")
ax1.axhline(kl_bits_final, color='green', ls='-', lw=2,
            label=f"MINE final = {kl_bits_final:.2f} bit")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("KL Divergence (bit)", fontsize=11)
ax1.set_title("MINE Egitim Egrisi — GW150914 15D Joint KL", fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Stabilite karsilastirma
ax2 = fig.add_subplot(gs[1, 0])
ns   = list(stability.keys())
vals = [stability[n][0] for n in ns]
errs = [stability[n][1] for n in ns]
ax2.errorbar(ns, vals, yerr=errs, fmt='o-', color='steelblue',
             capsize=5, lw=2, markersize=7, label='MINE')
ax2.axhline(41.5,  color='red',    ls='--', lw=1.2, label="F&H")
ax2.axhline(36.05, color='orange', ls='--', lw=1.2, label="Marjinal")
knn_vals = {5000: 25.92, 10000: 30.99, 20000: 37.62, 50000: 54.87}
ax2.plot(list(knn_vals.keys()), list(knn_vals.values()),
         's--', color='tomato', lw=1.5, markersize=7, label='k-NN (onceki)')
ax2.set_xscale('log')
ax2.set_xlabel("N", fontsize=10)
ax2.set_ylabel("KL (bit)", fontsize=10)
ax2.set_title("Stabilite: MINE vs k-NN", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Yontem ozeti
ax3 = fig.add_subplot(gs[1, 1])
yontemler = ['Gaussian\nJoint', 'Marjinal\nToplam', 'MINE\nJoint', 'F&H\nAnalitik']
degerler  = [34.96, 36.05, kl_bits_final, 41.5]
renkler   = ['tomato', 'orange', 'steelblue', 'green']
bars = ax3.bar(yontemler, degerler, color=renkler, alpha=0.8, edgecolor='k', lw=0.8)
ax3.errorbar([2], [kl_bits_final], yerr=[kl_std_bits],
             fmt='none', color='navy', capsize=6, lw=2)
for bar, val in zip(bars, degerler):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.3,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.set_ylabel("KL (bit)", fontsize=10)
ax3.set_title("Yontem Karsilastirmasi", fontsize=10)
ax3.set_ylim(0, 50)
ax3.grid(axis='y', alpha=0.3)

fig.suptitle(
    f"MINE — GW150914 15D Joint KL\n"
    f"Final: {kl_bits_final:.2f} +/- {kl_std_bits:.2f} bit  |  "
    f"F&H ~41.5 bit  |  TC = {kl_bits_final - 36.05:.2f} bit",
    fontsize=12, fontweight='bold'
)

outfile = "/mnt/user-data/outputs/mine_kl_gw150914.png"
plt.plot(outfile, dpi=150, bbox_inches='tight')
print(f"\nPlot kaydedildi: {outfile}")
plt.close()