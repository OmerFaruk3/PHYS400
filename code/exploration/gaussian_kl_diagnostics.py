"""
WHY does the analytic 15D Gaussian KL differ from the previous (KDE/kNN group)
estimates?  Per-parameter & per-event diagnostics.

For every event we measure, parameter by parameter:
  * posterior non-Gaussianity   -> skewness, excess kurtosis
  * 1D Gaussian-KL vs 1D KDE-KL -> how wrong the Gaussian shape assumption is
  * prior support overflow       -> fraction of posterior outside the stored
                                    prior's [min,max]  (the thing the hibrit
                                    pipeline "fixed" with an analytic prior)
Event level:
  * Gaussian 15D KL  vs  sum of 1D Gaussian KL   -> Gaussian correlation term
  * Gaussian 15D KL  vs  previous joint estimate -> the delta we want to explain
Outputs JSON + a readable TXT report + figures.
"""
import os, re, glob, json, csv
import numpy as np
import h5py
from scipy.stats import skew, kurtosis, gaussian_kde
from numpy import trapezoid
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA_DIR = "/sessions/lucid-clever-fermat/mnt/PHYS400/Data"
MASTER_CSV = "/sessions/lucid-clever-fermat/mnt/PHYS400/claude_codes/grup_kld/oto_master_ozet.csv"
G150914 = "/sessions/lucid-clever-fermat/mnt/PHYS400/claude_codes/grup_kld/results_grup_kld_hibrit_GW150914.json"
OUT = "/sessions/lucid-clever-fermat/mnt/Codes"

PAR = ["mass_1_source","mass_2_source","a_1","a_2","tilt_1","tilt_2","phi_12",
       "phi_jl","luminosity_distance","theta_jn","psi","azimuth","zenith",
       "geocent_time","phase"]
SHORT = ["m1","m2","a1","a2","t1","t2","phi12","phiJL","dL","theta_jn","psi","az","zen","tc","phase"]

prev = {}
for row in csv.DictReader(open(MASTER_CSV)):
    prev[row["event"]] = row
d = json.load(open(G150914))
prev["GW150914"] = {"joint_mean_bits": d["joint_kld_estimate_mean_bits"],
                    "marginal_1d_total_bits": d["marginal_1d_total_bits"]}

def plk(ev): return prev.get(ev) or prev.get(ev.split("_")[0], {})

def gauss1d_kl_bits(p, q):
    mp, mq, sp, sq = p.mean(), q.mean(), p.std(), q.std()
    sp = max(sp, 1e-300); sq = max(sq, 1e-300)
    kl = 0.5*((sp**2 + (mp-mq)**2)/sq**2 - 1.0 + np.log(sq**2/sp**2))
    return kl/np.log(2)

_RNG = np.random.default_rng(0)
def kde1d_kl_bits(p, q, n=700, cap=3000):
    # KDE shape estimate; subsample posterior for speed (cov/mean use full data elsewhere)
    if p.size > cap: p = p[_RNG.choice(p.size, cap, replace=False)]
    if q.size > cap: q = q[_RNG.choice(q.size, cap, replace=False)]
    lo = min(p.min(), q.min()); hi = max(p.max(), q.max())
    pad = 0.05*(hi-lo if hi>lo else 1.0)
    grid = np.linspace(lo-pad, hi+pad, n)
    fp = gaussian_kde(p)(grid); fq = gaussian_kde(q)(grid)
    m = (fp>1e-300)&(fq>1e-300)
    integ = np.zeros_like(fp); integ[m] = fp[m]*np.log2(fp[m]/fq[m])
    return float(trapezoid(integ, grid))

def gauss15d_kl_bits(P, Q):
    d = P.shape[1]; mp, mq = P.mean(0), Q.mean(0)
    Sp, Sq = np.cov(P, rowvar=False), np.cov(Q, rowvar=False)
    iSq = np.linalg.inv(Sq)
    kl = 0.5*(np.trace(iSq@Sp) + (mq-mp)@iSq@(mq-mp) - d
              + np.linalg.slogdet(Sq)[1] - np.linalg.slogdet(Sp)[1])
    return float(kl/np.log(2))

def pick_label(f, pref):
    ks=list(f.keys())
    if pref in ks: return pref
    for k in ks:
        if k.startswith("C01:IMRPhenomXPHM"): return k
    return "C01:Mixed" if "C01:Mixed" in ks else ks[0]

def evname(fn):
    m=re.search(r"(GW\d{6}_\d{6})",fn); return m.group(1) if m else fn

files = sorted(f for f in glob.glob(os.path.join(DATA_DIR,"*_cosmo.h5")) if "nocosmo" not in f)

rows = []
# accumulators per-parameter across events
acc = {k: {"skew":[], "kurt":[], "g1d":[], "k1d":[], "overflow":[]} for k in PAR}

for fn in files:
    ev = evname(fn)
    with h5py.File(fn,"r") as f:
        lab = pick_label(f, plk(ev).get("analysis_group","C01:IMRPhenomXPHM"))
        ps = f[lab]["posterior_samples"]; pr = f[lab]["priors"]["samples"]
        P = np.column_stack([np.asarray(ps[p],float) for p in PAR])
        Q = np.column_stack([np.asarray(pr[p][:],float) for p in PAR])

    per = {}
    sum_g1d = sum_k1d = 0.0
    for i,p in enumerate(PAR):
        pc, qc = P[:,i], Q[:,i]
        sk = float(skew(pc)); ku = float(kurtosis(pc))   # excess kurtosis
        g1 = float(gauss1d_kl_bits(pc, qc)); k1 = float(kde1d_kl_bits(pc, qc))
        ovf = float(np.mean((pc < qc.min()) | (pc > qc.max())))*100.0  # % outside prior support
        per[p] = {"skew":sk,"kurt":ku,"gauss1d_bits":g1,"kde1d_bits":k1,
                  "kde_minus_gauss_1d":k1-g1,"overflow_pct":ovf}
        sum_g1d += g1; sum_k1d += k1
        for key,val in [("skew",sk),("kurt",ku),("g1d",g1),("k1d",k1),("overflow",ovf)]:
            acc[p][key].append(val)

    g15 = gauss15d_kl_bits(P, Q)
    pj = plk(ev).get("joint_mean_bits"); pj = float(pj) if pj not in (None,"") else None
    pm = plk(ev).get("marginal_1d_total_bits"); pm = float(pm) if pm not in (None,"") else None
    rows.append({"event":ev,"gauss15d_bits":g15,"sum_gauss1d_bits":sum_g1d,
                 "sum_kde1d_bits":sum_k1d,"gauss_corr_term_bits":g15-sum_g1d,
                 "prev_joint_bits":pj,"prev_marg1d_bits":pm,
                 "delta_g15_minus_joint":(g15-pj) if pj else None,
                 "mean_abs_skew":float(np.mean(np.abs([per[p]["skew"] for p in PAR]))),
                 "mean_abs_kurt":float(np.mean(np.abs([per[p]["kurt"] for p in PAR]))),
                 "params":per})
    print(f"{ev:16} g15={g15:6.2f}  Sg1d={sum_g1d:6.2f}  Sk1d={sum_k1d:6.2f}  corr={g15-sum_g1d:+6.2f}  d(joint)={(g15-pj) if pj else float('nan'):+6.2f}")

json.dump(rows, open(os.path.join(OUT,"gaussian_kl_diagnostics.json"),"w"), indent=2)

# ---------- per-parameter aggregate ----------
print("\n================ PER-PARAMETER AGGREGATE (mean over 35 events) ================")
print(f"{'param':20}{'|skew|':>8}{'exkurt':>8}{'gauss1d':>9}{'kde1d':>8}{'kde-g':>8}{'ovf%':>7}")
agg = {}
for p,s in zip(PAR,SHORT):
    a=acc[p]
    agg[p]={"abs_skew":float(np.mean(np.abs(a["skew"]))),
            "exkurt":float(np.mean(a["kurt"])),
            "gauss1d":float(np.mean(a["g1d"])),"kde1d":float(np.mean(a["k1d"])),
            "kde_minus_gauss":float(np.mean(np.array(a["k1d"])-np.array(a["g1d"]))),
            "overflow":float(np.mean(a["overflow"]))}
    print(f"{s:20}{agg[p]['abs_skew']:>8.2f}{agg[p]['exkurt']:>8.2f}{agg[p]['gauss1d']:>9.2f}{agg[p]['kde1d']:>8.2f}{agg[p]['kde_minus_gauss']:>8.2f}{agg[p]['overflow']:>7.1f}")
json.dump(agg, open(os.path.join(OUT,"gaussian_kl_param_aggregate.json"),"w"), indent=2)

# ---------- figures ----------
fig,axs=plt.subplots(2,2,figsize=(16,11))
x=np.arange(len(PAR))
axs[0,0].bar(x,[agg[p]["abs_skew"] for p in PAR],color='indianred')
axs[0,0].set_title("Mean |skewness| of posterior (non-Gaussianity)"); axs[0,0].set_xticks(x); axs[0,0].set_xticklabels(SHORT,rotation=90)
axs[0,1].bar(x,[agg[p]["exkurt"] for p in PAR],color='darkorange')
axs[0,1].axhline(0,color='k',lw=.6); axs[0,1].set_title("Mean excess kurtosis (0 = Gaussian)"); axs[0,1].set_xticks(x); axs[0,1].set_xticklabels(SHORT,rotation=90)
axs[1,0].bar(x,[agg[p]["kde_minus_gauss"] for p in PAR],color='steelblue')
axs[1,0].axhline(0,color='k',lw=.6); axs[1,0].set_title("Mean (KDE 1D KL  -  Gaussian 1D KL)  [bit]\n>0: Gaussian UNDER-estimates info"); axs[1,0].set_xticks(x); axs[1,0].set_xticklabels(SHORT,rotation=90)
axs[1,1].bar(x,[agg[p]["overflow"] for p in PAR],color='seagreen')
axs[1,1].set_title("Mean % posterior outside stored-prior support\n(prior-definition confound)"); axs[1,1].set_xticks(x); axs[1,1].set_xticklabels(SHORT,rotation=90)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"gaussian_kl_diagnostics.png"),dpi=150,bbox_inches='tight')

# event-level scatter: what explains delta(g15 - joint)?
ok=[r for r in rows if r["delta_g15_minus_joint"] is not None]
dd=np.array([r["delta_g15_minus_joint"] for r in ok])
nong=np.array([r["sum_kde1d_bits"]-r["sum_gauss1d_bits"] for r in ok])  # total 1D non-Gauss correction
fig2,ax=plt.subplots(figsize=(8,7))
ax.scatter(nong,dd,c='purple',s=55)
for r,xx,yy in zip(ok,nong,dd): ax.annotate(r["event"].split("_")[0],(xx,yy),fontsize=6,alpha=.7)
ax.axhline(0,color='k',lw=.6)
ax.set_xlabel("Sum_1D (KDE - Gaussian)  [bit]  = total 1D non-Gaussianity")
ax.set_ylabel("delta = Gauss15D - previous joint  [bit]")
ax.set_title(f"Does 1D non-Gaussianity explain the gap?  r={np.corrcoef(nong,dd)[0,1]:.3f}")
ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"gaussian_kl_delta_explained.png"),dpi=150,bbox_inches='tight')

print("\nGlobal sums (mean over events):")
print(f"  sum 1D Gaussian = {np.mean([r['sum_gauss1d_bits'] for r in rows]):.2f} bit")
print(f"  sum 1D KDE      = {np.mean([r['sum_kde1d_bits'] for r in rows]):.2f} bit")
print(f"  Gaussian 15D    = {np.mean([r['gauss15d_bits'] for r in rows]):.2f} bit")
print(f"  Gaussian corr term (15D - sum1D) = {np.mean([r['gauss_corr_term_bits'] for r in rows]):+.2f} bit")
print("Saved diagnostics json/png.")
