"""
15D Multivariate (analytic) Gaussian KL divergence  KL(posterior || prior)
for ALL GWTC events.

Same method as the single-event "gaussian kl" script:
    KL = 0.5 * [ tr(Sq^-1 Sp) + (mu_q-mu_p)^T Sq^-1 (mu_q-mu_p) - d + ln(detSq/detSp) ]
computed in bits, using ALL posterior samples and the file's prior samples.

Reads the .h5 PE data releases directly with h5py (no pesummary needed).
Compares against previously found results in oto_master_ozet.csv.
"""

import os, re, glob, json
import numpy as np
import h5py
import csv

DATA_DIR = "/sessions/lucid-clever-fermat/mnt/PHYS400/Data"
MASTER_CSV = "/sessions/lucid-clever-fermat/mnt/PHYS400/claude_codes/grup_kld/oto_master_ozet.csv"
OUT_DIR = "/sessions/lucid-clever-fermat/mnt/Codes"

PARAMS_15D = [
    "mass_1_source", "mass_2_source",
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
    "luminosity_distance", "theta_jn", "psi",
    "azimuth", "zenith", "geocent_time", "phase",
]

# ── previously found results (analysis group + reference bits) ────────────
prev = {}
with open(MASTER_CSV) as fh:
    for row in csv.DictReader(fh):
        ev = row["event"]
        prev[ev] = row
# GW150914 was stored separately (its own hibrit json)
g150914_json = "/sessions/lucid-clever-fermat/mnt/PHYS400/claude_codes/grup_kld/results_grup_kld_hibrit_GW150914.json"
if os.path.exists(g150914_json):
    d = json.load(open(g150914_json))
    prev["GW150914"] = {
        "analysis_group": d["analysis_group"],
        "joint_mean_bits": d["joint_kld_estimate_mean_bits"],
        "group_total_mean_bits": d["group_total_mean_bits"],
        "marginal_1d_total_bits": d["marginal_1d_total_bits"],
    }


def pick_label(f, preferred):
    """Choose the analysis group label inside the h5 file."""
    keys = list(f.keys())
    if preferred and preferred in keys:
        return preferred
    for k in keys:
        if k.startswith("C01:IMRPhenomXPHM"):
            return k
    for k in keys:
        if k.startswith("C01:") and k != "C01:Mixed":
            return k
    return "C01:Mixed" if "C01:Mixed" in keys else keys[0]


def gaussian_kl_bits(P, Q):
    """Analytic multivariate-Gaussian KL(P||Q) in bits."""
    d = P.shape[1]
    mu_p = np.mean(P, axis=0)
    mu_q = np.mean(Q, axis=0)
    Sp = np.cov(P, rowvar=False)
    Sq = np.cov(Q, rowvar=False)
    inv_Sq = np.linalg.inv(Sq)

    term1 = np.trace(inv_Sq @ Sp)
    diff = mu_q - mu_p
    term2 = diff @ inv_Sq @ diff
    term3 = -d
    _, logdet_p = np.linalg.slogdet(Sp)
    _, logdet_q = np.linalg.slogdet(Sq)
    term4 = logdet_q - logdet_p

    kl_nat = 0.5 * (term1 + term2 + term3 + term4)
    kl_bit = kl_nat / np.log(2)
    diag = {
        "term1_trace": float(term1),
        "term2_mahal": float(term2),
        "term4_logdet_ratio": float(term4),
        "cond_prior": float(np.linalg.cond(Sq)),
        "cond_post": float(np.linalg.cond(Sp)),
    }
    return float(kl_bit), float(kl_nat), diag


def event_name(fn):
    m = re.search(r"(GW\d{6}_\d{6})", fn) or re.search(r"(GW\d{6})", fn)
    return m.group(1) if m else os.path.basename(fn)


def prev_lookup(ev):
    if ev in prev:
        return prev[ev]
    short = ev.split("_")[0]
    return prev.get(short, {})


files = sorted(glob.glob(os.path.join(DATA_DIR, "*_cosmo.h5")))
# drop nocosmo duplicates (none match *_cosmo.h5 anyway, but be safe)
files = [f for f in files if "nocosmo" not in f]

results = []
print(f"{'event':16} {'Npost':>8} {'Nprior':>7} {'KL_15D_bits':>12} {'condQ':>10}")
print("-" * 60)

for fn in files:
    ev = event_name(fn)
    rec = {"event": ev, "file": os.path.basename(fn)}
    try:
        with h5py.File(fn, "r") as f:
            preferred = prev_lookup(ev).get("analysis_group", "C01:IMRPhenomXPHM")
            lab = pick_label(f, preferred)
            ps = f[lab]["posterior_samples"]
            pr = f[lab]["priors"]["samples"]

            post_names = set(ps.dtype.names)
            pr_names = set(pr.keys())
            missing = [p for p in PARAMS_15D if p not in post_names or p not in pr_names]
            if missing:
                rec.update(status="skip", error=f"missing params: {missing}")
                results.append(rec)
                print(f"{ev:16} SKIP missing {missing}")
                continue

            P = np.column_stack([np.asarray(ps[p], float) for p in PARAMS_15D])
            Q = np.column_stack([np.asarray(pr[p][:], float) for p in PARAMS_15D])

        kl_bit, kl_nat, diag = gaussian_kl_bits(P, Q)
        rec.update(
            status="ok", analysis_group=lab,
            n_posterior=int(P.shape[0]), n_prior=int(Q.shape[0]),
            kl_15D_bits=kl_bit, kl_15D_nats=kl_nat, **diag,
        )
        print(f"{ev:16} {P.shape[0]:>8} {Q.shape[0]:>7} {kl_bit:>12.3f} {diag['cond_prior']:>10.2e}")
    except Exception as e:
        rec.update(status="error", error=str(e))
        print(f"{ev:16} ERROR {e}")
    results.append(rec)

# ── attach previous reference values and deltas ──────────────────────────
def asf(x):
    try: return float(x)
    except (TypeError, ValueError): return None

for r in results:
    p = prev_lookup(r["event"])
    r["prev_joint_mean_bits"] = asf(p.get("joint_mean_bits"))
    r["prev_group_total_mean_bits"] = asf(p.get("group_total_mean_bits"))
    r["prev_marginal_1d_bits"] = asf(p.get("marginal_1d_total_bits"))
    if r.get("status") == "ok" and r["prev_joint_mean_bits"] is not None:
        r["delta_gauss_minus_joint_bits"] = r["kl_15D_bits"] - r["prev_joint_mean_bits"]

with open(os.path.join(OUT_DIR, "gaussian_kl_all_events.json"), "w") as fh:
    json.dump(results, fh, indent=2)

cols = ["event", "status", "n_posterior", "n_prior", "kl_15D_bits", "kl_15D_nats",
        "prev_joint_mean_bits", "prev_group_total_mean_bits", "prev_marginal_1d_bits",
        "delta_gauss_minus_joint_bits", "cond_prior", "cond_post", "error"]
with open(os.path.join(OUT_DIR, "gaussian_kl_all_events.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in results:
        w.writerow(r)

ok = [r for r in results if r.get("status") == "ok"]
print("-" * 60)
print(f"Done: {len(ok)}/{len(results)} events computed.")
print("Saved gaussian_kl_all_events.json / .csv")
