"""
15D multivariate (analytic) Gaussian KL  KL(posterior || prior), ALL events,
but using the SAME posterior data as the previous (hibrit) analysis:
the identical de-duplication ("Tekilleştirme") step is applied so that
n_posterior matches oto_master_ozet.csv exactly.

Prior = file's original prior samples (same as the 'gaussian kl' method).
Only the posterior is made identical to the previous pipeline.
"""

import os, re, glob, json, csv
from collections import Counter
import numpy as np
import h5py

DATA_DIR = "/sessions/lucid-clever-fermat/mnt/PHYS400/Data"
MASTER_CSV = "/sessions/lucid-clever-fermat/mnt/PHYS400/claude_codes/grup_kld/oto_master_ozet.csv"
G150914_JSON = "/sessions/lucid-clever-fermat/mnt/PHYS400/claude_codes/grup_kld/results_grup_kld_hibrit_GW150914.json"
OUT_DIR = "/sessions/lucid-clever-fermat/mnt/Codes"

PARAMS_15D = [
    "mass_1_source", "mass_2_source",
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
    "luminosity_distance", "theta_jn", "psi",
    "azimuth", "zenith", "geocent_time", "phase",
]

# previous results -------------------------------------------------------
prev = {}
with open(MASTER_CSV) as fh:
    for row in csv.DictReader(fh):
        prev[row["event"]] = row
if os.path.exists(G150914_JSON):
    d = json.load(open(G150914_JSON))
    prev["GW150914"] = {
        "analysis_group": d["analysis_group"],
        "n_posterior": d["n_posterior"],
        "joint_mean_bits": d["joint_kld_estimate_mean_bits"],
        "group_total_mean_bits": d["group_total_mean_bits"],
        "marginal_1d_total_bits": d["marginal_1d_total_bits"],
    }


def prev_lookup(ev):
    return prev.get(ev) or prev.get(ev.split("_")[0], {})


def pick_label(f, preferred):
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


def dedup_posterior(P_raw):
    """Exact replica of the hibrit pipeline's 'Tekilleştirme' step."""
    n0, npar = P_raw.shape
    uniq_counts = [len(np.unique(P_raw[:, j])) for j in range(npar)]
    cnt = Counter(c for c in uniq_counts if c < 0.99 * n0)
    if cnt:
        modal, freq = cnt.most_common(1)[0]
        if freq >= 2:
            key_cols = [j for j in range(npar) if uniq_counts[j] == modal]
            _, keep = np.unique(P_raw[:, key_cols], axis=0, return_index=True)
            keep = np.sort(keep)
            if len(keep) < n0:
                return P_raw[keep], n0 - len(keep)
    return P_raw, 0


def gaussian_kl_bits(P, Q):
    d = P.shape[1]
    mu_p, mu_q = P.mean(0), Q.mean(0)
    Sp, Sq = np.cov(P, rowvar=False), np.cov(Q, rowvar=False)
    inv_Sq = np.linalg.inv(Sq)
    term1 = np.trace(inv_Sq @ Sp)
    diff = mu_q - mu_p
    term2 = diff @ inv_Sq @ diff
    _, ldp = np.linalg.slogdet(Sp)
    _, ldq = np.linalg.slogdet(Sq)
    kl_nat = 0.5 * (term1 + term2 - d + ldq - ldp)
    return float(kl_nat / np.log(2)), float(kl_nat)


def event_name(fn):
    m = re.search(r"(GW\d{6}_\d{6})", fn) or re.search(r"(GW\d{6})", fn)
    return m.group(1) if m else os.path.basename(fn)


files = sorted(f for f in glob.glob(os.path.join(DATA_DIR, "*_cosmo.h5")) if "nocosmo" not in f)

results = []
print(f"{'event':16}{'N_raw':>9}{'N_dedup':>9}{'N_prev':>9}{'KL_bits':>10}{'KL_prevN':>10}")
print("-" * 63)

for fn in files:
    ev = event_name(fn)
    rec = {"event": ev, "file": os.path.basename(fn)}
    try:
        with h5py.File(fn, "r") as f:
            lab = pick_label(f, prev_lookup(ev).get("analysis_group", "C01:IMRPhenomXPHM"))
            ps = f[lab]["posterior_samples"]
            pr = f[lab]["priors"]["samples"]
            P_raw = np.column_stack([np.asarray(ps[p], float) for p in PARAMS_15D])
            Q = np.column_stack([np.asarray(pr[p][:], float) for p in PARAMS_15D])

        n_raw = P_raw.shape[0]
        P, n_removed = dedup_posterior(P_raw)
        kl_bit, kl_nat = gaussian_kl_bits(P, Q)

        prev_n = prev_lookup(ev).get("n_posterior")
        prev_n = int(prev_n) if prev_n not in (None, "") else None
        rec.update(
            status="ok", analysis_group=lab,
            n_posterior_raw=n_raw, n_posterior_dedup=int(P.shape[0]),
            n_removed=int(n_removed), n_prior=int(Q.shape[0]),
            n_posterior_prev=prev_n,
            kl_15D_bits=kl_bit, kl_15D_nats=kl_nat,
            match_prev_N=(prev_n == P.shape[0]),
        )
        print(f"{ev:16}{n_raw:>9}{P.shape[0]:>9}{str(prev_n):>9}{kl_bit:>10.3f}{kl_bit:>10.3f}")
    except Exception as e:
        rec.update(status="error", error=str(e))
        print(f"{ev:16} ERROR {e}")
    results.append(rec)

# attach previous reference bits + delta ---------------------------------
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

json.dump(results, open(os.path.join(OUT_DIR, "gaussian_kl_all_events_matched.json"), "w"), indent=2)

cols = ["event", "status", "n_posterior_raw", "n_posterior_dedup", "n_posterior_prev",
        "match_prev_N", "n_removed", "n_prior", "kl_15D_bits", "kl_15D_nats",
        "prev_joint_mean_bits", "prev_group_total_mean_bits", "prev_marginal_1d_bits",
        "delta_gauss_minus_joint_bits", "error"]
with open(os.path.join(OUT_DIR, "gaussian_kl_all_events_matched.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in results:
        w.writerow(r)

ok = [r for r in results if r.get("status") == "ok"]
nmatch = sum(1 for r in ok if r.get("match_prev_N"))
print("-" * 63)
print(f"Done: {len(ok)}/{len(results)} ok.  N_posterior matches previous: {nmatch}/{len(ok)}")
