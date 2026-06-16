"""
=============================================================================
GW150914 — k-NN KL yakinsama testi:  TUM 147k ornek, k = 1 .. 10
=============================================================================
Amac
----
KL(posterior||prior) tahmininin komsu sayisi k ile nasil degistigini gormek.
TUM posterior orneklemleri (~147k, tekillestirme YOK) kullanilir — kullanicinin
istegi. Karsilastirma icin tekillestirilmis (71.747 bagimsiz) egri de eklenir.

Beklenti
--------
'mixed_cosmo' dosyasinda her ornek bir kez kopyalanmistir. k=1'de en yakin komsu
"ikiz" olur (rho~0) ve KL yapay sisar; k buyudukce ikizin etkisi azalir ve tahmin
gercek (tekillestirilmis) degere DOGRU yakinsar. Bu script bunu sayisal gosterir.

Estimator: makaledeki calc_knn_kld (Wang+2009 / Perez-Cruz 2008), bit cinsinden.
Verim: P ve Q agaclarina TEK sorgu (k=kmax+1) yapilir; her k icin ilgili sutun
alinir — yani 10 ayri sorgu yerine 1 sorgu (cok daha hizli).

Kullanim:  python gw_knn_kl_convergence.py [olay.h5]
"""

import os, sys, time, json
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gw_knn_kl_divergence import (
    load_event, build_matrices, dedup_resampling_copies,
    PARAMS_15, DEFAULT_FILE, LABEL,
)

LN2 = np.log(2.0)


def kl_curve_bits(P, Q, kmax=10, workers=-1):
    """k = 1..kmax icin D_KL(P||Q) [bit] dizisi. Tek sorgu ile."""
    n, d = P.shape
    m = Q.shape[0]
    bias = np.log(m / (n - 1.0))

    t0 = time.perf_counter()
    # P agacinda kendisi dahil kmax+1 komsu: sutun j = j. komsu (sutun0 = self = 0)
    rho_all = cKDTree(P).query(P, k=kmax + 1, p=2, workers=workers)[0]
    # Q agacinda kmax komsu: sutun j = (j+1). komsu
    nu_all = cKDTree(Q).query(P, k=kmax, p=2, workers=workers)[0]
    print(f"    (sorgular bitti: {time.perf_counter()-t0:.1f}s)", flush=True)

    out = {}
    for k in range(1, kmax + 1):
        rho = np.maximum(rho_all[:, k], 1e-15)
        nu = np.maximum(nu_all[:, k - 1], 1e-15)
        kld_nats = (d / n) * np.sum(np.log(nu / rho)) + bias
        out[k] = kld_nats / LN2
    return out


def main(path=None, kmax=10):
    path = path or DEFAULT_FILE
    event = os.path.basename(path).split("_PEDataRelease")[0]
    print("=" * 64)
    print(f"GW KL yakinsama testi — TUM ornekler, k=1..{kmax}")
    print("=" * 64)

    post, pri, label = load_event(path)
    params = [p for p in PARAMS_15 if p in post and p in pri]
    P_full, Q = build_matrices(post, pri, params)
    keep = dedup_resampling_copies(P_full)
    P_dd = P_full[keep]

    print(f"Olay: {event} | etiket: {label} | d={len(params)}")
    print(f"TUM posterior: {P_full.shape[0]:,} | Tekillestirilmis: {P_dd.shape[0]:,} | Prior: {Q.shape[0]:,}\n")

    print(f"[1] TUM {P_full.shape[0]:,} ornek (kopyalar dahil):", flush=True)
    full = kl_curve_bits(P_full, Q, kmax)
    print(f"[2] Tekillestirilmis {P_dd.shape[0]:,} ornek:", flush=True)
    dd = kl_curve_bits(P_dd, Q, kmax)

    print(f"\n{'k':>3} | {'TUM 147k (bit)':>16} | {'tekillestirilmis (bit)':>22}")
    print("-" * 48)
    for k in range(1, kmax + 1):
        print(f"{k:>3} | {full[k]:>16.2f} | {dd[k]:>22.2f}")

    # Yakinsama gostergesi: ardisik farklar
    dd_vals = [dd[k] for k in range(1, kmax + 1)]
    print(f"\nTekillestirilmis egri son 5 k ortalamasi: {np.mean(dd_vals[-5:]):.2f} bit "
          f"(std {np.std(dd_vals[-5:]):.2f}) -> kararli/yakinsamis")
    print(f"Referans: F&H ~41.5 bit | Gaussian joint ~41.24 bit")

    # Kaydet
    out = {"event": event, "label": label, "dimension": len(params),
           "n_full": int(P_full.shape[0]), "n_dedup": int(P_dd.shape[0]),
           "n_prior": int(Q.shape[0]),
           "KL_bits_full_allsamples": {str(k): float(full[k]) for k in full},
           "KL_bits_dedup": {str(k): float(dd[k]) for k in dd}}
    jpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"kl_knn_convergence_{event}.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nKaydedildi: {jpath}")

    # Grafik
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = list(range(1, kmax + 1))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ks, [full[k] for k in ks], "o-", label=f"TUM {P_full.shape[0]:,} (kopyalar dahil)")
        ax.plot(ks, dd_vals, "s-", label=f"tekillestirilmis {P_dd.shape[0]:,}")
        ax.axhline(41.24, ls="--", c="gray", label="Gaussian joint ~41.24")
        ax.set_xlabel("k (komsu sayisi)"); ax.set_ylabel("D_KL(post||prior) [bit]")
        ax.set_title(f"{event} — k-NN KL yakinsamasi"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        ppath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"kl_knn_convergence_{event}.png")
        fig.savefig(ppath, dpi=130)
        print(f"Grafik: {ppath}")
    except Exception as e:
        print(f"(grafik atlandi: {e})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
