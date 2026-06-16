"""
RESULT TEST CV — Çapraz doğrulama (model gerçekten sağlam mı?)
==============================================================

Soru: I = sabit + a·ln(SNR) + c·ln(Mtot) + d·ndet modeli "zorlama" mı,
yoksa GERÇEKTEN genelleyen bir ilişki mi?

Tek başına R² aldatıcıdır: modeli fit ettiğin veriyle test edersen iyi
görünür (in-sample). Asıl sınav: modeli GÖRMEDİĞİ olaylarda test etmek
(out-of-sample). İki yöntem:

  1) YARI-YARI (tekrarlı):  olayları rastgele yarıya böl, bir yarıda fit et,
     diğer yarıda tahmin et. Bunu yüzlerce kez tekrarla, ortalama test R²
     ve test MAPE'sini al. (Senin istediğin test.)

  2) LEAVE-ONE-OUT (LOOCV):  her olayı sırayla dışarı çıkar, kalan 33 ile
     fit et, çıkardığın olayı tahmin et. 34 tahminin tamamı "görülmemiş".

Ayrıca SNR-tek-başına modeliyle kıyaslar: kütle+dedektör eklemek
out-of-sample'da da yardım ediyor mu, yoksa sadece fit'i mi şişiriyor?

Sağlık ölçütü: in-sample R² ile out-of-sample R² birbirine yakınsa model
sağlıklı (overfit yok). Out-of-sample çökerse model zorlama/ezberlemedir.

Çıktı: konsol raporu + result_test_cv.png  (LOO tahmin vs gerçek)
Bağımlılıklar: numpy, h5py, matplotlib
"""

import os, csv, numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data"
MASTER = os.path.join(HERE, "oto_master_ozet.csv")
METRIC = "joint_mean_bits"
RANDOM_SPLITS = 1000          # yarı-yarı testin tekrar sayısı
SEED = 42


def med(ps, c):
    return float(np.median(np.asarray(ps[c], float))) if c in (ps.dtype.names or []) else np.nan


def load():
    rows = [r for r in csv.DictReader(open(MASTER, encoding="utf-8")) if r["status"] == "ok"]
    ev, snr, mt, nd, I = [], [], [], [], []
    for r in rows:
        p = os.path.join(DATA_DIR, r["file"])
        with h5py.File(p, "r") as f:
            g = r["analysis_group"]
            if g not in f:
                g = [k for k in f if isinstance(f[k], h5py.Group) and "posterior_samples" in f[k]][0]
            ps = f[g]["posterior_samples"]; nm = ps.dtype.names
            s = med(ps, "network_matched_filter_snr"); M = med(ps, "total_mass_source")
            n = sum(1 for d in ("H1", "L1", "V1")
                    if d + "_optimal_snr" in nm and np.median(np.asarray(ps[d + "_optimal_snr"], float)) > 1)
        ev.append(r["event"]); snr.append(s); mt.append(M); nd.append(n); I.append(float(r[METRIC]))
    return ev, np.array(snr), np.array(mt), np.array(nd, float), np.array(I)


def design(name, snr, mt, nd):
    """İstenen modelin tasarım matrisini kurar (sabit terim dahil)."""
    cols = [np.ones(len(snr)), np.log(snr)]
    if name in ("snr_mass", "full"):
        cols.append(np.log(mt))
    if name == "full":
        cols.append(nd)
    return np.column_stack(cols)


def r2(y, yhat):
    y = np.asarray(y, float)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def mape(y, yhat):
    return float(np.mean(np.abs((yhat - y) / y)) * 100)


def fit_predict(Xtr, ytr, Xte):
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xte @ beta, beta


def loocv(X, y):
    """Her noktayı sırayla dışarıda bırakıp tahmin et."""
    n = len(y); pred = np.empty(n)
    idx = np.arange(n)
    for i in range(n):
        tr = idx != i
        pred[i], _ = fit_predict(X[tr], y[tr], X[i:i + 1])
    return pred


def repeated_half(X, y, reps, seed):
    """Tekrarlı yarı-yarı: rastgele yarıda fit, diğer yarıda test."""
    rng = np.random.default_rng(seed)
    n = len(y); half = n // 2
    r2s, mapes = [], []
    for _ in range(reps):
        perm = rng.permutation(n)
        tr, te = perm[:half], perm[half:]
        yhat, _ = fit_predict(X[tr], y[tr], X[te])
        r2s.append(r2(y[te], yhat)); mapes.append(mape(y[te], yhat))
    return np.array(r2s), np.array(mapes)


def main():
    ev, snr, mt, nd, I = load()
    n = len(I)
    print(f"Olay sayısı: {n}   bilgi ölçüsü: {METRIC}   aralık: {I.min():.1f}–{I.max():.1f} bit\n")

    models = {"snr": "I = a·ln(SNR)",
              "snr_mass": "I = a·ln(SNR) + c·ln(Mtot)",
              "full": "I = a·ln(SNR) + c·ln(Mtot) + d·ndet"}

    print("=" * 86)
    print(f"{'MODEL':40s}{'in-sample':>11}{'LOO R²':>9}{'LOO MAPE':>10}{'yarı R²':>9}{'yarı MAPE':>11}")
    print("-" * 86)
    loo_full = None
    for key, label in models.items():
        X = design(key, snr, mt, nd)
        # in-sample
        beta, *_ = np.linalg.lstsq(X, I, rcond=None)
        r2_in = r2(I, X @ beta)
        # LOO
        pred_loo = loocv(X, I)
        r2_loo, mape_loo = r2(I, pred_loo), mape(I, pred_loo)
        # tekrarlı yarı-yarı
        r2h, mapeh = repeated_half(X, I, RANDOM_SPLITS, SEED)
        print(f"{label:40s}{r2_in:11.3f}{r2_loo:9.3f}{mape_loo:9.1f}%"
              f"{np.median(r2h):9.3f}{np.median(mapeh):10.1f}%")
        if key == "full":
            loo_full = pred_loo

    print("-" * 86)
    print("'in-sample'  = modeli tüm veriyle fit edip yine tüm veride ölçmek (iyimser).")
    print("'LOO'        = her olay, onu görmemiş modelle tahmin edildi (dürüst).")
    print("'yarı'       = 1000 rastgele 17/17 bölmenin medyanı (dürüst).")
    print("Sağlık: in-sample ≈ LOO ise model GERÇEK; LOO çökerse zorlama/ezber.")

    # ---- Grafik: LOO (görülmemiş) tahmin vs gerçek, full model ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(loo_full, I, c=mt, cmap="plasma_r", s=60, edgecolor="k", linewidth=0.3, zorder=3)
    lim = [min(I.min(), loo_full.min()) - 2, max(I.max(), loo_full.max()) + 2]
    ax.plot(lim, lim, "k--", lw=1.2, label="y = x (mükemmel tahmin)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("LOO tahmini (olay modele görünmezken)  [bit]")
    ax.set_ylabel(f"gerçek {METRIC}  [bit]")
    ax.set_title(f"Leave-one-out: görülmemiş tahmin vs gerçek\n"
                 f"out-of-sample R²={r2(I, loo_full):.2f},  MAPE={mape(I, loo_full):.1f}%")
    cb = fig.colorbar(ax.collections[0], ax=ax); cb.set_label("toplam kütle (Msun)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "result_test_cv.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nGrafik kaydedildi: {out}")


if __name__ == "__main__":
    main()
