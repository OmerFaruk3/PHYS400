"""
RESULT TEST EXTENDED — modeli fiziksel değişkenlerle genişlet (dürüst LOO testi)
================================================================================

Soru: SNR + kütle dışındaki fiziksel değişkenler (chirp kütlesi, simetrik
kütle-oranı η, etkin spin χ_eff, kırmızıya kayma z) bilgi tahminini
GERÇEKTEN iyileştiriyor mu, yoksa sadece fit'i mi şişiriyor?

Yöntem — her şey OUT-OF-SAMPLE (leave-one-out):
  * Nested modeller kuruyoruz, her birinin LOO R² ve LOO MAPE'sini ölçüyoruz.
    In-sample R² ile yan yana koyuyoruz (fark büyükse overfit).
  * chirp kütlesi vs toplam kütle baş başa kıyas (teori chirp'i öne koyar).
  * NULL TESTİ (bilimsel dürüstlük): bir değişken eklemenin LOO kazancını,
    yerine RASTGELE (çöp) bir değişken eklemenin kazancıyla kıyaslıyoruz.
    Gerçek değişkenin kazancı, çöp dağılımının içinde kalıyorsa -> kazanç
    şanstan ayırt edilemez, o değişkeni REDDEDİYORUZ.

Bilimsel uyarılar kodda ve raporda. Sadece OKUR.
Çıktı: konsol raporu + result_test_extended.png
Bağımlılıklar: numpy, h5py, matplotlib
"""

import os, csv, numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data"
MASTER = os.path.join(HERE, "oto_master_ozet.csv")
METRIC = "joint_mean_bits"
SEED = 7
N_NULL = 2000          # rastgele değişken null testi tekrar sayısı


def med(ps, c):
    return float(np.median(np.asarray(ps[c], float))) if c in (ps.dtype.names or []) else np.nan


def load():
    rows = [r for r in csv.DictReader(open(MASTER, encoding="utf-8")) if r["status"] == "ok"]
    d = {k: [] for k in ["ev", "snr", "mtot", "mc", "eta", "chieff", "chip", "z", "nd", "I"]}
    for r in rows:
        with h5py.File(os.path.join(DATA_DIR, r["file"]), "r") as f:
            g = r["analysis_group"]
            if g not in f:
                g = [k for k in f if isinstance(f[k], h5py.Group) and "posterior_samples" in f[k]][0]
            ps = f[g]["posterior_samples"]; nm = ps.dtype.names
            n = sum(1 for dd in ("H1", "L1", "V1")
                    if dd + "_optimal_snr" in nm and np.median(np.asarray(ps[dd + "_optimal_snr"], float)) > 1)
        d["ev"].append(r["event"])
        with h5py.File(os.path.join(DATA_DIR, r["file"]), "r") as f:
            ps = f[g]["posterior_samples"]
            d["snr"].append(med(ps, "network_matched_filter_snr"))
            d["mtot"].append(med(ps, "total_mass_source"))
            d["mc"].append(med(ps, "chirp_mass_source"))
            d["eta"].append(med(ps, "symmetric_mass_ratio"))
            d["chieff"].append(med(ps, "chi_eff"))
            d["chip"].append(med(ps, "chi_p"))
            d["z"].append(med(ps, "redshift"))
        d["nd"].append(n)
        d["I"].append(float(r[METRIC]))
    return {k: np.array(v, float) if k != "ev" else v for k, v in d.items()}


def r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


def mape(y, yh):
    return float(np.mean(np.abs((yh - y) / y)) * 100)


def loo(cols, y):
    """cols: liste of 1D diziler (sabit terim otomatik). LOO tahminleri döndürür."""
    X = np.column_stack([np.ones(len(y))] + cols)
    n = len(y); pred = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[i] = X[i] @ beta
    return pred


def insample_r2(cols, y):
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return r2(y, X @ beta)


def main():
    d = load()
    y = d["I"]; n = len(y)
    lnS = np.log(d["snr"]); lnMt = np.log(d["mtot"]); lnMc = np.log(d["mc"])
    eta = d["eta"]; chie = d["chieff"]; chip = d["chip"]; z = d["z"]; nd = d["nd"]

    print(f"Olay: {n}   bilgi: {METRIC}  ({y.min():.1f}-{y.max():.1f} bit)\n")

    # 0) chirp vs toplam kütle baş başa
    print("=" * 78)
    print("ADIM 1 — chirp kütlesi mi, toplam kütle mi? (teori: chirp daha temel)")
    print("=" * 78)
    for name, cols in [("ln(SNR)+ln(Mtot)", [lnS, lnMt]),
                       ("ln(SNR)+ln(Mc)  ", [lnS, lnMc])]:
        p = loo(cols, y)
        print(f"  {name}   in-R²={insample_r2(cols,y):.3f}   LOO R²={r2(y,p):.3f}   LOO MAPE={mape(y,p):.1f}%")

    # 1) nested ekleme (chirp tabanlı)
    print("\n" + "=" * 78)
    print("ADIM 2 — değişken EKLEDİKÇE out-of-sample iyileşiyor mu? (chirp tabanlı)")
    print("=" * 78)
    print(f"{'model':46s}{'in-R²':>8}{'LOO R²':>9}{'LOO MAPE':>10}")
    steps = [
        ("ln(SNR)",                                   [lnS]),
        ("+ ln(Mc)",                                  [lnS, lnMc]),
        ("+ ndet",                                    [lnS, lnMc, nd]),
        ("+ eta (simetrik kütle-oranı)",              [lnS, lnMc, nd, eta]),
        ("+ chi_eff (etkin spin)",                    [lnS, lnMc, nd, eta, chie]),
        ("+ chi_p (presesyon)",                       [lnS, lnMc, nd, eta, chie, chip]),
        ("+ z (kırmızıya kayma) [her şey]",           [lnS, lnMc, nd, eta, chie, chip, z]),
    ]
    loo_r2s = []
    for name, cols in steps:
        p = loo(cols, y); rr = r2(y, p); loo_r2s.append(rr)
        print(f"{name:46s}{insample_r2(cols,y):8.3f}{rr:9.3f}{mape(y,p):9.1f}%")
    print("-" * 78)
    print("NOT: LOO R² artmıyor/düşüyorsa o değişken out-of-sample'da İŞE YARAMIYOR.")

    # 2) NULL TESTİ: her eklemenin kazancını rastgele değişkenle kıyasla
    print("\n" + "=" * 78)
    print("ADIM 3 — NULL TESTİ: gerçek değişkenin kazancı ÇÖP değişkeninkinden büyük mü?")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    base = [lnS, lnMc, nd]
    base_loo = r2(y, loo(base, y))
    # rastgele değişken eklenince LOO R² değişiminin dağılımı
    null = []
    for _ in range(N_NULL):
        rnd = rng.standard_normal(n)
        null.append(r2(y, loo(base + [rnd], y)) - base_loo)
    null = np.array(null)
    print(f"Taban model [ln(SNR)+ln(Mc)+ndet]  LOO R² = {base_loo:.3f}")
    print(f"ÇÖP değişken eklemenin LOO R² değişimi: ort={null.mean():+.3f}  "
          f"%95 üst sınır={np.quantile(null,0.95):+.3f}  max={null.max():+.3f}\n")
    print(f"{'eklenen gerçek değişken':28s}{'ΔLOO R²':>10}{'çöpü geçti mi? (>%95)':>24}")
    for name, col in [("eta", eta), ("chi_eff", chie), ("chi_p", chip),
                      ("z (redshift)", z), ("ln(Mtot) ek", lnMt)]:
        delta = r2(y, loo(base + [col], y)) - base_loo
        verdict = "EVET (gerçek katkı)" if delta > np.quantile(null, 0.95) else "hayır (şanstan farksız)"
        print(f"  {name:26s}{delta:+10.3f}{verdict:>24}")

    # ---- Grafik: LOO R² vs model karmaşıklığı ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    labels = ["lnSNR", "+lnMc", "+ndet", "+eta", "+chi_eff", "+chi_p", "+z"]
    inss = [insample_r2(c, y) for _, c in steps]
    ax[0].plot(labels, inss, "o-", label="in-sample R² (iyimser)", color="#95a5a6")
    ax[0].plot(labels, loo_r2s, "s-", label="LOO R² (dürüst)", color="#2980b9", lw=2)
    ax[0].set_ylabel("R²"); ax[0].set_title("Değişken ekledikçe R²\n(LOO platoya ulaşırsa fazlası overfit)")
    ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].tick_params(axis="x", rotation=35)

    ax[1].hist(null, bins=40, color="#bdc3c7", label="çöp değişken ΔLOO R² (null)")
    ax[1].axvline(np.quantile(null, 0.95), color="k", ls="--", label="null %95")
    for name, col, c in [("eta", eta, "#27ae60"), ("chi_eff", chie, "#e74c3c"),
                         ("z", z, "#8e44ad")]:
        delta = r2(y, loo(base + [col], y)) - base_loo
        ax[1].axvline(delta, color=c, lw=2, label=f"{name}: {delta:+.3f}")
    ax[1].set_xlabel("ΔLOO R² (taban modele bir değişken eklemenin kazancı)")
    ax[1].set_title("Null testi: gerçek değişkenler çöp dağılımını geçiyor mu?")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "result_test_extended.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nGrafik: {out}")


if __name__ == "__main__":
    main()
