"""
RESULT TEST SURPRISE — "Sürpriz" istatistiği ile anomali tespiti
=================================================================

Fikir (Seehars+ 2014, Nicola+ 2019; kozmolojide tansiyon testi):
    Surprise  S = D_KL  −  <D_KL>
    D_KL     = bir olayda GERÇEKTEN ölçülen bilgi (joint KLD, bit)
    <D_KL>   = o olaydan BEKLENEN bilgi
    S > 0    = beklenenden FAZLA bilgi  (olay "şaşırtıcı")
    S < 0    = beklenenden AZ bilgi
Anlamlılık:  z = S / σ ,   σ = beklentinin tipik saçılması.
|z| büyükse olay popülasyon eğiliminden sapıyor → ya dalga-modeli
sistematiği, ya ilginç astrofizik/GR, ya da tahmin tutarsızlığı.

UYARLAMA: Kozmolojide <D_KL> ve σ analitik Gauss formüllerinden gelir.
Burada bunları VERİDEN kestiriyoruz:
  <D_KL>  = çoklu modelin (I = a·ln SNR + c·ln Mtot + d·ndet) tahmini,
            AMA her olay için LEAVE-ONE-OUT (olayı görmemiş modelle) —
            böylece olay kendi beklentisini şişirmez (Seehars'ın
            "tutarlılık varsayımı altında beklenen" fikrinin ampirik karşılığı).
  σ       = LOO artıklarının standart sapması (popülasyonun tipik saçılması).
Yani z_i = studentize edilmiş out-of-sample artık. |z|>2 ≈ %95 dışı.

Sadece OKUR; mevcut sonuçlara dokunmaz.
Çıktı: result_test_surprise.csv , result_test_surprise.png
Bağımlılıklar: numpy, h5py, matplotlib
"""

import os, csv, numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data"
MASTER = os.path.join(HERE, "oto_master_ozet.csv")
METRIC = "joint_mean_bits"


def med(ps, c):
    return float(np.median(np.asarray(ps[c], float))) if c in (ps.dtype.names or []) else np.nan


def load():
    rows = [r for r in csv.DictReader(open(MASTER, encoding="utf-8")) if r["status"] == "ok"]
    ev, snr, mt, nd, I = [], [], [], [], []
    for r in rows:
        with h5py.File(os.path.join(DATA_DIR, r["file"]), "r") as f:
            g = r["analysis_group"]
            if g not in f:
                g = [k for k in f if isinstance(f[k], h5py.Group) and "posterior_samples" in f[k]][0]
            ps = f[g]["posterior_samples"]; nm = ps.dtype.names
            s = med(ps, "network_matched_filter_snr"); M = med(ps, "total_mass_source")
            n = sum(1 for d in ("H1", "L1", "V1")
                    if d + "_optimal_snr" in nm and np.median(np.asarray(ps[d + "_optimal_snr"], float)) > 1)
        ev.append(r["event"]); snr.append(s); mt.append(M); nd.append(n); I.append(float(r[METRIC]))
    return ev, np.array(snr), np.array(mt), np.array(nd, float), np.array(I)


def design(snr, mt, nd):
    return np.column_stack([np.ones(len(snr)), np.log(snr), np.log(mt), nd])


def loo_predictions(X, y):
    """Her olay için, onu DIŞARIDA bırakarak fit edilen modelin tahmini."""
    n = len(y); pred = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[i] = X[i] @ beta
    return pred


def main():
    ev, snr, mt, nd, I = load()
    short = [e.split("_")[0] for e in ev]
    X = design(snr, mt, nd)

    Ipred = loo_predictions(X, I)        # <D_KL>  (out-of-sample beklenti)
    S = I - Ipred                        # Surprise (bit)
    sigma = S.std(ddof=1)                # beklentinin tipik saçılması
    z = S / sigma                        # standartlaştırılmış sürpriz

    # sırala (en şaşırtıcıdan)
    order = np.argsort(-np.abs(z))

    # CSV
    with open(os.path.join(HERE, "result_test_surprise.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event", "SNR", "Mtot", "ndet", "I_gercek_bit",
                    "I_beklenen_bit", "Surprise_S_bit", "z_sigma", "bayrak"])
        for i in order:
            flag = ">2sigma" if abs(z[i]) > 2 else (">1sigma" if abs(z[i]) > 1 else "")
            w.writerow([ev[i], f"{snr[i]:.2f}", f"{mt[i]:.1f}", int(nd[i]),
                        f"{I[i]:.2f}", f"{Ipred[i]:.2f}", f"{S[i]:+.2f}",
                        f"{z[i]:+.2f}", flag])

    print(f"Olay: {len(I)}   σ(Surprise) = {sigma:.2f} bit\n")
    print(f"{'event':17s}{'SNR':>6}{'Mtot':>7}{'I_ger':>8}{'I_bek':>8}{'S':>7}{'z':>7}  bayrak")
    print("-" * 70)
    for i in order:
        flag = "  <-- >2σ ANOMALİ" if abs(z[i]) > 2 else ("  (>1σ)" if abs(z[i]) > 1 else "")
        print(f"{short[i]:17s}{snr[i]:6.1f}{mt[i]:7.1f}{I[i]:8.1f}{Ipred[i]:8.1f}"
              f"{S[i]:+7.1f}{z[i]:+7.2f}{flag}")
    print("-" * 70)
    n2 = int(np.sum(np.abs(z) > 2)); n1 = int(np.sum(np.abs(z) > 1))
    print(f">2σ (güçlü anomali): {n2} olay    >1σ: {n1} olay    (Gauss beklentisi: ~%5 ve ~%32)")

    # ---- Grafik ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Sol: olay başına z, ±1σ ±2σ bantları
    oi = np.argsort(I)  # bilgiye göre sırala (okunur)
    y = np.arange(len(oi))
    cols = ["#e74c3c" if abs(z[i]) > 2 else ("#f39c12" if abs(z[i]) > 1 else "#3498db") for i in oi]
    ax[0].barh(y, z[oi], color=cols)
    for s in (1, 2):
        ax[0].axvline(s, color="#aaa", ls=":"); ax[0].axvline(-s, color="#aaa", ls=":")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{short[i]}" for i in oi], fontsize=7)
    ax[0].set_xlabel("standartlaştırılmış Sürpriz  z = (gerçek − beklenen)/σ")
    ax[0].set_title(f"Olay başına Sürpriz\nkırmızı |z|>2, sarı |z|>1, mavi normal  •  σ={sigma:.1f} bit")
    ax[0].grid(axis="x", alpha=0.3)

    # Sağ: Sürpriz kütleye/SNR'ye bağlı mı? (sistematik kontrolü)
    sc = ax[1].scatter(snr, S, c=mt, cmap="plasma_r", s=60, edgecolor="k", linewidth=0.3)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].axhline(sigma, color="#aaa", ls=":"); ax[1].axhline(-sigma, color="#aaa", ls=":")
    ax[1].axhline(2 * sigma, color="#e74c3c", ls=":"); ax[1].axhline(-2 * sigma, color="#e74c3c", ls=":")
    for i in order[:4]:
        ax[1].annotate(short[i], (snr[i], S[i]), fontsize=8, xytext=(4, 4),
                       textcoords="offset points")
    ax[1].set_xlabel("medyan SNR"); ax[1].set_ylabel("Sürpriz S = gerçek − beklenen (bit)")
    ax[1].set_title("Sürpriz, SNR/kütleyle örüntü gösteriyor mu?\n(rastgele dağılmalı; örüntü → eksik değişken)")
    cb = fig.colorbar(sc, ax=ax[1]); cb.set_label("toplam kütle (Msun)")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, "result_test_surprise.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nGrafik: {out}\nCSV: {os.path.join(HERE, 'result_test_surprise.csv')}")


if __name__ == "__main__":
    main()
