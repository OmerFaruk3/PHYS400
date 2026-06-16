"""
RESULT TEST — SNR ↔ elde edilen bilgi (KLD) ilişkisi
=====================================================

Ne yapar:
  1) oto_master_ozet.csv'deki her olayın bilgi ölçülerini (KLD, bit) okur.
  2) Her olayın .h5 dosyasından SNR'yi okur:
        posterior_samples["network_matched_filter_snr"]  (medyan)
     ve karşılaştırma için network_optimal_snr medyanını da alır.
     SNR, KLD hesabında kullanılan AYNI analiz grubundan (CSV'deki
     analysis_group, ör. C01:IMRPhenomXPHM) okunur — tutarlılık için.
  3) SNR ↔ bilgi (bit) saçılım grafiğini çizer.
  4) Üç model dener ve en iyisini işaretler:
        log     :  I = a·ln(SNR) + b      (teorik beklenti: Fisher ~ SNR²)
        linear  :  I = m·SNR + c
        power   :  I = A·SNR^p
     Her biri için R² hesaplar; Pearson & Spearman korelasyonu raporlar.

Çıktılar (bu .py ile aynı klasöre):
  result_test_snr_info.csv   olay başına SNR + bilgi tablosu
  result_test_plot.png       saçılım + fitler (2 panel)

Bu kod yalnızca OKUR; mevcut sonuç/master dosyalarına yazmaz/dokunmaz.

Kullanım:
  python result_test.py
  python result_test.py --metric group_total_mean_bits   # bilgi ölçüsünü değiştir
  python result_test.py --snr optimal                    # matched yerine optimal SNR
  python result_test.py --master oto_master_ozet_gwtc2p1.csv   # başka bir master

Bağımlılıklar: numpy, scipy, h5py, matplotlib
"""

import os
import sys
import csv
import argparse

import numpy as np
import h5py
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data"

# CSV'de "bilgi" olarak kullanılabilecek sayısal sütunlar
INFO_METRICS = [
    "joint_mean_bits",            # joint 15D KLD tahmini (yöntem-ort.)  [VARSAYILAN]
    "group_total_mean_bits",      # ≤5D grup-toplam KLD (yöntem-ort.)
    "marginal_1d_total_bits",     # 1D marjinal toplam
]


# ---------------------------------------------------------------------
# SNR okuma
# ---------------------------------------------------------------------
def read_snr_median(h5_path, analysis_group, which="matched"):
    """h5'ten medyanları döndürür: (matched_snr, optimal_snr, total_mass, n, grp).

    analysis_group veriliyse o grubu kullanır; yoksa posterior_samples içeren
    ilk grubu bulur. İstenen kolon yoksa np.nan döner.
    """
    with h5py.File(h5_path, "r") as f:
        grp = analysis_group
        if not grp or grp not in f or "posterior_samples" not in f[grp]:
            grp = None
            for k in f.keys():
                g = f[k]
                if isinstance(g, h5py.Group) and "posterior_samples" in g:
                    grp = k
                    break
        if grp is None:
            return np.nan, np.nan, np.nan, 0, None
        ps = f[grp]["posterior_samples"]
        names = ps.dtype.names or []

        def med(cname):
            if cname in names:
                return float(np.median(np.asarray(ps[cname], dtype=float)))
            return np.nan

        mfa = med("network_matched_filter_snr")
        opt = med("network_optimal_snr")
        mtot = med("total_mass_source")
        # dedektör sayısı: tek-dedektör optimal_snr medyanı belirgin (>1) olanlar
        ndet = 0
        for d in ("H1", "L1", "V1"):
            c = d + "_optimal_snr"
            if c in names and np.median(np.asarray(ps[c], dtype=float)) > 1.0:
                ndet += 1
        n = ps.shape[0]
    return mfa, opt, mtot, ndet, n, grp


# ---------------------------------------------------------------------
# F&H / Shannon kapasite formülü ve "etkin parametre sayısı" N_eff
# ---------------------------------------------------------------------
#   I(N) = (1/2) N log2(1 + SNR^2 / N)   [bit]
# N eşit-bölüşülmüş Gauss kanalının toplam bilgisi. N büyüdükçe I, üst sınıra
# (SNR^2 / (2 ln2)) doğru DOYAR. Bu yüzden ölçülen bilgi I_meas verildiğinde,
# tek bir N_eff için çözülebilir (I_meas < asimptot ise). N_eff, olayın SNR'den
# bağımsız "etkin kısıtlama gücünü" verir.
def channel_capacity(N, snr):
    return 0.5 * N * np.log2(1.0 + snr * snr / N)


def solve_Neff(I_meas, snr):
    """I(N)=I_meas denklemini N>0 için çözer. Çözüm yoksa inf/nan döner."""
    if not np.isfinite(I_meas) or not np.isfinite(snr) or I_meas <= 0 or snr <= 0:
        return np.nan
    asymptote = snr * snr / (2.0 * np.log(2.0))   # N->inf limiti
    if I_meas >= asymptote:
        return np.inf      # ölçülen bilgi formül tavanını aşıyor
    from scipy.optimize import brentq
    try:
        return brentq(lambda N: channel_capacity(N, snr) - I_meas, 1e-6, 1e7)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------
# Modeller + fit
# ---------------------------------------------------------------------
def f_log(x, a, b):     return a * np.log(x) + b
def f_lin(x, m, c):     return m * x + c
def f_pow(x, A, p):     return A * np.power(x, p)


def r2(y, yhat):
    y = np.asarray(y, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_model(func, x, y, p0):
    try:
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=20000)
        yhat = func(x, *popt)
        return popt, r2(y, yhat)
    except Exception as e:
        return None, np.nan


def fit_all(x, y):
    """log / linear / power modellerini fitler. Sözlük döndürür."""
    res = {}
    # log: I = a ln x + b   (lineer çözüm, sağlam başlangıç)
    a0, b0 = np.polyfit(np.log(x), y, 1)
    res["log"] = {"func": f_log, "label": "I = a·ln(SNR) + b",
                  **dict(zip(["popt", "r2"], fit_model(f_log, x, y, [a0, b0])))}
    # linear
    m0, c0 = np.polyfit(x, y, 1)
    res["linear"] = {"func": f_lin, "label": "I = m·SNR + c",
                     **dict(zip(["popt", "r2"], fit_model(f_lin, x, y, [m0, c0])))}
    # power: ln I = ln A + p ln x
    p0p = np.polyfit(np.log(x), np.log(y), 1)
    res["power"] = {"func": f_pow, "label": "I = A·SNR^p",
                    **dict(zip(["popt", "r2"], fit_model(f_pow, x, y, [np.exp(p0p[1]), p0p[0]])))}
    return res


def fmt_popt(name, popt):
    if popt is None:
        return "(fit başarısız)"
    if name == "log":
        return f"a={popt[0]:.2f}, b={popt[1]:.2f}"
    if name == "linear":
        return f"m={popt[0]:.3f}, c={popt[1]:.2f}"
    if name == "power":
        return f"A={popt[0]:.3f}, p={popt[1]:.2f}"
    return str(popt)


# ---------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------
def main(args):
    master = args.master if os.path.isabs(args.master) else os.path.join(HERE, args.master)
    if not os.path.exists(master):
        print(f"HATA: master CSV bulunamadı: {master}")
        return 1
    metric = args.metric
    if metric not in INFO_METRICS:
        print(f"HATA: --metric şunlardan biri olmalı: {INFO_METRICS}")
        return 1

    with open(master, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]
    print(f"Master: {os.path.basename(master)}  |  ok olay: {len(rows)}  |  bilgi ölçüsü: {metric}")

    data = []   # (event, snr_matched, snr_optimal, mass, ndet, n_post, info_bits)
    missing = []
    for r in rows:
        ev = r["event"]
        fname = r["file"]
        h5p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(h5p):
            missing.append((ev, "h5 yok"))
            continue
        try:
            mfa, opt, mtot, ndet, n, used = read_snr_median(h5p, r.get("analysis_group"), which="matched")
            info = float(r[metric])
            if not np.isfinite(mfa):
                missing.append((ev, "SNR kolonu yok"))
                continue
            data.append((ev, mfa, opt, mtot, ndet, int(r.get("n_posterior") or n or 0), info))
        except Exception as e:
            missing.append((ev, str(e)))

    if missing:
        print(f"\nUYARI: {len(missing)} olay atlandı:")
        for ev, why in missing:
            print(f"   {ev}: {why}")
    if len(data) < 3:
        print("Yeterli veri yok (≥3 olay gerekli).")
        return 1

    events = [d[0] for d in data]
    snr_m = np.array([d[1] for d in data])
    snr_o = np.array([d[2] for d in data])
    mass = np.array([d[3] for d in data])
    ndet = np.array([d[4] for d in data], dtype=float)
    npost = np.array([d[5] for d in data])
    info = np.array([d[6] for d in data])

    x = snr_m if args.snr == "matched" else snr_o
    xlabel = ("medyan network matched-filter SNR" if args.snr == "matched"
              else "medyan network optimal SNR")

    # F&H kapasite formülünden olay başına etkin parametre sayısı N_eff
    neff = np.array([solve_Neff(I, p) for I, p in zip(info, x)])

    # CSV kaydet (N_eff dahil)
    out_csv = os.path.join(HERE, "result_test_snr_info.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event", "snr_matched_median", "snr_optimal_median",
                    "total_mass_source_median", "n_posterior", metric,
                    "capacity_asymptote_bits", "N_eff"])
        order = np.argsort(x)
        for i in order:
            asym = x[i] ** 2 / (2 * np.log(2))
            nv = ("inf" if np.isinf(neff[i]) else
                  ("nan" if not np.isfinite(neff[i]) else f"{neff[i]:.3f}"))
            w.writerow([events[i], f"{snr_m[i]:.4f}", f"{snr_o[i]:.4f}",
                        f"{mass[i]:.3f}", npost[i], f"{info[i]:.4f}",
                        f"{asym:.3f}", nv])
    print(f"\nTablo kaydedildi: {out_csv}")

    # ---- N_eff tablosu ve ilişki ----
    print("\n" + "=" * 72)
    print(f"ETKİN PARAMETRE SAYISI  N_eff   (formül: I=(N/2)log2(1+SNR²/N), {metric})")
    print("=" * 72)
    print(f"{'event':17s}{'SNR':>7}{'I(bit)':>9}{'asimptot':>10}{'N_eff':>9}")
    print("-" * 72)
    for i in np.argsort(x):
        asym = x[i] ** 2 / (2 * np.log(2))
        nv = "inf" if np.isinf(neff[i]) else (f"{neff[i]:.2f}" if np.isfinite(neff[i]) else "nan")
        print(f"{events[i]:17s}{x[i]:7.2f}{info[i]:9.2f}{asym:10.1f}{nv:>9}")
    fin = np.isfinite(neff) & ~np.isinf(neff)
    if fin.sum() >= 3:
        Nf = neff[fin]
        print("-" * 72)
        print(f"N_eff: medyan={np.median(Nf):.2f}  ort={Nf.mean():.2f}  "
              f"[{Nf.min():.2f}, {Nf.max():.2f}]   (analizdeki parametre sayısı = 15)")
        print("\nN_eff ile korelasyon:")
        for nm, xv in [("SNR", x[fin]), ("ln(SNR)", np.log(x[fin])),
                       ("toplam_kütle", mass[fin]), ("ln(kütle)", np.log(mass[fin]))]:
            ok = np.isfinite(xv)
            pr = pearsonr(xv[ok], Nf[ok])[0]
            sr = spearmanr(xv[ok], Nf[ok])[0]
            print(f"  N_eff ~ {nm:14s} r={pr:+.3f}  rho={sr:+.3f}")

    # Korelasyon
    pr, pp = pearsonr(x, info)
    sr, sp = spearmanr(x, info)
    print("\n" + "=" * 64)
    print(f"KORELASYON  ({xlabel}  ↔  {metric})")
    print("=" * 64)
    print(f"  Pearson  r = {pr:+.3f}  (p={pp:.2e})")
    print(f"  Spearman ρ = {sr:+.3f}  (p={sp:.2e})")

    # Fitler
    fits = fit_all(x, info)
    print("\n" + "=" * 64)
    print("MODEL FİTLERİ")
    print("=" * 64)
    best_name, best_r2 = None, -np.inf
    for name in ("log", "linear", "power"):
        d = fits[name]
        print(f"  {name:7s} {d['label']:22s}  R²={d['r2']:.4f}   {fmt_popt(name, d.get('popt'))}")
        if np.isfinite(d["r2"]) and d["r2"] > best_r2:
            best_r2, best_name = d["r2"], name
    print(f"\n  >>> En iyi model: {best_name}  (R²={best_r2:.4f})")

    # ---- ÇOKLU MODEL:  I = a·ln(SNR) + c·ln(Mtot) + d·ndet + sabit ----
    def ols(cols):
        """Sabit terimli OLS. (beta, yhat, R², düzeltilmiş R²) döndürür."""
        X = np.column_stack([np.ones(len(info))] + cols)
        beta, *_ = np.linalg.lstsq(X, info, rcond=None)
        yhat = X @ beta
        ss_res = np.sum((info - yhat) ** 2)
        ss_tot = np.sum((info - info.mean()) ** 2)
        R2 = 1 - ss_res / ss_tot
        k = X.shape[1] - 1
        n = len(info)
        adj = 1 - (1 - R2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else np.nan
        return beta, yhat, R2, adj

    lnS = np.log(x)
    lnM = np.log(mass)
    okm = np.isfinite(lnM)
    print("\n" + "=" * 64)
    print("ÇOKLU MODEL  (bilgiyi açıklayan değişkenler eklenince R²)")
    print("=" * 64)
    _, _, r2a, adja = ols([lnS])
    print(f"  I = a·ln(SNR) + sabit                     R²={r2a:.3f}  (düz.R²={adja:.3f})")
    if okm.all():
        _, _, r2b, adjb = ols([lnS, lnM])
        print(f"  I = a·ln(SNR) + c·ln(Mtot) + sabit        R²={r2b:.3f}  (düz.R²={adjb:.3f})")
        beta, yhat_full, r2c, adjc = ols([lnS, lnM, ndet])
        print(f"  I = a·ln(SNR) + c·ln(Mtot) + d·ndet + sb. R²={r2c:.3f}  (düz.R²={adjc:.3f})")
        print("\n  Tam model katsayıları:")
        print(f"    sabit     = {beta[0]:+.2f} bit")
        print(f"    a [lnSNR] = {beta[1]:+.2f}   (SNR ↑ → bilgi ↑)")
        print(f"    c [lnMtot]= {beta[2]:+.2f}   (kütle ↑ → bilgi ↓)")
        print(f"    d [ndet]  = {beta[3]:+.2f}   (dedektör ↑ → bilgi ↑)")
        # her değişkenin tek başına bıraktığı katkı (kısmi)
        from scipy.stats import pearsonr as _pr
        print("\n  Tek değişkenli korelasyon (bilgi ile):")
        for nm, xv in [("ln(SNR)", lnS), ("ln(Mtot)", lnM), ("ndet", ndet)]:
            print(f"    {nm:9s} r={_pr(xv, info)[0]:+.3f}")
        multi_ok = True
    else:
        multi_ok = False
        yhat_full = None
        print("  (kütle okunamadığı için çoklu model atlandı)")

    # ---- Grafik ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: seçilen ölçü + fitler
    ax = axes[0]
    ax.scatter(x, info, s=45, c="#2c3e50", zorder=3, label="events")
    # birkaç uç olayı etiketle (en yüksek/en düşük SNR ve bilgi)
    idx_label = set(np.argsort(x)[-3:]) | set(np.argsort(info)[-3:]) | set(np.argsort(x)[:2])
    for i in idx_label:
        ax.annotate(events[i].split("_")[0], (x[i], info[i]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points", color="#555")
    xx = np.linspace(x.min() * 0.98, x.max() * 1.02, 200)
    colors = {"log": "#e74c3c", "linear": "#27ae60", "power": "#8e44ad"}
    for name in ("log", "linear", "power"):
        d = fits[name]
        if d.get("popt") is not None and np.isfinite(d["r2"]):
            lw = 2.6 if name == best_name else 1.4
            ls = "-" if name == best_name else "--"
            star = "  ★en iyi" if name == best_name else ""
            ax.plot(xx, d["func"](xx, *d["popt"]), ls, lw=lw, color=colors[name],
                    label=f"{d['label']}  (R²={d['r2']:.3f}){star}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{metric}  (bit)")
    ax.set_title(f"SNR ↔ Obtained Information\nPearson r={pr:.3f}, Spearman ρ={sr:.3f}, n={len(x)}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # Panel B: tüm bilgi ölçüleri karşılaştırma (log-fit eğilimiyle)
    ax2 = axes[1]
    metric_cols = {}
    with open(master, encoding="utf-8") as fh:
        mrows = {r["event"]: r for r in csv.DictReader(fh) if r.get("status") == "ok"}
    mk = {"joint_mean_bits": ("#2980b9", "o"),
          "group_total_mean_bits": ("#e67e22", "s"),
          "marginal_1d_total_bits": ("#16a085", "^")}
    for col, (cc, mm) in mk.items():
        yv = np.array([float(mrows[e][col]) for e in events])
        ax2.scatter(x, yv, s=35, color=cc, marker=mm, alpha=0.8, label=col)
        a0, b0 = np.polyfit(np.log(x), yv, 1)
        ax2.plot(xx, a0 * np.log(xx) + b0, "-", color=cc, lw=1.3, alpha=0.7)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("information (bit)")
    ax2.set_title("Three information measures vs SNR  (lines: log-fit)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(HERE, "result_test_plot.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nGrafik kaydedildi: {out_png}")

    # ---- N_eff grafiği: SNR'ye karşı (düz olmalı) ve kütleye karşı ----
    fin = np.isfinite(neff) & ~np.isinf(neff)
    if fin.sum() >= 3:
        fig2, ax = plt.subplots(1, 2, figsize=(15, 6))
        Nf, xf, mf = neff[fin], x[fin], mass[fin]

        a0 = ax[0]
        sc = a0.scatter(xf, Nf, c=mf, cmap="viridis", s=55, zorder=3)
        a0.axhline(15, color="#888", ls=":", lw=1.5, label="analiz parametre sayısı = 15")
        a0.axhline(np.median(Nf), color="#c0392b", ls="--", lw=1.5,
                   label=f"medyan N_eff = {np.median(Nf):.1f}")
        r_s = pearsonr(xf, Nf)[0]
        a0.set_xlabel(xlabel)
        a0.set_ylabel("N_eff  (etkin parametre sayısı)")
        a0.set_title(f"N_eff vs SNR  (r={r_s:+.2f} → SNR'den BAĞIMSIZ)\nrenk = toplam kütle")
        a0.legend(fontsize=8)
        a0.grid(alpha=0.3)
        cb = fig2.colorbar(sc, ax=a0); cb.set_label("toplam kütle (Msun)")

        a1 = ax[1]
        a1.scatter(mf, Nf, c="#2c3e50", s=55, zorder=3)
        ok = np.isfinite(mf)
        a, b = np.polyfit(np.log(mf[ok]), Nf[ok], 1)
        mm = np.linspace(mf.min() * 0.95, mf.max() * 1.05, 100)
        a1.plot(mm, a * np.log(mm) + b, "r-", lw=2,
                label=f"N_eff = {a:.1f}·ln(M) + {b:.1f}")
        rm = pearsonr(np.log(mf[ok]), Nf[ok])[0]
        a1.set_xlabel("medyan toplam kütle (Msun, kaynak çerçeve)")
        a1.set_ylabel("N_eff")
        a1.set_title(f"N_eff vs toplam kütle  (ln-fit, r={rm:+.2f})")
        a1.legend(fontsize=8)
        a1.grid(alpha=0.3)

        fig2.tight_layout()
        out_png2 = os.path.join(HERE, "result_test_neff.png")
        fig2.savefig(out_png2, dpi=140, bbox_inches="tight")
        print(f"N_eff grafiği kaydedildi: {out_png2}")

    # ---- Çoklu model grafiği: kütle-renkli saçılım + tahmin-gerçek ----
    if multi_ok and yhat_full is not None:
        fig3, ax = plt.subplots(1, 2, figsize=(15, 6))

        # Sol: SNR vs bilgi, KÜTLEye göre renkli + tek-değişkenli log fit
        a0 = ax[0]
        sc = a0.scatter(x, info, c=mass, cmap="plasma_r", s=60, zorder=3,
                        edgecolor="k", linewidth=0.3)
        dlog = fits["log"]
        if dlog.get("popt") is not None:
            a0.plot(xx, dlog["func"](xx, *dlog["popt"]), "k--", lw=1.6,
                    label=f"I=a·ln(SNR)+b  (R²={dlog['r2']:.2f})")
        a0.set_xlabel(xlabel)
        a0.set_ylabel(f"{metric}  (bit)")
        a0.set_title("SNR ↔ color, color = total mass\n"
                     "(same SNR dark=heavy→less information, light=light→more information)")
        a0.legend(fontsize=8)
        a0.grid(alpha=0.3)
        cb = fig3.colorbar(sc, ax=a0); cb.set_label("total mass (Msun)")

        # Sağ: çoklu modelin tahmini vs gerçek
        a1 = ax[1]
        a1.scatter(yhat_full, info, c=mass, cmap="plasma_r", s=60, zorder=3,
                   edgecolor="k", linewidth=0.3)
        lim = [min(info.min(), yhat_full.min()) - 1, max(info.max(), yhat_full.max()) + 1]
        a1.plot(lim, lim, "k--", lw=1.2, label="y = x")
        a1.set_xlim(lim); a1.set_ylim(lim)
        a1.set_xlabel("multi-model prediction  a·ln(SNR)+c·ln(Mtot)+d·ndet  (bit)")
        a1.set_ylabel(f"true {metric}  (bit)")
        a1.set_title(f"Multi-model agreement  (R²={r2c:.2f}, adjusted R²={adjc:.2f})\n"
                     "SNR alone R²={:.2f} → adding mass+detector makes it jump".format(r2a))
        a1.legend(fontsize=8)
        a1.grid(alpha=0.3)

        fig3.tight_layout()
        out_png3 = os.path.join(HERE, "result_test_multi.png")
        fig3.savefig(out_png3, dpi=140, bbox_inches="tight")
        print(f"Multi-model graph saved: {out_png3}")

        # ---- Sol panel'i ayrı olarak kaydet ----
        fig_left, ax_left = plt.subplots(1, 1, figsize=(8, 6))
        sc_left = ax_left.scatter(x, info, c=mass, cmap="plasma_r", s=60, zorder=3,
                                   edgecolor="k", linewidth=0.3)
        if dlog.get("popt") is not None:
            ax_left.plot(xx, dlog["func"](xx, *dlog["popt"]), "k--", lw=1.6,
                        label=f"I=a·ln(SNR)+b  (R²={dlog['r2']:.2f})")
        ax_left.set_xlabel(xlabel)
        ax_left.set_ylabel(f"{metric}  (bit)")
        ax_left.set_title("SNR ↔ color, color = total mass\n"
                         "(same SNR dark=heavy→less information, light=light→more information)")
        ax_left.legend(fontsize=9)
        ax_left.grid(alpha=0.3)
        cb_left = fig_left.colorbar(sc_left, ax=ax_left); cb_left.set_label("total mass (Msun)")
        
        fig_left.tight_layout()
        out_png_left = os.path.join(HERE, "result_test_multi_left.png")
        fig_left.savefig(out_png_left, dpi=140, bbox_inches="tight")
        plt.close(fig_left)
        print(f"Left panel saved separately: {out_png_left}")

    print("=" * 64)
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description="SNR ↔ KLD bilgi ilişkisi + fit")
    p.add_argument("--master", default="oto_master_ozet.csv",
                   help="master CSV dosyası (varsayılan: oto_master_ozet.csv)")
    p.add_argument("--metric", default="joint_mean_bits", choices=INFO_METRICS,
                   help="bilgi ölçüsü sütunu")
    p.add_argument("--snr", default="matched", choices=["matched", "optimal"],
                   help="SNR türü (matched-filter veya optimal)")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_args(sys.argv[1:])))
