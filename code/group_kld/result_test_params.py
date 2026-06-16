"""
RESULT TEST PARAMS — hangi parametre en çok bilgi taşıyor? (34 olay ortalaması)
================================================================================

Her olayın sonuç JSON'undaki 'marginal_kld_1d_nats' (parametre başına 1D KLD)
değerlerini 34 olay üzerinde toplar, bit'e çevirir, sıralar.

NEDEN GEÇERLİ: 1D marginal KLD, standartlaştırılmış (whitened) sütunlarda
hesaplandı; KL ıraksaması yeniden-parametrizasyona göre değişmezdir, yani
hem posterior hem prior aynı dönüşümden geçtiği için değerler boyutsuz bilgi
(nat) olarak parametreler arası KIYASLANABİLİR.

KRİTİK UYARI (raporda detaylı): 1D marginal KLD = ln(prior_genişlik /
posterior_genişlik) mertebesinde. Yani "iyi ölçülen" demek DEĞİL; GENİŞ prior'lu
bir parametre (örn. luminosity_distance) kötü ölçülse bile yüksek KL verir.
Ayrıca hibrit-prior'da bazı parametreler analitik prior'a çevrildi — bu da
karşılaştırmayı etkiler. Bu yüzden her parametrenin kaç olayda analitik-prior
kullanıldığını da işaretliyoruz.

Sadece OKUR. Çıktı: result_test_params.csv , result_test_params.png
Bağımlılıklar: numpy, matplotlib
"""

import os, csv, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "oto_master_ozet.csv")
NAT2BIT = 1.0 / np.log(2.0)

PARAMS = ["mass_1_source", "mass_2_source", "a_1", "a_2", "tilt_1", "tilt_2",
          "phi_12", "phi_jl", "luminosity_distance", "theta_jn", "psi",
          "azimuth", "zenith", "geocent_time", "phase"]


def load_jsons():
    events = [r["event"] for r in csv.DictReader(open(MASTER, encoding="utf-8")) if r["status"] == "ok"]
    out = []
    for ev in events:
        full = os.path.join(HERE, f"results_grup_kld_hibrit_{ev}.json")
        short = os.path.join(HERE, f"results_grup_kld_hibrit_{ev.split('_')[0]}.json")
        p = full if os.path.exists(full) else (short if os.path.exists(short) else None)
        if p:
            out.append(json.load(open(p, encoding="utf-8")))
    return out


def main():
    data = load_jsons()
    n = len(data)
    print(f"Yüklenen olay: {n}\n")

    # parametre başına marginal KLD (bit) matrisi
    M = np.array([[d["marginal_kld_1d_nats"][p] * NAT2BIT for p in PARAMS] for d in data])
    # her parametre kaç olayda analitik-prior kullanıldı?
    analytic_count = {p: sum(1 for d in data if p in d.get("hybrid_analytic_params", [])) for p in PARAMS}

    mean_ = M.mean(axis=0); med_ = np.median(M, axis=0); std_ = M.std(axis=0)
    order = np.argsort(-mean_)

    # CSV
    with open(os.path.join(HERE, "result_test_params.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["parametre", "ort_marginal_KL_bit", "medyan_bit", "std_bit",
                    "analitik_prior_olay_sayisi(/%d)" % n])
        for i in order:
            w.writerow([PARAMS[i], f"{mean_[i]:.3f}", f"{med_[i]:.3f}", f"{std_[i]:.3f}",
                        analytic_count[PARAMS[i]]])

    print(f"{'parametre':22s}{'ort(bit)':>9}{'medyan':>8}{'std':>7}{'analitik-prior':>16}")
    print("-" * 62)
    for i in order:
        print(f"{PARAMS[i]:22s}{mean_[i]:9.2f}{med_[i]:8.2f}{std_[i]:7.2f}{analytic_count[PARAMS[i]]:>12}/{n}")
    print("-" * 62)

    # Hiyerarşi: marginal toplam < grup-toplam < joint  (korelasyonda saklı bilgi)
    mt = np.mean([d["marginal_1d_total_bits"] for d in data])
    gt = np.mean([d["group_total_mean_bits"] for d in data])
    jt = np.mean([d["joint_kld_estimate_mean_bits"] for d in data])
    print(f"\nORTALAMA BİLGİ HİYERARŞİSİ (bit):")
    print(f"  1D marjinal toplam   = {mt:.1f}")
    print(f"  ≤5D grup-toplam      = {gt:.1f}")
    print(f"  joint 15D (+TC)      = {jt:.1f}")
    print(f"  => 1D marjinallerin kaçırdığı (korelasyon/birlikte) bilgi ≈ {jt-mt:.1f} bit "
          f"(joint'in %{100*(jt-mt)/jt:.0f}'i)")

    # ---- Grafik ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Sol: parametre sıralaması (analitik-prior oranına göre renk)
    yo = order[::-1]
    frac = np.array([analytic_count[PARAMS[i]] / n for i in yo])
    cols = plt.cm.coolwarm(frac)
    ax[0].barh(range(len(yo)), mean_[yo], xerr=std_[yo], color=cols,
               error_kw=dict(alpha=0.4))
    ax[0].set_yticks(range(len(yo))); ax[0].set_yticklabels([PARAMS[i] for i in yo], fontsize=9)
    ax[0].set_xlabel("ortalama 1D marjinal KLD (bit)  ±std")
    ax[0].set_title("Parametre başına bilgi (34 olay ort.)\n"
                    "renk: kırmızı=çoğunlukla ANALİTİK prior (kıyas dikkat), mavi=orijinal")
    ax[0].grid(axis="x", alpha=0.3)

    # Sağ: bilgi hiyerarşisi
    ax[1].bar(["1D marjinal\ntoplam", "≤5D grup\ntoplam", "joint 15D\n(+TC)"],
              [mt, gt, jt], color=["#16a085", "#e67e22", "#2980b9"], alpha=0.85)
    ax[1].set_ylabel("ortalama bilgi (bit)")
    ax[1].set_title("Bilgi hiyerarşisi: marjinaller joint'i eksik temsil eder\n"
                    f"korelasyonda saklı ≈ {jt-mt:.0f} bit")
    for k, v in enumerate([mt, gt, jt]):
        ax[1].text(k, v + 0.3, f"{v:.1f}", ha="center")
    ax[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, "result_test_params.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nGrafik: {out}\nCSV: {os.path.join(HERE,'result_test_params.csv')}")


if __name__ == "__main__":
    main()
