"""
Binning (histogram) tabanlı bilgi-teorik tahmin ediciler — entropi & KL-ıraksaması
===================================================================================

Kaynak: Álvarez Chaves et al. (2024), "On the Accurate Estimation of
Information-Theoretic Quantities from Multi-Dimensional Sample Data",
Entropy 26(5), 387.  Orijinal kod: UNITE Toolbox (MIT lisansı),
unite_toolbox/bin_estimators.py.

Bu dosya, makalenin BIN tahmin edicilerinin TÜRKÇE AÇIKLAMALI ve TEK-DOSYA
halidir; knn_estimators_reference.py ile aynı tarzdadır. Bağımlılık: yalnızca numpy.

Temel fikir: veriyi d boyutlu bir histograma böl, her hücredeki yoğunluğu
(sayım / (N * hücre_hacmi)) kullanarak entropi/KLD'yi topla. Bin sayısı seçimi
(scott/fd/sturges) sonucu doğrudan etkiler — bu yüzden birden çok kural denenir.
"""

import numpy as np


def estimate_ideal_bins(data: np.ndarray, *, counts: bool = True) -> dict:
    """Her boyut (sütun) için ideal bin sayısını/kenarlarını tahmin et.

    numpy.histogram_bin_edges'in kuralları kullanılır. `counts=True` ise her
    boyut için bin SAYISI, `counts=False` ise bin KENARLARI (array) döner.
    Dönen sözlük: {"fd": [...], "scott": [...], "sturges": [...], "doane": [...]}
    """
    _, d_features = data.shape
    methods = ["fd", "scott", "sturges", "doane"]
    ideal_bins = []
    for m in methods:
        d_bins = []
        for d in range(d_features):
            edges = np.histogram_bin_edges(data[:, d], bins=m)
            d_bins.append(len(edges) if counts else edges)
        ideal_bins.append(d_bins)
    return dict(zip(methods, ideal_bins))


def calc_vol_array(edges: list) -> np.ndarray:
    """`edges` (her boyut için 1B kenar dizisi) ile tanımlı çok-boyutlu ızgaranın
    her hücresinin HACMİNİ hesapla. Dönen ızgara, np.histogramdd çıktısıyla aynı
    indislerle indekslenebilir (diferansiyel entropi düzeltmesi için gerekli)."""
    vol = np.diff(edges[0])
    for e in edges[1:]:
        vol = np.stack([vol] * (len(e) - 1), axis=-1)
        for idx, val in enumerate(np.diff(e)):
            vol[..., idx] = vol[..., idx] * val
    return vol


def calc_bin_entropy(data: np.ndarray, edges) -> tuple:
    """Binning ile (ortak) diferansiyel entropi [nats].

    H = -Σ delta * f * log(f)   (f: hücre yoğunluğu, delta: hücre hacmi)
    `edges`: her boyut için kenar dizileri listesi ya da bin sayısı (int/list).
    Dönüş: (h, corr_fact). Toplam entropi genelde h + corr_fact olarak alınır;
    makaledeki kullanımda sum(calc_bin_entropy(...)) = h + corr_fact'tır.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    f, edges = np.histogramdd(data, bins=edges, density=True)
    volume = calc_vol_array(edges)
    idx = f.nonzero()
    delta = volume[idx]
    f = f[idx]
    h = -1.0 * np.sum(delta * f * np.log(f))
    corr_fact = -1.0 * np.sum(f * delta * np.log(delta))
    return h, corr_fact


def calc_bin_kld(p: np.ndarray, q: np.ndarray, edges: list) -> float:
    """Binning ile Kullback-Leibler ıraksaması  D_KL(p||q) [nats].

    Hem p hem q, q'nun DESTEĞİNE göre belirlenen `edges` ile aynı histograma
    bölünür. KLD yalnızca p ve q'nun ORTAK dolu olduğu hücrelerde toplanır:
        D = Σ_bin  pp * log(pp / pq)
    `edges`: estimate_ideal_bins(q, counts=False) çıktısından bir kural
    (ör. nbins["scott"]). q desteği p'yi kapsamalıdır.
    """
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    p_binned = np.empty(shape=p.shape, dtype=np.int64)
    q_binned = np.empty(shape=q.shape, dtype=np.int64)
    for idy in range(q.shape[1]):
        p_binned[:, idy] = np.digitize(p[:, idy], edges[idy])
        q_binned[:, idy] = np.digitize(q[:, idy], edges[idy])

    bins_p, counts_p = np.unique(p_binned, return_counts=True, axis=0)
    bins_q, counts_q = np.unique(q_binned, return_counts=True, axis=0)

    set_p = set(tuple(x) for x in bins_p)
    set_q = set(tuple(x) for x in bins_q)
    matching_bins = [x for x in set_p & set_q]

    density_p = counts_p / p.shape[0]
    density_q = counts_q / q.shape[0]

    kld = 0.0
    for idx in matching_bins:
        idx = np.array(idx)
        a = np.where((bins_p == idx).all(axis=1))[0][0]
        b = np.where((bins_q == idx).all(axis=1))[0][0]
        kld += density_p[a] * np.log(density_p[a] / density_q[b])
    return kld


# --------------------------------------------------------------------------
# Hızlı doğrulama: 4D Gaussian KLD (analitik referansla)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy import stats

    d = 4
    mu = np.zeros(d)
    S1 = 0.6 * np.ones((d, d)); np.fill_diagonal(S1, 1.0)
    S2 = 0.2 * np.ones((d, d)); np.fill_diagonal(S2, 1.0)
    P = stats.multivariate_normal(mu, S1).rvs(100_000, random_state=1)
    Q = stats.multivariate_normal(mu, S2).rvs(100_000, random_state=2)

    true = 0.5 * (np.log(np.linalg.det(S2) / np.linalg.det(S1))
                  + np.trace(np.linalg.inv(S2) @ S1) - d)
    nbins = estimate_ideal_bins(Q, counts=False)
    for rule in ("scott", "fd"):
        est = calc_bin_kld(P, Q, nbins[rule])
        print(f"KLD bin-{rule:7s}: tahmin {est:.3f} | analitik {true:.3f} nats")
