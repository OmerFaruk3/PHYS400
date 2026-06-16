"""
kNN tabanlı bilgi-teorik tahmin ediciler (entropi, KL-ıraksaması, karşılıklı bilgi)
=====================================================================================

Kaynak: Álvarez Chaves et al. (2024), "On the Accurate Estimation of
Information-Theoretic Quantities from Multi-Dimensional Sample Data",
Entropy 26(5), 387.  Orijinal kod: UNITE Toolbox (MIT lisansı).

Bu dosya, makalenin kNN tahmin edicilerinin TÜRKÇE AÇIKLAMALI ve TEK-DOSYA
halidir. Amaç: yüksek boyutlu (ör. 10D) örnek verisinden, olasılık yoğunluğunu
(PDF) AÇIKÇA tahmin etmeden bilgi-teorik nicelikleri hesaplamak.

Temel fikir: bir noktanın k. en yakın komşusuna olan uzaklığı (rho_k) yerel bir
yoğunluk ölçüsüdür. Yoğunluk seyrekse komşular uzaktır, yoğunsa yakındır.
Bütün yöntem KDTree üzerinde komşu sorgularına dayanır -> yüksek boyutta bile
hızlı ve uygulaması basit.

Bağımlılıklar: numpy, scipy.  (pip install unite-toolbox ile hazır halini de
kurabilirsiniz; bu dosya bağımsız çalışsın diye kopyalanmıştır.)
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.special import digamma, gamma

EPS = 1e-12


def vol_lp_ball(r: float, d: int, p_norm: float) -> float:
    """d boyutlu L^p topunun hacmi (formüllerdeki c_1(d) terimi).

    p_norm = 2 (Öklid) -> normal "yuvarlak" küre hacmi.
    p_norm = inf (Chebyshev/maksimum norm) -> hiperküp; (2r)^d ile çok hızlı.
    Entropi/KLD'de p=2, karşılıklı bilgide p=inf kullanılır.
    """
    if p_norm == np.inf:
        return (2 ** d) * (r ** d)
    a = (2 * gamma(1 / p_norm + 1)) ** d
    b = gamma(d / p_norm + 1)
    return (r ** d) * a / b


def calc_knn_entropy(data: np.ndarray, k: int = 1, p_norm: float = 2) -> float:
    """Kozachenko-Leonenko (1987) entropi tahmincisi [nats].

    H = psi(N) - psi(k) + log(c_1(d)) + (d/N) * sum_i log(rho_k(i))

    data : (N, d) dizi.  k : komşu sayısı (makale entropi için k=1 önerir).
    Her nokta için kendisi hariç k. komşuya uzaklık (rho_k) hesaplanır;
    bu yüzden ağaca k+1 komşu sorulur ve [:, k] indeksi alınır (ilk komşu
    noktanın kendisidir, uzaklığı 0).
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, d = data.shape
    knn_tree = KDTree(data)
    radius = knn_tree.query(data, k + 1, p=p_norm)[0][:, k]
    h = (
        digamma(n)
        - digamma(k)
        + np.log(vol_lp_ball(1.0, d, p_norm))
        + d * np.mean(np.log(radius))
    )
    return h


def calc_knn_kld(p: np.ndarray, q: np.ndarray, k: int = 1, p_norm: float = 2) -> float:
    """Wang et al. (2009) Kullback-Leibler ıraksaması tahmincisi  D_KL(p||q) [nats].

    D = (d/n) * sum_i log( nu_k(i) / rho_k(i) ) + log( m / (n-1) )

    rho_k(i): i. noktanın KENDİ kümesindeki (p) k. komşuya uzaklığı.
    nu_k(i) : i. noktanın DİĞER kümedeki (q) k. komşuya uzaklığı.
    p ve q'nun örnek sayıları (n, m) farklı olabilir; boyut (d) aynı olmalı.
    Negatif çıkarsa 0'a kırpılır (KLD >= 0 olmalı). Makale KLD için k=1 önerir.
    """
    n, m = len(p), len(q)
    d = len(p[0])
    rho, _ = KDTree(p).query(p, k + 1, p=p_norm)   # kendi kümesi -> k+1 (kendini at)
    nu, _ = KDTree(q).query(p, k, p=p_norm)        # diğer küme   -> k (kendisi yok)
    rho = rho.reshape(-1, k + 1)[:, -1]
    nu = nu.reshape(-1, k)[:, -1]
    kld = (d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1))
    return max(0.0, kld)


def calc_knn_mutual_information(x: np.ndarray, y: np.ndarray, k: int = 15) -> float:
    """Kraskov et al. (2004) karşılıklı bilgi (KSG) tahmincisi  I(X;Y) [nats].

    I = psi(N) + psi(k) - <psi(n_x+1) + psi(n_y+1)>

    - x, y aynı sayıda örneğe sahip; her biri >=1 boyutlu olabilir.
    - Ortak (x,y) uzayında MAKSİMUM norm (p=inf) ile k. komşu uzaklığı (radius) bulunur.
    - Sonra X ve Y alt-uzaylarında bu yarıçap içine düşen komşu sayıları (nx, ny)
      sayılır (query_ball_point). EPS, sınır noktalarını dışarıda tutmak için.
    - Makale karşılıklı bilgi için k=15 önerir.
    """
    assert len(x) == len(y), "x ve y aynı örnek sayısına sahip olmalı."
    n_samples = len(x)
    xy = np.hstack((x, y))
    xy_tree, x_tree, y_tree = KDTree(xy), KDTree(x), KDTree(y)
    radius = xy_tree.query(xy, k=[k + 1], p=np.inf)[0].flatten()
    nx = x_tree.query_ball_point(x, radius - EPS, p=np.inf, return_length=True)
    ny = y_tree.query_ball_point(y, radius - EPS, p=np.inf, return_length=True)
    return digamma(n_samples) + digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1))


# --------------------------------------------------------------------------
# Hızlı doğrulama (makaledeki 10D Gaussian referans değerleriyle):
#   H ~ 4.93 nats, D_KL ~ 7.00 nats, I ~ 1.10 nats
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy import stats

    d = 10
    mu = np.zeros(d)
    S1 = 0.9 * np.ones((d, d)); np.fill_diagonal(S1, 1.0)
    S2 = 0.1 * np.ones((d, d)); np.fill_diagonal(S2, 1.0)

    Xp = stats.multivariate_normal(mu, S1).rvs(size=50_000, random_state=1)
    Xq = stats.multivariate_normal(mu, S2).rvs(size=50_000, random_state=2)

    true_H = 0.5 * np.log((2 * np.pi * np.e) ** d * np.linalg.det(S1))
    print(f"Entropi : tahmin {calc_knn_entropy(Xp, k=1):.3f} | gerçek {true_H:.3f}")
    print(f"KLD     : tahmin {calc_knn_kld(Xp, Xq, k=1):.3f} | gerçek 7.00")
    print(f"MI(9;1) : tahmin {calc_knn_mutual_information(Xp[:, :9], Xp[:, 9:], k=15):.3f} | gerçek 1.10")
