"""
KDE (çekirdek yoğunluk kestirimi) tabanlı bilgi-teorik tahmin ediciler
=======================================================================

Kaynak: Álvarez Chaves et al. (2024), Entropy 26(5), 387.
Orijinal kod: UNITE Toolbox (MIT), unite_toolbox/kde_estimators.py.

Tek-dosya, Türkçe açıklamalı; knn_estimators_reference.py ile aynı tarzda.
Bağımlılıklar: numpy, scipy.

Temel fikir: dağılımı Gauss çekirdekli KDE ile pürüzsüzce tahmin et, sonra
entropi/KLD'yi ÖRNEK NOKTALARINDA yeniden-yerine-koyma (resubstitution) ile
ortalamadan hesapla. Yüksek boyutta integral yerine resubstitution kullanılır
(makaledeki yaklaşım); bant genişliği "scott" veya "silverman" ile seçilir.
"""

import numpy as np
from scipy.stats import gaussian_kde


def _as2d(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1, 1) if a.ndim == 1 else a


def calc_kde_entropy(data: np.ndarray, bandwidth=None) -> float:
    """KDE ile (ortak) diferansiyel entropi [nats], resubstitution.

    H = -<log p(x_i)>,  p: data'nın Gauss-KDE yoğunluğu.
    bandwidth: None (scipy varsayılan=scott), "silverman" veya bir skaler.
    """
    data = _as2d(data)
    kde = gaussian_kde(data.T, bw_method=bandwidth)
    p = kde.evaluate(data.T)
    return -1.0 * np.mean(np.log(p))


def calc_kde_kld(p: np.ndarray, q: np.ndarray, bandwidth=None) -> float:
    """KDE ile Kullback-Leibler ıraksaması  D_KL(p||q) [nats], resubstitution.

    p ve q ayrı ayrı KDE ile modellenir; ıraksama D(p||q) tanımı gereği p
    ÖRNEKLERİNDE (beklenen değer P altında) değerlendirilir:
        D = < log( p_kde(p_i) / q_kde(p_i) ) >_i
    Bu yön bin/knn estimatörleriyle TUTARLIDIR. (UNITE toolbox'ın orijinali
    beklenen değeri q üzerinden alıp |.| uyguluyordu; bu, posterior priordan
    çok dar olduğunda yanlış yöne kayıp patladığından burada düzeltilmiştir.)
    p ve q farklı sayıda örneğe sahip olabilir; boyut (d) aynı olmalı.
    Negatifse 0'a kırpılır. bandwidth: None=scott, "silverman" veya skaler.

    NOT: KDE resubstitution KLD'si boyut arttıkça (yaklaşık d>=4) yukarı doğru
    yanlılık gösterir (makalede belgelenmiştir). ≤3D'de güvenilir; daha yüksek
    boyutta kNN'i referans alın.
    """
    p = _as2d(p)
    q = _as2d(q)
    p_kde = gaussian_kde(p.T, bw_method=bandwidth)
    q_kde = gaussian_kde(q.T, bw_method=bandwidth)
    pi = p_kde.evaluate(p.T)
    qi = q_kde.evaluate(p.T)
    mask = (pi > 0) & (qi > 0)
    kld = np.mean(np.log(pi[mask] / qi[mask]))
    return max(0.0, kld)


# --------------------------------------------------------------------------
# Hızlı doğrulama: 4D Gaussian KLD (analitik referansla)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy import stats

    d = 4
    mu = np.zeros(d)
    S1 = 0.6 * np.ones((d, d)); np.fill_diagonal(S1, 1.0)
    S2 = 0.2 * np.ones((d, d)); np.fill_diagonal(S2, 1.0)
    P = stats.multivariate_normal(mu, S1).rvs(20_000, random_state=1)
    Q = stats.multivariate_normal(mu, S2).rvs(20_000, random_state=2)
    true = 0.5 * (np.log(np.linalg.det(S2) / np.linalg.det(S1))
                  + np.trace(np.linalg.inv(S2) @ S1) - d)
    for bw in (None, "silverman"):
        est = calc_kde_kld(P, Q, bw)
        print(f"KLD kde-{str(bw):9s}: tahmin {est:.3f} | analitik {true:.3f} nats")
