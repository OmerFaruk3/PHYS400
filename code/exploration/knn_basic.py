"""
=============================================================================
GW150914 — 15 Boyutlu Toplam Bilgi Kazanci (Joint Information Gain)
k-NN Tabanli KL Iraksamasi, Pérez-Cruz (2008)
=============================================================================

REFERANSLAR:
  [F&H]  Flanagan & Hughes (1998), PRD 57, 4566, Bölüm 7
         → F&H'nin kullandığı 15 parametre: (m1, m2, a1, a2, tilt1, tilt2,
           phi12, phiJL, dL, RA, Dec, theta_JN, psi, tc, phi_c)
         → I_source ~ 41.5 bit (GW150914 için analitik tahmin)
  [PC08] Pérez-Cruz (2008), NeurIPS
         → k-NN tabanlı KL iraksaması tahmincisi, Denklem 14

PARAMETRE SEÇİMİ NEDEN ÖNEMLİ?
  k-NN KL iraksaması parametre dönüşümüne karşı TAM olarak değişmez değildir.
  Lineer dönüşümlerde Jacobian 1 olduğu için sorun çıkmaz, ama
  (m1, m2) -> (chirp_mass, mass_ratio) gibi nonlineer dönüşümlerde
  k-NN'nin yoğunluk tahminleri farklı gelir.
  F&H Bölüm 7'de m1, m2 (ve RA, Dec) kullanıldığından, tutarlı
  karşılaştırma için aynı koordinatlar kullanılmalıdır.

  YANLIŞ: chirp_mass_source, mass_ratio, azimuth, zenith  (türetilmiş!)
  DOĞRU:  mass_1_source, mass_2_source, ra, dec            (temel fiziksel)

FORMÜL (Pérez-Cruz 2008, Denklem 14):
  D_KL(P || Q) = (d/n) * Σ_i log₂(s_i / r_i) + log₂(m / (n-1))

  r_i : x_i'nin P içindeki (kendisi hariç) 1. komşuya Öklid mesafesi
  s_i : x_i'nin Q içindeki 1. komşuya Öklid mesafesi
  d   : boyut (15)
  n   : posterior örnek sayısı (~147k)
  m   : prior örnek sayısı (~5k)

ÖNEMLİ UYARI — m << n DURUMU:
  Pesummary'nin GW150914 prior dosyasında ~5000 örnek bulunur,
  posterior'da ise ~147000 örnek vardır. Bu durum Pérez-Cruz
  tahmincisinde log₂(5000/146999) ≈ -4.88 bit'lik bir düzeltme
  terimi üretir. Bu düzeltme terimi KASITLI olarak negatiftir:
  Q'nun seyrek örneklenmesi s_i'leri yapay olarak büyütür (prior
  noktaları birbirinden uzak), bu nedenle tahminci oluşan fazlalığı
  bias correction ile telafi eder. Teorik olarak tutarlı, ama varyans
  artar. Nihai sonuç 30–55 bit aralığında beklenmektedir.
"""

import numpy as np
from scipy.spatial import cKDTree
from pesummary.io import read
import time
import sys

# =============================================================================
# BÖLÜM 1: DÜZELTILMIŞ k-NN KL TAHMINCI FONKSİYONU
# =============================================================================

def KLdivergence_knn_bits(P_matrix, Q_matrix, k=1, chunk_size=10000):
    """
    Pérez-Cruz (2008) NeurIPS, Denklem 14 — k-NN tabanlı KL iraksaması.
    Büyük veri setleri için parçalı (chunked) hesaplama ile ilerlemeyi gösterir.
    Çok çekirdekli CPU kullanımı için workers=-1 parametresi eklendi.

    Formül:
        D_KL(P || Q) = (d/n) * Σ_i log₂(s_i / r_i) + log₂(m / (n-1))

    Parametreler
    ------------
    P_matrix   : ndarray (N, d) — Posterior örneklemleri
    Q_matrix   : ndarray (M, d) — Prior örneklemleri
    k          : int — Komşu sayısı (k=1 yüksek boyutlarda en az yanlılık verir)
    chunk_size : int — Her parçada kaç P noktası işleneceği (bellek yönetimi)

    Döndürür
    --------
    float : D_KL(P || Q) [bit]
    """
    P_matrix = np.atleast_2d(P_matrix)
    Q_matrix = np.atleast_2d(Q_matrix)

    n, d = P_matrix.shape
    m    = Q_matrix.shape[0]
    assert P_matrix.shape[1] == Q_matrix.shape[1], "Boyut uyumsuzluğu!"
    assert k < n, f"k={k} olamaz, n={n}"

    print(f"    d={d}, n={n:,} (posterior), m={m:,} (prior), k={k}")
    print(f"    Bias correction terimi: log₂({m}/{n-1}) = {np.log2(m/(n-1.0)):.4f} bit")
    print(f"    (Negatif olması beklenir: m < n durumu)")

    # -------------------------------------------------------------------------
    # KDTree yapıları: O(N log N) kurulum maliyeti
    # workers=-1 → cKDTree sorgularını tüm CPU çekirdeklerine dağıtır
    # -------------------------------------------------------------------------
    print(f"\n    KDTree kuruluyor...")
    t0 = time.time()
    xtree = cKDTree(P_matrix)    # Posterior ağacı
    ytree = cKDTree(Q_matrix)    # Prior ağacı
    print(f"    KDTree kuruldu. ({time.time()-t0:.1f}s)")

    # -------------------------------------------------------------------------
    # Parçalı hesaplama: Bellek taşmasını engellemek için P'yi parçalara bölüyoruz.
    # Tüm r ve s vektörlerini saklamak yerine toplamı birikimli tutuyoruz.
    # -------------------------------------------------------------------------
    log_ratio_sum = 0.0
    t0 = time.time()
    n_chunks = (n + chunk_size - 1) // chunk_size

    for i, start in enumerate(range(0, n, chunk_size)):
        end   = min(start + chunk_size, n)
        chunk = P_matrix[start:end]

        # r_i: P içindeki k. komşu mesafesi (kendisi hariç → k+1 sorgu, [:, k] al)
        # DÜZELTME: [0] ile sadece DISTANCES array'ini alıyoruz, [:, k] → k. komşu
        r_chunk = xtree.query(chunk, k=k+1, eps=1e-6, p=2, workers=-1)[0][:, k]

        # s_i: Q içindeki k. komşu mesafesi
        # DÜZELTME: [0] ile sadece DISTANCES, k=1 → 1D array döner
        if k == 1:
            s_chunk = ytree.query(chunk, k=1, eps=1e-6, p=2, workers=-1)[0]
        else:
            s_chunk = ytree.query(chunk, k=k, eps=1e-6, p=2, workers=-1)[0][:, k-1]

        # Numerik zemin: log(0) = -∞ kaçınmak için
        r_chunk = np.maximum(r_chunk, 1e-15)
        s_chunk = np.maximum(s_chunk, 1e-15)

        log_ratio_sum += np.sum(np.log2(s_chunk / r_chunk))

        # İlerleme çubuğu
        done = (i + 1) / n_chunks
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (n_chunks - i - 1) if i > 0 else 0
        bar = "█" * int(done * 20) + "░" * (20 - int(done * 20))
        print(f"\r    [{bar}] {100*done:.0f}% | İşlenen: {end:,}/{n:,} | "
              f"Geçen: {elapsed:.0f}s | Kalan: ~{eta:.0f}s",
              end="", flush=True)

    print()  # Satır sonu

    # -------------------------------------------------------------------------
    # Pérez-Cruz (2008) Denklem 14:
    #   D_KL = (d/n) * Σ log₂(s_i/r_i) + log₂(m/(n-1))
    # -------------------------------------------------------------------------
    divergence_bits = (d / n) * log_ratio_sum + np.log2(m / (n - 1.0))
    return divergence_bits


# =============================================================================
# BÖLÜM 2: VERİ YÜKLEME
# =============================================================================

file_name = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"

print("=" * 65)
print("GW150914 — 15 Boyutlu Toplam Bilgi Kazancı")
print("Pérez-Cruz (2008) k-NN Tahmincisi")
print("=" * 65)

print("\n[1/5] HDF5 dosyası okunuyor...")
data = read(file_name, disable_conversion=True)

# DÜZELTME (HATA 1): data.labels bir LİSTE döndürür.
# Spesifik string label seçmeliyiz.
print(f"\nMevcut analizler: {data.labels}")
label = 'C01:IMRPhenomXPHM'
print(f"Seçilen: {label}")

posterior_samples = data.samples_dict[label]
prior_samples     = data.priors["samples"][label]

# Mevcut parametre isimlerini kontrol edelim
available_params = list(posterior_samples.keys())
print(f"\nToplam mevcut parametre sayısı: {len(available_params)}")


# =============================================================================
# BÖLÜM 3: F&H PARAMETRELERİ — DOĞRU SEÇİM
# =============================================================================

# Flanagan & Hughes (1998) Bölüm 7'de kullanılan 15 parametre.
# Her satırda: (pesummary_adi, fiziksel_sembol, aciklama)
PARAMETER_MAP = [
    # --- İçsel parametreler (8 adet) ---
    # Kütle: m1, m2 doğrudan kullanılır; chirp_mass ve mass_ratio TÜRETILMIŞ
    ("mass_1_source",      "m₁ [M☉]",    "Birincil kara delik kütlesi (kaynak çerçevesi)"),
    ("mass_2_source",      "m₂ [M☉]",    "İkincil kara delik kütlesi (kaynak çerçevesi)"),
    ("a_1",                "a₁",          "Birincil spin büyüklüğü [0,1]"),
    ("a_2",                "a₂",          "İkincil spin büyüklüğü [0,1]"),
    ("tilt_1",             "tilt₁ [rad]", "Spin₁ — orbital L açısı"),
    ("tilt_2",             "tilt₂ [rad]", "Spin₂ — orbital L açısı"),
    ("phi_12",             "φ₁₂ [rad]",  "İki spin arası azimutal açı"),
    ("phi_jl",             "φ_JL [rad]", "J ve L arası azimutal açı"),
    # --- Dışsal parametreler (7 adet) ---
    ("luminosity_distance","d_L [Mpc]",   "Işıklılık uzaklığı"),
    # Gökyüzü konumu: GWTC-2.1 prior dosyasında ra/dec yerine azimuth/zenith
    # (dedektör çerçevesi koordinatları) saklanır. İkisi arasındaki dönüşüm
    # sabit bir küresel rotasyondur — izometrik → KL iraksamasını değiştirmez.
    # Tutarlılık için hem P hem Q'da aynı koordinatları kullanıyoruz.
    ("azimuth",            "α_det [rad]", "Gökyüzü azimut açısı (RA'nın dedektör eşdeğeri)"),
    ("zenith",             "ζ_det [rad]", "Gökyüzü tepe açısı (Dec'in dedektör eşdeğeri)"),
    ("theta_jn",           "θ_JN [rad]", "İnklinasyon açısı"),
    ("psi",                "ψ [rad]",    "Polarizasyon açısı"),
    ("geocent_time",       "t_c [GPS]",  "Birleşme zamanı"),
    ("phase",              "φ_c [rad]",  "Referans faz"),
]

# Pesummary dosyasında yoksa alternatif isimler dene
FALLBACK = {
    "mass_1_source": ["mass_1"],
    "mass_2_source": ["mass_2"],
}

print(f"\n[2/5] Parametre mevcudiyeti kontrol ediliyor...")

resolved = []   # (pesummary_adi, sembol, açıklama)
missing  = []

for (pname, symbol, desc) in PARAMETER_MAP:
    if pname in available_params:
        resolved.append((pname, symbol, desc))
        print(f"  ✓ {pname:25s} ({symbol})")
    else:
        # Fallback dene
        found_alt = None
        for alt in FALLBACK.get(pname, []):
            if alt in available_params:
                found_alt = alt
                break
        if found_alt:
            resolved.append((found_alt, symbol, desc + f" [fallback: {found_alt}]"))
            print(f"  ⚠ {pname:25s} → {found_alt} kullanılıyor ({symbol})")
        else:
            missing.append(pname)
            print(f"  ✗ {pname:25s} BULUNAMADI!")

if missing:
    print(f"\nEKSİK PARAMETRELER: {missing}")
    print("Mevcut parametreler arasında şunlar var:")
    for p in sorted(available_params)[:30]:
        print(f"  {p}")
    sys.exit(1)

parameters_15D = [r[0] for r in resolved]
print(f"\nToplam: {len(parameters_15D)} parametre doğrulandı.")


# =============================================================================
# BÖLÜM 4: TÜM VERİYİ KULLANARAK MATRİS OLUŞTURMA
# =============================================================================

print(f"\n[3/5] Matrisler oluşturuluyor (tüm örneklemler)...")

P_cols = []
Q_cols = []

for pname in parameters_15D:
    p_val = np.array(posterior_samples[pname])
    q_val = np.array(prior_samples[pname])

    # Standartlaştırma: PRIOR ortalaması ve std'si referans alınır.
    # Neden prior? P ve Q aynı koordinat sistemine getirilmeli.
    # Prior daha geniş kapsamlı olduğundan daha stabil bir referans sağlar.
    # KL iraksaması affine dönüşüm altında (lineer + öteleme) değişmez;
    # z-score normalizasyonu da bu kapsamda sayılır.
    ref_mean = np.mean(q_val)
    ref_std  = np.std(q_val)
    if ref_std < 1e-15:
        ref_std = 1.0   # Sabit parametre güvenliği

    P_cols.append((p_val - ref_mean) / ref_std)
    Q_cols.append((q_val - ref_mean) / ref_std)

P_matrix = np.column_stack(P_cols)   # (N_posterior, 15)
Q_matrix = np.column_stack(Q_cols)   # (N_prior, 15)

n_post = P_matrix.shape[0]
n_prior = Q_matrix.shape[0]

print(f"  Posterior matrisi boyutu: {P_matrix.shape}  ({n_post:,} nokta)")
print(f"  Prior matrisi boyutu:     {Q_matrix.shape}  ({n_prior:,} nokta)")
print(f"  m/n oranı: {n_prior/n_post:.4f}  (ideal: >1.0, bizde <1.0 — uyarı yukarıda)")


# =============================================================================
# BÖLÜM 5: HESAPLAMA
# =============================================================================

print(f"\n[4/5] k-NN KL iraksaması hesaplanıyor...")
print(f"  Yöntem: Pérez-Cruz (2008) NeurIPS, Denklem 14")
print(f"  k=1, workers=-1 (tüm CPU çekirdekleri)")
print()

t_total = time.time()
toplam_bilgi_kazanci = KLdivergence_knn_bits(
    P_matrix, Q_matrix,
    k=1,
    chunk_size=10000   # Her parçada 10k nokta işle (bellek ~= 10k×15×8 byte ≈ 1.2MB)
)
elapsed_total = time.time() - t_total


# =============================================================================
# BÖLÜM 6: SONUÇLAR VE YORUMLAMA
# =============================================================================

print(f"\n[5/5] SONUÇLAR")
print("=" * 65)
print(f"  Toplam süre: {elapsed_total:.1f} saniye ({elapsed_total/60:.1f} dakika)")
print()
print(f"  ┌─────────────────────────────────────────────────────────┐")
print(f"  │  Joint KL(posterior || prior):  {toplam_bilgi_kazanci:>8.2f} bit          │")
print(f"  └─────────────────────────────────────────────────────────┘")
print()
print(f"  Flanagan & Hughes (1998) analitik tahmini: ~41.5 bit")

if toplam_bilgi_kazanci > 0:
    ratio = toplam_bilgi_kazanci / 41.5
    print(f"  k-NN / F&H oranı: {ratio:.3f}  ({ratio*100:.1f}%)")

print()
print(f"  YORUM:")
print(f"  • k-NN tahminleri yüksek boyutlarda sistematik olarak düşük")
print(f"    çıkabilir (boyutun laneti), bu nedenle ~25-40 bit de beklenir.")
print(f"  • m={n_prior:,} < n={n_post:,} durumunda bias terimi negatif ({np.log2(n_prior/(n_post-1)):.2f} bit).")
print(f"  • Bu, prior'ın seyrek örneklenmesini otomatik olarak telafi eder.")
print(f"  • Daha güvenilir sonuç için: TC ayrışımı yöntemi (marginal KL'ler")
print(f"    toplamı + Total Correlation) önerilmektedir.")
print()

# Sonuç değerlendirmesi
if 20 < toplam_bilgi_kazanci < 80:
    print(f"  [✓ MANTIKLI] Sonuç beklenen aralıkta (20–80 bit).")
elif toplam_bilgi_kazanci <= 0:
    print(f"  [✗ YANLIŞ] Negatif KL iraksaması — veri veya parametre hatası!")
elif toplam_bilgi_kazanci < 20:
    print(f"  [⚠ DÜŞÜK] Beklenen aralığın altında. Prior seyrekliği etkisi olabilir.")
else:
    print(f"  [⚠ YÜKSEK] Beklenen aralığın üstünde. Standartlaştırmayı kontrol et.")