# GW olaylarından bilgi kazancı — kNN KL ıraksaması (bit)

> Bu dosya başka bir sohbette devam edebilmek için durum kaydıdır.
> Önceki `ANALIZ_GW150914.md` (entropi / total-correlation yaklaşımı) **geçersizdir** —
> bizim hesapladığımız nicelik **KL(posterior‖prior) = bilgi kazancı**dır.

## Amaç
Bir GW olayının gözleminden parametreler hakkında kazanılan **toplam bilgiyi (bit)**
hesaplamak:  `I = D_KL(posterior ‖ prior)`.
- GW150914 referansları: Flanagan & Hughes (1998) ~**41.5 bit**, Gaussian ortak yaklaşım ~**41.24 bit**.

## Yöntem — makaledeki estimator
Álvarez Chaves et al. (2024, *Entropy* 26(5):387) toolbox'undaki `calc_knn_kld` ile
**birebir aynı** estimator = Wang+2009 = Pérez-Cruz 2008 (NeurIPS, Denk. 14):

```
D_KL(P‖Q) = (d/n) · Σ_i log( ν_k(i) / ρ_k(i) ) + log( m / (n−1) )
```
- ρ_k(i): x_i'nin **posterior** içindeki k. komşuya uzaklığı (kendisi hariç)
- ν_k(i): x_i'nin **prior** içindeki k. komşuya uzaklığı
- d=15, n=posterior, m=prior; p-norm=2 (Öklid); makale KLD için **k=1** önerir.
- Doğal log (nats) → **bit = nat/ln2**. Sonuç bit cinsinden verilir.

## Proje konvansiyonları (mevcut `Codes/` ile tutarlı)
- Veri: `Data/IGWN-GWTC2p1-v2-GW150914_..._mixed_cosmo.h5`, etiket **`C01:IMRPhenomXPHM`**
- 15 F&H parametresi (kaynak-çerçevesi kütleler + dedektör-çerçevesi gökyüzü):
  `mass_1_source, mass_2_source, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
  luminosity_distance, azimuth, zenith, theta_jn, psi, geocent_time, phase`
  → türetilmiş `chirp_mass/mass_ratio` ve `ra/dec` KULLANILMAZ (prior dosyası
  azimuth/zenith saklar; bu seçim mevcut kodlarınızla aynı).
- Standardizasyon: prior referanslı z-skor (KL affine-değişmez; sadece sayısal).
- Okuma h5py ile (pesummary gerekmez), yalnızca gereken 15 alan okunur (137 alanı
  okumak yükleme süresini patlatıyordu).

## ⚠️ Kritik bulgu: yeniden-örnekleme kopyaları
`mixed_cosmo` posterioru **147.634 satır** içerir ama bunların yalnızca **71.747'si
istatistiksel olarak bağımsızdır**. Her örnek bir kez kopyalanmış: redshift'ten
bağımsız iç parametreler (a_1,a_2,tilt_1,tilt_2,phi_12,phi_jl,theta_jn,psi,phase)
ikizler arasında **birebir aynı**, yalnızca dL/gökyüzü/zaman yeniden çizilmiş.

Bu kopyalar k-NN'in iid varsayımını bozar: **k=1'de en yakın komşu örneğin "ikizi"
olur → ρ≈0 → KL yapay olarak şişer.** Ham 147k ile sonuçlar:
`k=1 → 100.8 bit, k=3 → 48.2, k=5 → 29.5` (güçlü k-bağımlılığı = artefakt işareti).

**Çözüm:** ikiz-özdeş sütun bloğuna göre tekilleştirme (`dedup_resampling_copies`,
otomatik tespit). Sonra **71.747 gerçek bağımsız örnek** ile sonuç k'dan bağımsız kararlı:

| k | I (bit) |
|---|---------|
| **1** | **29.41** |
| 3 | 27.83 |
| 5 | 27.34 |

## Sonuç (GW150914)
**I = D_KL(posterior‖prior) ≈ 29.4 bit** (k=1, 15D, 71.747 bağımsız örnek, prior 5.000).

### Neden Gaussian/F&H'nin (~41 bit) altında?
1. **Seyrek prior (m=5.000):** 15B'de prior çok seyrek örneklenmiş; bias terimi
   −3.84 bit ve ν_k uzaklıkları yüksek varyanslı. Asıl sınırlayıcı budur.
2. **Boyut laneti:** Makale de 10B'de kNN KLD'nin yeterli örnek yoksa düşük
   tahmin verdiğini gösteriyor (k=1 aşağı yönlü, daha çok örnekle yakınsar).
3. **Gaussian 41.24 yalnızca bir yaklaşımdır** (dağılımları Gaussian sayar);
   kNN gerçek (Gaussian-olmayan) dağılımları kullanır.
→ Gerçek değer büyük olasılıkla **~29 (kNN, örnek-sınırlı) ile ~41 (Gaussian) arası**.

## Çalıştırma
```
python gw_knn_kl_divergence.py            # varsayılan: GW150914
python gw_knn_kl_divergence.py <olay.h5>  # başka olay (aynı format)
```
Çıktı: konsol + `kl_knn_<olay>.json`. ~10 sn (yükleme dahil), tüm gerçek örnekler.

## Sonraki adımlar / yapılacaklar
- [ ] Diğer GWTC olayları için aynı scripti çalıştırıp olay-başına bit tablosu.
- [ ] Prior örnek sayısını artırma yolu (prior'dan yeniden örnekleme) → kNN yakınsaması.
- [ ] İstenirse: marjinal KL toplamı (mevcut `15 parameter total`) ile karşılaştırma;
      fark ≈ parametreler arası bağımlılık (total correlation).
- [ ] k-NN'i `unite_toolbox`'un `calc_knn_kld`'siyle çapraz-doğrulama (aynı formül).

## Dosyalar (`claude_codes/`)
- `gw_knn_kl_divergence.py` — ana script (estimator + yükleme + tekilleştirme)
- `kl_knn_<olay>.json` — sayısal sonuçlar
- `knn_estimators_reference.py` — makalenin estimatörlerinin açıklamalı tek-dosya hali
