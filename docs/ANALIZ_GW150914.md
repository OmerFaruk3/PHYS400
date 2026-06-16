# GW150914 — kNN tabanlı bilgi-teorik analiz

**Yöntem:** Álvarez Chaves et al. (2024, *Entropy* 26(5):387) makalesindeki kNN
tahmin edicileri (Kozachenko–Leonenko entropi, Kraskov/KSG karşılıklı bilgi).
Kod: `knn_estimators_reference.py` + `gw150914_knn_information.py`.

## Veri
- Dosya: `IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5` (GWTC-2.1)
- Kullanılan posterior: `C01:IMRPhenomXPHM` (147.634 örnek)
- **Tekilleştirme:** Veri yayını her örneği bir kez kopyalamış (iç parametreler
  aynı, mesafe/ra/zaman yeniden çizilmiş). Bu, alt-uzaylarda bire bir aynı
  noktalar üretip kNN uzaklığını sıfırlıyordu. Kopya bloğuna göre tekilleştirerek
  **71.747 benzersiz** örnek elde edildi; hız için 20.000'e alt-örneklendi.
- 15 standart bağımsız CBC parametresi: chirp_mass, mass_ratio, a_1, a_2,
  tilt_1, tilt_2, phi_12, phi_jl, luminosity_distance, theta_jn, ra, dec, psi,
  phase, geocent_time. (Her parametre z-skoruna standardize edildi.)

## Gruplama (en bağımsız / en az korele gruplara göre)
Algoritma korele parametreleri AYNI grupta toplar; böylece gruplar birbirinden
olabildiğince bağımsız olur. GW150914'te en güçlü korelasyonlar fizikseldir:

| çift | \|Spearman\| | yorum |
|------|------------|-------|
| ra – geocent_time | 0.97 | gökyüzü konumu ↔ varış zamanı gecikmesi |
| luminosity_distance – geocent_time | 0.71 | konum-mesafe bağı |
| luminosity_distance – ra | 0.70 | konum-mesafe bağı |
| tilt_1 – tilt_2 | 0.51 | spin yönelimleri |
| luminosity_distance – theta_jn | 0.42 | **klasik mesafe–eğim dejenerasyonu** |
| chirp_mass – mass_ratio | 0.30 | kütle parametreleri |

Geri kalan parametreler (kütleler, spin büyüklükleri, açılar) neredeyse
korelasyonsuz — GW150914 yüksek SNR'lı, iyi konumlanmış bir olay olduğundan
posterior büyük ölçüde ayrışıktır.

## Sonuçlar

### Birincil bölme: 10D + 5D  (gruplar arası ort. |korelasyon| = 0.037)
- **G1 (10D):** luminosity_distance, geocent_time, ra, psi, theta_jn,
  chirp_mass, mass_ratio, tilt_2, tilt_1, a_2 → H = **9.87 nats**
- **G2 (5D):** phi_jl, a_1, phi_12, dec, phase → H = **5.33 nats**
- **H(tüm 15D) = 14.84 nats = 21.41 bit**  ← *toplam bilgi içeriği (diferansiyel entropi)*
- **Toplam korelasyon C = Σ H(grup) − H(tüm) = 0.36 nats (0.52 bit)** ← gruplar arası paylaşılan bilgi
- Karşılıklı bilgi I(G1;G2) = 0.46 nats (0.66 bit) [Kraskov, k=15]

### İkincil bölme: 5D + 5D + 5D  (gruplar arası ort. |korelasyon| = 0.042)
- **G1:** luminosity_distance, geocent_time, ra, psi, theta_jn → H = 3.40 nats *(konum/mesafe bloğu)*
- **G2:** chirp_mass, tilt_1, tilt_2, mass_ratio, a_2 → H = 5.80 nats *(kütle + tilt)*
- **G3:** phi_jl, a_1, phi_12, dec, phase → H = 5.33 nats *(spin açıları)*
- H(tüm 15D) = 14.84 nats
- Toplam korelasyon C = −0.32 nats (≈0, tahmin hatası içinde — aşağıya bakın)
- I(G1;G2)=0.17, I(G1;G3)=0.90, I(G2;G3)=0.30 nats

## Yorum
1. **Toplam bilgi:** Posteriorun ortak diferansiyel entropisi ~**14.8 nats (21.4 bit)**
   (standardize birimlerde). Bu, 15 parametredeki toplam belirsizliğin/bilginin ölçüsüdür.
2. **Bağımsızlık:** Seçilen gruplar arası ortalama |korelasyon| ≈ 0.04 ve gruplar
   arası karşılıklı bilgi < 1 nat — yani 15 parametre ~bağımsız bloklara ayrışıyor.
   Toplam korelasyonun küçük olması (0.5 bite yakın) bunu doğrular.
3. **Estimatör yanlılığı uyarısı:** İki grup için teoride C = I(G1;G2) olmalı; burada
   0.36 (entropilerden) vs 0.46 (Kraskov) — farklı estimatörlerin farklı yanlılığı.
   5+5+5'te C'nin hafif negatif çıkması, makalede de belirtilen kNN entropi
   yanlılığının boyutla artmasından kaynaklanır. **Bu yüzden gruplar arası bağı
   ölçmek için Kraskov karşılıklı bilgisi (I) daha güvenilirdir; toplam entropi
   farkından gelen C yalnızca kaba bir göstergedir.**
4. **Fizik:** En çok bilgi paylaşan grup, gökyüzü-konumu/mesafe bloğudur
   (ra, geocent_time, luminosity_distance, theta_jn) — bilinen konum ve
   mesafe–eğim dejenerasyonlarını yansıtır.

## Çıktı dosyaları
- `results_gw150914_knn.json` — tüm sayısal sonuçlar
- `gw150914_correlation.png` — |Spearman| korelasyon ısı haritası
