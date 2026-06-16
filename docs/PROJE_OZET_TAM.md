# PHYS400 — GW Bilgi Teorisi Projesi · TAM ÖZET (Master Doküman)

> **Bu dosyanın amacı:** Tüm sohbetlerde tek kaynak. Neyin nerede olduğu, hangi
> kodun hangi çıktıyı ürettiği, hangi yöntemlerin kullanıldığı, artı/eksileri ve
> şu ana kadar bulunan tüm sonuçlar burada. Yeni bir sohbette **önce bunu oku**.
>
> Son güncelleme: 2026-06-09 · Güncelleyen analiz: 15D analitik Gaussian KL, 35 event.

---

## 1. Proje Amacı ve Bilimsel İddia

Kütleçekimsel dalga (GW) gözlemlerinden kaç **bit** bilgi kazanıldığını ölçmek:

```
I = D_KL( posterior(θ|s)  ‖  prior(θ) )   [bit]      (Flanagan & Hughes 1998, Eq. 7.1)
```

θ = 15 boyutlu kaynak parametre vektörü. Merkezî hedef: GW150914 (ve artık tüm
GWTC-2.1/3 BBH kataloğu) için **gerçek 15B joint KL**'yi hesaplayıp F&H 1998'in
~**41.5 bit** analitik tahminiyle karşılaştırmak.

**Ana iddia:** F&H'nin yaptığı *Gaussian posterior* varsayımının kaçırdığı
non-Gaussian / nonlineer katkıları, MCMC posteriorları üzerinden ölçmek. Yani
"Gaussian ne kadar iyi/kötü bir yaklaşım?" sorusu projenin kalbidir — bu yüzden
analitik Gaussian KL hesabı bir *referans/temel çizgi (baseline)* olarak kritik.

---

## 2. Veri

- **Konum:** `PHYS400/Data/` — IGWN GWTC PE Data Release `.h5` dosyaları.
- **Event sayısı:** **35** BBH (`*_mixed_cosmo.h5`). GW150914 (GWTC-2.1) + 34 GWTC-3.
  - `..._nocosmo.h5` bir kopya, kullanılmıyor (yalnız GW191103 için var).
- **Label (analiz grubu):** çoğunlukla `C01:IMRPhenomXPHM`; GW191219 için
  `C01:IMRPhenomXPHM:HighSpin`.
- **Posterior:** her event'te ~3k–250k örnek → `f[label]["posterior_samples"]`
- **Prior:** dosyada saklı ~5000 örnek → `f[label]["priors"]["samples"]`
- **Yükleme iki yolla:**
  - pesummary: `read(path, disable_conversion=True).samples_dict[label]`
  - doğrudan h5py (pesummary yoksa): `h5py.File(path)[label]["posterior_samples"]`
    — yapısal dizi; sütunlar `ps[param]`. Prior: `[label]["priors"]["samples"][param]`.

### 15 Parametre (F&H sırası)
```
mass_1_source, mass_2_source,        # kütleler [M☉]
a_1, a_2,                            # spin büyüklükleri [0,1]
tilt_1, tilt_2,                      # spin eğim açıları [rad]
phi_12, phi_jl,                     # azimuthal spin açıları [rad]
luminosity_distance,                # dL [Mpc]
theta_jn,                           # inclination [rad]
psi,                                # polarizasyon açısı [rad]
azimuth, zenith,                    # gökyüzü konumu (≡ ra/dec, izometrik)
geocent_time,                       # tc [GPS s]
phase                               # [rad]
```
Not: prior tablosunda `ra/dec` yok, `azimuth/zenith` var — matematiksel eşdeğer.

---

## 3. Klasör Haritası — Neyin Nerede Olduğu

```
PHYS400/
├── Data/                              ← 35 event .h5 (girdi)
├── Codes/                             ← ANA çalışma klasörü (çoğu güncel kod burada)
│   ├── PROJE_OZET_TAM.md              ← BU DOSYA
│   ├── gaussian kl                    ← tek-event analitik 15D Gaussian KL (orijinal)
│   ├── gaussian_kl_all_events.py      ← YENİ: 35 event, TÜM posterior noktaları
│   ├── gaussian_kl_all_events_matched.py  ← YENİ: 35 event, önceki pipeline ile AYNI posterior (dedup)
│   ├── gaussian_kl_diagnostics.py     ← YENİ: "neden Gaussian sapıyor?" tanılama
│   ├── 15 parameter total             ← 1D marjinal KDE KL (15 panel plot)
│   ├── kNN                            ← Pérez-Cruz 2008 k-NN KL tahmincisi
│   ├── MINE                           ← Mutual Information Neural Estimator (torch)
│   ├── fisher_eigen_info.py           ← Fisher özdeğer bilgi sıralaması
│   ├── covariance, 2D *, histogram 2D PLOT, deneme*  ← korelasyon/keşif denemeleri
│   ├── SNR_values, approximation formula, prior*, posterior*  ← yardımcı parçalar
│   └── *.png, *.json                  ← çıktı görseller/sonuçlar
├── claude_codes/                      ← grup-tabanlı & kNN ana boru hattı
│   ├── CLAUDE.md                      ← (eski) proje notları
│   ├── gw_knn_kl_divergence.py        ← full 15D kNN KL
│   ├── gw_knn_kl_convergence.py       ← kNN k=1..10 yakınsama testi
│   ├── gw150914_knn_information.py     ← kNN bilgi analizi (Álvarez-Chaves 2024)
│   └── grup_kld/
│       ├── gw_grup_kld_hibrit.py      ← ⭐ HİBRİT grup KLD + TC düzeltmesi (ana yöntem)
│       ├── gw_kld_hibrit_oto_katalog.py ← tüm kataloğu otomatik işleyen sürücü
│       ├── oto_master_ozet.csv        ← ⭐ ÖNCEKİ SONUÇLAR (34 event özet tablosu)
│       ├── results_grup_kld_hibrit_GW*.json ← her event ayrıntılı hibrit sonucu (+GW150914)
│       ├── {bin,kde,knn}_estimators_reference.py ← KL tahminci kütüphaneleri
│       ├── gw_mi_gruplama.py          ← MI tabanlı parametre gruplama (dendrogram/heatmap)
│       └── result_test_*.{py,csv}     ← yan analizler (surprise, model compare, snr)
└── (GW Information Theory/PHYS400/claude codes/grup_kld/)  ← erken sürümler
    ├── gw_1d_kld_analizi.py + results_1d_kld.json
    └── gw_grup_kld_analizi.py + results_grup_kld.json
```

---

## 4. Yöntemler — Her Biri Ayrıntılı (kod → çıktı, artı/eksi)

15B joint KL'yi tahmin etmek zor (boyut laneti). Proje boyunca denenen yöntemler:

### 4.1 Marjinal 1D KL (alt sınır)
- **Ne:** Her parametre için ayrı ayrı `KL(post_i ‖ prior_i)`, sonra topla.
  `I_joint = Σ I_marginal + TC`, TC = Total Correlation ≥ 0 ⇒ **marjinal toplam = alt sınır**.
- **Kod:** `Codes/15 parameter total` (KDE), `GW.../grup_kld/gw_1d_kld_analizi.py`
- **Çıktı:** `results_1d_kld.json`, `marginal_kl_15params.png`; özet tablodaki
  `marginal_1d_total_bits` sütunu (KDE-scott ile, **tüm posterior** üzerinden).
- **+** Basit, kararlı, hızlı. Her parametrenin katkısı görünür.
- **–** Korelasyonları (TC) tamamen atar → gerçeği olduğundan düşük gösterir.
- **GW150914:** ~36.9 bit.

### 4.2 Grup KLD (≤5B bloklar)
- **Ne:** 15 parametreyi düşük-korelasyonlu ≤5B gruplara böl, her grupta
  KDE/kNN ile KL hesapla, topla. Boyut lanetini grup başına azaltır.
- **Kod:** `grup_kld/gw_grup_kld_analizi.py` (+ `_v2`)
- **Çıktı:** `results_grup_kld.json`; özette `group_total_*_bits`.
- **+** Gruplar arası küçük korelasyonu ihmal ederek 15B'yi parçalara böler.
- **–** Gruplar arası korelasyonu kaçırır; grup seçimi öznel.
- **GW150914:** ~38.9 bit.

### 4.3 ⭐ HİBRİT Grup KLD + TC Düzeltmesi  (en güvendiğimiz "joint" tahmini)
- **Ne:** 4.2'nin üstüne (a) **hibrit prior** ve (b) **TC düzeltmesi** ekler.
  - **Hibrit prior:** posterior, saklı prior desteğini %0.5'ten fazla aşıyorsa o
    parametre için **analitik prior** (örn. dL, m1, m2), aksi halde orijinal
    prior bootstrap'i (N=30000). Bu, "prior örtüşme" artefaktını düzeltir.
  - **TC düzeltmesi:** `joint = grup_toplam + (TC_post − TC_prior)` — gruplar
    arası kalan korelasyonu ekler. `joint_kld_estimate_mean_bits`.
- **Kod:** `grup_kld/gw_grup_kld_hibrit.py`; tüm katalog: `gw_kld_hibrit_oto_katalog.py`
- **Çıktı:** **`oto_master_ozet.csv`** (34 event) + `results_grup_kld_hibrit_GW150914.json`.
  Önemli sütunlar: `joint_mean_bits`, `group_total_mean_bits`, `marginal_1d_total_bits`,
  `tc_correction_nats`, `between_group_abs_corr`, `hybrid_analytic_params`, `n_posterior`.
  Yöntemler: `kde-scott`, `kde-silverman`, `knn-k1` (ortalaması alınır).
  **Posterior:** tüm örnekler okunur, sonra **tekilleştirme** (tekrar eden satırlar atılır)
  → `n_posterior` bu yüzden ham sayıdan düşük (örn. GW150914 147634→71747).
- **+** Korelasyon + prior düzeltmesi içeren en gerçekçi tahmin. Tüm katalog için var.
- **–** KDE/kNN bant genişliği ve grup seçimine duyarlı; TC kendisi de tahmin.
- **GW150914:** **39.63 bit** (joint mean).

### 4.4 ⭐ Analitik Çok Değişkenli (15B) Gaussian KL  — BU ANALİZ
- **Ne:** Posterior ve prior'ı 15B Gaussian kabul edip kapalı-form KL:
  ```
  KL = ½[ tr(Σq⁻¹Σp) + (μq−μp)ᵀΣq⁻¹(μq−μp) − d + ln(detΣq/detΣp) ] / ln2   [bit]
  ```
  Tüm posterior noktalarıyla μ, Σ tahmin edilir. Prior = dosyanın orijinal prior'u.
- **Kod:**
  - `Codes/gaussian kl` — orijinal tek event.
  - `Codes/gaussian_kl_all_events.py` — 35 event, **TÜM** posterior noktaları.
  - `Codes/gaussian_kl_all_events_matched.py` — 35 event, hibrit ile **AYNI** posterior
    (aynı tekilleştirme adımı), `n_posterior` 35/35 eşleşir.
- **Çıktı:** `gaussian_kl_all_events.{csv,json}`, `..._matched.{csv,json}`,
  `gaussian_kl_comparison.png`, `gaussian_kl_comparison_matched.png`.
- **+** Kapalı-form, hızlı, kararlı (Cholesky ile makine hassasiyetinde doğrulandı),
  boyut lanetinden etkilenmez, tüm katalog için tek tipte baseline verir.
- **–** Gaussian varsayımı: non-Gaussian kuyrukları, çok-tepeli açısal dağılımları
  ve **nonlineer korelasyonları** kaçırır; saklı prior'ın yetersiz desteğine
  (dL, m2 örtüşme) çok duyarlı (bkz. §8).
- **GW150914:** **41.25 bit** (tüm nokta) / **41.30 bit** (eşleşmiş) — F&H ~41.5 ile uyumlu.
- **Sayısal doğrulama:** inverse-yöntem vs Cholesky/solve farkı ~1e-14 bit.

### 4.5 Full 15B k-NN KL (Pérez-Cruz 2008)
- **Ne:** Komşu mesafelerinden non-parametrik 15B KL. `m≪n` bias düzeltmesi gerekir.
- **Kod:** `Codes/kNN`, `claude_codes/gw_knn_kl_divergence.py`, `_convergence.py`,
  `gw150914_knn_information.py`; `results_gw150914_knn.json`.
- **+** Dağılım-bağımsız, teoride tutarlı.
- **–** 15B'de **ıraksıyor** (n: 5k→50k iken 25.9→54.9 bit), büyük varyans, prior
  örnek sayısı (5000) yetersiz. Tek başına güvenilmez.

### 4.6 MINE (Belghazi 2018, sinir ağı)
- **Ne:** Donsker-Varadhan ile KL'yi bir sinir ağıyla alttan kestir.
- **Kod:** `Codes/MINE` (torch).
- **–** d=1'de ~%9, d=5'te ~%34 hata; **d=15 için yetersiz**. Terk edildi.

### 4.7 Fisher Özdeğer Bilgi Sıralaması
- **Ne:** Prior ile beyazlatılmış Fisher matrisinin özdeğerleri; her özyön için
  `I_k = ½ log2(1+λ_k)`. Marjinal KLD'deki "geniş-prior artefaktını" çözer.
- **Kod:** `Codes/fisher_eigen_info.py` (+ `fisher_eigen_PLAN.md`).
- **Çıktı:** `fisher_eigen_GW150914.{json,png}`.
- **+** Prior-bağımsız yön sıralaması; hangi parametre kombinasyonu bilgi taşıyor görünür.
- **–** Yine Gaussian/lineer-manifold varsayımı.

### 4.8 MI Gruplama / 2D Korelasyon (yardımcı)
- **Kod:** `grup_kld/gw_mi_gruplama.py` (heatmap+dendrogram), `Codes/2D *`,
  `covariance`, `histogram 2D PLOT`. Grup KLD için grupları belirler.

### 4.9 histogramdd (terk)
- b=5 binde ~240 GB RAM → **imkansız**. Boyut lanetinin doğrudan kanıtı.

---

## 5. GW150914 — Tüm Yöntemler Karşılaştırması

| Yöntem | Sonuç (bit) | Durum / Not |
|---|---|---|
| F&H 1998 analitik | ~41.5 | Referans (Gaussian + lineer manifold) |
| **Gaussian 15B (bu analiz)** | **41.25 / 41.30** | ✅ F&H ile uyumlu; prior'a duyarlı |
| Hibrit joint (grup+TC) | 39.63 | ⭐ en gerçekçi tahmin |
| Grup toplamı | 38.90 | korelasyonsuz blok toplamı |
| Marjinal 1D (KDE) | 36.90 | alt sınır |
| full 15B kNN | 25.9→54.9 | ❌ ıraksıyor |
| MINE | — | ❌ d=15 yetersiz |

Sıralama beklendiği gibi: `marjinal ≤ grup ≤ joint ≲ Gaussian ≈ F&H`.

---

## 6. 15B Gaussian — Tüm Katalog Sonucu (35 event)

- **Çalıştırma:** `python3 gaussian_kl_all_events_matched.py` (Codes/ içinde).
- **Genel:** Gaussian 15B ile önceki **joint** tahmini arasında **Pearson r = 0.965**.
  Gaussian ortalama **2.3 bit daha düşük** (medyan −2.9). N_posterior 35/35 eşleşiyor.
- **Tüm-nokta vs eşleşmiş posterior:** ortalama fark sadece +0.016 bit (tekilleştirme
  KL'yi neredeyse değiştirmiyor — μ/Σ'yi az etkiliyor).
- **En yüksek bilgi:** GW191204_171526 (48.8/49.4), GW200202 (42.7/44.2), GW150914 (41.3).
- **En düşük:** GW200322 (6.3), GW200308 (7.4–7.5).
- **Dosyalar:** `gaussian_kl_all_events_matched.csv` (event başına N_raw, N_dedup,
  N_prev, match bayrağı, KL, önceki referanslar, delta).

---

## 7. Sonuç Dosyaları Haritası (kod → çıktı)

| Çıktı dosyası | Üreten kod | İçerik |
|---|---|---|
| `oto_master_ozet.csv` | `gw_kld_hibrit_oto_katalog.py` → `gw_grup_kld_hibrit.py` | 34 event hibrit joint/grup/marjinal |
| `results_grup_kld_hibrit_GW*.json` | `gw_grup_kld_hibrit.py` | event başına ayrıntı (TC, gruplar, …) |
| `results_grup_kld.json` / `results_1d_kld.json` | erken `gw_grup_kld_analizi.py` / `gw_1d_kld_analizi.py` | GW150914 grup / 1D |
| `results_gw150914_knn.json` | `gw150914_knn_information.py` | kNN bilgi |
| `gaussian_kl_all_events.{csv,json}` | `gaussian_kl_all_events.py` | 15B Gaussian, tüm nokta |
| `gaussian_kl_all_events_matched.{csv,json}` | `gaussian_kl_all_events_matched.py` | 15B Gaussian, eşleşmiş posterior |
| `gaussian_kl_diagnostics.json`, `..._param_aggregate.json` | `gaussian_kl_diagnostics.py` | neden-sapıyor tanıları |
| `gaussian_kl_comparison*.png`, `gaussian_kl_diagnostics.png`, `..._delta_explained.png` | yukarıdakiler | görseller |
| `fisher_eigen_GW150914.{json,png}` | `fisher_eigen_info.py` | Fisher özyön bilgisi |
| `marginal_kl_15params.png` | `15 parameter total` | 1D marjinal paneller |

---

## 8. ⭐ NEDEN GAUSSIAN SAPIYOR? — Ayrıntılı Bulgular

`gaussian_kl_diagnostics.py` ile 35 event × 15 parametre tanılaması. Gaussian KL
dört varsayıma dayanır; hangisinin nerede çöktüğü:

### KL'yi oluşturan terimler (hangisi baskın?)
- `term2` = **ortalama kayması** (μq−μp)ᵀΣq⁻¹(μq−μp): **en baskın ve en değişken**
  (ortalama 16, aralık 5–43 bit). Olaylar arası farkın çoğu buradan.
- `term4` = **hacim** ln(detΣq/detΣp): ortalama 21, aralık −0.2…40 (posterior priordan
  ne kadar küçüldü).
- `term1` = iz tr(Σq⁻¹Σp): ortalama 10 (varyans oranı).

### Varsayım 1 — "Saklı prior gerçek prior'u temsil eder" → **EN BÜYÜK SORUN**
- `luminosity_distance` saklı prior'u ≈ **[682, 10000] Mpc**, ama posterior düşük
  mesafede: GW150914'te **%99.9**, GW191204_171526'da **%66** posterior prior desteği
  DIŞINDA; üstelik posterior **ortalaması prior min'in altında**.
- `mass_2` posterioru da %4.7 (ort.), bazı eventte **%43** prior dışında.
- Sonuç: Gaussian, dL/m2 yönünde **yanlış (çok dar) bir prior varyansı** uyduruyor;
  kötü-koşullu Σq⁻¹ (koşul sayısı ~1e9) ile birleşince **term2 patlıyor**. Bu
  **gerçek bilgi değil, prior-tanımı artefaktı**.
- Hibrit yöntem tam da bunu "analitik prior" ile düzeltir → bu yüzden iki yöntem
  en çok **örtüşen (overflow) eventlerde** ayrışır.
- **Kanıt (event-düzeyi korelasyon):** delta(Gauss15B − joint) ile
  **m2 prior-dışı % → r=0.62**, Gaussian korelasyon terimi → r=0.41,
  dL prior-dışı % → r=0.33. Saf posterior non-Gaussianlığı (|kurtosis|) → r≈−0.08
  (yani olay-düzeyi farkı **non-Gaussianlık değil, prior tanımı** sürüyor).

### Varsayım 2 — "Posterior Gaussian" → kısmen çöker (etki orta)
- Aşırı basıklık (excess kurtosis; 0 = Gaussian): **dL 38, m2 24, zenith 27, m1 11**
  (ağır kuyruk + sivri tepe). `tc` çarpık (skew 1.2), sivri non-Gaussian tepe.
- 1D'de Gaussian vs KDE farkı (bit): Gaussian **fazla** sayıyor → dL −1.33, m2 −1.06,
  m1 −0.20; **az** sayıyor → tc +0.95, azimuth +0.41, theta_jn +0.12.
- Net etki olay başına büyük ölçüde **birbirini götürüyor** → Gaussian, KDE/kNN
  joint'in yalnız ~2 bit altında kalıyor.

### Varsayım 3 — "Prior Gaussian" → açıkça yanlış (ama düşük-bilgi yönlerde)
- Açısal priorlar neredeyse **düzgün (uniform)**: phi12, phiJL, psi, phase aşırı
  basıklığı ≈ −1.1 (platykurtik). Gaussian, [min,max] dışına olasılık atıp düz içi
  hafife alır. Neyse ki bu yönlerde bilgi küçük → KL etkisi sınırlı.

### Varsayım 4 — "Bağımlılık = lineer korelasyon" → çoğunlukla küçük, ara sıra patlar
- Gaussian yalnız Pearson korelasyonu modeller; m1–m2 (muz şekli), dL–theta_jn
  (yozlaşma) gibi **nonlineer** bağları kaçırır.
- Gaussian korelasyon terimi (15B − Σ1D) ortalama yalnız **+1.45 bit**, AMA
  kötü-koşullu kovaryansta patlıyor: **GW191204_171526 = +12 bit** → bu eventin
  Gaussian'ı joint'in 8.1 bit üstüne çıkmasının sebebi.

### Özet cümlesi
> Gaussian 15B, F&H ile çok uyumlu (GW150914 ≈ 41.3 vs 41.5) ve joint tahminle güçlü
> korelasyonlu (r≈0.97), ama **saf bir baseline**. Olaylar arası sapmanın baskın
> sebebi posteriorun non-Gaussianlığı **değil**, saklı prior'ın yetersiz desteği
> (özellikle `luminosity_distance` ve `mass_2`), ki bu term2'yi (ortalama kayması)
> ve kötü-koşullu kovaryansta korelasyon terimini istikrarsız kılar. Hibrit yöntem
> bu prior sorununu analitik prior ile çözdüğü için "doğru" referanstır.

---

## 9. Bilinen Tuzaklar / Dikkat

- **dL & m2 prior örtüşmesi:** saklı prior posterioru kapsamaz → Gaussian/kNN'de
  artefakt. Hibrit prior (analitik) kullan.
- **Koşul sayısı ~1e9:** Σ prior kötü-koşullu; KL'yi `np.linalg.solve`/Cholesky ile
  hesapla (inverse yerine) — sonuç aynı (1e-14), ama daha güvenli.
- **kNN `m≪n`:** bias düzeltmesi `log2(m/n)` büyük negatif & yüksek varyans.
- **`geocent_time`:** ~1e9 GPS; padding aralık-tabanlı olmalı, mutlak değil.
- **Negatif TC imkansız** → Gaussian joint, gerçeğin bir *alt sınırı* sayılmaz ama
  TC≥0 nedeniyle `marjinal ≤ joint`.
- **Tekilleştirme:** hibrit boru hattı tekrar eden posterior satırlarını atar;
  Gaussian'ı eşleştirmek için aynı adım `..._matched.py` içinde.
- **1D KDE marjinalleri zaten hesaplı:** `oto_master_ozet.csv → marginal_1d_total_bits`
  (KDE-scott, tüm posterior). Yeniden hesaplamaya gerek yok; tanılamada per-parametre
  kırılım için alt-örneklemeli KDE kullanıldı (event toplamı için master CSV esastır).

---

## 10. Referanslar

- **[F&H]** Flanagan & Hughes (1998), PRD 57, 4566 — Böl. 7, App. B (~41.5 bit).
- **[PC08]** Pérez-Cruz (2008), NeurIPS — k-NN KL, Eq. 14.
- **[MINE]** Belghazi et al. (2018), ICML, arXiv:1801.04062.
- **[GWTC]** Abbott et al. (2023), PRX 13, 011048.
- **[Álvarez-Chaves]** (2024), Entropy 26(5):387 — kNN bilgi tahmincileri.

---

## 11. Sonraki Adımlar (öneri)

- [ ] Gaussian'ı **hibrit prior** ile yeniden koş (dL/m2/m1 analitik) → prior
      artefaktını ayıkla, kalan farkı saf non-Gaussianlığa indir.
- [ ] Katalog geneli: Gaussian vs joint farkını `between_group_abs_corr` ve
      SNR ile ilişkilendir (yüksek-SNR → daha Gaussian beklenir).
- [ ] Faz 2: BBH popülasyon bilgisi (GWTC-3 geneli) — şimdi 35 event hazır.
```
