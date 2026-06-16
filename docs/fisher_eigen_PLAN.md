# Fisher Özdeğer Bilgi Sıralaması — Araştırma, Zorluk ve Plan

**Amaç:** GW olaylarının bilgi içeriğini, prior seçiminden olabildiğince bağımsız biçimde,
*parametre kombinasyonları* (özyönler) bazında sıralamak. Çıkış noktası senin formülün:

$$ I \;=\; \tfrac{1}{2}\sum_k \ln\!\big(1+\lambda_k\big) $$

Burada $\lambda_k$, her olayın posterior kovaryansından çıkarılan **Fisher özdeğerleridir**.
Bu, marjinal KLD sıralamasındaki "mesafe (luminosity_distance) artefaktını" ortadan
kaldırır ve **chirp-kütlesi yönünün baskınlığını doğrudan görünür kılar**.

> Bu döküman yanında çalışan bir prototip (`fisher_eigen_info.py`) ve GW150914 üzerinde
> alınmış gerçek sonuçlar var. Aşağıdaki sayılar o koşudan.

---

## 1. Neden bu? Mevcut yöntemin sorunu

Şu an bilgi içeriğini iki şekilde ölçüyorsun:

- **Marjinal KLD** (`15 parameter total`): her parametre için ayrı ayrı
  $D(p_i\|q_i)$. Toplam ~birkaç bit. Sorun: bir parametrenin "bilgisi" **prior
  genişliğine** bağlı. `luminosity_distance` priorı çok geniş (hacim ağırlıklı
  $\propto d_L^2$) olduğundan, posterior orta darlıkta bile **yapay olarak yüksek**
  KLD verir. Oysa mesafe aslında *kötü* ölçülür (eğim–mesafe dejenerasyonu). Bu
  **"mesafe artefaktı"**.
- **Çok-değişkenli Gauss KLD** (`gaussian kl`): 15 boyutlu ortak dağılım,
  GW150914 için ~41 bit. Doğru bir *toplam* verir ama **parametre kombinasyonlarına
  ayrışmaz** — hangi yönün baskın olduğunu söylemez.

Marjinal yaklaşım yönleri göremez; ortak Gauss yön bilgisini bir sayıya gömer.
**Fisher özdeğer yöntemi tam ortada durur:** ortak yapıyı korur ama onu
fiziksel olarak yorumlanabilir *özyönlere* ayırır.

---

## 2. Matematik (kısa ve tam)

Gauss/Laplace yaklaşımında posterior kovaryansı $\Sigma_{\text{post}}$, prior
kovaryansı $\Sigma_{\text{prior}}$. Veri-baskın rejimde

$$ \Sigma_{\text{post}}^{-1} = F + \Sigma_{\text{prior}}^{-1}
\;\Rightarrow\; F = \Sigma_{\text{post}}^{-1} - \Sigma_{\text{prior}}^{-1}, $$

$F$ Fisher bilgi matrisi. Sıralamayı **genelleştirilmiş özproblemle** yaparız:

$$ \Sigma_{\text{post}}\, v_k = \mu_k\, \Sigma_{\text{prior}}\, v_k . $$

- $\mu_k$ = o özyöndeki **posterior/prior varyans oranı** (ne kadar daraldı).
- $\lambda_k = \dfrac{1}{\mu_k} - 1$ = **prior ile beyazlatılmış Fisher özdeğeri**
  = o yöndeki **etkin SNR², $\rho_k^2$**.
- Yön başına bilgi (senin formülün, bit cinsinden):

$$ \boxed{\,I_k = \tfrac12\log_2(1+\lambda_k) = -\tfrac12\log_2\mu_k\,} $$

- Toplam: $\sum_k I_k = \tfrac12\log_2\dfrac{\det\Sigma_{\text{prior}}}{\det\Sigma_{\text{post}}}$
  — yani **hacim/Occam (ölçülebilirlik) terimi**. (Doğrulandı: makine hassasiyetinde eşleşiyor.)

**Önemli ayrım — iki tür "bilgi":** Tam Gauss KLD'yi *aynı özbazda* ayrıştırınca

$$ \mathrm{KLD}(p\|q)=\tfrac12\sum_k\Big[\underbrace{-\ln\mu_k}_{\text{ölçülebilirlik}}
\;+\;\underbrace{(\mu_k-1)}_{\text{iz}}\;+\;\underbrace{\delta_k^2}_{\text{ortalama-kayması / sürpriz}}\Big], $$

$\delta_k$ = posterior ortalamasının prior ortalamasından, prior-sigma biriminde
kayması. Yani:

| Büyüklük | Ne ölçer | Prior-bağımlılığı |
|---|---|---|
| $I_k=\tfrac12\log_2(1+\lambda_k)$ | **ölçülebilirlik** (yön ne kadar daraldı) | düşük → **sıralama için doğru olan bu** |
| $\mathrm{KLD}_k$ (tam) | ölçülebilirlik + sürprizin nereye düştüğü | yüksek (toplam ~41 bit buradan gelir) |

"Mesafe artefaktı"nı çözen şey, sıralamayı **prior genişliğine değil, prior'a göre
göreli daralmaya ($\lambda_k$)** dayandırmaktır.

---

## 3. Ne kadar zor? — **Düşük.** (~yarım gün)

Altyapı zaten sende: `covariance` kovaryansı, `gaussian kl` slogdet/inv akışını,
`approximation formula` ise tam olarak $\tfrac12 N\log_2(1+\rho^2/N)$'i kuruyor.
Eklenen tek şey **tek satırlık genelleştirilmiş özçözüm**:

```python
from scipy.linalg import eigh
mu, V = eigh(Sigma_post, Sigma_prior)   # mu = varyans oranları, V = özyönler
lam   = 1/mu - 1                          # = SNR_k^2
I_k   = 0.5*np.log2(1+lam)                # yön başına bit
```

| İş | Zorluk | Not |
|---|---|---|
| Kovaryanslar + özçözüm | Çok kolay | 10 satır, zaten var |
| Koordinat dönüşümü (Mc, q, log dL, cos θ) | Kolay | Gauss'luğu iyileştirir, baskın yönü temizler |
| Özvektör → fiziksel yorum | Orta | prior-sigma birimine çevir, en baskın 3 parametreyi yaz |
| Periyodik açılar (φ, ψ, az) | Orta | Gauss yaklaşımı zayıf; ya dışla ya circular-cov kullan |
| 36 olaya ölçekleme | Kolay | döngü; her olay <1 sn |

Gerçek zorluk teknik değil, **kavramsal/yorumsal**: Gauss yaklaşımının geçerliliği
ve "prior-bağımsız" ifadesinin sınırları (bkz. §6).

---

## 4. Hesaplama nasıl işliyor — adım adım

1. **Yükle:** posterior (147 634 örnek) + prior (5 000 örnek), 15 parametre.
2. **Koordinatla:** `chirp_mass→log`, `luminosity_distance→log`, eğimler `cos`. Bu,
   baskın yönü chirp-kütle ekseninde temiz tutar.
3. **Tekilleştir (dedup):** GWTC posterior'u iç parametreleri ~2× kopyalar
   (147 634 → 71 747). Kovaryans yanlı olmaz ama doğru $N$ ile çalışmak temizdir
   (bu senin `grup_kld` notundaki tuzakla aynı).
4. **Kovaryans:** $\Sigma_{\text{post}},\ \Sigma_{\text{prior}}$ (+ pozitif-tanımlılık için minik jitter).
5. **Genelleştirilmiş özçözüm:** `eigh(Σ_post, Σ_prior)` → $\mu_k, v_k$.
6. **Bilgi:** $\lambda_k=1/\mu_k-1$, $I_k=\tfrac12\log_2(1+\lambda_k)$. Büyükten küçüğe sırala.
7. **Yorumla:** $v_k$'yi prior-sigma birimine çevir, en baskın parametreleri yaz
   (örn. "$+0.98\,M_c$").
8. **Çapraz kontroller:** (A) $\sum I_k$ vs slogdet hacim terimi; (B) $\sum\lambda_k,\sum\delta_k^2$ vs $\rho^2$;
   (C) equipartition $\tfrac12 N\log_2(1+\rho^2/N)$ vs gerçek.

---

## 5. Prototipten gerçek sonuçlar (GW150914)

**İç (intrinsic) alt-uzay — chirp-kütle baskınlığı net:**

| # | $\lambda_k$ | SNR_k | $I_k$ [bit] | %I | baskın kombinasyon |
|---|---|---|---|---|---|
| 1 | 56.6 | 7.5 | **2.92** | **57%** | **+0.98·Mc** −0.14·cosθ1 |
| 2 | 4.43 | 2.1 | 1.22 | 24% | −0.96·q (kütle oranı) |
| 3 | 1.88 | 1.4 | 0.76 | 15% | +0.79·cosθ1 +0.54·cosθ2 (eğimler) |
| 4–6 | <0.25 | <0.5 | <0.16 | <4% | spin büyüklükleri a1,a2 |

→ İç bilginin **%57'si tek bir yönde: chirp-kütle**. Hipotez doğrulandı.

**Tam 15 parametre:** en büyük özyönler **varış zamanı $t_c$ ve gök açısı (zenith)**
($\lambda\sim 10^3$), ardından **Mc–azimuth bloğu** ($\lambda\sim 50$–80), sonra
mesafe–eğim dejenere bloğu ($\lambda\sim 1$–5). `luminosity_distance` artık tek
başına parlamıyor — yalnızca cosθJN/cosθ ile karışık, *düşük* özdeğerli yönlerde
çıkıyor. **Mesafe artefaktı çözüldü.**

**Çapraz kontroller (doğrulandı):**
- (A) $\sum I_k = 20.50$ bit $=\tfrac12\log_2(\det\Sigma_q/\det\Sigma_p)$, fark $\sim10^{-13}$. ✓
- Senin ham koordinatlarında (m1,m2,…) tam Gauss KLD = **41.25 bit** → önceki ~41 bit sonucunla birebir. ✓
- (B) **equipartition başarısız:** $\sum\lambda_k=3059 \neq \rho^2=669$. Çünkü $t_c$/zenith
  özdeğerleri SNR değil **prior-genişliği** kaynaklı. Yani senin $I_{\text{equi}}=\tfrac12 N\log_2(1+\rho^2/N)$
  formülün bilgiyi **fazla sayıyor** (41 vs gerçek dağılım). Bu başlı başına bir bulgu.

---

## 6. Getiri (neden değer)

1. **Mesafe artefaktının çözülmesi.** Sıralama prior genişliğine değil göreli
   daralmaya dayanır; mesafe doğru biçimde dejenere/düşük-bilgi blokta görünür.
2. **Fiziksel yorumlanabilirlik.** "dL şu kadar bit" yerine "en iyi ölçülen
   kombinasyon $0.98 M_c$, $X$ bit" — çok daha güçlü bilimsel ifade. Bilinen GW
   fenomenolojisini (önce Mc, sonra q/χ_eff, sonra gök/mesafe) sayısal olarak üretir.
3. **Senin equipartition formülünün rigor hali.** $\tfrac12 N\log_2(1+\rho^2/N)$,
   $\rho^2$'yi eşit dağıttığını varsayar. Özdeğer spektrumu gerçek **eşitsiz**
   dağılımı verir; formülün ne zaman/neden saptığını gösterir.
4. **Olaylar-arası ortak zemin.** 36 olayın hepsini aynı özyön tabanında sıralayıp
   "hangi olay hangi yönde ne kadar bilgi" haritası çıkarılabilir.
5. **Reparametrizasyon teşhisi.** Gauss-KLD koordinata bağlı (41 vs 17.5 bit); özdeğer
   *sıralaması* sağlam kalır. Bu, hangi sonuçların koordinat-yapaylığı olduğunu ayırt eder.

---

## 7. Sınırlar / dürüst uyarılar

- **"Prior-bağımsız" yaklaşık.** Metrik olarak hâlâ $\Sigma_{\text{prior}}$ kullanılır;
  tam bağımsızlık için sabit fiziksel ölçek (örn. Fisher'ın kendi özdeğerleri,
  $\Sigma_{\text{post}}^{-1}$) seçilebilir ama o da birimler arası karşılaştırma için
  bir metrik gerektirir. Chirp-kütle baskınlığı bu seçime karşı **gürbüz**.
- **Gauss yaklaşımı.** Kütleler için iyi; periyodik açılar (φ, ψ, azimuth) için zayıf —
  $\Sigma$ bu yönlerde anlamsızlaşır. Çözüm: ya iç+iyi-ölçülen alt-uzaya kısıtla, ya
  circular istatistik kullan.
- **Negatif $I_k$.** Posterior'un prior'dan *geniş* olduğu yönler (örn. φjl). Veri o
  kombinasyonu kısıtlamıyor; doğru davranış, sıralamada en altta.

---

## 8. Yol haritası

1. **(Bitti)** Prototip + GW150914 doğrulaması. → `fisher_eigen_info.py`
2. Koordinat setini sabitle (intrinsic için Mc, q, a1, a2, cosθ1, cosθ2; tam set için
   ek olarak log dL, cosθJN, t_c, gök açıları).
3. 36 olaya döngüyle uygula; her olay için (özdeğer spektrumu, baskın yön, $I_1$/toplam
   oranı) tablosu üret.
4. **Olaylar-arası analiz:** $I_1$ (baskın yön biti) ve spektrum eğimini SNR ile
   karşılaştır; equipartition'dan sapmayı SNR'a göre haritala.
5. (Opsiyonel) Gauss yaklaşımını sınamak için aynı özyönlerde kNN-KLD ile karşılaştır
   (mevcut `grup_kld` referans estimatörlerinle).
6. Yazım: yöntem + "mesafe artefaktının çözümü" + chirp-kütle baskınlığı şekli.

---

*Prototip: `fisher_eigen_info.py` · Çıktılar: `fisher_eigen_GW150914.json`, `fisher_eigen_GW150914.png`*
