# PHYS400 Final Sunumu — Konuşma Taslağı (ver1 deck'e göre)

> Hedef: 20 dk sunum + 5 dk soru. Revize yapı: 17 slayt (s9+s10 birleşik, s15 yedeğe,
> s6 dağıtıldı). Süreler revize slayt numarasına göre. Anahtar cümleler İngilizce
> (sunum dili), yönlendirmeler Türkçe.

---

## AÇILIŞ

### S1 · Başlık — 30 sn
- Kendini ve danışmanı tanıt, başlığı OKUMA; alt sorudan gir:
- **"How many bits does a black-hole merger reveal? This entire project is an attempt to answer that one question with a single number."**
- Geçiş: "Let me first convince you the question is well-posed."

### S2 · Motivasyon — 1 dk
- Corner plot'u göster: "Before the signal, these distributions were the broad priors. After — this."
- **"The size of that update, measured in bits, IS the information the detector extracted from the Universe."**
- İki hedefi söyle: (1) compute it for real posteriors, (2) learn what controls it.
- Geçiş: "The right tool for 'size of an update' is the KL divergence."

### S3 · Teori (KL) — 1 dk
- Formülü bir kez oku, üç kutuyu (posterior/prior/bits) 10'ar sn ile geç.
- Vurgu: **"Log base 2 — so the answer comes out in bits, a number anyone can interpret."**
- Burada OYALANMA; teori slaytı tuzaktır, sayılar seni bekliyor.

### S4 · 15 parametre — 45 sn
- Tek tek okumak yasak. "Two masses, four spin magnitudes and tilts, two more spin angles, distance, orientation, sky position, time, phase — fifteen in total."
- Vurgu: F&H 1998 ile aynı konvansiyon → karşılaştırma elma-elma.

### S5 · F&H benchmark — 1.5 dk
- **"In 1998, before any detection, Flanagan & Hughes estimated ≈41.5 bits for a GW150914-like event — from a formula you can write on one line."**
- Üç varsayımı say, sonra punchline: "They themselves call it 'a rather crude approximation'. Nobody had checked it against real posteriors. That is exactly what we do."
- (Eski s6'nın marjinal-alt-sınır yarısını burada TEK cümleyle önceden haber ver:
  "We'll also use one exact inequality as a safety net — more on that in a minute.")

### S6 (eski s7) · Veri & pipeline — 1 dk
- 35 / 15 / 10⁵ sayılarını göster. "Public IGWN data, read straight from the HDF5 releases."
- A→E akışını parmakla izle, her adım ≤5 kelime. Detaya girme — B ve C'nin
  kendi slaytları var, "two of these steps hide the most interesting bugs of the project" diyerek merak bırak.

---

## PART I — 15D'de KL TAHMİNİ

### S7 (divider) — 15 sn
- **"Part one: the estimation problem. Five estimators, two data traps, and the method that finally scaled."** (Bu cümle zaten slaytta — sadece oku, mükemmel.)

### S8 (eski s9+s10 birleşik) · Estimator zoo + boyut laneti — 2.5 dk
- Sunumun ilk büyük anı. Tabloyu satır satır:
  - Histogram: **"At five bins per axis, 5^15 cells — 240 gigabytes of RAM. The curse of dimensionality is not a metaphor; it is a memory allocation error."** (En iyi repliğin, acele etme.)
  - kNN: "25.9 to 54.9 bits depending on sample size — it never converges. Hold that thought."
  - Marginal: "36 bits — but provably a floor: joint = Σ marginals + total correlation, and TC ≥ 0."  ← eski s6'nın sol yarısı buraya
  - Gaussian: "41.3 — suspiciously close to F&H. We'll come back to why."
  - Group-KLD: "and this one we adopted."
- **YENİ eklenecek merdiven görseli:** "Notice the ordering: 36.9 ≤ 38.9 ≤ 39.6 ≤ 41.3 ≈ 41.5 — marginal ≤ group ≤ joint ≲ Gaussian. Theory predicts exactly this ordering. The pipeline passes its own consistency test."

### S9 (eski s11) · Literatür benchmark — 1 dk
- "We didn't trust our implementations blindly — they adapt Álvarez-Chaves et al. 2024, validated against analytic ground truth up to 10-D."
- Punchline + gerilim: **"In their benchmark, k-NN converges beautifully. In our data, it diverged. Why? The data itself was hiding two traps."** → mükemmel geçiş.

### S10 (eski s12) · İki veri tuzağı — 1.5 dk
- Trap 1: "147,634 samples — but only 71,747 unique. Every point's nearest neighbour was its own twin, at distance zero. k-NN sees infinite density."
- Trap 2: "The stored prior for distance starts at 681 Mpc — but the posterior lives at 137–773. The data release's own prior doesn't cover its own posterior."
- Bu slaytta debugging hikayesi anlat (jüri süreç görmek ister): nasıl fark ettin, kaç gün aldı — 2 cümle, samimi ton.

### S11 (eski s13) · Hibrit prior — 1 dk
- **GRAFİĞİ DEĞİŞTİR: sadece dL paneli (+ belki m2), büyük.**
- "Where the posterior overflows the stored samples, we substitute the analytic prior — only there. The other parameters keep the file prior."
- 24.9 → 26.6: "1.7 bits of real information was being silently destroyed by missing support."

### S12 (eski s14) · Ana yöntem — 2 dk
- Üç adımı say; **DÖRDÜNCÜ ADIMI EKLE:**
  - "Step four: we add back what the blocks miss — joint = Σ groups + ΔTC, the between-group total-correlation correction. For GW150914 that lands at **39.6 bits**."
- "Three estimators — KDE-Scott, KDE-Silverman, k-NN — now agree. Method-independence is the whole point."

### S13 (eski s16) · Gaussian baseline, 35 event — 1.5 dk
- "Now the suspicious one. Treat everything as 15-D Gaussian — closed form, immune to dimensionality."
- r = 0.965 büyük göster: "It tracks the real joint across the whole catalogue, sitting ~2.3 bits low."
- GW150914: 41.3 ≈ F&H 41.5 → **"Twenty-six years later, with real data: Flanagan & Hughes were right to within a quarter of a bit."** (İkinci büyük an.)
- Scatter'daki GW191204 outlier'ını işaretle: "one event sits 8 bits HIGH — ill-conditioned covariance inflating the correlation term; diagnostics on the next slide."

### S14 (eski s17) · Neden sapıyor — 2 dk (PUNCHLINE SLAYTI)
- Beklentiyi kur: "You'd assume the gap comes from non-Gaussianity — heavy tails, banana-shaped correlations. **It doesn't.**"
- "Non-Gaussianity correlates with the gap at r ≈ −0.1: essentially nothing. The per-parameter biases largely cancel."
- "The real driver is the prior's missing support — m2 and dL spilling outside the stored samples (on average a few percent; for GW150914's distance, **99.9%** of the posterior lies outside). The Gaussian fit then invents a too-narrow prior variance and the mean-shift term explodes."
- Kapanış: **"So the surprise of this project: the hard part of measuring information is not modelling the posterior — it's defining the prior."**
- (r sayısını master dokümanla eşitle: −0.08 mi −0.15 mi — sunumdan önce kontrol!)

---

## PART II — POPÜLASYON

### S15 (divider, eski s18) — 15 sn
- "Part two: run the validated pipeline on all 35 events, and ask what controls the answer."

### S16 (eski s19) · I ~ ln(SNR) — 1.5 dk
- "Across the catalogue, information spans 6.7 to 44.6 bits — a factor of seven. The first-order explanation is loudness."
- Model seçimini FİZİĞE yasla, R²'ye değil: **"A logarithmic law is what the capacity formula predicts — Fisher information scales as SNR², so bits scale as log SNR. The data agrees: log beats linear and power-law fits."**
- "But R² = 0.58 means SNR is barely half the story."  → geçiş.

### S17 (eski s20) · Tam yasa — 1.5 dk
- "Add two physically motivated terms: total mass and detector count. R² jumps to 0.88."
- İşaretleri fizikle anlat: "Heavier binaries merge at lower frequency — fewer cycles in band — less information. More detectors — tighter sky localisation."
- **"Three numbers you can read off a detection alert — SNR, chirp mass, detector count — predict the information content to ±2 bits."**

### S18 (eski s21) · Neff — 1.5 dk
- Kapasite formülünü BURADA tanıt (eski s6'nın sağ yarısı): "Invert the channel-capacity formula and ask: how many effective parameters did the event measure?"
- "Median 15.9 — strikingly close to the 15 we analyse. **Treat this as a consistency check, not a measurement** — the equal-channel assumption is crude."
- "It's flat in SNR but falls steeply with mass — heavy events measure fewer directions."

### S19 (eski s22) · Özet & outlook — 1.5 dk
- Dört maddeyi HIZLI geç (zaten anlattın), her biri tek cümle.
- **Outlook ekle (slayta da 2 madde):** "Next: re-run the Gaussian with the hybrid prior to isolate pure non-Gaussianity; and extend to O4 — the pipeline is event-count agnostic."
- Kapanış cümlesi: **"A black-hole merger writes about forty bits into our instruments. We can now say that with a method that survives fifteen dimensions — and we know what controls the number. Thank you."**

**Toplam: ~19.5 dk**

---

## SORU-CEVAP HAZIRLIĞI (yedek slaytlarla)

| Beklenen soru | Cevabın özü | Yedek slayt |
|---|---|---|
| "Gruplara bölünce gruplar arası korelasyonu atmıyor musun?" | TC düzeltmesi: joint = Σgrup + ΔTC; GW150914'te grup 38.9 → joint 39.6 | TC matematiği |
| "Grupları nasıl seçtin, öznel değil mi?" | MI tabanlı gruplama: dendrogram + heatmap | gw_mi_gruplama çıktıları |
| "Neff > 15 nasıl olabilir?" | Eşit-kanal varsayımı kaba; inversiyon yaklaşık; 15'e yakınlık tutarlılık göstergesi | Neff dağılımı |
| "Neden log fit? R² farkları küçük." | Teorik beklenti (Fisher ~ SNR²); R² sadece doğruluyor | fit karşılaştırma |
| "Neden 35 event? GWTC-3'te ~90 var." | BBH + mixed_cosmo PE release'i olanlar; seçim kriteri veri ürünü, fizik değil | event listesi |
| "kNN'i neden tamamen atmadınız/tutmadınız?" | Diagnostik olarak tutuldu; ≤5D bloklarda çalışıyor, 15D'de değil | yakınsama eğrisi |
| "Başka neural yöntem denediniz mi?" | MINE denendi: d=5'te %34 hata → d=15 için terk | MINE sonuçları |
| "Prior'a bu kadar duyarlıysa 'bilgi' iyi tanımlı mı?" | En derin soru. Cevap: KL prior'a GÖRE tanımlı; analiz prior'ı LVK'nın kendi seçimi; biz onların prior'ının tutarlı (tam destekli) versiyonunu kullanıyoruz | hibrit prior detayı |
| "Tekilleştirme sonucu değiştiriyor mu?" | Gaussian'da fark +0.016 bit (matched analizi); kNN'de kritik | matched karşılaştırma |

## SUNUM ÖNCESİ KONTROL LİSTESİ

- [ ] r = −0.08 / −0.15 tutarsızlığını çöz (s14)
- [ ] "38–39 bit" ifadelerini joint 39.6 ile netleştir (s8, s12)
- [ ] dL %16 (ortalama) vs %99.9 (GW150914) ayrımını netleştir (s14)
- [ ] m2'nin en güçlü öngörücü olduğunu (r=0.62) dL'nin önüne al (s14)
- [ ] S11 grafiğini dL-odaklı tek panele indir
- [ ] Merdiven (number line) görseli ekle (s8)
- [ ] S12'ye TC düzeltmesi 4. adımı ekle
- [ ] Özete 2 outlook maddesi ekle
- [ ] Eski s15 (pros&cons) → yedek slaytlara
- [ ] GW191204 outlier'ını s13 scatter'ında işaretle
- [ ] Yedek slaytları hazırla (yukarıdaki tablo)
- [ ] Sesli prova: hedef 19-20 dk, s8 ve s14'te yavaşla
