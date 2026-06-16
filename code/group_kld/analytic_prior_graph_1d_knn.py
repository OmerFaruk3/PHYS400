"""
HDF5 dosyasındaki analytic prior tanımlarını oku ve göster.
Hangi fonksiyonları neden kullandığımızı açıklar.
"""

import h5py
import re

FILE = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data/IGWN-GWTC3p0-v2-GW191103_012549_PEDataRelease_mixed_cosmo.h5"

PARAMS_15 = [
    "mass_1_source", "mass_2_source", "a_1", "a_2", "tilt_1", "tilt_2",
    "phi_12", "phi_jl", "luminosity_distance", "theta_jn", "psi",
    "azimuth", "zenith", "geocent_time", "phase",
]

with h5py.File(FILE, "r") as f:
    an = f["C01:IMRPhenomXPHM"]["priors"]["analytic"]

    print("=" * 70)
    print("HDF5 → priors/analytic  içeriği")
    print("Bilby PE analizinin kullandığı gerçek prior tanımları burada yazıyor.")
    print("=" * 70)

    print("\n── 15 parametremiz ──────────────────────────────────────────────────")
    for p in PARAMS_15:
        if p in an:
            raw = an[p][()][0].decode()
            # tip ismini çek
            tip = re.match(r"(\w+)\(", raw).group(1)
            print(f"\n  {p}")
            print(f"    Tanım : {raw}")
            print(f"    Tip   : {tip}")
            if tip == "Uniform":
                print(f"    Örnekleme: rng.uniform(minimum, maximum)")
            elif tip == "Sine":
                print(f"    Örnekleme: θ = arccos(cos(min) - u*(cos(min)-cos(max)))")
                print(f"    (p(θ) ∝ sin(θ), küresel izotropi için doğal prior)")
            elif tip == "PowerLaw":
                m = re.search(r"alpha=([\d.]+)", raw)
                a = m.group(1) if m else "?"
                print(f"    Örnekleme: ters-CDF, p(x) ∝ x^{a}")
        else:
            print(f"\n  {p}")
            print(f"    Tanım : *** dosyada YOK — türetilmiş parametre ***")

    print("\n── Kütle için gerçek primary prior değişkenleri ──────────────────────")
    for k in ["chirp_mass", "mass_ratio", "mass_1", "mass_2"]:
        if k in an:
            raw = an[k][()][0].decode()
            tip = re.match(r"(\w+)\(", raw).group(1)
            print(f"\n  {k}")
            print(f"    Tanım : {raw}")
            print(f"    Tip   : {tip}")
            if tip == "Constraint":
                print(f"    → Bu bir prior DEĞİL, sadece fiziksel kısıt (m > 0).")
                print(f"      Gerçek prior chirp_mass + mass_ratio üzerinde.")
            elif "UniformInComponents" in tip:
                print(f"    → Bileşen kütlelerinde düzgün prior anlamına gelir.")
                print(f"      Örnekleme: Mc ~ Uniform(min, max), q ~ Uniform(0.05, 1)")
                print(f"      Sonra: m1 = Mc*(1+q)^(1/5)/q^(3/5),  m2 = q*m1")
                print(f"      Source frame: m_source = m_detector / (1 + z(d_L))")

    print("\n── Neden mass_1/2_SOURCE dosyada yok? ───────────────────────────────")
    print("""
  Bilby analizi şu zinciri kurar:
    1. chirp_mass (detector frame) ~ UniformInComponentsChirpMass
    2. mass_ratio                  ~ UniformInComponentsMassRatio
    3. luminosity_distance         ~ PowerLaw(alpha=2)
       → redshift z = z(d_L)  [Planck15 kozmolojisi]
    4. mass_1_source = mass_1_detector / (1 + z)
    5. mass_2_source = mass_2_detector / (1 + z)

  mass_1/2_source analytic prior'da YOK çünkü bunlar PRIMARY değil,
  yukardaki zincirin ÇIKTISI. Biz de aynı zinciri taklit ederek
  mass_1/2_source için doğru prior dağılımını üretiyoruz.
    """)

    print("── Özet: hangi fonksiyon → hangi prior tanımı ───────────────────────")
    print("""
  Uniform(min, max)              → rng.uniform(min, max)
  Sine(min, max)                 → arccos(cos(min) - u*(cos(min)-cos(max)))
  PowerLaw(alpha=2, min, max)    → ters-CDF ile örnekleme
  UniformInComponentsChirpMass   → Mc ~ Uniform, q ~ Uniform → m1, m2
  + luminosity_distance          → z(d_L) ile source frame'e çevir
    """)