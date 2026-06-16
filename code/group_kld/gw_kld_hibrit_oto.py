"""
GW KLD HİBRİT OTO  —  toplu (batch) indir → hesapla → kaydet
============================================================

Zenodo record 8177023 (GWTC-3 PE Data Release) içindeki BBH olaylarını
TEK TEK otomatik olarak:
    1) Zenodo'dan indirir   (Data klasörüne, varsa yeniden indirmez)
    2) gw_grup_kld_hibrit.main() ile ≤5D grup KLD(posterior||prior) hesabını yapar
    3) sonuçları kaydeder    (olay başına JSON+PNG, ayrıca master özet tablo)
    4) bir sonraki olaya geçer

Adım adım ilerler; herhangi bir olayda hata olursa onu loglar ve durmadan
diğerlerine devam eder. Çalışma yarıda kesilirse tekrar başlatıldığında
kaldığı yerden ilerleyebilir (--resume).

Çıktılar (hepsi bu .py ile aynı klasöre yazılır):
  results_grup_kld_hibrit_<EVENT>.json   olay başına tam sonuç (main() üretir)
  grup_kld_hibrit_<EVENT>.png            olay başına grafik     (main() üretir)
  oto_master_ozet.json                   tüm olayların derli toplu özeti
  oto_master_ozet.csv                    aynı özet, Excel/LibreOffice için
  oto_log.txt                            ilerleme ve hata günlüğü

NOT: İndirilen .h5 dosyaları Data klasöründe TUTULUR (silinmez).

Kullanım:
  python gw_kld_hibrit_oto.py                 # tüm BBH olaylarını işle;
                                              #   sonucu OLAN olayları otomatik ATLAR
                                              #   (yarıda kesilirse aynı komutla devam eder)
  python gw_kld_hibrit_oto.py --force         # bitmiş olanları da baştan hesapla / üzerine yaz
  python gw_kld_hibrit_oto.py --only GW191105 # sadece adında bu geçen olay(lar)
  python gw_kld_hibrit_oto.py --limit 3       # ilk 3 olayı işle (test için)
  python gw_kld_hibrit_oto.py --dry-run       # hiçbir şey indirme/hesaplama; sadece listele
  python gw_kld_hibrit_oto.py --redownload    # mevcut .h5'leri yok say, yeniden indir

Bağımlılıklar: gw_grup_kld_hibrit.py + (kde/knn)_estimators_reference.py
               numpy, scipy, h5py, astropy, matplotlib  (sadece stdlib ile indirir)
"""

import os
import sys
import csv
import json
import time
import shutil
import hashlib
import argparse
import datetime
import urllib.request
import urllib.error

# --- kendi klasörünü import yoluna ekle (gw_grup_kld_hibrit'i bulmak için) ---
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# =====================================================================
# AYARLAR
# =====================================================================
RECORD_ID = "8177023"                      # GWTC-3 PE Data Release (Zenodo)
FILES_API = f"https://zenodo.org/api/records/{RECORD_ID}/files"

# .h5 dosyalarının indirileceği klasör (mevcut kodla aynı yer).
DATA_DIR = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data"

# Hangi sürüm dosyalar? Mevcut analiz "mixed_cosmo" kullanıyor.
FILE_SUFFIX = "_PEDataRelease_mixed_cosmo.h5"

# BBH DIŞI olaylar (NSBH) — "sadece BBH" seçimi gereği atlanır.
EXCLUDE_EVENTS = {
    "GW200105_162426",   # NSBH
    "GW200115_042309",   # NSBH
}

# İndirme dayanıklılığı
DOWNLOAD_RETRIES = 3
RETRY_WAIT_S = 5
HTTP_TIMEOUT = 60          # bağlantı/okuma zaman aşımı (s)
CHUNK = 1024 * 1024        # 1 MiB

LOG_PATH = os.path.join(HERE, "oto_log.txt")
MASTER_JSON = os.path.join(HERE, "oto_master_ozet.json")
MASTER_CSV = os.path.join(HERE, "oto_master_ozet.csv")


# =====================================================================
# YARDIMCILAR
# =====================================================================
def log(msg):
    """Hem ekrana hem oto_log.txt'ye zaman damgalı yaz."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0


def event_id_from_key(key):
    """'IGWN-...-GW191105_143521_PEDataRelease_mixed_cosmo.h5' -> 'GW191105_143521'."""
    import re
    m = re.search(r"(GW\d{6}_\d{6})", key)
    return m.group(1) if m else key


def short_event(event_full):
    """'GW191105_143521' -> 'GW191105' (main() bu kısa adı kullanıyor)."""
    return event_full.split("_")[0]


def md5_of_file(path, chunk=CHUNK):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


# =====================================================================
# ZENODO: DOSYA LİSTESİ
# =====================================================================
def fetch_remote_file_list():
    """Zenodo files API'sini çağırıp BBH cosmo dosyalarının listesini döndürür.

    Dönüş: [{event, key, size, md5, url}, ...]  (event'e göre sıralı)
    """
    log(f"Zenodo dosya listesi çekiliyor: {FILES_API}")
    req = urllib.request.Request(FILES_API, headers={"User-Agent": "gw-kld-oto/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    entries = data.get("entries", data if isinstance(data, list) else [])
    out = []
    for e in entries:
        key = e.get("key", "")
        if not key.endswith(FILE_SUFFIX):
            continue
        event = event_id_from_key(key)
        if event in EXCLUDE_EVENTS:
            continue
        # checksum "md5:...." biçiminde gelir
        chk = e.get("checksum", "") or ""
        md5 = chk.split(":", 1)[1] if chk.startswith("md5:") else None
        # indirme bağlantısı: API "content" linki varsa onu, yoksa kararlı public URL
        links = e.get("links", {}) or {}
        url = links.get("content") or f"https://zenodo.org/records/{RECORD_ID}/files/{key}?download=1"
        out.append({
            "event": event,
            "key": key,
            "size": int(e.get("size", 0) or 0),
            "md5": md5,
            "url": url,
        })
    out.sort(key=lambda d: d["event"])
    log(f"Bulunan BBH cosmo olay sayısı: {len(out)}")
    return out


# =====================================================================
# İNDİRME
# =====================================================================
def file_ok(path, expected_size=0, expected_md5=None):
    """Dosya var ve (varsa) boyut/md5 ile uyumlu mu?"""
    if not os.path.exists(path):
        return False
    if expected_size and os.path.getsize(path) != expected_size:
        return False
    if expected_md5:
        try:
            if md5_of_file(path) != expected_md5:
                return False
        except Exception:
            return False
    return True


def download_one(info, dest, redownload=False):
    """Tek dosyayı indirir (gerekirse), boyut/md5 ile doğrular. Başarı -> True."""
    key, url = info["key"], info["url"]
    size, md5 = info["size"], info["md5"]

    if not redownload and file_ok(dest, size, md5):
        log(f"  zaten mevcut, atlanıyor: {key}  ({human(os.path.getsize(dest))})")
        return True

    tmp = dest + ".part"
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            log(f"  indiriliyor (deneme {attempt}/{DOWNLOAD_RETRIES}): {key}"
                + (f"  ~{human(size)}" if size else ""))
            req = urllib.request.Request(url, headers={"User-Agent": "gw-kld-oto/1.0"})
            t0 = time.time()
            done = 0
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp, open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length", size) or 0)
                last_print = 0
                while True:
                    blk = resp.read(CHUNK)
                    if not blk:
                        break
                    fh.write(blk)
                    done += len(blk)
                    if total and (done - last_print) >= 25 * 1024 * 1024:  # her ~25MB
                        pct = 100.0 * done / total
                        log(f"    ... {human(done)}/{human(total)} (%{pct:.0f})")
                        last_print = done
            dt = time.time() - t0
            log(f"    indirildi: {human(done)} / {dt:.0f} s")

            # doğrula
            if size and os.path.getsize(tmp) != size:
                raise IOError(f"boyut uyumsuz: {os.path.getsize(tmp)} != {size}")
            if md5:
                got = md5_of_file(tmp)
                if got != md5:
                    raise IOError(f"md5 uyumsuz: {got} != {md5}")
                log("    md5 doğrulandı ✓")
            os.replace(tmp, dest)
            return True

        except Exception as e:
            log(f"    HATA (indirme): {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(RETRY_WAIT_S)
            else:
                log(f"    VAZGEÇİLDİ: {key} indirilemedi.")
                return False
    return False


# =====================================================================
# ÖZET ÇIKARMA
# =====================================================================
def summarize(out):
    """main()'in döndürdüğü tam sonuçtan master tabloya kısa satır çıkar."""
    gt = out.get("kld_group_total_bits", {}) or {}
    jt = out.get("joint_kld_estimate_bits", {}) or {}
    return {
        "event": out.get("event"),         # kısa ad; çağıran tam adla ezer
        "file": out.get("file"),
        "n_posterior": out.get("n_posterior"),
        "n_prior": out.get("n_prior"),
        "analysis_group": out.get("analysis_group"),
        "between_group_abs_corr": round(out.get("avg_between_group_abs_corr", float("nan")), 4),
        "group_total_mean_bits": round(out.get("group_total_mean_bits", float("nan")), 3),
        "joint_mean_bits": round(out.get("joint_kld_estimate_mean_bits", float("nan")), 3),
        "marginal_1d_total_bits": round(out.get("marginal_1d_total_bits", float("nan")), 3),
        "tc_correction_nats": round(out.get("tc_correction_nats", float("nan")), 3),
        "grp_total_kde_scott_bits": round(gt.get("kde-scott", float("nan")), 3),
        "grp_total_kde_silverman_bits": round(gt.get("kde-silverman", float("nan")), 3),
        "grp_total_knn_k1_bits": round(gt.get("knn-k1", float("nan")), 3),
        "joint_kde_scott_bits": round(jt.get("kde-scott", float("nan")), 3),
        "joint_kde_silverman_bits": round(jt.get("kde-silverman", float("nan")), 3),
        "joint_knn_k1_bits": round(jt.get("knn-k1", float("nan")), 3),
        "hybrid_analytic_params": ", ".join(out.get("hybrid_analytic_params", []) or []),
        "status": "ok",
        "error": "",
    }


def write_master(rows):
    """Master özeti JSON + CSV olarak yaz (her olaydan sonra güncellenir)."""
    rows = sorted(rows, key=lambda r: r.get("event") or "")
    with open(MASTER_JSON, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    if rows:
        fields = list(rows[0].keys())
        # eksik anahtarları tamamla (hata satırları daha az alan içerebilir)
        for r in rows:
            for k in fields:
                r.setdefault(k, "")
        with open(MASTER_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def load_existing_master():
    if os.path.exists(MASTER_JSON):
        try:
            with open(MASTER_JSON, encoding="utf-8") as fh:
                return {r["event"]: r for r in json.load(fh)}
        except Exception:
            return {}
    return {}


# =====================================================================
# ANA AKIŞ
# =====================================================================
def run(args):
    # gw_grup_kld_hibrit'i içe aktar (main fonksiyonu burada)
    try:
        import gw_grup_kld_hibrit as hibrit
    except Exception as e:
        log(f"KRİTİK: gw_grup_kld_hibrit içe aktarılamadı: {e}")
        log("Bu .py dosyasının gw_grup_kld_hibrit.py ile AYNI klasörde olduğundan emin ol.")
        return 1

    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        files = fetch_remote_file_list()
    except Exception as e:
        log(f"KRİTİK: Zenodo dosya listesi alınamadı: {e}")
        return 1

    # --only filtresi
    if args.only:
        sel = [f for f in files if args.only.lower() in f["event"].lower()]
        log(f"--only '{args.only}' -> {len(sel)} olay seçildi")
        files = sel
    if args.limit:
        files = files[: args.limit]

    if not files:
        log("İşlenecek olay yok. (filtreyi kontrol et)")
        return 0

    log("=" * 70)
    log(f"İŞLENECEK OLAYLAR ({len(files)}):")
    for f in files:
        log(f"  - {f['event']:18s} {human(f['size']) if f['size'] else '?':>9}")
    log("=" * 70)

    if args.dry_run:
        log("--dry-run: indirme/hesaplama yapılmadı. Çıkılıyor.")
        return 0

    master = load_existing_master()

    n_ok = n_skip = n_err = 0
    for i, info in enumerate(files, 1):
        event = info["event"]
        short = short_event(event)
        dest = os.path.join(DATA_DIR, info["key"])
        # main() 6 haneli kısa adla yazar; biz TAM olay adıyla kopya saklarız
        # (aynı 6 haneyi paylaşan iki olay birbirini ezmesin diye).
        result_short = os.path.join(HERE, f"results_grup_kld_hibrit_{short}.json")
        png_short = os.path.join(HERE, f"grup_kld_hibrit_{short}.png")
        result_full = os.path.join(HERE, f"results_grup_kld_hibrit_{event}.json")
        png_full = os.path.join(HERE, f"grup_kld_hibrit_{event}.png")

        log("")
        log("#" * 70)
        log(f"[{i}/{len(files)}]  {event}")
        log("#" * 70)

        # Varsayılan: sonucu zaten olan olayı ATLA (tekrar hesaplama yok).
        # Sadece --force verilirse baştan hesaplar/üzerine yazar.
        if (not args.force) and os.path.exists(result_full):
            log(f"  sonuç zaten var, atlanıyor ({os.path.basename(result_full)}) "
                f"— baştan istersen --force")
            n_skip += 1
            continue

        # 1) İNDİR
        ok = download_one(info, dest, redownload=args.redownload)
        if not ok:
            n_err += 1
            master[event] = {"event": event, "file": info["key"],
                             "status": "download_failed", "error": "indirilemedi"}
            write_master(list(master.values()))
            continue

        # 2) HESAPLA
        try:
            log(f"  hesaplanıyor: gw_grup_kld_hibrit.main('{os.path.basename(dest)}')")
            t0 = time.time()
            out = hibrit.main(dest)
            dt = time.time() - t0
            if not isinstance(out, dict):
                raise RuntimeError("main() beklenen sözlüğü döndürmedi")
            log(f"  bitti: {dt:.0f} s  |  joint(ort.)={out.get('joint_kld_estimate_mean_bits', float('nan')):.2f} bit"
                f"  grup-toplam(ort.)={out.get('group_total_mean_bits', float('nan')):.2f} bit")
            # main()'in kısa adlı çıktısını TAM olay adıyla da sakla (çakışmayı önler)
            if event != short:
                for src, dst in ((result_short, result_full), (png_short, png_full)):
                    try:
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                    except Exception as ce:
                        log(f"  (uyarı: {os.path.basename(src)} kopyalanamadı: {ce})")
                log(f"  tam-adlı kopya: {os.path.basename(result_full)}")
            row = summarize(out)
            row["event"] = event           # TAM olay adı (6-haneli çakışmaları ayırır)
            master[event] = row
            n_ok += 1
        except Exception as e:
            import traceback
            log(f"  HATA (hesaplama): {e}")
            log(traceback.format_exc())
            master[event] = {"event": event, "file": info["key"],
                             "status": "compute_failed", "error": str(e)}
            n_err += 1

        # 3) Her olaydan sonra master özeti güncelle (yarıda kesilse bile korunur)
        write_master(list(master.values()))
        log(f"  master özet güncellendi: {os.path.basename(MASTER_CSV)}")

    log("")
    log("=" * 70)
    log(f"TAMAMLANDI.  başarılı={n_ok}  atlanan={n_skip}  hatalı={n_err}")
    log(f"Master özet : {MASTER_JSON}")
    log(f"            : {MASTER_CSV}")
    log(f"Günlük      : {LOG_PATH}")
    log("=" * 70)
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description="GW KLD hibrit — toplu indir/hesapla/kaydet")
    p.add_argument("--force", action="store_true",
                   help="sonucu zaten olan olayları da BAŞTAN hesapla / üzerine yaz")
    p.add_argument("--resume", action="store_true",
                   help="(artık varsayılan davranış; geriye uyumluluk için kabul edilir)")
    p.add_argument("--only", type=str, default=None,
                   help="sadece adında bu metin geçen olay(lar) (ör. GW191105)")
    p.add_argument("--limit", type=int, default=None,
                   help="ilk N olayı işle (test için)")
    p.add_argument("--dry-run", action="store_true",
                   help="indirme/hesaplama yapma; sadece işlenecek olayları listele")
    p.add_argument("--redownload", action="store_true",
                   help="mevcut .h5 olsa bile yeniden indir")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))
