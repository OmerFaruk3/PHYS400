"""
GW KLD HİBRİT OTO — ÇOK KATALOG  (GWTC-2.1 / GWTC-3)
====================================================

gw_kld_hibrit_oto.py ile AYNI mantık; tek fark: hangi Zenodo kataloğunu
çekeceğini seçebilirsin. Böylece GWTC-3'te OLMAYAN olaylara (örn. GW150914)
de ulaşırsın.

Katalog -> Zenodo kaydı eşleşmesi:
  gwtc3    = record 8177023  -> O3b dönemi  (GW191xxx, GW200xxx)
  gwtc2p1  = record 6513631  -> O1 + O2 + O3a  ("deep extended catalog";
             GW150914, GW151226, GW170814 ... GW190xxx hepsi burada)

İki katalog birlikte O3 sonuna kadar tüm güvenilir olayları kapsar.

ÖNEMLİ:
  * Bu kod, mevcut gw_kld_hibrit_oto.py dosyasına ve onun ürettiği
    oto_master_ozet.* / oto_log.txt dosyalarına DOKUNMAZ.
  * Her katalog kendi özet/günlük dosyasını yazar:
        oto_master_ozet_<katalog>.json / .csv
        oto_log_<katalog>.txt
  * Olay başına sonuç dosyaları (results_grup_kld_hibrit_<EVENT>.json) olay
    adıyla adlandırılır; GWTC-2.1 ve GWTC-3 olayları farklı olduğundan
    mevcut 34 GWTC-3 sonucunun ÜZERİNE YAZILMAZ.
  * Varsayılan davranış: sonucu zaten olan olayı ATLAR (--force ile baştan).

Hesabı yapan motor yine gw_grup_kld_hibrit.main()'dir (hiç değişmedi).

Kullanım:
  python gw_kld_hibrit_oto_katalog.py --catalog gwtc2p1            # tüm BBH (O1/O2/O3a)
  python gw_kld_hibrit_oto_katalog.py --catalog gwtc2p1 --dry-run  # önce listeyi gör
  python gw_kld_hibrit_oto_katalog.py --catalog gwtc2p1 --only GW150914
  python gw_kld_hibrit_oto_katalog.py --catalog gwtc2p1 --force    # bitmişleri de yeniden
  python gw_kld_hibrit_oto_katalog.py --catalog gwtc3              # (istersen GWTC-3 tekrar)

Bağımlılıklar: gw_grup_kld_hibrit.py + (kde/knn)_estimators_reference.py
               numpy, scipy, h5py, astropy, matplotlib  (indirme: sadece stdlib)
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# =====================================================================
# KATALOG TANIMLARI
# =====================================================================
# Her katalog: Zenodo record id, dosya öneki ve "BBH değil" diye atlanacaklar.
# EXCLUDE listelerini istediğin gibi düzenleyebilirsin (örn. GW190814'ü dahil
# etmek istersen aşağıdaki kümeden çıkar).
CATALOGS = {
    "gwtc3": {
        "record": "8177023",
        "prefix": "IGWN-GWTC3p0-v2-",
        "exclude": {
            "GW200105_162426",   # NSBH
            "GW200115_042309",   # NSBH
        },
    },
    "gwtc2p1": {
        "record": "6513631",
        "prefix": "IGWN-GWTC2p1-v2-",
        "exclude": {
            "GW170817_124104",   # BNS
            "GW190425_081805",   # BNS
            "GW190426_152155",   # düşük anlamlı NSBH adayı
            "GW190814_211039",   # kütle-boşluğu (ikincil ~2.6 Msun); BBH dışı sayıldı
            "GW190917_114630",   # NSBH adayı
        },
    },
}

# .h5 dosyalarının indirileceği klasör (mevcut kodla aynı yer).
DATA_DIR = "/Users/omerfaruk/Desktop/PHYS400-Code/PHYS400/Data"

# Mevcut analiz "mixed_cosmo" kullanıyor (kaynak-çerçeve kütleleri için gerekli).
FILE_SUFFIX = "_PEDataRelease_mixed_cosmo.h5"

# İndirme dayanıklılığı
DOWNLOAD_RETRIES = 3
RETRY_WAIT_S = 5
HTTP_TIMEOUT = 60
CHUNK = 1024 * 1024


# =====================================================================
# YARDIMCILAR
# =====================================================================
def make_log(path):
    def log(msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
    return log


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0


def event_id_from_key(key):
    import re
    m = re.search(r"(GW\d{6}_\d{6})", key)
    return m.group(1) if m else key


def short_event(event_full):
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
def fetch_remote_file_list(catalog, log):
    cfg = CATALOGS[catalog]
    record, prefix, exclude = cfg["record"], cfg["prefix"], cfg["exclude"]
    api = f"https://zenodo.org/api/records/{record}/files"
    log(f"Zenodo dosya listesi çekiliyor [{catalog}, record {record}]: {api}")
    req = urllib.request.Request(api, headers={"User-Agent": "gw-kld-oto/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    entries = data.get("entries", data if isinstance(data, list) else [])
    out = []
    for e in entries:
        key = e.get("key", "")
        if not key.endswith(FILE_SUFFIX):
            continue
        if prefix not in key:
            continue
        event = event_id_from_key(key)
        if event in exclude:
            continue
        chk = e.get("checksum", "") or ""
        md5 = chk.split(":", 1)[1] if chk.startswith("md5:") else None
        links = e.get("links", {}) or {}
        url = links.get("content") or f"https://zenodo.org/records/{record}/files/{key}?download=1"
        out.append({"event": event, "key": key,
                    "size": int(e.get("size", 0) or 0), "md5": md5, "url": url})
    out.sort(key=lambda d: d["event"])
    log(f"Bulunan BBH cosmo olay sayısı: {len(out)}  (atlanan BBH-dışı: {len(exclude)})")
    return out


# =====================================================================
# İNDİRME
# =====================================================================
def file_ok(path, expected_size=0, expected_md5=None):
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


def download_one(info, dest, log, redownload=False):
    key, url, size, md5 = info["key"], info["url"], info["size"], info["md5"]
    if not redownload and file_ok(dest, size, md5):
        log(f"  zaten mevcut, atlanıyor: {key}  ({human(os.path.getsize(dest))})")
        return True
    tmp = dest + ".part"
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            log(f"  indiriliyor (deneme {attempt}/{DOWNLOAD_RETRIES}): {key}"
                + (f"  ~{human(size)}" if size else ""))
            req = urllib.request.Request(url, headers={"User-Agent": "gw-kld-oto/1.0"})
            t0, done, last = time.time(), 0, 0
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp, open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length", size) or 0)
                while True:
                    blk = resp.read(CHUNK)
                    if not blk:
                        break
                    fh.write(blk)
                    done += len(blk)
                    if total and (done - last) >= 25 * 1024 * 1024:
                        log(f"    ... {human(done)}/{human(total)} (%{100.0*done/total:.0f})")
                        last = done
            log(f"    indirildi: {human(done)} / {time.time()-t0:.0f} s")
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
# ÖZET
# =====================================================================
def summarize(out):
    gt = out.get("kld_group_total_bits", {}) or {}
    jt = out.get("joint_kld_estimate_bits", {}) or {}
    return {
        "event": out.get("event"),
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


def write_master(rows, master_json, master_csv):
    rows = sorted(rows, key=lambda r: r.get("event") or "")
    with open(master_json, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    if rows:
        fields = list(rows[0].keys())
        for r in rows:
            for k in fields:
                r.setdefault(k, "")
        with open(master_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def load_existing_master(master_json):
    if os.path.exists(master_json):
        try:
            with open(master_json, encoding="utf-8") as fh:
                return {r["event"]: r for r in json.load(fh)}
        except Exception:
            return {}
    return {}


# =====================================================================
# ANA AKIŞ
# =====================================================================
def run(args):
    catalog = args.catalog
    if catalog not in CATALOGS:
        print(f"Bilinmeyen katalog: {catalog}. Seçenekler: {', '.join(CATALOGS)}")
        return 2

    master_json = os.path.join(HERE, f"oto_master_ozet_{catalog}.json")
    master_csv = os.path.join(HERE, f"oto_master_ozet_{catalog}.csv")
    log = make_log(os.path.join(HERE, f"oto_log_{catalog}.txt"))

    try:
        import gw_grup_kld_hibrit as hibrit
    except Exception as e:
        log(f"KRİTİK: gw_grup_kld_hibrit içe aktarılamadı: {e}")
        log("Bu .py, gw_grup_kld_hibrit.py ile AYNI klasörde olmalı.")
        return 1

    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        files = fetch_remote_file_list(catalog, log)
    except Exception as e:
        log(f"KRİTİK: Zenodo dosya listesi alınamadı: {e}")
        return 1

    if args.only:
        files = [f for f in files if args.only.lower() in f["event"].lower()]
        log(f"--only '{args.only}' -> {len(files)} olay seçildi")
    if args.limit:
        files = files[: args.limit]

    if not files:
        log("İşlenecek olay yok. (filtre/katalog kontrol et)")
        return 0

    log("=" * 70)
    log(f"KATALOG: {catalog}   İŞLENECEK OLAYLAR ({len(files)}):")
    for f in files:
        log(f"  - {f['event']:18s} {human(f['size']) if f['size'] else '?':>9}")
    log("=" * 70)

    if args.dry_run:
        log("--dry-run: indirme/hesaplama yapılmadı. Çıkılıyor.")
        return 0

    master = load_existing_master(master_json)
    n_ok = n_skip = n_err = 0

    for i, info in enumerate(files, 1):
        event = info["event"]
        short = short_event(event)
        dest = os.path.join(DATA_DIR, info["key"])
        result_short = os.path.join(HERE, f"results_grup_kld_hibrit_{short}.json")
        png_short = os.path.join(HERE, f"grup_kld_hibrit_{short}.png")
        result_full = os.path.join(HERE, f"results_grup_kld_hibrit_{event}.json")
        png_full = os.path.join(HERE, f"grup_kld_hibrit_{event}.png")

        log("")
        log("#" * 70)
        log(f"[{i}/{len(files)}]  {event}   ({catalog})")
        log("#" * 70)

        if (not args.force) and os.path.exists(result_full):
            log(f"  sonuç zaten var, atlanıyor ({os.path.basename(result_full)}) — baştan: --force")
            n_skip += 1
            continue

        if not download_one(info, dest, log, redownload=args.redownload):
            n_err += 1
            master[event] = {"event": event, "file": info["key"],
                             "status": "download_failed", "error": "indirilemedi"}
            write_master(list(master.values()), master_json, master_csv)
            continue

        try:
            log(f"  hesaplanıyor: gw_grup_kld_hibrit.main('{os.path.basename(dest)}')")
            t0 = time.time()
            out = hibrit.main(dest)
            dt = time.time() - t0
            if not isinstance(out, dict):
                raise RuntimeError("main() beklenen sözlüğü döndürmedi")
            log(f"  bitti: {dt:.0f} s  |  joint(ort.)={out.get('joint_kld_estimate_mean_bits', float('nan')):.2f} bit"
                f"  grup-toplam(ort.)={out.get('group_total_mean_bits', float('nan')):.2f} bit")
            if event != short:
                for src, dst in ((result_short, result_full), (png_short, png_full)):
                    try:
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                    except Exception as ce:
                        log(f"  (uyarı: {os.path.basename(src)} kopyalanamadı: {ce})")
                log(f"  tam-adlı kopya: {os.path.basename(result_full)}")
            row = summarize(out)
            row["event"] = event
            row["catalog"] = catalog
            master[event] = row
            n_ok += 1
        except Exception as e:
            import traceback
            log(f"  HATA (hesaplama): {e}")
            log(traceback.format_exc())
            master[event] = {"event": event, "file": info["key"], "catalog": catalog,
                             "status": "compute_failed", "error": str(e)}
            n_err += 1

        write_master(list(master.values()), master_json, master_csv)
        log(f"  master özet güncellendi: {os.path.basename(master_csv)}")

    log("")
    log("=" * 70)
    log(f"TAMAMLANDI [{catalog}].  başarılı={n_ok}  atlanan={n_skip}  hatalı={n_err}")
    log(f"Master özet : {master_json}")
    log(f"            : {master_csv}")
    log("=" * 70)
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description="GW KLD hibrit — çok-katalog indir/hesapla/kaydet")
    p.add_argument("--catalog", choices=list(CATALOGS), default="gwtc2p1",
                   help="hangi Zenodo kataloğu (varsayılan: gwtc2p1 = O1/O2/O3a)")
    p.add_argument("--force", action="store_true",
                   help="sonucu zaten olan olayları da BAŞTAN hesapla / üzerine yaz")
    p.add_argument("--only", type=str, default=None,
                   help="sadece adında bu metin geçen olay(lar) (ör. GW150914)")
    p.add_argument("--limit", type=int, default=None, help="ilk N olayı işle (test için)")
    p.add_argument("--dry-run", action="store_true",
                   help="indirme/hesaplama yapma; sadece işlenecek olayları listele")
    p.add_argument("--redownload", action="store_true",
                   help="mevcut .h5 olsa bile yeniden indir")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))
