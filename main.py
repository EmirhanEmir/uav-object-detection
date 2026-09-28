"""Ana tespit girişi — HER ZAMAN kararlaştırılmış / o anki en iyi sürüm.

Kullanım:
    python main.py --source kare.jpg
    python main.py --source data/sanity/images --out preds.json
    python main.py --source data/sanity/images --coco-gt data/coco/instances_sanity.json

Model, ağırlık ve çıkarım düzeni burada sabittir; komut satırından değiştirilemez.
Deney için başka ağırlık/ayar denenecekse scripts/ altındaki araçlar kullanılır.
Yeni bir sürüm kararlaştırılınca SADECE aşağıdaki AKTİF SÜRÜM bloğu güncellenir ve
eski sürüm SÜRÜM GEÇMİŞİ'ne eklenir.

Çıktı: kutular orijinal kare pikselinde (x1, y1, x2, y2); sınıf id'leri şartname
sırası 0=vehicle 1=human 2=uap 3=uai.

SÜRÜM GEÇMİŞİ (sanity, 175 kare, mAP@0.5):
  v1  Faz 4 ağırlığı + SAHI 6×1024 @1280 + tam kare      0.8419  human 0.5664  303 ms/kare
  v2  Faz 4 ağırlığı + 2×1080 @1280 + tam kare (V2)       0.8457  human 0.5431  136 ms/kare
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# ── AKTİF SÜRÜM ──────────────────────────────────────────────────────────────
SURUM = "v2"
TARIH = "2026-09-28"
# Faz 4: COCO → VisDrone-DET → bizim veri, YOLO11s-p2, imgsz=1280
WEIGHTS = ROOT / "uav_runs/visdrone_pre/faz4/baseline-visdrone-yolo11s-p2/weights/best.pt"
# Çıkarım düzeni: uav_vision.detector (V2 — sabitler.md → CIKARIM_DUZENI)
CONF = 0.001  # mAP puanlaması için düşük eşik
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_detector(device: str = "0"):
    """Aktif sürümün detektörü — diğer modüller (ör. sunucu istemcisi) buradan alır."""
    from uav_vision.detector import Detector

    if not WEIGHTS.exists():
        raise SystemExit(f"Ağırlık bulunamadı: {WEIGHTS}")
    return Detector(WEIGHTS, device=device, conf=CONF)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Ana tespit ({SURUM}, {TARIH})")
    parser.add_argument("--source", required=True, help="Görsel veya klasör")
    parser.add_argument("--out", default="preds.json")
    parser.add_argument("--device", default="0")
    parser.add_argument("--coco-gt", help="GT json — verilirse COCO results formatında yaz")
    args = parser.parse_args()

    import cv2

    src = Path(args.source)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]

    name_to_id = None
    if args.coco_gt:
        with open(args.coco_gt, encoding="utf-8") as f:
            name_to_id = {Path(im["file_name"]).name: im["id"] for im in json.load(f)["images"]}
        paths = [p for p in paths if p.name in name_to_id]

    det = load_detector(args.device)
    print(f"[main] sürüm {SURUM} ({TARIH}) — {WEIGHTS.relative_to(ROOT)}")

    out: list[dict] | dict[str, list[dict]] = [] if name_to_id else {}
    t0 = time.perf_counter()
    for p in paths:
        dets = det(cv2.imread(str(p)))
        if name_to_id is not None:
            out.extend(
                {
                    "image_id": name_to_id[p.name],
                    "category_id": d.cls,
                    "bbox": [round(d.x1, 2), round(d.y1, 2), round(d.x2 - d.x1, 2), round(d.y2 - d.y1, 2)],
                    "score": round(d.score, 5),
                }
                for d in dets
            )
        else:
            out[p.name] = [
                {"bbox_xyxy": [round(v, 2) for v in (d.x1, d.y1, d.x2, d.y2)],
                 "score": round(d.score, 5), "cls": d.cls, "name": d.name}
                for d in dets
            ]
    elapsed = time.perf_counter() - t0

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    n = len(paths)
    print(f"{n} görsel, {elapsed / max(n, 1) * 1000:.0f} ms/kare (okuma dahil) -> {args.out}")


if __name__ == "__main__":
    main()
