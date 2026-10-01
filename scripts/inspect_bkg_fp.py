"""Arka plan (Bkg) yanlış pozitiflerini gözle incelemek için kırpıntı panosu üretir.

    python scripts/inspect_bkg_fp.py preds.json data/coco/instances_sanity.json \
        --images data/sanity/images --cls 1 --top 60 --out results/bkg_fp_v2

Bkg tanımı TIDE ile aynı: tahmin, herhangi bir sınıftaki hiçbir GT ile IoU ≥ 0.1
yapmıyor. Skora göre sıralanır, en yüksek skorlu `--top` tanesi kırpılır.
Kırpıntı: kutu merkezli, kenarı max(4×kutu, 128 px) — bağlam görünsün diye.
Kırmızı = FP kutusu, yeşil = kırpıntıya düşen GT kutuları.
Ayrıca aynı sınıfın TP skor dağılımıyla karşılaştırma özeti yazılır.
Model çalıştırılmaz; offline.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BG_IOU = 0.1  # TIDE arka plan eşiği
POS_IOU = 0.5  # METRIK_G1
TILE = 192  # panodaki kırpıntı kenarı (px)


def iou(a, b) -> float:
    """a, b: COCO xywh."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def classify(preds, gts_by_img, cls):
    """Sınıfın tahminlerini skor sırasıyla TP / Bkg / diğer olarak ayırır (açgözlü eşleşme)."""
    used = set()
    tp, bkg = [], []
    for p in sorted((p for p in preds if p["category_id"] == cls), key=lambda p: -p["score"]):
        gts = gts_by_img[p["image_id"]]
        ious = [iou(p["bbox"], g["bbox"]) for g in gts]
        if not ious or max(ious) < BG_IOU:
            bkg.append(p)
            continue
        best = max(
            (i for i, g in enumerate(gts) if g["category_id"] == cls and (p["image_id"], i) not in used),
            key=lambda i: ious[i], default=None,
        )
        if best is not None and ious[best] >= POS_IOU:
            used.add((p["image_id"], best))
            tp.append(p)
    return tp, bkg


def crop(img: Image.Image, p, gts) -> Image.Image:
    x, y, w, h = p["bbox"]
    side = max(4 * max(w, h), 128)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = cx - side / 2, cy - side / 2
    c = img.crop((int(x0), int(y0), int(x0 + side), int(y0 + side))).resize((TILE, TILE))
    s = TILE / side
    d = ImageDraw.Draw(c)
    for g in gts:
        gx, gy, gw, gh = g["bbox"]
        d.rectangle([(gx - x0) * s, (gy - y0) * s, (gx + gw - x0) * s, (gy + gh - y0) * s], outline=(0, 255, 0))
    d.rectangle([(x - x0) * s, (y - y0) * s, (x + w - x0) * s, (y + h - y0) * s], outline=(255, 0, 0), width=2)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("gt")
    ap.add_argument("--images", required=True)
    ap.add_argument("--cls", type=int, default=1, help="1 = human")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", default="results/bkg_fp")
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    preds = json.loads(Path(args.preds).read_text(encoding="utf-8"))
    files = {im["id"]: im["file_name"] for im in gt["images"]}
    gts_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gts_by_img[a["image_id"]].append(a)

    tp, bkg = classify(preds, gts_by_img, args.cls)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Özet: Bkg FP'ler skor eşiği üstünde kaç tane, TP'lerle karşılaştırma
    tps = np.array([p["score"] for p in tp])
    bks = np.array([p["score"] for p in bkg])
    lines = [f"sınıf {args.cls}: TP={len(tps)}  Bkg={len(bks)}"]
    for t in (0.1, 0.25, 0.5, 0.7):
        lines.append(f"  skor ≥ {t}: TP {int((tps >= t).sum())}  Bkg {int((bks >= t).sum())}")
    if len(tps):
        lines.append(f"  TP skor medyan {np.median(tps):.3f}  min {tps.min():.3f}")

    # Pano + tablo
    top = bkg[: args.top]
    rows = -(-len(top) // args.cols)
    sheet = Image.new("RGB", (args.cols * TILE, rows * (TILE + 16)), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    cache: dict[int, Image.Image] = {}
    table = ["| # | skor | görsel | kutu (x,y,w,h) |", "|---|---|---|---|"]
    for i, p in enumerate(top):
        if p["image_id"] not in cache:
            cache = {p["image_id"]: Image.open(Path(args.images) / files[p["image_id"]]).convert("RGB")}
        c = crop(cache[p["image_id"]], p, gts_by_img[p["image_id"]])
        r, k = divmod(i, args.cols)
        sheet.paste(c, (k * TILE, r * (TILE + 16)))
        d.text((k * TILE + 4, r * (TILE + 16) + TILE + 2), f"#{i} {p['score']:.2f} im{p['image_id']}", fill=(255, 255, 255))
        bb = ", ".join(f"{v:.0f}" for v in p["bbox"])
        table.append(f"| {i} | {p['score']:.3f} | {files[p['image_id']]} | {bb} |")

    sheet.save(out / f"bkg_fp_cls{args.cls}_top{args.top}.jpg", quality=92)
    (out / f"bkg_fp_cls{args.cls}_top{args.top}.md").write_text(
        "\n".join(["```", *lines, "```", "", *table, ""]), encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
