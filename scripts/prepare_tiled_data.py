"""Faz 5b — dilimli (tiled) eğitim verisi hazırlığı.

SAHI ile kilitlenen tile parametreleriyle (tile=1024, overlap=%30 — bkz.
sabitler.md C4 kararı) train/val split'lerini diliyor, YOLO formatına çeviriyor.
Amaç: modelin eğitim sırasında da çıkarımdaki (SAHI) tile boyutunu görmesi.

`test`/`sanity` DİLİMLENMEZ — değerlendirme zaten SAHI ile tam görüntü üzerinden
yapılıyor (bkz. scripts/ablate_tiling.py).

Görsel görsel işlenir (sahi.slicing.slice_image) — sahi.slicing.slice_coco'nun
aksine tüm split'i tek seferde belleğe almaz; büyük (4K) görsellerle OOM'a
yol açtığı için bu yola geçildi.

Kullanım:
    python scripts/prepare_tiled_data.py

Çıktı: data/uav_ldz_tiled/{train,val}/{images,labels} + configs/data_tiled.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

TILE = 1024
OVERLAP = 0.30
MIN_AREA_RATIO = 0.2       # sınırda kesilen kutu, orijinalin en az %20'si kalmalı
NEGATIVE_KEEP_RATIO = 0.15  # boş (nesnesiz) tile'ların ne kadarı tutulsun
SEED = 42
JPEG_QUALITY = 90

SPLITS = {
    "train": ("data/uav_ldz/train/images", "data/coco/instances_train.json"),
    "val": ("data/uav_ldz/val/images", "data/coco/instances_val.json"),
}

OUT_ROOT = Path("data/uav_ldz_tiled")


def _slice_split(
    split: str, img_dir: str, gt_json: str, start: int = 0, end: int | None = None
) -> None:
    from PIL import Image
    from sahi.slicing import slice_image
    from sahi.utils.coco import CocoAnnotation

    with open(gt_json, encoding="utf-8") as f:
        gt = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    anns_by_image_id: dict[int, list] = {}
    for a in gt["annotations"]:
        anns_by_image_id.setdefault(a["image_id"], []).append(a)

    out_img_dir = OUT_ROOT / split / "images"
    out_lbl_dir = OUT_ROOT / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    n_pos = n_neg_kept = n_neg_dropped = n_skipped_missing = 0

    total = len(gt["images"])
    images = gt["images"][start:end]
    for idx, im in enumerate(images, start=start):
        img_path = Path(img_dir) / Path(im["file_name"]).name
        if not img_path.exists():
            n_skipped_missing += 1
            continue

        coco_anns = [
            CocoAnnotation.from_coco_bbox(
                bbox=a["bbox"],
                category_id=a["category_id"],
                category_name=id_to_name[a["category_id"]],
            )
            for a in anns_by_image_id.get(im["id"], [])
            if a["bbox"][2] >= 1 and a["bbox"][3] >= 1  # sub-piksel/sıfır en-boy — bozuk etiket, atla
        ]

        result = slice_image(
            image=str(img_path),
            coco_annotation_list=coco_anns,
            slice_height=TILE,
            slice_width=TILE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            min_area_ratio=MIN_AREA_RATIO,
            verbose=False,
        )

        stem = Path(im["file_name"]).stem
        for sliced in result.sliced_image_list:
            has_ann = len(sliced.coco_image.annotations) > 0
            if not has_ann:
                if rng.random() >= NEGATIVE_KEEP_RATIO:
                    n_neg_dropped += 1
                    continue
                n_neg_kept += 1
            else:
                n_pos += 1

            sx, sy = sliced.starting_pixel
            tile_name = f"{stem}_{sx}_{sy}.jpg"
            Image.fromarray(sliced.image).save(
                out_img_dir / tile_name, quality=JPEG_QUALITY
            )

            w, h = sliced.coco_image.width, sliced.coco_image.height
            lines = []
            for ann in sliced.coco_image.annotations:
                x, y, bw, bh = ann.bbox
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                nw, nh = bw / w, bh / h
                lines.append(f"{ann.category_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            (out_lbl_dir / f"{Path(tile_name).stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )

        del result, coco_anns
        if idx % 200 == 0:
            gc.collect()
            print(f"[{split}] {idx}/{total} görsel işlendi...", flush=True)

    print(
        f"[{split}] tamam — pozitif={n_pos}  negatif_tutulan={n_neg_kept}  "
        f"negatif_atılan={n_neg_dropped}  eksik_dosya={n_skipped_missing}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dilimli eğitim verisi hazırlığı")
    parser.add_argument(
        "--split", choices=[*SPLITS, "all"], default="all",
        help="Tek split işle (bellek sorunlarında tek Python süreci içinde küçük "
        "gruplar halinde --start/--end ile çağırmak için)",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--no-config", action="store_true",
        help="configs/data_tiled.yaml yazma (batch modunda ara çağrılarda kullan)",
    )
    args = parser.parse_args()

    splits = SPLITS.items() if args.split == "all" else [(args.split, SPLITS[args.split])]
    for split, (img_dir, gt_json) in splits:
        _slice_split(split, img_dir, gt_json, start=args.start, end=args.end)

    if not args.no_config:
        cfg = f"""path: {OUT_ROOT.resolve()}
train: train/images
val: val/images
names:
  0: vehicle
  1: human
  2: uap
  3: uai
"""
        Path("configs/data_tiled.yaml").write_text(cfg, encoding="utf-8")
        print("\n[config] configs/data_tiled.yaml yazıldı.")


if __name__ == "__main__":
    main()
