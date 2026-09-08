"""Eğitilmiş modeli bir split üzerinde çalıştır -> COCO "results" JSON üret.

    python scripts/predict_to_coco.py runs/faz3/baseline-yolo11s-p2/weights/best.pt test
    python scripts/predict_to_coco.py <best.pt> sanity --out preds_sanity.json --conf 0.001

Çıktı scripts/evaluate.py'ye girer:
    python scripts/evaluate.py preds_test.json data/coco/instances_test.json --cross-check

image_id eşlemesi GT json'daki file_name -> id üzerinden yapılır (sıra/isim garanti).
Sınıf id'leri modelin çıktısıyla aynı: 0=vehicle 1=human 2=uap 3=uai.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# split -> (görsel klasörü, GT json)
SPLITS = {
    "train": ("data/uav_ldz/train/images", "data/coco/instances_train.json"),
    "val": ("data/uav_ldz/val/images", "data/coco/instances_val.json"),
    "test": ("data/uav_ldz/test/images", "data/coco/instances_test.json"),
    "sanity": ("data/sanity/images", "data/coco/instances_sanity.json"),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Model -> COCO results JSON")
    p.add_argument("weights", help="Eğitilmiş .pt")
    p.add_argument("split", choices=SPLITS, help="Değerlendirme split'i")
    p.add_argument("--out", help="Çıktı JSON (varsayılan preds_<split>.json)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.001, help="mAP için düşük tut")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    p.add_argument("--device", default="0")
    args = p.parse_args()

    img_dir, gt_json = SPLITS[args.split]
    out = args.out or f"preds_{args.split}.json"

    with open(gt_json, encoding="utf-8") as f:
        gt = json.load(f)
    name_to_id = {Path(im["file_name"]).name: im["id"] for im in gt["images"]}

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.predict(
        source=img_dir, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        device=args.device, stream=True, verbose=False,
    )

    preds: list[dict] = []
    seen = 0
    for r in results:
        key = Path(r.path).name
        if key not in name_to_id:
            continue
        seen += 1
        image_id = name_to_id[key]
        b = r.boxes
        if b is None:
            continue
        for xyxy, cls, score in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist(), strict=True):
            x1, y1, x2, y2 = xyxy
            preds.append({
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(float(score), 5),
            })

    with open(out, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    print(f"[{args.split}] {seen}/{len(name_to_id)} görsel eşleşti, {len(preds)} kutu -> {out}")


if __name__ == "__main__":
    main()
