"""Faz 5 — SAHI tile boyutu/overlap ablasyonu (yerel, GPU üzerinde FPS ölçümlü).

Colab'da FPS önemsiz ama nihai pipeline yerel makinede (RTX 4060 Laptop) çalışacak
— bu yüzden bu ablasyon Colab'da değil burada, yerelde koşturulur.

Kullanım:
    python scripts/ablate_tiling.py uav_runs/visdrone_pre/faz4/baseline-visdrone-yolo11s-p2/weights/best.pt

FPS uçtan uca ölçülür: disk okuma + dilimleme + çıkarım + birleştirme (görsel
başına). En iyi kombinasyon bulununca `sabitler.md`'ye elle kilitlenir (C4 kararı).
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

SPLITS = {
    "test": ("data/uav_ldz/test/images", "data/coco/instances_test.json"),
    "sanity": ("data/sanity/images", "data/coco/instances_sanity.json"),
}


def _run_combo(
    weights: str, img_dir: str, name_to_id: dict, tile: int, overlap: float,
    conf: float, device: str,
) -> tuple[list[dict], float, int]:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=weights,
        confidence_threshold=conf, device=device,
    )

    preds: list[dict] = []
    n = 0
    t0 = time.perf_counter()
    for img_path in sorted(Path(img_dir).iterdir()):
        if img_path.name not in name_to_id:
            continue
        n += 1
        result = get_sliced_prediction(
            str(img_path), detection_model,
            slice_height=tile, slice_width=tile,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            verbose=0,
        )
        image_id = name_to_id[img_path.name]
        for obj in result.object_prediction_list:
            x, y, w, h = obj.bbox.to_xywh()
            preds.append({
                "image_id": image_id,
                "category_id": int(obj.category.id),
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "score": round(float(obj.score.value), 5),
            })
    elapsed = time.perf_counter() - t0
    fps = n / elapsed if elapsed > 0 else float("nan")

    # Her kombinasyonda taze model yükleniyor — GPU/host belleği bir sonrakine
    # sızmasın diye açıkça serbest bırak (art arda 6 kombinasyon OOM'a yol açabiliyor).
    del detection_model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    return preds, fps, n


def main() -> None:
    parser = argparse.ArgumentParser(description="SAHI tile boyutu/overlap ablasyonu")
    parser.add_argument("weights", help="Eğitilmiş .pt")
    parser.add_argument("--split", default="sanity", choices=SPLITS)
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=[512, 640, 1024])
    parser.add_argument("--overlaps", type=float, nargs="+", default=[0.2, 0.3])
    parser.add_argument("--conf", type=float, default=0.001, help="mAP için düşük tut")
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="tile_ablation.json")
    args = parser.parse_args()

    from uav_vision.eval import evaluate

    img_dir, gt_json = SPLITS[args.split]
    with open(gt_json, encoding="utf-8") as f:
        gt = json.load(f)
    name_to_id = {Path(im["file_name"]).name: im["id"] for im in gt["images"]}

    results = []
    for tile in args.tile_sizes:
        for overlap in args.overlaps:
            print(f"\n=== tile={tile} overlap={overlap:.0%} ===")
            preds, fps, n = _run_combo(
                args.weights, img_dir, name_to_id, tile, overlap, args.conf, args.device,
            )
            preds_path = f"preds_{args.split}_tile{tile}_ov{int(overlap * 100)}.json"
            with open(preds_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, ensure_ascii=False)

            res = evaluate(gt_json, preds_path, iou_thr=0.5)
            print(f"mAP@0.5={res.map_voc:.4f}  FPS={fps:.2f}  ({n} görsel, {len(preds)} kutu)")
            results.append({
                "tile": tile,
                "overlap": overlap,
                "map_voc": res.map_voc,
                "map_coco101": res.map_coco101,
                "fps": fps,
                "n_images": n,
                "per_class_ap_voc": {c.name: round(c.ap_voc, 4) for c in res.per_class},
            })

    results.sort(key=lambda r: -r["map_voc"])
    print("\n=== Özet (mAP@0.5'e göre sıralı) ===")
    print(f"{'tile':>6} {'overlap':>8} {'mAP@0.5':>9} {'FPS':>7}")
    for r in results:
        print(f"{r['tile']:>6} {r['overlap']:>8.0%} {r['map_voc']:>9.4f} {r['fps']:>7.2f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[json] {args.out}")
    print("\nEn iyi kombinasyonu seçtikten sonra sabitler.md'ye kilitle (C4 kararı).")


if __name__ == "__main__":
    main()
