"""Değerlendirme girişi — mAP@0.5 (bbox + sınıf).

Kullanım:
    python scripts/evaluate.py preds.json gt.json
    python scripts/evaluate.py preds.json gt.json --json sonuc.json --cross-check

- preds.json : COCO "results" formatı [{image_id, category_id, bbox, score}, ...]
- gt.json    : COCO instances formatı (data/coco/instances_*.json)

Öznitelik (hareket/iniş) skorlaması bu sürümde yok — yalnız bbox + sınıf.
"""

from __future__ import annotations

import argparse
import json

from uav_vision.eval import evaluate


def _fmt(x: float) -> str:
    return "  -  " if x != x else f"{x:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="mAP@0.5 değerlendirme (bbox + sınıf)")
    parser.add_argument("preds", help="Tahmin JSON (COCO results formatı)")
    parser.add_argument("gt", help="Ground-truth JSON (COCO instances formatı)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU eşiği (varsayılan 0.5)")
    parser.add_argument("--json", dest="json_out", help="Sonucu bu yola JSON yaz")
    parser.add_argument(
        "--cross-check",
        action="store_true",
        help="faster-coco-eval ile çapraz kontrol (AP50 / AP@[.5:.95])",
    )
    args = parser.parse_args()

    res = evaluate(args.gt, args.preds, iou_thr=args.iou)

    print(f"\n=== mAP@{args.iou} — bbox + sınıf ===")
    print(f"{'sınıf':<10} {'#GT':>6} {'#pred':>6} {'TP':>5} {'FP':>5} {'FN':>5}  {'AP(voc)':>8} {'AP(coco101)':>11}")
    for c in res.per_class:
        print(
            f"{c.name:<10} {c.n_gt:>6} {c.n_pred:>6} {c.tp:>5} {c.fp:>5} {c.fn:>5}  "
            f"{_fmt(c.ap_voc):>8} {_fmt(c.ap_coco101):>11}"
        )
    print("-" * 66)
    print(f"{'mAP':<10} {'':>6} {'':>6} {'':>5} {'':>5} {'':>5}  {_fmt(res.map_voc):>8} {_fmt(res.map_coco101):>11}")

    if args.cross_check:
        from uav_vision.eval.cocowrap import coco_cross_check

        print("\n--- faster-coco-eval çapraz kontrol ---")
        cc = coco_cross_check(args.gt, args.preds)
        print(f"AP50        = {cc['AP50']:.4f}   (bizim coco101 mAP: {_fmt(res.map_coco101)})")
        print(f"AP@[.5:.95] = {cc['AP@[.5:.95]']:.4f}")
        print(f"AP75        = {cc['AP75']:.4f}")

    if args.json_out:
        out = res.to_dict()
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[json]  {args.json_out}")


if __name__ == "__main__":
    main()
