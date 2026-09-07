"""faster-coco-eval sarmalayıcı — kendi hesabımız için çapraz kontrol.

Kendi VOC-tarzı evaluator'ımızın (core.evaluate) sonucunu, endüstri standardı
COCO protokolüyle karşılaştırmak için. Beklenti: bizim mAP@0.5 (coco101) ~=
buradaki AP50 (±%1-2 sayısal gürültü).
"""

from __future__ import annotations

import json
from pathlib import Path


def coco_cross_check(gt_json: str | Path, pred_json: str | Path) -> dict:
    """faster-coco-eval ile AP50 + AP@[.5:.95] + sınıf-başı AP50 döndür.

    faster-coco-eval kurulu değilse RuntimeError.
    """
    try:
        from faster_coco_eval import COCO, COCOeval_faster
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("faster-coco-eval kurulu değil (pip install faster-coco-eval)") from e

    coco_gt = COCO(str(gt_json))
    preds = json.loads(Path(pred_json).read_text(encoding="utf-8"))
    if isinstance(preds, dict) and "annotations" in preds:
        preds = preds["annotations"]
    coco_dt = coco_gt.loadRes(preds)

    ev = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # ev.stats: [AP@.5:.95, AP50, AP75, AP_s, AP_m, AP_l, AR1, AR10, AR100, ...]
    stats = list(ev.stats)
    out = {"AP@[.5:.95]": stats[0], "AP50": stats[1], "AP75": stats[2]}

    # Sınıf-başı AP50: precision boyutu [T, R, K, A, M]; T=0 -> IoU 0.5, A=0 all, M=-1
    try:
        import numpy as np

        prec = ev.eval["precision"]  # [T,R,K,A,M]
        cat_ids = coco_gt.getCatIds()
        per_class = {}
        for k, cid in enumerate(cat_ids):
            p = prec[0, :, k, 0, -1]
            p = p[p > -1]
            name = coco_gt.loadCats([cid])[0]["name"]
            per_class[name] = float(np.mean(p)) if p.size else float("nan")
        out["per_class_AP50"] = per_class
    except Exception:  # pragma: no cover - çapraz kontrol, kritik değil
        pass

    return out
