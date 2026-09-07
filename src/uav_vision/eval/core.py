"""Uçtan uca değerlendirme: COCO GT json + COCO tahmin json -> mAP@0.5.

Tahmin json formatı (COCO "results" formatı): liste, her eleman
    {"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}

GT json: standart COCO instances formatı (images / annotations / categories).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .ap import average_precision, mean_ap
from .matching import IOU_THR, Box, match_image


@dataclass
class ClassScore:
    class_id: int
    name: str
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    ap_voc: float
    ap_coco101: float


@dataclass
class EvalResult:
    iou_thr: float
    per_class: list[ClassScore] = field(default_factory=list)
    map_voc: float = float("nan")
    map_coco101: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "iou_thr": self.iou_thr,
            "mAP@0.5_voc": self.map_voc,
            "mAP@0.5_coco101": self.map_coco101,
            "per_class": [vars(c) for c in self.per_class],
        }


def _load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _index_gt(gt: dict) -> tuple[dict[int, dict[int, list[Box]]], dict[int, str], set[int]]:
    """Dönüş: {class_id: {image_id: [box, ...]}}, {class_id: name}, tüm image_id'ler."""
    names = {c["id"]: c["name"] for c in gt["categories"]}
    image_ids = {im["id"] for im in gt["images"]}
    by_class: dict[int, dict[int, list[Box]]] = defaultdict(lambda: defaultdict(list))
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        by_class[ann["category_id"]][ann["image_id"]].append((x, y, w, h))
    return by_class, names, image_ids


def _index_preds(preds: list[dict]) -> dict[int, dict[int, list[tuple[Box, float]]]]:
    by_class: dict[int, dict[int, list[tuple[Box, float]]]] = defaultdict(lambda: defaultdict(list))
    for p in preds:
        x, y, w, h = p["bbox"]
        by_class[p["category_id"]][p["image_id"]].append(((x, y, w, h), float(p["score"])))
    return by_class


def evaluate(
    gt_json: str | Path,
    pred_json: str | Path,
    iou_thr: float = IOU_THR,
) -> EvalResult:
    gt = _load_json(gt_json)
    preds = _load_json(pred_json)
    if isinstance(preds, dict) and "annotations" in preds:  # yanlışlıkla GT formatı verilirse
        preds = [
            {**a, "score": a.get("score", 1.0)}
            for a in preds["annotations"]
        ]

    gt_by_class, names, image_ids = _index_gt(gt)
    pred_by_class = _index_preds(preds)

    result = EvalResult(iou_thr=iou_thr)
    all_class_ids = sorted(set(names) | set(gt_by_class) | set(pred_by_class))

    ap_voc_map: dict[int, float] = {}
    ap_coco_map: dict[int, float] = {}

    for cid in all_class_ids:
        gt_imgs = gt_by_class.get(cid, {})
        pred_imgs = pred_by_class.get(cid, {})
        n_gt = sum(len(v) for v in gt_imgs.values())

        tp_all: list[int] = []
        fp_all: list[int] = []
        sc_all: list[float] = []

        for img_id in image_ids:
            g_boxes = gt_imgs.get(img_id, [])
            pp = pred_imgs.get(img_id, [])
            p_boxes = [b for b, _ in pp]
            p_scores = [s for _, s in pp]
            m = match_image(g_boxes, p_boxes, p_scores, iou_thr)
            tp_all.extend(m.tp)
            fp_all.extend(m.fp)
            sc_all.extend(m.scores)

        n_pred = len(tp_all)
        n_tp = sum(tp_all)
        n_fp = sum(fp_all)
        ap_voc = average_precision(tp_all, fp_all, sc_all, n_gt, "voc")
        ap_coco = average_precision(tp_all, fp_all, sc_all, n_gt, "coco101")
        ap_voc_map[cid] = ap_voc
        ap_coco_map[cid] = ap_coco

        result.per_class.append(
            ClassScore(
                class_id=cid,
                name=names.get(cid, str(cid)),
                n_gt=n_gt,
                n_pred=n_pred,
                tp=n_tp,
                fp=n_fp,
                fn=n_gt - n_tp,
                ap_voc=ap_voc,
                ap_coco101=ap_coco,
            )
        )

    result.map_voc = mean_ap(ap_voc_map)
    result.map_coco101 = mean_ap(ap_coco_map)
    return result
