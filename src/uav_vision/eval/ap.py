"""Precision-Recall eğrisinden Average Precision (AP).

Şartname "klasik nesne tespiti yöntemlerindeki gibi mAP, IoU eşiği 0.5" diyor
ama AP eğrisi altındaki alanın hangi yöntemle hesaplanacağını belirtmiyor. Bu
yüzden iki yaygın yöntemi de raporluyoruz:

  - "voc"     : PASCAL VOC 2012 / tüm-nokta interpolasyonu (monoton precision
                zarfının recall'e göre integrali).
  - "coco101" : COCO'nun 101-nokta interpolasyonu (recall 0.00..1.00, adım 0.01).

Pratikte ikisi arasındaki fark %1-3 seviyesindedir.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Method = str  # "voc" | "coco101"


def pr_curve(
    tp: Sequence[int],
    fp: Sequence[int],
    scores: Sequence[float],
    n_gt: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Güven skoruna göre azalan sıralanmış tp/fp bayraklarından PR eğrisi.

    Dönüş: (recall, precision) — kümülatif, tahmin sayısıyla aynı uzunlukta.
    """
    idx = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    tp_c = np.cumsum(np.asarray(tp, dtype=float)[idx])
    fp_c = np.cumsum(np.asarray(fp, dtype=float)[idx])

    recall = tp_c / n_gt if n_gt > 0 else np.zeros_like(tp_c)
    precision = np.divide(tp_c, tp_c + fp_c, out=np.zeros_like(tp_c), where=(tp_c + fp_c) > 0)
    return recall, precision


def average_precision(
    tp: Sequence[int],
    fp: Sequence[int],
    scores: Sequence[float],
    n_gt: int,
    method: Method = "coco101",
) -> float:
    """Tek sınıf için AP. n_gt == 0 ise sınıf tanımsız (NaN döner)."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0

    recall, precision = pr_curve(tp, fp, scores, n_gt)

    if method == "voc":
        # VOC 2012 / tüm-nokta: precision zarfı + recall değişimlerinin integrali.
        mpre = np.concatenate([[0.0], precision, [0.0]])
        mrec = np.concatenate([[0.0], recall, [1.0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        change = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))

    if method == "coco101":
        # pycocotools ile aynı: precision'ı sağdan monoton yap, 101 recall
        # noktasında searchsorted ile örnekle, max recall'un ötesinde 0.
        pr = np.maximum.accumulate(precision[::-1])[::-1]
        rec_pts = np.linspace(0.0, 1.0, 101)
        idx = np.searchsorted(recall, rec_pts, side="left")
        out = np.zeros(101)
        valid = idx < len(pr)
        out[valid] = pr[idx[valid]]
        return float(out.mean())

    raise ValueError(f"Bilinmeyen AP yöntemi: {method!r}")


def mean_ap(per_class_ap: dict[int, float]) -> float:
    """Tanımlı (NaN olmayan) sınıf AP'lerinin ortalaması."""
    vals = [v for v in per_class_ap.values() if not _isnan(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _isnan(x: float) -> bool:
    return x != x
