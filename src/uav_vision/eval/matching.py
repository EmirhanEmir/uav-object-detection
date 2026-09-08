"""Tespit <-> gerçek referans kutu eşleştirmesi (tek görsel, tek sınıf).

Puanlama:
  - IoU eşiği 0.5 (sabit).
  - Tahminler güven skoruna göre azalan sırada işlenir.
  - Her tahmin, henüz eşleşmemiş GT'ler içinde IoU'su en yüksek ve >= eşik olana
    atanır -> TP. Eşleşemeyen tahmin -> FP. Aynı GT'ye ikinci tahmin -> FP
    Hiç eşleşmeyen GT -> FN.

Bu modül öznitelik (hareket/iniş) bakmaz; sadece kutu + sınıf. Öznitelik-koşullu
skorlama sonraki bir sürüme bırakıldı.
"""

from __future__ import annotations

from dataclasses import dataclass

Box = tuple[float, float, float, float]  # x_min, y_min, w, h (mutlak piksel)

IOU_THR = 0.5


def iou_xywh(a: Box, b: Box) -> float:
    """[x, y, w, h] formatındaki iki kutunun IoU'su."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class MatchResult:
    """Tek görsel + tek sınıf için eşleştirme sonucu.

    tp / fp: tahmin sayısıyla aynı uzunlukta, güven skoruna göre azalan sıralı
    0/1 bayrak listeleri (bir tahmin ya TP ya FP'dir).
    scores: aynı sıradaki güven skorları (AP eğrisi için).
    n_gt: bu görseldeki gerçek kutu sayısı (FN = n_gt - sum(tp)).
    """

    tp: list[int]
    fp: list[int]
    scores: list[float]
    n_gt: int

    @property
    def n_fn(self) -> int:
        return self.n_gt - sum(self.tp)


def match_image(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    pred_scores: list[float],
    iou_thr: float = IOU_THR,
) -> MatchResult:
    """Bir görselde tek bir sınıfın tahminlerini GT'lerle eşleştir."""
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)
    matched_gt: set[int] = set()
    tp = [0] * len(pred_boxes)
    fp = [0] * len(pred_boxes)

    for rank, pi in enumerate(order):
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            v = iou_xywh(gt, pred_boxes[pi])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_gi)
            tp[rank] = 1
        else:
            fp[rank] = 1

    scores = [pred_scores[pi] for pi in order]
    return MatchResult(tp=tp, fp=fp, scores=scores, n_gt=len(gt_boxes))
