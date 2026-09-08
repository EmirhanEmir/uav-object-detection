"""Değerlendirme katmanı — mAP@IoU=0.5 (bbox + sınıf).

Bu sürüm kapsamı: sadece kutu + sınıf eşlemesi. Öznitelik-koşullu skorlama
(hareket/iniş durumu) sonraki bir sürüme bırakıldı.

  - matching : greedy tahmin<->GT eşleştirme
  - ap       : PR eğrisinden AP (VOC tüm-nokta + COCO 101-nokta)
  - core     : evaluate(gt_json, pred_json) -> EvalResult
  - cocowrap : faster-coco-eval ile çapraz kontrol
"""

from .ap import average_precision, mean_ap, pr_curve
from .core import ClassScore, EvalResult, evaluate
from .matching import IOU_THR, iou_xywh, match_image

__all__ = [
    "IOU_THR",
    "ClassScore",
    "EvalResult",
    "average_precision",
    "evaluate",
    "iou_xywh",
    "match_image",
    "mean_ap",
    "pr_curve",
]
