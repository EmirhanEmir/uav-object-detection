"""Ana detektör — YOLO11s-p2 + V2 dilimleme + tam kare, NMS ile birleştirme.

V2 düzeni (seçim gerekçesi: `results/cikarim_maliyet_ablasyonu/`):
  - 1920×1080 karede 2 kare tile, 1080×1080: x ∈ {0, 840}, y = 0 (yatay %22, dikey %0 örtüşme)
  - Tile'lar modelin imgsz'ine (1280) büyütülür (1.19×) — büyütme human AP için şart
  - Ek olarak tam kare geçişi: 1920×1080 → 1280×720 (letterbox 736×1280)
  - 3 forward/kare, ~298 GFLOPs (eski 6×1024 akışına göre −%61)
  - Birleştirme: sınıf bazlı NMS, IoU 0.5 (GREEDYNMM/NMM ablasyonda hep daha kötüydü)

Birimler: tüm kutular orijinal kare pikselinde, (x1, y1, x2, y2), sol-üst köşe orijin.
Sınıf id'leri: 0=vehicle 1=human 2=uap 3=uai (model çıktısıyla aynı, bkz. uav_vision.classes).
Giriş karesi OpenCV BGR (H, W, 3) uint8 bekler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uav_vision.classes import CLASS_NAMES

# V2 sabitleri — değiştirmek ablasyonla seçilen maliyet/doğruluk dengesini bozar
TILE_SIZE = 1080  # px, kare tile kenarı
IMGSZ = 1280  # modelin eğitim çözünürlüğü; tile'lar ve tam kare buna ölçeklenir
MERGE_IOU = 0.5  # tile + tam kare tahminlerini birleştiren sınıf bazlı NMS eşiği
MODEL_NMS_IOU = 0.7  # her forward'ın kendi içindeki NMS'i (Ultralytics varsayılanı)
MAX_DET = 300  # forward başına kutu sınırı (Ultralytics varsayılanı)


@dataclass(frozen=True, slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int

    @property
    def name(self) -> str:
        return CLASS_NAMES[self.cls]


def tile_origins(length: int, tile: int) -> list[int]:
    """Bir eksen boyunca kapsayan en az sayıda tile'ın başlangıç koordinatları.

    Tile'lar eşit aralıklı dağılır; ilk 0'da, son `length - tile`'da başlar.
    1920 / 1080 → [0, 840];  1080 / 1080 → [0].
    """
    if length <= tile:
        return [0]
    n = math.ceil(length / tile)
    step = (length - tile) / (n - 1)
    return [round(i * step) for i in range(n)]


class Detector:
    """Tek kare → birleştirilmiş tespit listesi.

    >>> det = Detector("weights/best.pt")
    >>> dets = det(cv2.imread("kare.jpg"))
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "0",
        conf: float = 0.001,
        full_frame: bool = True,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        self.conf = conf
        self.full_frame = full_frame

    def _forward(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Görseller → her biri için (N, 6) [x1, y1, x2, y2, score, cls] (görselin kendi pikselinde)."""
        results = self.model.predict(
            images, imgsz=IMGSZ, conf=self.conf, iou=MODEL_NMS_IOU, max_det=MAX_DET,
            device=self.device, verbose=False,
        )
        return [r.boxes.data[:, :6].cpu().numpy() for r in results]

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        tile = min(TILE_SIZE, h, w)
        origins = [(x, y) for y in tile_origins(h, tile) for x in tile_origins(w, tile)]

        crops = [frame[y:y + tile, x:x + tile] for x, y in origins]
        parts = []
        for (x, y), boxes in zip(origins, self._forward(crops), strict=True):
            boxes[:, [0, 2]] += x
            boxes[:, [1, 3]] += y
            parts.append(boxes)
        if self.full_frame and len(origins) > 1:
            parts.extend(self._forward([frame]))

        merged = _merge(np.concatenate(parts), w, h)
        return [
            Detection(float(x1), float(y1), float(x2), float(y2), float(s), int(c))
            for x1, y1, x2, y2, s, c in merged
        ]


def _merge(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """Kareye kırp, sıfır alanlıları at, sınıf bazlı NMS (IoU > MERGE_IOU bastırılır)."""
    import torch
    from torchvision.ops import batched_nms

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    if len(boxes) == 0:
        return boxes
    t = torch.from_numpy(boxes)
    keep = batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), MERGE_IOU)
    return boxes[keep.numpy()]
