"""Detektör katmanı — eğitim ve çıkarım (baseline: Ultralytics YOLO11 + P2 başlığı)."""

from .detector import Detection, Detector, tile_origins

__all__ = ["Detection", "Detector", "tile_origins"]
