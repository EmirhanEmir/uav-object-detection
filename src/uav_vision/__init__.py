"""uav_vision — İHA alt-görüş kamerasından nesne tespiti pipeline'ı.

TEKNOFEST Havacılıkta Yapay Zeka — Görev 1: nesne tespiti (taşıt/insan/UAP/UAİ),
tespit edilen nesnelerin öznitelikleri (hareket durumu, iniş alanı uygunluğu).

Alt paketler:
- data:     veri yükleme, sınıf remap, YOLO <-> COCO dönüşümü
- detector: model tanımı, eğitim, çıkarım
- eval:     mAP@0.5 değerlendirme
- utils:    ortak yardımcılar
"""

__version__ = "0.1.0"
