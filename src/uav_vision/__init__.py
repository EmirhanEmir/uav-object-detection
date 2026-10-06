"""uav_vision — İHA alt-görüş kamerasından nesne tespiti pipeline'ı.

Alt-görüş (yaklaşık dik) drone karelerinde taşıt / insan / iniş alanı işareti
tespiti; küçük nesne için dilimlemeli (tiling) eğitim ve çıkarım.

Alt paketler:
- classes:  sınıf tanımları
- detector: dilimlemeli çıkarım (YOLO11s-p2 + tile + tam kare)
- eval:     mAP@0.5 değerlendirme
- utils:    ortak yardımcılar
"""

__version__ = "0.1.0"
