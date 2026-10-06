# SAHI Tile Ablasyonu

Yerelde (RTX 4060 Laptop) `scripts/analysis/ablate_tiling.py` ile çalıştırıldı.
Ağırlık: iki aşamalı transfer modeli (COCO → VisDrone-DET → kendi veri seti, YOLO11s-p2).
Değerlendirme: `holdout` seti (eğitimde hiç görülmeyen ayrı kaynak, 175 görsel, 1920×1080).

## Sonuçlar

| tile | overlap | mAP@0.5 | human AP | vehicle AP | FPS |
|---|---|---|---|---|---|
| **1024** | **%30** | **0.8419** | **0.5664** | **0.8923** | 2.64 |
| 1024 | %20 | 0.8371 | 0.5567 | 0.8919 | 2.66 |
| 640 | %30 | 0.8178 | 0.4977 | 0.8870 | 2.29 |
| 640 | %20 | 0.8176 | 0.5087 | 0.8885 | 2.24 |
| 512 | %20 | 0.7762 | 0.4538 | 0.8590 | 1.41 |
| 512 | %30 | 0.7649 | 0.4591 | 0.8599 | 1.41 |

**tile=1024, overlap=%30** en iyi sonucu verdi. Bu düzen daha sonra çıkarım maliyeti
ablasyonunda (`../cikarim_maliyet_ablasyonu/`) V0 olarak referans alındı ve yerini
daha ucuz V2 düzenine bıraktı.

## Aşama aşama ilerleme (holdout, mAP@0.5)

| Aşama | mAP@0.5 | human AP |
|---|---|---|
| Baseline (COCO → veri seti) | 0.8048 | 0.3776 |
| + VisDrone ara-domain transfer | 0.8203 | 0.4559 |
| **+ SAHI dilimleme (tile=1024, %30)** | **0.8419** | **0.5664** |

## Gözlemler

- **Beklenmedik bulgu:** küçük tile (512) daha kötü sonuç veriyor. Aşırı parçalanma
  her dilimde ayrı düşük güvenli gürültü üretip toplam yanlış pozitifi patlatıyor
  (512'de 175 görselde ~38K kutu, 1024'te ~6.9K kutu). "Küçük dilim = küçük nesnede
  daha iyi" varsayımı bu veri setinde tersine döndü.
- `human` sınıfı en çok kazanan oldu (baseline'a göre kümülatif +0.19 AP): holdout
  görsellerindeki küçük, düşük çözünürlüklü insanlar dilimlemeden en çok fayda gören grup.

## Dosyalar

- `tile_ablation_holdout.json` — özet (6 kombinasyon, mAP + FPS + sınıf başına AP)
