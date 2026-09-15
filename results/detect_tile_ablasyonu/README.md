# Faz 5 — SAHI Tile Ablasyonu

Yerelde (RTX 4060 Laptop) `scripts/ablate_tiling.py` ile çalıştırıldı. Ağırlık:
Faz 4 çıktısı (`uav_runs/visdrone_pre/faz4/baseline-visdrone-yolo11s-p2/weights/best.pt`,
COCO→VisDrone-DET→bizim veri). Değerlendirme split'i: `sanity` (yarışma verisi, 175 görsel).

## Sonuçlar

| tile | overlap | mAP@0.5 | human AP | vehicle AP | FPS |
|---|---|---|---|---|---|
| **1024** | **%30** | **0.8419** | **0.5664** | **0.8923** | 2.64 |
| 1024 | %20 | 0.8371 | 0.5567 | 0.8919 | 2.66 |
| 640 | %30 | 0.8178 | 0.4977 | 0.8870 | 2.29 |
| 640 | %20 | 0.8176 | 0.5087 | 0.8885 | 2.24 |
| 512 | %20 | 0.7762 | 0.4538 | 0.8590 | 1.41 |
| 512 | %30 | 0.7649 | 0.4591 | 0.8599 | 1.41 |

## Karar

**tile=1024, overlap=%30** kazandı — `sabitler.md`'ye kilitlendi (C4 kararı).

## Kıyas (aşama aşama ilerleme, sanity mAP@0.5)

| Aşama | mAP@0.5 | human AP |
|---|---|---|
| Faz 3 (baseline, COCO→bizim veri) | 0.8048 | 0.3776 |
| Faz 4 (COCO→VisDrone→bizim veri) | 0.8203 | 0.4559 |
| **Faz 5 (SAHI, tile=1024/%30, Faz 4 ağırlığı üstünde)** | **0.8419** | **0.5664** |

## Gözlemler

- **Beklenmedik bulgu:** küçük tile (512) daha kötü sonuç veriyor. Aşırı parçalanma
  her dilimde ayrı düşük-conf gürültü üretip toplam false-positive'i patlatıyor
  (512'de 175 görselde ~38K kutu, 1024'te ~6.9K kutu). "Küçük dilim = küçük nesnede
  daha iyi" varsayımı bu veri setinde tersine döndü.
- `human` sınıfı en çok kazanan oldu (+0.19 AP, Faz 3'e göre kümülatif) — sanity
  görsellerindeki küçük/düşük çözünürlüklü insan örnekleri SAHI'den en çok fayda
  gören grup.
- FPS (2.64) hız bütçesinin (~0.6 FPS gerekli, `sabitler.md` → FPS_BUTCESI) çok
  üzerinde — daha büyük/ağır bir model veya TTA için hâlâ pay var.

## Dosyalar

- `tile_ablation_sanity.json` — özet (6 kombinasyon, mAP + FPS + sınıf-başı AP)
- `preds_sanity_tile<N>_ov<M>.json` — her kombinasyonun ham COCO-results tahminleri
