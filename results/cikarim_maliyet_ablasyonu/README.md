# Çıkarım Maliyeti Ablasyonu

Kısıt: kare başına çıkarım maliyeti mevcut akışın üstüne **çıkamaz**. Amaç, doğruluğu
koruyarak maliyeti düşürmek. Yeniden eğitim yok; tüm varyantlar aynı transfer ağırlığıyla
(COCO → VisDrone-DET → kendi veri seti, YOLO11s-p2) çalıştırıldı.

- Donanım: RTX 4060 Laptop (8 GB), FP32; süre = wall-clock ms/kare (`cuda.synchronize`)
- Değerlendirme: `holdout` seti (175 kare, 1920×1080), mAP@0.5 (kendi VOC evaluator'ümüz)
- GFLOPs: forward başına thop ölçümü (1280² = 115.8, 736×1280 = 66.6), piksel sayısıyla ölçeklendi
- Yöntem: SAHI'nin dilimleme ve çıkarım kodu kullanıldı. Birleştirilmemiş (ham) tahminler bir kez
  kaydedildi, birleştirme offline olarak SAHI'nin kendi postprocess sınıflarıyla yapıldı.
  V0, SAHI tile ablasyonunun sonucunu birebir tekrar etti (0.8419 / human 0.5664).

## Mevcut akışın maliyeti (V0)

- 1024'lük tile, %30 overlap → 6 tile: x ∈ {0, 717, 896}, y ∈ {0, 56}. Satırlar %95, orta ile sağ
  sütun %82 örtüşüyor.
- Checkpoint `imgsz=1280` olduğu için tile'lar 1.25× büyütülüyor, tam kare 736×1280'de işleniyor.
- 7 forward/kare, ~761 GFLOPs, 303 ms/kare (%91'i model hesabı), tepe VRAM 430 MB.

## Varyantlar (birleştirme: NMS / IOU / 0.5 / sınıf bazlı)

| # | Düzen | GFLOPs | ms/kare | mAP@0.5 | human | vehicle |
|---|---|---|---|---|---|---|
| V0 | 6 × 1024 @1280 + tam kare (mevcut) | 761 | 303 | 0.8419 | 0.5664 | 0.8923 |
| V1 | 6 × 1024 @1024 + tam kare | 511 | 216 | 0.8125 | 0.4830 | 0.8818 |
| **V2** | **2 × 1080 @1280 + tam kare** | **298** | **154** | **0.8457** | 0.5432 | 0.8989 |
| V3 | 2 × 1080 @1088 + tam kare | 234 | 136 | 0.8386 | 0.5148 | 0.8893 |
| V4 | 6 × 1024 @1280 | 695 | 251 | 0.8366 | 0.5532 | 0.8857 |
| V5 | 6 × 1024 @1024 | 445 | 164 | 0.8029 | 0.4566 | 0.8727 |
| V6 | 2 × 1080 @1280 | 232 | 102 | 0.8343 | 0.5047 | 0.8940 |
| V7 | 2 × 1080 @1088 | 167 | 84 | 0.8341 | 0.5085 | 0.8812 |

2 × 1080 düzeni: x ∈ {0, 840}, y = 0 (yatay %20, dikey %0 overlap). ms/kare değerleri,
tile ve tam kare geçişlerinin ayrı ayrı ölçülüp toplanmasıyla elde edildi.

## Gözlemler

- Asıl israf tekrarlanan tile'larda. V2 maliyeti %61 düşürüyor ve mAP'yi koruyor.
- Tile büyütmesi human için şart: büyütme kaldırılınca (@1024/@1088) human AP belirgin düşüyor.
- V2'de büyütme oranı 1.25×'ten 1.19×'e iniyor. Human kaybının bir kısmı buradan geliyor olabilir
  (@1344 denemesi açık).
- Birleştirme (`varyant_birlestirme_grid.json`, varyant başına 36 ayar): NMS/IOU en iyi aile,
  GREEDYNMM ve NMM her varyantta daha kötü. V0'da NMS/IOS/0.7 → mAP 0.8596 (kazanç
  vehicle/uap/uai'de, human değişmiyor). Aynı sette seçildiği için iyimser bir sonuç.
- SAHI, `conf=0.001` verildiğinde birleştirme yöntemini sessizce GREEDYNMM/IOS'tan NMS/IOU'ya çeviriyor.
- FP16 (`fp16.json`): sadece ~%7 hızlanma, human AP ~−0.023 (V0 ve V2'de). Tercih edilmedi.

## Belirsizlik

Holdout setindeki GT: vehicle 306, **human 57**, uap 15, uai 12. Eşli bootstrap (200 tekrar, %95 GA):

| Karşılaştırma | Δ mAP | Δ human AP |
|---|---|---|
| V2 − V0 | +0.003 [−0.020, +0.025] | −0.023 [−0.071, +0.028] |
| V4 − V0 | −0.004 [−0.016, +0.009] | −0.012 [−0.042, +0.016] |
| V0 (NMS/IOS/0.7) − V0 | +0.017 [+0.002, +0.033] | −0.002 [−0.013, +0.007] |

**Seçim: V2.** Maliyeti %61 düşürüp mAP'yi koruduğu için ana çıkarım düzeni oldu (`main.py`,
`src/uav_vision/detector/`). human farkı bu örneklemde anlamlı değil; daha büyük bir
değerlendirme setinde yeniden doğrulanması planlanıyor.

## Dosyalar

- `varyant_birlestirme_grid.json`: 8 varyant × 36 birleştirme ayarı (mAP, sınıf başına AP, GFLOPs, ms)
- `fp16.json`: V0 ve V2 için FP16 sonuçları
