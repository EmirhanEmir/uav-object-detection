# Mimari

Pipeline bağımsız katmanlardan oluşur:

| Katman | Yer | Sorumluluk |
|---|---|---|
| Sınıflar | `uav_vision.classes` | Sınıf id ↔ ad eşlemesi (tek kaynak) |
| Detektör | `uav_vision.detector` | YOLO11s-p2 + dilimlemeli çıkarım + birleştirme |
| Değerlendirme | `uav_vision.eval` | mAP@0.5 (bbox + sınıf), sınıf başına AP, VOC + COCO-101, faster-coco-eval çapraz kontrol |
| Giriş | `main.py` | Seçilmiş sürümü (ağırlık + düzen) sabitler, kare/klasör üzerinde çalıştırır |

Veri hazırlama (export, sınıf eşleme, format dönüşümü) depo dışında yürütülür.

## Model: YOLO11s + P2

Standart YOLO11 üç seviyede tespit yapar (P3/8, P4/16, P5/32). Alt-görüş karelerinde
insanlar onlarca piksel boyutunda olduğu için stride-8 bile kaba kalıyor. P2 başlığı
stride-4 seviyesini ekler; tespit dört seviyede yapılır (P2, P3, P4, P5).

- Omurga (katman 0–10) COCO ağırlığından transfer edilir.
- Kaydırılan ve yeni eklenen head katmanları sıfırdan öğrenir (kısmi transfer).
- Tanım: `configs/yolo11-p2.yaml`

## Eğitim

1. **COCO → VisDrone-DET** (60 epoch): omurgayı drone bakışına ve küçük nesneye alıştırır.
2. **VisDrone → kendi veri seti** (150 epoch, 4 sınıf): son fine-tune.

Her iki aşama imgsz 1280, tam kare (dilimleme yok). Dilimlenmiş veriyle üçüncü bir
fine-tune denendi, holdout'ta geriledi; dilimleme yalnız çıkarımda kullanılıyor.

## Çıkarım düzeni (V2)

Kaynak: `src/uav_vision/detector/detector.py`

```
1920×1080 kare
 ├─ tile A: x=0..1080,    y=0..1080  ─┐
 ├─ tile B: x=840..1920,  y=0..1080  ─┼─ her biri 1280'e büyütülür (1.19×)
 └─ tam kare: 1280×720 (letterbox 736×1280)
          │
          ▼  YOLO11s-p2 (3 forward, conf 0.001, forward içi NMS IoU 0.7)
          ▼  tile kutuları kare koordinatına kaydırılır
          ▼  kareye kırpma, sıfır alanlıları atma
          ▼  sınıf bazlı NMS, IoU 0.5
       tespitler (x1, y1, x2, y2, skor, sınıf)
```

- **Tile yerleşimi:** bir ekseni kapsayan en az sayıda tile, eşit aralıklı
  (`tile_origins`). 1920 genişlikte 2 tile, 1080 yükseklikte tek satır.
- **Büyütme:** model 1280'de eğitildi; 1080'lik tile 1280'e büyütülünce küçük nesneler
  modelin gördüğü ölçeğe yaklaşıyor. Büyütmeyi kaldıran varyantlar (@1024, @1088)
  human AP'de belirgin kayıp verdi.
- **Tam kare geçişi:** tile sınırında bölünen büyük nesneleri (taşıt, iniş alanı) kurtarır.
- **Birleştirme:** SAHI'nin GREEDYNMM / NMM / NMS aileleri × IoU/IOS × eşik (36 ayar)
  tarandı; sınıf bazlı NMS / IoU 0.5 seçildi.

### Neden V2

Önceki düzen (SAHI, 6 × 1024 tile, %30 örtüşme) 7 forward ile ~761 GFLOPs ve 303 ms/kare
tutuyordu. Örtüşmenin büyük kısmı gereksizdi (satırlar %95, sütunlar %82 örtüşüyordu).
8 varyant ölçüldü:

| Düzen | GFLOPs | ms/kare | mAP@0.5 | human |
|---|---|---|---|---|
| 6 × 1024 @1280 + tam kare (önceki) | 761 | 303 | 0.8419 | 0.5664 |
| **2 × 1080 @1280 + tam kare (V2)** | **298** | **154** | **0.8457** | 0.5432 |
| 2 × 1080 @1280, tam kare yok | 232 | 102 | 0.8343 | 0.5047 |

Tam tablo ve bootstrap aralıkları: `results/cikarim_maliyet_ablasyonu/`.

## Değerlendirme

- **Eşleştirme** (`eval/matching.py`): sınıf başına, tahminler güven sırasıyla işlenir;
  her tahmin eşleşmemiş GT'ler içinde IoU'su en yüksek ve ≥ 0.5 olana atanır.
- **AP** (`eval/ap.py`): iki yöntem birden raporlanır. VOC tüm-nokta interpolasyonu ve
  COCO 101-nokta. README'deki tablolar VOC değerleridir.
- **Çapraz kontrol** (`eval/cocowrap.py`): faster-coco-eval AP50 ile kendi COCO-101
  sonucumuz karşılaştırılır.
- **Hata analizi** (`scripts/eval/tide_analysis.py`): TIDE ile mAP kaybı hata türlerine
  ayrılır (sınıf, konum, kopya, arka plan, kaçırma).

Ana değerlendirme seti **holdout**: eğitimde hiç görülmeyen ayrı bir kaynaktan 175 kare.
Kutu sayıları küçük olduğu için (human 57) sonuçlar eşli bootstrap ile güven aralığıyla
verilir.
