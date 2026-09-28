# TIDE Hata Analizi — v2 (2026-09-28)

- Model: `main.py` v2 (Faz 4 ağırlığı, 2×1080 tile @1280 + tam kare, NMS/IOU/0.5), conf=0.001
- Set: `sanity` (175 kare), IoU 0.5 (ön plan) / 0.1 (arka plan)
- Script: `scripts/tide_analysis.py` (tidecv 1.0.1; tek koşu, sınıf başına dAP)
- Tutarlılık: TIDE AP50 = evaluate.py AP(coco101) (TÜMÜ 84.45, human 54.34)

## dAP — bu hata türü düzelseydi AP kaç puan artardı

| | AP50 | Cls | Loc | Both | Dupe | Bkg | Miss | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| TÜMÜ | 84.45 | 0.05 | 1.85 | 0.04 | 0.01 | **6.80** | 2.27 | 10.85 | 3.24 |
| vehicle | 89.51 | 0.20 | 1.10 | 0.00 | 0.02 | 2.57 | **3.88** | 4.55 | 5.01 |
| human | 54.34 | 0.01 | 4.52 | 0.14 | 0.00 | **20.48** | 5.19 | 32.79 | 7.94 |
| uap | 97.67 | 0.00 | 0.00 | 0.00 | 0.00 | 2.33 | 0.00 | 2.33 | 0.00 |
| uai | 96.28 | 0.00 | 1.79 | 0.00 | 0.00 | 1.82 | 0.00 | 3.72 | 0.00 |

## Hata sayıları (tahmin edilen sınıfa yazılır; Miss GT'nin sınıfına)

| | Cls | Loc | Both | Dupe | Bkg | Miss |
|---|---|---|---|---|---|---|
| vehicle | 8 | 193 | 6 | 5 | 1611 | 13 |
| human | 3 | 38 | 21 | 2 | 1397 | 5 |
| uap | 0 | 0 | 0 | 0 | 75 | 0 |
| uai | 0 | 1 | 0 | 0 | 21 | 0 |

## Gözlemler

- human kaybının açık ara en büyük kaynağı **Bkg** (+20.5 AP): model, yakınında hiçbir
  GT olmayan yerlere yüksek skorla "insan" diyor. Sayı (1397) conf=0.001 yüzünden şişik;
  dAP ise skor sırasına duyarlı → yüksek skorlu arka plan FP'leri gerçekten var.
- human Loc +4.5 (38 kutu), Miss +5.2 (5 insan hiç bulunamadı).
- Cls ve Dupe ~0: sınıf karışıklığı yok; tile + tam kare birleştirmesi kopya bırakmıyor
  (IOS birleştirmesi human'a katkı yapmaz — ablasyonla tutarlı).
- vehicle'da en büyük kalem Miss (+3.9, 13 araç).
- Uyarı: 57 human GT → oranlar kaba. Bkg FP'lerinin bir kısmı etiketlenmemiş gerçek
  insan olabilir — yüksek skorlu Bkg örneklerine gözle bakılmadan kesin yorum yapılmamalı.
