# Mimari Genel Bakış

Pipeline bağımsız katmanlardan oluşur:

| Katman | Paket | Sorumluluk |
|---|---|---|
| Veri | (yerel) | Etiketli export → temiz YOLO/COCO, sınıf remap |
| Detektör | `uav_vision.detector` | YOLO11 + P2 başlığı; dilimlemeli (SAHI) eğitim/çıkarım |
| Değerlendirme | `uav_vision.eval` | mAP@IoU=0.5 (bbox + sınıf), sınıf-başı AP, VOC + COCO-101, faster-coco-eval çapraz kontrol |

İniş alanı uygunluğu ve hareket durumu (geometri + ego-motion) sonraki bir
aşamada, tespit çekirdeği oturduktan sonra ele alınacak.
