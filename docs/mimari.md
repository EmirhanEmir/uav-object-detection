# Mimari Genel Bakış

Pipeline üç bağımsız katman + değerlendirme:

| Katman | Paket | Sorumluluk |
|---|---|---|
| Veri | `uav_vision.data` | Roboflow export → temiz YOLO/COCO, sınıf remap (uai/uap ↔ UAP/UAİ) |
| Detektör | `uav_vision.detector` | YOLO11 + P2 başlığı; SAHI tiling eğitim/çıkarım |
| Değerlendirme | `uav_vision.eval` | mAP@IoU=0.5 (bbox + sınıf), sınıf-başı AP, VOC + COCO-101, faster-coco-eval çapraz kontrol |
| (sonra) İniş uygunluğu | — | UAP/UAİ geometri + üstünde cisim kontrolü (Faz 7) |
| (sonra) Hareket durumu | — | ego-motion telafisi (Görev 2 ile ortak, Faz 7) |

Detaylı yol haritası ve kararlar: kök dizindeki `notes.md`.
