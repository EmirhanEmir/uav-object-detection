# UAV Object Detection

İHA alt-görüş kamera karelerinde **nesne tespiti** ve **iniş alanı uygunluğu** 

- **Girdi:** alt-görüş drone kareleri
- **Çıktı:** taşıt / insan / UAP / UAİ tespitleri (bbox + sınıf), iniş alanı uygunluğu
- **Metrik:** mAP @ IoU 0.5

> Durum: **baseline sonucu alındı** — sanity (yarışma verisi) referans mAP@0.5 = 0.8048.

## Kurulum

Yerel ortam **hafiftir** (veri katmanı + değerlendirme, `torch` yok). Eğitim/çıkarım
Colab'da yapılır (`requirements-train.txt`, bkz. `notebooks/colab_egitim.ipynb`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # = requirements.txt + ruff/pytest/pre-commit
pip install -e .
pre-commit install
```

`.env.example` → `.env` kopyalayıp anahtarları doldurun.

## Kullanım

```powershell
python scripts/train.py --config configs/baseline.yaml   # eğitim
python scripts/evaluate.py preds.json gt.json            # mAP@0.5
python scripts/predict.py --source frames/ --weights runs/best.pt
```

## Baseline sonuçları

YOLO11s + P2 başlığı, imgsz 1280, 150 epoch, COCO ağırlığından fine-tune
(`configs/baseline.yaml`, `uav_runs/faz3/baseline-yolo11s-p2`).

Referans skor **sanity** verisi (`data/sanity`, yarışma verisiyle aynı kaynak,
model tarafından hiç görülmemiş) üzerinden alınır; kendi test split'imiz
(`data/uav_ldz/test`, eğitimle aynı dağılım) yalnızca yardımcı kontroldür.

| Split | mAP@0.5 (VOC) | vehicle | human | uap | uai |
|---|---|---|---|---|---|
| **sanity (referans)** | **0.8048** | 0.8622 | 0.3776 | 0.9922 | 0.9872 |
| test (kendi split) | 0.9658 | 0.9802 | 0.9210 | 0.9798 | 0.9823 |

`human` sınıfı sanity verisinde ciddi geneleme sorunu gösteriyor (çok yüksek
false-positive) — açık iyileştirme konusu.

## Yapı

```
src/uav_vision/     paket: detector / eval / utils
configs/             yaml deney konfigleri
scripts/             train / evaluate / predict girişleri
notebooks/           Colab defterleri
docs/                dokümantasyon
data/                veri (git'e girmez)
```

## Geliştirme

```powershell
ruff check .
```

CI: `.github/workflows/ci.yml` (ruff).

## Lisans

[AGPL-3.0-or-later](LICENSE). Eğitim/çıkarım Ultralytics YOLO'ya (AGPL-3.0) dayandığı
için proje de aynı güçlü copyleft lisansıyla yayınlanır: bu kodu dağıtan veya ağ
servisi olarak sunan, kaynağını AGPL ile açmak zorundadır.

Veri seti (Roboflow export) ayrı şartlara tabidir (CC BY 4.0) ve bu depoya dahil değildir.
