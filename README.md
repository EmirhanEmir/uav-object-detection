# UAV Landing Zone Detection

İHA alt-görüş kamera karelerinde **nesne tespiti** ve **iniş alanı uygunluğu** —
TEKNOFEST Havacılıkta Yapay Zeka Yarışması, **Görev 1**.

- **Girdi:** alt-görüş (70–90°) drone kareleri, 1920×1080
- **Çıktı:** taşıt / insan / UAP / UAİ tespitleri (bbox + sınıf), iniş alanı uygunluğu
- **Metrik:** mAP @ IoU 0.5

> Durum: **altyapı kurulumu (Faz 0)**. Yol haritası ve kararlar kök dizindeki
> `notes.md` dosyasında.

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
python scripts/train.py --config configs/baseline.yaml   # eğitim (Faz 3)
python scripts/evaluate.py preds.json gt.json            # mAP@0.5 (Faz 2)
python scripts/predict.py --source frames/ --weights runs/best.pt
```

## Yapı

```
src/uav_landing/     paket: data / detector / eval / utils
configs/             yaml deney konfigleri
scripts/             train / evaluate / predict girişleri
tests/               pytest
notebooks/           Colab defterleri
docs/                dokümantasyon
data/                veri (git'e girmez)
```

## Geliştirme

```powershell
ruff check .
pytest
```

CI: `.github/workflows/ci.yml` (ruff + pytest, her push/PR).
