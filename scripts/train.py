"""Eğitim girişi — baseline (YOLO11s + P2 başlığı, tam kare fine-tune).

Kullanım (Colab):
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --batch -1 --epochs 200

Sıra:
  1. configs/yolo11-p2.yaml mimarisi kurulur (arch = n/s/m/l/x).
  2. COCO ağırlığı (yolo11s.pt) omurgaya transfer edilir (kısmi — P2 head sıfırdan).
  3. configs/data.yaml üzerinde fine-tune.
  4. Eğitim bitince test split'inde Ultralytics val (hızlı kontrol).

Resmi referans skor için eğitimden sonra:
    python scripts/predict_to_coco.py <best.pt> test  -> preds_test.json
    python scripts/evaluate.py preds_test.json data/coco/instances_test.json --cross-check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detektör eğitimi (baseline)")
    parser.add_argument("--config", default="configs/baseline.yaml", help="YAML deney konfigi")
    # Colab'da hızlı override — verilmezse config'teki değer kullanılır.
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true", help="Son checkpoint'ten devam")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    m, d, t = cfg["model"], cfg["data"], cfg["train"]

    arch = m["arch"]                       # yolo11s
    size = arch.replace("yolo11", "")      # s
    # Ultralytics ölçek harfini dosya adının kökünden okur: "yolo11s-p2" -> scale=s,
    # sonra aynı klasördeki "yolo11-p2.yaml"yı yükler. Dosya fiziksel olmasa da olur.
    model_yaml = f"configs/yolo11{size}-p2.yaml" if m.get("p2_head") else f"{arch}.yaml"

    from ultralytics import YOLO

    model = YOLO(model_yaml)

    if m.get("pretrained") == "coco":
        weights = f"{arch}.pt"            # Ultralytics otomatik indirir
        print(f"[transfer] COCO ağırlığı yükleniyor: {weights} (omurga; P2 head sıfırdan)")
        model.load(weights)

    imgsz = args.imgsz or d["imgsz"]
    epochs = args.epochs or t["epochs"]
    batch = args.batch if args.batch is not None else t["batch"]
    device = args.device or t.get("device", 0)

    data_yaml = str(Path(d["yaml"]).resolve())
    print(f"[eğitim] arch={arch}+P2 imgsz={imgsz} epochs={epochs} batch={batch} device={device}")

    model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=t.get("patience", 40),
        close_mosaic=t.get("close_mosaic", 15),
        device=device,
        seed=t.get("seed", 42),
        workers=t.get("workers", 8),
        project=t.get("project", "runs/baseline"),
        name=cfg["name"],
        resume=args.resume,
        plots=True,
    )

    # Eğitim sonrası hızlı kontrol — test split (Ultralytics kendi mAP'i).
    metrics = model.val(data=data_yaml, imgsz=imgsz, split="test", iou=cfg["eval"]["iou"])
    print(f"\n[test / Ultralytics val] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    print("Sınıf-başı AP50:", {model.names[i]: round(float(ap), 4)
                               for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap50, strict=True)})
    print("\nResmi referans skor için: scripts/predict_to_coco.py + scripts/evaluate.py")


if __name__ == "__main__":
    main()
