"""Eğitim girişi.

Kullanım:
    python scripts/train.py --config configs/baseline.yaml

Not: Faz 3'te doldurulacak (YOLO11 + P2 başlığı fine-tune).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Detektör eğitimi")
    parser.add_argument("--config", required=True, help="YAML deney konfigi")
    args = parser.parse_args()
    raise SystemExit(f"Eğitim henüz uygulanmadı (config={args.config}). Bkz. notes.md Faz 3.")


if __name__ == "__main__":
    main()
