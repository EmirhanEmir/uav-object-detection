"""Değerlendirme girişi.

Kullanım:
    python scripts/evaluate.py preds.json gt.json

Not: Faz 2'de doldurulacak (mAP@0.5, 3 protokol + sınıf-başı AP).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="mAP@0.5 değerlendirme")
    parser.add_argument("preds", help="Tahmin JSON (COCO formatı)")
    parser.add_argument("gt", help="Ground-truth JSON (COCO formatı)")
    args = parser.parse_args()
    raise SystemExit(f"Değerlendirme henüz uygulanmadı (preds={args.preds}, gt={args.gt}). Bkz. notes.md Faz 2.")


if __name__ == "__main__":
    main()
