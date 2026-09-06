"""Çıkarım girişi (tek görsel / klasör / video karesi).

Kullanım:
    python scripts/predict.py --source path/to/frames --weights runs/best.pt

Not: Faz 3+ ile doldurulacak (SAHI tiling + tam-kare birleşimi).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Detektör çıkarımı")
    parser.add_argument("--source", required=True, help="Görsel/klasör yolu")
    parser.add_argument("--weights", required=True, help="Model ağırlıkları (.pt)")
    args = parser.parse_args()
    raise SystemExit(f"Çıkarım henüz uygulanmadı (source={args.source}, weights={args.weights}).")


if __name__ == "__main__":
    main()
