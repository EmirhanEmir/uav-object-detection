"""Ortak yardımcılar (yol çözümleme, .env yükleme, loglama)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """Veri kök dizini — .env'deki DATA_ROOT ya da <proje>/data."""
    return Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data")).resolve()
