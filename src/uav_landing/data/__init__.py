"""Veri katmanı.

Şartname sınıf sırası (sabit):  0=car (taşıt), 1=human, 2=UAP, 3=UAİ
Dataset (emirhanin_donu v2) sırası:  0=car, 1=human, 2=uai, 3=uap  → 2<->3 remap gerekir.
"""

# Şartname sınıf isimleri — indeks = sınıf id
CLASS_NAMES = ("car", "human", "uap", "uai")

# dataset id -> şartname id
DATASET_TO_SPEC = {0: 0, 1: 1, 2: 3, 3: 2}
