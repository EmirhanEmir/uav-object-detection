"""Sınıf remap testi — dataset (uai/uap) sırası şartname (UAP/UAİ) sırasına dönmeli."""

from uav_landing.data import CLASS_NAMES, DATASET_TO_SPEC


def test_spec_class_order():
    assert CLASS_NAMES == ("car", "human", "uap", "uai")


def test_car_and_human_unchanged():
    assert DATASET_TO_SPEC[0] == 0
    assert DATASET_TO_SPEC[1] == 1


def test_uai_uap_swapped():
    # dataset 2=uai -> şartname 3=uai ; dataset 3=uap -> şartname 2=uap
    assert DATASET_TO_SPEC[2] == 3
    assert DATASET_TO_SPEC[3] == 2


def test_remap_is_bijective():
    assert sorted(DATASET_TO_SPEC.values()) == [0, 1, 2, 3]
