"""TIDE hata analizi — mAP kaybını hata türlerine ayırır (Bolya ve ark., ECCV 2020).

    python scripts/tide_analysis.py preds.json data/coco/instances_sanity.json --out tide.json

Girdi: COCO "results" tahmin json'u (main.py --coco-gt çıktısı) + COCO GT json'u.
Model çalıştırılmaz; tamamen offline, çıkarım maliyetine etkisi yok.

Hata türleri (IoU eşikleri: ön plan 0.5 = METRIK_G1, arka plan 0.1):
  Cls   kutu doğru yerde (IoU ≥ 0.5), sınıf yanlış
  Loc   sınıf doğru, kutu kaymış (0.1 ≤ IoU < 0.5)
  Both  sınıf yanlış + kutu kaymış
  Dupe  doğru ama aynı GT zaten daha yüksek skorlu bir tahminle eşleşmiş
  Bkg   hiçbir GT'ye yakın değil (IoU < 0.1) — arka plan nesne sanılmış
  Miss  hiçbir tahminle bulunamayan GT
dAP = o hata türü tamamen düzeltilseydi AP kaç puan artardı (0–100 ölçeği).
FP / FN = tüm yanlış pozitifler / negatifler düzeltilseydi dAP.

Not: tidecv'nin kendi COCO yükleyicisi `segmentation` alanı şart koşuyor (bizim GT'de
yok) ve görsel başına 100 tahminle sınırlıyor — bu yüzden veri doğrudan `Data`'ya
yükleniyor ve sınır kaldırılıyor (bizim evaluator'ümüz de sınırsız). TIDE AP'si
COCO 101-nokta enterpolasyonu kullanır → evaluate.py'deki AP(coco101) ile karşılaştır.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

POS_IOU = 0.5  # METRIK_G1 = mAP @ IoU 0.5
BG_IOU = 0.1  # TIDE varsayılanı
MAX_DETS = 100_000  # pratikte sınırsız


def _load(gt: dict, preds: list[dict]):
    from tidecv import Data

    gt_data = Data("gt", max_dets=MAX_DETS)
    pred_data = Data("pred", max_dets=MAX_DETS)
    for cat in gt["categories"]:
        gt_data.add_class(cat["id"], cat["name"])
        pred_data.add_class(cat["id"], cat["name"])
    for im in gt["images"]:
        gt_data.add_image(im["id"], im["file_name"])
        pred_data.add_image(im["id"], im["file_name"])
    for a in gt["annotations"]:
        gt_data.add_ground_truth(a["image_id"], a["category_id"], a["bbox"])
    for p in preds:
        pred_data.add_detection(p["image_id"], p["category_id"], p["score"], p["bbox"])
    return gt_data, pred_data


def _error_class(err) -> int:
    """Hatanın ait olduğu sınıf: tahmini varsa tahminin sınıfı, yoksa (Miss) GT'nin."""
    pred = getattr(err, "pred", None)
    return pred["class"] if pred is not None else err.gt["class"]


def _analyze(gt: dict, preds: list[dict]) -> dict[str, dict]:
    """Tek TIDE koşusu (tüm sınıflar) → {"TÜMÜ": ..., "<sınıf>": ...}.

    Sınıf başına dAP, hatalar düzeltildiğinde o sınıfın AP'sindeki değişimdir. Tek koşu
    kullanıldığı için sınıf karışıklığı (Cls) doğru sayılır: yanlış sınıflı tahmin,
    tahmin edilen sınıfın satırında görünür, düzeltilince GT'nin sınıfına geçer.
    """
    from tidecv import TIDE
    from tidecv.errors.main_errors import FalsePositiveError

    gt_data, pred_data = _load(gt, preds)
    run = TIDE(pos_threshold=POS_IOU, background_threshold=BG_IOU, mode=TIDE.BOX).evaluate(
        gt_data, pred_data,
    )
    base = run.ap_data
    fixed = {e: run.fix_errors(lambda x, e=e: isinstance(x, e)) for e in TIDE._error_types}
    fixed_fp = run.fix_errors(transform=FalsePositiveError.fix)
    fixed_fn = run.fix_errors(false_neg_dict=run.false_negatives)

    def row(ap_of) -> dict:
        ap = ap_of(base)
        return {
            "ap50": round(ap, 2),
            # Negatif fark binning kaynaklıdır, TIDE da 0'a kırpar
            "dAP": {e.short_name: round(max(ap_of(d) - ap, 0), 2) for e, d in fixed.items()},
            "dAP_special": {"FalsePos": round(max(ap_of(fixed_fp) - ap, 0), 2),
                            "FalseNeg": round(max(ap_of(fixed_fn) - ap, 0), 2)},
        }

    rows = {"TÜMÜ": row(lambda d: d.get_mAP())}
    rows["TÜMÜ"]["count"] = {e.short_name: len(run.error_dict[e]) for e in TIDE._error_types}
    for cat in sorted(gt["categories"], key=lambda c: c["id"]):
        cid = cat["id"]
        rows[cat["name"]] = row(lambda d, cid=cid: d.objs[cid].get_ap())
        rows[cat["name"]]["count"] = {
            e.short_name: sum(_error_class(x) == cid for x in run.error_dict[e])
            for e in TIDE._error_types
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="TIDE hata analizi (bbox, IoU 0.5)")
    parser.add_argument("preds", help="COCO results json")
    parser.add_argument("gt", help="COCO GT json")
    parser.add_argument("--out", help="Sonuçları json olarak yaz")
    args = parser.parse_args()

    with open(args.gt, encoding="utf-8") as f:
        gt = json.load(f)
    with open(args.preds, encoding="utf-8") as f:
        preds = json.load(f)

    rows = _analyze(gt, preds)

    types = list(rows["TÜMÜ"]["dAP"])
    special = list(rows["TÜMÜ"]["dAP_special"])
    print(f"\n=== TIDE — dAP (bu hata düzelseydi AP kaç puan artardı), IoU {POS_IOU} ===")
    print(f"{'':10}{'AP50':>7}" + "".join(f"{t:>7}" for t in types + special))
    for name, r in rows.items():
        vals = [r["dAP"][t] for t in types] + [r["dAP_special"][t] for t in special]
        print(f"{name:10}{r['ap50']:>7.2f}" + "".join(f"{v:>7.2f}" for v in vals))

    print("\n=== TIDE — hata sayısı ===")
    print(f"{'':10}" + "".join(f"{t:>7}" for t in types))
    for name, r in rows.items():
        print(f"{name:10}" + "".join(f"{r['count'][t]:>7}" for t in types))
    print("\nSayılar: hata, tahmin edilen sınıfa yazılır (Miss: GT'nin sınıfına).")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[json] {args.out}")


if __name__ == "__main__":
    main()
