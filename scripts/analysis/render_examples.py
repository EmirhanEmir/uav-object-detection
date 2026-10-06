"""Örnek tespit görselleri — aktif detektörün (main.py) çıktısını karelerin üstüne çizer.

    python scripts/analysis/render_examples.py kare1.jpg kare2.jpg --out docs/assets
    python scripts/analysis/render_examples.py data/uav_ldz/test/images --limit 8 --out ornekler/

Çizim için güven eşiği (--min-score) yalnız görsellik içindir; mAP hesabı main.py'nin
düşük eşiğiyle yapılır. Kutu renkleri sınıfa sabittir (plot_ap_history.py ile aynı palet).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# BGR — vehicle, human, uap, uai
COLORS = ((214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237))


def draw(frame, dets, min_score: float):
    import cv2

    h = frame.shape[0]
    thick = max(2, round(h / 540))
    font = 0.45 * h / 1080 * 1.6
    for d in dets:
        if d.score < min_score:
            continue
        color = COLORS[d.cls]
        p1, p2 = (round(d.x1), round(d.y1)), (round(d.x2), round(d.y2))
        cv2.rectangle(frame, p1, p2, color, thick, cv2.LINE_AA)
        text = f"{d.name} {d.score:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
        y0 = max(p1[1], th + base + 2)
        cv2.rectangle(frame, (p1[0], y0 - th - base - 2), (p1[0] + tw + 4, y0), color, -1)
        cv2.putText(frame, text, (p1[0] + 2, y0 - base), cv2.FONT_HERSHEY_SIMPLEX, font,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    p = argparse.ArgumentParser(description="Örnek tespit görselleri")
    p.add_argument("sources", nargs="+", help="Görsel(ler) veya klasör")
    p.add_argument("--out", default="docs/assets")
    p.add_argument("--min-score", type=float, default=0.35)
    p.add_argument("--width", type=int, default=1280, help="Çıktı genişliği (px)")
    p.add_argument("--limit", type=int, help="Klasörden en fazla bu kadar kare")
    p.add_argument("--device", default="0")
    args = p.parse_args()

    import cv2
    from main import load_detector

    paths: list[Path] = []
    for s in map(Path, args.sources):
        if s.is_dir():
            paths += sorted(q for q in s.iterdir() if q.suffix.lower() in IMG_EXTS)[: args.limit]
        else:
            paths.append(s)

    det = load_detector(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        frame = draw(frame, det(frame), args.min_score)
        scale = args.width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        dst = out / f"ornek_{i + 1:02d}.jpg"
        cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{path.name} -> {dst}")


if __name__ == "__main__":
    main()
