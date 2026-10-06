"""Deney geçmişi grafiği — aşama aşama mAP@0.5 ve sınıf başına AP.

    python scripts/analysis/plot_ap_history.py

Girdi: results/deney_gecmisi.json
Çıktı: docs/assets/ap_history_light.png + ap_history_dark.png (README'de tema bazlı seçilir)

Renkler doğrulanmış kategorik paletten (sınıf sırası sabit); mAP birincil mürekkeple
çizilir. Kimlik yalnız renge bırakılmaz: her çizginin sağ ucunda adı ve son değeri yazar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = ("vehicle", "human", "uap", "uai")

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500"),
    },
}

LABEL_GAP = 0.034  # uç etiketleri arasında en az dikey boşluk (AP birimi)


def _spread(ys: list[float], gap: float) -> list[float]:
    """Uç etiketlerini çakışmayacak şekilde dikeyde aç (sıra korunur)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:], strict=False):
        out[b] = max(out[b], out[a] + gap)
    # Üstten taşanları aşağı it
    top = 1.0
    for i in reversed(order):
        out[i] = min(out[i], top)
        top = out[i] - gap
    return out


def plot(stages: list[dict], theme: str, out: Path) -> None:
    t = THEMES[theme]
    x = list(range(len(stages)))
    labels = [s["ad"] for s in stages]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    series = [(c, [s[c] for s in stages], t["series"][i], 2.0) for i, c in enumerate(CLASSES)]
    series.append(("mAP", [s["map"] for s in stages], t["ink"], 2.8))

    for _name, ys, color, lw in series:
        ax.plot(x, ys, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.scatter(x, ys, s=46, color=color, edgecolors=t["surface"], linewidths=2, zorder=4)

    # Sağ uç etiketleri: kısa renk anahtarı + metin mürekkebinde ad ve değer
    ends = [ys[-1] for _, ys, _, _ in series]
    label_y = _spread(ends, LABEL_GAP)
    for (name, ys, color, _), ly in zip(series, label_y, strict=True):
        xe = x[-1] + 0.12
        ax.plot([x[-1] + 0.04, xe], [ys[-1], ly], color=t["muted"], lw=0.8, zorder=2)
        ax.scatter([xe + 0.06], [ly], s=30, color=color, zorder=4)
        weight = "bold" if name == "mAP" else "normal"
        ax.text(xe + 0.13, ly, f"{name}  {ys[-1]:.3f}", va="center", ha="left",
                color=t["ink"], fontsize=10, fontweight=weight)

    # human'ın başlangıç değeri: hikâyenin ana kalemi
    h0 = stages[0]["human"]
    ax.text(x[0] - 0.08, h0, f"{h0:.3f}", va="center", ha="right", color=t["ink2"], fontsize=9)

    ax.set_xlim(-0.35, x[-1] + 0.95)
    ax.set_ylim(0.3, 1.02)
    ax.set_xticks(x, labels)
    ax.set_ylabel("AP @ IoU 0.5", color=t["ink2"], fontsize=10)
    ax.tick_params(colors=t["muted"], labelsize=9.5, length=0)
    for lbl in ax.get_xticklabels():
        lbl.set_color(t["ink2"])
    ax.grid(axis="y", color=t["grid"], lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])

    ax.set_title("Holdout setinde aşama aşama AP (175 kare)", loc="left",
                 color=t["ink"], fontsize=12.5, fontweight="bold", pad=14)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print(f"-> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Deney geçmişi grafiği")
    p.add_argument("--history", default="results/deney_gecmisi.json")
    p.add_argument("--out-dir", default="docs/assets")
    args = p.parse_args()

    stages = json.loads(Path(args.history).read_text(encoding="utf-8"))["asamalar"]
    for theme in THEMES:
        plot(stages, theme, Path(args.out_dir) / f"ap_history_{theme}.png")


if __name__ == "__main__":
    main()
