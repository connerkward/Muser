"""Pareto-frontier plot: step envelope + shaded dominated region + Wilson-CI error bars.

data: list of {name, ms_img, hits1, n, lic}. Frontier computed from point estimates
(minimize ms_img, maximize hits1); y error bars are 95% Wilson intervals on hits@1.
"""

from __future__ import annotations

import math


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def _dominated(p, pts):
    return any(
        q is not p and q["ms_img"] <= p["ms_img"] and q["hits1"] >= p["hits1"]
        and (q["ms_img"] < p["ms_img"] or q["hits1"] > p["hits1"])
        for q in pts
    )


def pareto_plot(data: list[dict], title: str, outpath: str, notes: str | None = None, logx: bool = False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    front = sorted([p for p in data if not _dominated(p, data)], key=lambda p: p["ms_img"])
    xmax = max(p["ms_img"] for p in data) * (1.5 if logx else 1.12)
    xmin = min(p["ms_img"] for p in data) * 0.6 if logx else 0
    ys = [p["hits1"] for p in data]
    ymin, ymax = min(ys) - 0.06, min(1.0, max(ys) + 0.06)

    fig, ax = plt.subplots(figsize=(9.5, 6.4), dpi=140)

    # step envelope (achievable best)
    sx, sy = [], []
    for i, p in enumerate(front):
        if i:
            sx.append(p["ms_img"]); sy.append(front[i - 1]["hits1"])
        sx.append(p["ms_img"]); sy.append(p["hits1"])
    sx.append(xmax); sy.append(front[-1]["hits1"])
    ax.fill_between(sx, sy, ymin, color="#c62828", alpha=0.07, zorder=0)
    ax.plot(sx, sy, color="#2e7d32", lw=2.4, zorder=2, label="Pareto frontier (achievable best)")
    ax.text(xmax * 0.6, ymin + (ymax - ymin) * 0.12, "DOMINATED REGION",
            color="#c62828", alpha=0.6, fontsize=9, style="italic", ha="center")

    names_front = {p["name"] for p in front}
    texts = []
    for p in data:
        on = p["name"] in names_front
        lo, hi = wilson(p["hits1"], p["n"])
        c = "#2e7d32" if on else "#c62828"
        ax.errorbar([p["ms_img"]], [p["hits1"]],
                    yerr=[[p["hits1"] - lo], [hi - p["hits1"]]],
                    fmt="o" if on else "X", color=c, ms=10, capsize=4,
                    elinewidth=1.3, zorder=5, markeredgecolor="white", markeredgewidth=1.2)
        texts.append(ax.text(p["ms_img"], p["hits1"],
                             f"{p['name']}\n{p['hits1']:.3f} · {p['ms_img']:.0f}ms · {p['lic']}",
                             fontsize=8.3, color=c, weight="bold" if on else "normal"))
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand=(1.3, 1.6),
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
    except Exception:
        pass

    ax.set_xlabel("Index cost  →  ms / image   (← cheaper / faster is better)")
    ax.set_ylabel("Quality  →  hits@1   (higher is better ↑;  bars = 95% Wilson CI)")
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", framealpha=0.95)
    if notes:
        ax.text(0.015, 0.015, notes, transform=ax.transAxes, fontsize=7.5, color="#555",
                va="bottom", ha="left", bbox=dict(boxstyle="round", fc="white", ec="#bbb", alpha=0.9))
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight")
    return [p["name"] for p in front]
