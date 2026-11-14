"""Figures for the results report.

One visual system for every figure:

* colour does one job per chart. Methods keep the same colour everywhere
  (``METHOD_COLORS``); ordered series use one hue from light to dark; risk bands
  use the reserved status colours; a single series is one hue; references and
  context are neutral grey;
* thin marks, solid hairline grid, recessive axes, text in ink colours only;
* two or more series get a legend, and lines are also labelled at their ends;
* one y-axis per plot, never two scales.

The categorical order and the ordinal ramp were checked with a colour-vision
validator (adjacent-pair separation under protan/deutan simulation). Two of the
method colours are below 3:1 contrast on the light surface, so they are always
accompanied by direct labels, and every figure has a table in RESULTS.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#b4b2aa"  # de-emphasised marks (context, not the subject)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
#: a method keeps its colour in every figure (fixed categorical order: blue, orange, aqua, yellow)
METHOD_COLORS = {"fedavg": BLUE, "local": ORANGE, "pooled_local_features": AQUA, "fedavg_finetuned": YELLOW}
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "local": "Local-only",
    "pooled_local_features": "Pooled model on local features",
    "fedavg_finetuned": "FedAvg + local fine-tuning",
    "fedprox": "FedProx",
    "centralised": "Centralised MLP",
}
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]  # one hue, light to dark, for ordered series
STATUS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#eef4fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

DPI = 160


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.labelsize": 9.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2.0, "lines.markersize": 7, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2, "figure.dpi": DPI, "savefig.dpi": DPI,
    })


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


BAR = 0.36  # bar thickness as a share of its slot: thin marks, the rest is air


def _header(ax: plt.Axes, title: str, handles: list[Any] | None = None, labels: list[str] | None = None, ncols: int = 2) -> None:
    """Title at the top left, with the legend (if any) in its own row between title and plot."""
    rows = 0
    if handles is None and labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if handles:
        rows = int(np.ceil(len(handles) / ncols))
        ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=ncols, columnspacing=1.6, handlelength=1.5, handletextpad=0.5,
                  borderaxespad=0.3)
    ax.set_title(title, loc="left", pad=10 + 17 * rows)


def _only_y_grid(ax: plt.Axes) -> None:
    ax.grid(axis="x", visible=False)


def _only_x_grid(ax: plt.Axes) -> None:
    ax.grid(axis="y", visible=False)


def _marker(ax: plt.Axes, x: Sequence[float], y: Sequence[float], color: str, **kw: Any) -> None:
    """Filled markers with a ring in the surface colour, so they stay legible where they overlap."""
    ax.plot(x, y, linestyle="none", marker="o", markersize=8, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4, **kw)


def _end_labels(ax: plt.Axes, items: list[tuple[float, float, str]], x_text: float, min_gap: float) -> None:
    """Direct labels at line ends; labels that would collide are spread and joined by leader lines."""
    items = sorted(items, key=lambda t: t[1])
    placed: list[float] = []
    for _, y, _ in items:
        placed.append(y if not placed else max(y, placed[-1] + min_gap))
    shift = (np.mean(placed) - np.mean([y for _, y, _ in items])) if items else 0.0
    for (x, y, text), ty in zip(items, placed):
        ty -= shift
        if abs(ty - y) > min_gap * 0.25:
            ax.plot([x, x_text], [y, ty], color=AXIS, linewidth=0.7, clip_on=False, zorder=1)
        ax.annotate(text, (x_text, ty), xytext=(4, 0), textcoords="offset points", va="center", ha="left", fontsize=8.5, color=INK_2, annotation_clip=False)


def _reference_line(ax: plt.Axes, y: float, label: str, x_text: float | None = None) -> None:
    ax.axhline(y, color=MUTED, linewidth=1.1, zorder=1)
    x = ax.get_xlim()[0] if x_text is None else x_text
    ax.annotate(label, (x, y), xytext=(3, 4), textcoords="offset points", fontsize=8.5, color=INK_2, va="bottom", ha="left")


# --------------------------------------------------------------------------- core figures


def method_comparison(rows: list[dict[str, Any]], path: Path, title: str, subtitle: str) -> Path:
    """Ranked horizontal bars of F1 (mean, with the spread over seeds); methods that keep raw data local are emphasised."""
    apply_style()
    rows = sorted(rows, key=lambda r: r["f1"])
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(rows) + 1.5))
    y = np.arange(len(rows))
    colors = [BLUE if r["local_data"] else NEUTRAL for r in rows]
    ax.barh(y, [r["f1"] for r in rows], height=BAR, color=colors, zorder=3)
    ax.errorbar([r["f1"] for r in rows], y, xerr=[r["f1_std"] for r in rows], fmt="none", ecolor=INK_2, elinewidth=1.0, capsize=2.5, zorder=4)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['f1']:.3f}", (r["f1"] + r["f1_std"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlim(0, max(r["f1"] + r["f1_std"] for r in rows) * 1.14)
    ax.set_xlabel("F1 on the test period (mean of 3 seeds, whiskers show ± 1 std)" + (f"\n{subtitle}" if subtitle else ""))
    _only_x_grid(ax)
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=NEUTRAL)]
    _header(ax, title, handles, ["raw data stays with each institution", "needs pooled raw data"])
    return _save(fig, path)


def scenario_dumbbell(rows: list[dict[str, Any]], centralised: float, path: Path) -> Path:
    """Local-only to FedAvg per scenario: does federating help, and by how much?"""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 0.46 * len(rows) + 1.7))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r["local"], r["fedavg"]], [yi, yi], color=AXIS, linewidth=2.2, zorder=2, solid_capstyle="butt")
    _marker(ax, [r["local"] for r in rows], y, ORANGE, label=METHOD_LABELS["local"])
    _marker(ax, [r["fedavg"] for r in rows], y, BLUE, label=METHOD_LABELS["fedavg"])
    lo = min(min(r["local"], r["fedavg"]) for r in rows)
    hi = max(centralised, max(max(r["local"], r["fedavg"]) for r in rows))
    ax.set_xlim(lo - 0.03, hi + 0.02)
    for yi, r in zip(y, rows):  # the change, in one aligned column outside the plot
        ax.annotate(f"{r['fedavg'] - r['local']:+.3f}", (1.0, yi), xycoords=("axes fraction", "data"), xytext=(10, 0), textcoords="offset points", va="center",
                    ha="left", fontsize=9, color=INK, annotation_clip=False)
    ax.annotate("FedAvg −\nlocal-only", (1.0, len(rows) - 0.25), xycoords=("axes fraction", "data"), xytext=(10, 0), textcoords="offset points", va="bottom", ha="left",
                fontsize=8.5, color=INK_2, annotation_clip=False)
    ax.axvline(centralised, color=MUTED, linewidth=1.1, zorder=1)
    ax.annotate(f"centralised MLP {centralised:.3f}", (centralised, -0.62), xytext=(-4, 2), textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.set_xlabel("F1 on the test period (mean of 3 seeds)")
    _only_x_grid(ax)
    ax.spines["left"].set_visible(False)
    _header(ax, "Federating helps where data is spread thin, not where it is skewed")
    return _save(fig, path)


def f1_vs_round(panels: list[dict[str, Any]], centralised: float, path: Path) -> Path:
    """Test F1 per round, one small panel per data split, with the fixed references."""
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.05 * len(panels), 3.3), sharey=True)
    for ax, panel in zip(np.atleast_1d(axes), panels):
        rounds = np.arange(1, len(panel["mean"]) + 1)
        ax.fill_between(rounds, panel["mean"] - panel["std"], panel["mean"] + panel["std"], color=BLUE, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(rounds, panel["mean"], color=BLUE, zorder=3, label=METHOD_LABELS["fedavg"])
        ax.axhline(panel["local"], color=ORANGE, linewidth=1.6, zorder=2, label=METHOD_LABELS["local"])
        ax.axhline(centralised, color=MUTED, linewidth=1.1, zorder=1, label=METHOD_LABELS["centralised"])
        ax.set_title(panel["label"], fontsize=10, fontweight="normal", color=INK_2, pad=6)
        ax.set_xlabel("round")
        ax.set_xlim(1, len(rounds))
        _only_y_grid(ax)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("F1 on the test period")
    lows = [min(float((p["mean"] - p["std"]).min()), p["local"]) for p in panels]
    first.set_ylim(max(0.0, min(lows) - 0.03), centralised + 0.06)
    handles, labels = first.get_legend_handles_labels()
    fig.subplots_adjust(wspace=0.08, top=0.74)
    left = first.get_position().x0
    fig.suptitle("FedAvg converges in about ten rounds; the band is ± 1 std over 3 seeds", x=left, y=0.99, ha="left", va="top", fontsize=11, fontweight="bold")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(left, 0.925), ncols=3, handlelength=1.6, columnspacing=1.6, borderaxespad=0.0)
    return _save(fig, path)


def lines_by_setting(x_labels: list[str], series: dict[str, dict[str, np.ndarray]], centralised: float, path: Path, title: str, xlabel: str) -> Path:
    """F1 of each method across an ordered setting (data skew, number of clients)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(x_labels))
    ends = []
    for name, data in series.items():
        color = METHOD_COLORS[name]
        ax.fill_between(x, data["mean"] - data["std"], data["mean"] + data["std"], color=color, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(x, data["mean"], color=color, zorder=3, label=METHOD_LABELS[name])
        _marker(ax, x, data["mean"], color)
        ends.append((x[-1], float(data["mean"][-1]), METHOD_LABELS[name]))
    ax.set_xticks(x, x_labels)
    ax.set_xlim(-0.25, len(x_labels) - 1 + 0.25)
    values = np.concatenate([d["mean"] for d in series.values()])
    ax.set_ylim(min(values.min() - 0.04, centralised - 0.02), max(values.max(), centralised) + 0.035)
    _reference_line(ax, centralised, f"centralised MLP {centralised:.3f}", x_text=-0.25)
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    _end_labels(ax, ends, x[-1] + 0.3, min_gap=span * 0.055)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1 on the test period (mean ± 1 std, 3 seeds)")
    _only_y_grid(ax)
    _header(ax, title)
    fig.subplots_adjust(right=0.74)
    return _save(fig, path)


def typology_heatmap(methods: list[str], typologies: list[str], recall: np.ndarray, path: Path) -> Path:
    """Recall per typology and method: one hue, darker is higher, values printed in the cells."""
    apply_style()
    fig, ax = plt.subplots(figsize=(1.0 + 0.86 * len(typologies), 0.9 + 0.5 * len(methods)))
    image = ax.imshow(recall, cmap=SEQUENTIAL, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(typologies)), [t.replace("_", "\n") for t in typologies], fontsize=8.5)
    ax.set_yticks(np.arange(len(methods)), methods)
    ax.xaxis.tick_top()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(typologies)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    for i in range(recall.shape[0]):
        for j in range(recall.shape[1]):
            if np.isfinite(recall[i, j]):
                ax.text(j, i, f"{recall[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="#ffffff" if recall[i, j] >= 0.55 else INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.outline.set_visible(False)
    bar.set_label("recall", fontsize=9, color=INK_2)
    bar.ax.tick_params(labelsize=8, length=0)
    fig.suptitle("Which laundering patterns each method catches (recall on the test period, mean of 3 seeds)", x=0.02, y=1.04, ha="left", fontsize=11,
                 fontweight="bold")
    return _save(fig, path)


def communication(curves: list[dict[str, Any]], path: Path) -> Path:
    """Test F1 against megabytes exchanged, for federations of different size (ordered series: one hue)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ends = []
    for color, curve in zip(ORDINAL, curves):
        ax.plot(curve["mb"], curve["f1"], color=color, zorder=3, label=curve["label"])
        ends.append((float(curve["mb"][-1]), float(curve["f1"][-1]), curve["label"]))
    for _, (x, y, label) in zip(ORDINAL, ends):
        ax.annotate(label, (x, y), xytext=(6, 7), textcoords="offset points", va="center", fontsize=8.5, color=INK_2, annotation_clip=False)
    ax.set_xlabel("cumulative traffic of the whole federation, MB (model up and down, evaluation, setup)")
    ax.set_ylabel("F1 on the test period (mean of 3 seeds)")
    ax.set_xlim(0, max(c["mb"][-1] for c in curves) * 1.16)
    _only_y_grid(ax)
    ax.set_title("More clients need more traffic for a lower F1 (IID split, 30 rounds)")
    ax.legend(loc="lower right", handlelength=1.6)
    return _save(fig, path)


def ledger_modes(modes: list[str], throughput: list[float], gas: list[float], path: Path) -> Path:
    """Throughput and gas per record for each way of writing to the chain (two measures, two panels)."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 0.42 * len(modes) + 1.5), sharey=True)
    y = np.arange(len(modes))[::-1]
    for ax, values, label, fmt in [(axes[0], throughput, "records per second", "{:,.0f}"), (axes[1], gas, "gas per record", "{:,.0f}")]:
        ax.barh(y, values, height=BAR, color=BLUE, zorder=3)
        for yi, v in zip(y, values):
            ax.annotate(fmt.format(v), (v, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, max(values) * 1.22)
        ax.set_xlabel(label)
        _only_x_grid(ax)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v >= 1000 else f"{v:.0f}"))
    axes[0].set_yticks(y, modes)
    fig.suptitle("Batching raises throughput; a Merkle root makes the cost per document negligible (EVM backend)", x=0.02, y=1.04, ha="left", fontsize=11,
                 fontweight="bold")
    fig.subplots_adjust(wspace=0.12)
    return _save(fig, path)


def storage_growth(records: np.ndarray, direct_bytes: np.ndarray, merkle_bytes: np.ndarray, path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(records, direct_bytes / 1e6, color=BLUE, label="one record per document")
    ax.plot(records, merkle_bytes / 1e6, color=ORANGE, label="one Merkle root per 500 documents")
    ax.annotate(f"{direct_bytes[-1] / 1e6:.1f} MB", (records[-1], direct_bytes[-1] / 1e6), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.annotate(f"{merkle_bytes[-1] / 1e6:.2f} MB", (records[-1], merkle_bytes[-1] / 1e6), xytext=(6, 4), textcoords="offset points", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("trade documents registered")
    ax.set_ylabel("contract storage, MB")
    ax.set_xlim(0, records[-1] * 1.14)
    ax.set_ylim(0, None)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0"))
    _only_y_grid(ax)
    ax.set_title("On-chain storage grows linearly with the number of records")
    ax.legend(loc="upper left", handlelength=1.6)
    return _save(fig, path)


def privacy_utility(epsilons: list[float], mean: np.ndarray, std: np.ndarray, non_private: float, path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    x = np.arange(len(epsilons))
    ax.errorbar(x, mean, yerr=std, color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=3, zorder=3)
    _marker(ax, x, mean, BLUE)
    for xi, m in zip(x, mean):
        ax.annotate(f"{m:.3f}", (xi, m), xytext=(9, -11), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xticks(x, [f"ε = {e:g}" for e in epsilons])
    ax.set_xlim(-0.4, len(epsilons) - 0.6)
    ax.set_ylim(min(mean.min() - std.max() - 0.04, non_private - 0.05), non_private + 0.045)
    _reference_line(ax, non_private, f"FedAvg without DP {non_private:.3f}", x_text=-0.4)
    ax.set_xlabel("privacy budget of each client (smaller = more private), δ = 1e-5")
    ax.set_ylabel("F1 on the test period (mean ± 1 std, 3 seeds)")
    _only_y_grid(ax)
    ax.set_title("DP-SGD: what example-level privacy costs in F1")
    return _save(fig, path)


def condition_bars(labels: list[str], mean: np.ndarray, std: np.ndarray, highlight: list[bool], path: Path, title: str, xlabel: str,
                   notes: list[str] | None = None) -> Path:
    """A few conditions compared on one measure; the highlighted ones are the subject, the rest are context."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 0.5 * len(labels) + 1.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, mean, height=BAR, color=[BLUE if h else NEUTRAL for h in highlight], zorder=3)
    ax.errorbar(mean, y, xerr=std, fmt="none", ecolor=INK_2, elinewidth=1.0, capsize=2.5, zorder=4)
    for i, (yi, m, s) in enumerate(zip(y, mean, std)):
        text = f"{m:.3f}" + (f"   {notes[i]}" if notes and notes[i] else "")
        ax.annotate(text, (m + s, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, float((mean + std).max()) * (1.42 if notes else 1.16))
    ax.set_xlabel(xlabel)
    _only_x_grid(ax)
    ax.spines["left"].set_visible(False)
    ax.set_title(title)
    return _save(fig, path)


def risk_bands(bands: list[dict[str, Any]], reliability: list[dict[str, Any]], path: Path) -> Path:
    """Left: share of illicit trades per risk band. Right: predicted against observed probability."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    y = np.arange(len(bands))[::-1]
    values = [100 * (b["precision"] or 0.0) for b in bands]
    ax.barh(y, values, height=BAR, color=[STATUS[b["band"]] for b in bands], zorder=3)
    for yi, b, v in zip(y, bands, values):
        ax.annotate(f"{v:.0f}%   ({b['illicit']:,} of {b['trades']:,} trades)", (v, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [b["band"] for b in bands])
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of trades in the band that are illicit")
    _only_x_grid(ax)
    ax.spines["left"].set_visible(False)
    ax.set_title("Precision rises with the risk band", fontsize=10.5)

    ax = axes[1]
    predicted = np.array([r["mean_predicted"] for r in reliability])
    observed = np.array([r["observed_rate"] for r in reliability])
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.1, zorder=1)
    ax.annotate("perfectly calibrated", (0.62, 0.62), xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.plot(predicted, observed, color=BLUE, zorder=3)
    _marker(ax, predicted, observed, BLUE)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("risk score / 100 (mean per bucket)")
    ax.set_ylabel("observed share of illicit trades")
    ax.set_title("The score can be read as a probability", fontsize=10.5)
    fig.subplots_adjust(wspace=0.28)
    return _save(fig, path)
