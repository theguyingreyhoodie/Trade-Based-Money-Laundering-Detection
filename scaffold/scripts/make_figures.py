"""Phase 6: figures for the report, from the results the other scripts wrote.

    fig1_model_comparison   test PR-AUC of every setup, mean ± std over seeds
    fig2_training_curves    validation PR-AUC per FL round, mean over seeds with min-max band
    fig3_pattern_recall     recall per laundering pattern for four setups
    fig4_per_bank           each bank alone vs federated hybrid
    fig5_ablation           label_signal 0.1 vs 0 (only once the ablation results exist)

Each figure is written as PNG (300 dpi) and PDF to docs/figures/. They are made for print: every
series has a second encoding (line style, marker shape) besides its colour, so they read in
black and white too. Colours: the first four slots of a palette validated for colour-blind
separation; the method's own result is always slot 1 (blue).

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --ablation-config configs/ablation_signal0.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml_fl.utils import load_config, results_dir  # noqa: E402

OUT = ROOT / "docs" / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
GREY_DARK, GREY, GREY_LIGHT = "#6f6e69", "#a3a29c", "#cfcec8"

# setup, model, label, colour (ours in blue; the bounds that are not private in light grey)
SETUPS = [
    ("centralized", "xgboost", "Centralized XGBoost (not private)", GREY_LIGHT),
    ("centralized", "mlp", "Centralized MLP (not private)", GREY_LIGHT),
    ("local", "xgboost", "Each bank alone, XGBoost", GREY),
    ("fl_hybrid", "mlp", "Federated hybrid (ours)", BLUE),
    ("fl_zero_filled", "mlp", "Federated zero-filled", GREY_DARK),
    ("local", "mlp", "Each bank alone, MLP", GREY),
    ("fl_encoder_head", "mlp", "Federated, shared head only", GREY_DARK),
    ("fl_naive", "mlp", "Federated, common columns only", GREY_DARK),
]
MODES = [("hybrid", "Hybrid (ours)", BLUE, "-"), ("zero_filled", "Zero-filled", ORANGE, "--"),
         ("encoder_head", "Shared head only", AQUA, "-."), ("naive", "Common columns only", YELLOW, ":")]
PATTERN_NAMES = {"fan_out": "Fan-out", "fan_in": "Fan-in", "cycle": "Cycle", "scatter_gather": "Scatter-gather",
                 "gather_scatter": "Gather-scatter", "bipartite": "Bipartite", "stack": "Stack",
                 "random": "Random", "unlisted": "Not in a listed pattern"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote docs/figures/{name}.png / .pdf")


def headline(res: Path) -> pd.DataFrame:
    s = pd.read_csv(res / "seeds_summary.csv")
    return s[s["client"].isin(["pooled", "all"])].set_index(["setup", "model"])


def fig_model_comparison(res: Path) -> None:
    s = headline(res)
    rows = [(lab, s.loc[(st, m)], col) for st, m, lab, col in SETUPS if (st, m) in s.index]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r, col) in zip(y, rows):
        ax.barh(yi, r["pr_auc_mean"], height=0.62, color=col, edgecolor="white", linewidth=1)
        ax.errorbar(r["pr_auc_mean"], yi, xerr=r["pr_auc_std"], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
        ax.text(r["pr_auc_mean"] + r["pr_auc_std"] + 0.006, yi, f"{r['pr_auc_mean']:.3f} ± {r['pr_auc_std']:.3f}",
                va="center", fontsize=8, color=INK, fontweight="bold" if col == BLUE else "normal")
    ax.set_yticks(y, [lab for lab, _, _ in rows])
    for tick, (_, _, col) in zip(ax.get_yticklabels(), rows):
        if col == BLUE:
            tick.set_fontweight("bold")
    ax.set_xlabel(f"Test PR-AUC (pooled over 5 banks, mean ± std over {int(rows[0][1]['seeds'])} seeds)")
    ax.set_xlim(0, max(r["pr_auc_mean"] + r["pr_auc_std"] for _, r, _ in rows) * 1.28)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Detection quality of every setup")
    save(fig, "fig1_model_comparison")


def fig_training_curves(res: Path, seeds=(42, 43, 44)) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ends = []
    for mode, label, col, ls in MODES:
        runs = [pd.read_csv(p)["val_pr_auc"].to_numpy() for s in seeds
                if (p := res / f"fl_{mode}_s{s}" / "history.csv").exists()]
        if not runs:
            continue
        n = min(len(r) for r in runs)
        arr = np.vstack([r[:n] for r in runs])
        x = np.arange(1, n + 1)
        ax.fill_between(x, arr.min(0), arr.max(0), color=col, alpha=0.13, linewidth=0)
        ax.plot(x, arr.mean(0), color=col, linestyle=ls, linewidth=2 if mode == "hybrid" else 1.6,
                label=f"{label} ({len(runs)} seeds)")
        ends.append([arr.mean(0)[-1], label, col])
    # direct labels at the line ends, nudged apart so they never overlap
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.011)
    for yv, label, col in ends:
        ax.text(101.5, yv, label, color=INK, fontsize=8, va="center")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, None)
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Validation PR-AUC")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, fontsize=8, handlelength=2.6,
              title="Line: mean over seeds; band: min to max", title_fontsize=7.5, alignment="left")
    ax.set_title("Validation PR-AUC over 100 federated rounds")
    save(fig, "fig2_training_curves")


def fig_pattern_recall(res: Path) -> None:
    frames = [pd.read_csv(p) for p in res.glob("*_s*_by_pattern.csv")]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["client"].isin(["pooled", "all"])]
    agg = df.groupby(["setup", "model", "pattern"]).agg(recall=("recall", "mean"), n=("positives", "mean")).reset_index()
    show = [("fl_hybrid", "mlp", "Federated hybrid (ours)", BLUE, "o"), ("fl_zero_filled", "mlp", "Federated zero-filled", ORANGE, "s"),
            ("local", "mlp", "Each bank alone, MLP", AQUA, "^"), ("centralized", "xgboost", "Centralized XGBoost", YELLOW, "D")]
    ref = agg[(agg["setup"] == "fl_hybrid") & (agg["pattern"].isin(PATTERN_NAMES))]
    order = ref.sort_values("n", ascending=False)["pattern"].tolist()
    order = [p for p in order if p != "unlisted"] + (["unlisted"] if "unlisted" in order else [])
    n_of = ref.set_index("pattern")["n"]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    y = np.arange(len(order))[::-1]
    for yi in y:
        ax.axhline(yi, color=GRID, linewidth=0.7, zorder=0)
    offsets = np.linspace(-0.24, 0.24, len(show))
    for (st, m, label, col, mk), off in zip(show, offsets):
        sub = agg[(agg["setup"] == st) & (agg["model"] == m)].set_index("pattern")
        xs = [sub["recall"].get(p, np.nan) for p in order]
        ax.scatter(xs, y + off, s=34, marker=mk, color=col, edgecolor=INK, linewidth=0.5, label=label, zorder=3)
    ax.set_yticks(y, [f"{PATTERN_NAMES[p]}  (n={int(round(n_of[p]))})" for p in order])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Recall on the test set (mean over seeds, each bank's own threshold)")
    ax.legend(loc="lower right", fontsize=8, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("Which laundering patterns are caught")
    save(fig, "fig3_pattern_recall")


def fig_per_bank(res: Path, names: dict) -> None:
    pc = pd.read_csv(res / "seeds_per_client.csv").set_index(["setup", "model"])
    alone, fed = pc.loc[("local", "mlp")], pc.loc[("fl_hybrid", "mlp")]
    banks = sorted({c.rsplit("_", 1)[0] for c in pc.columns if c.endswith("_mean")})
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    y = np.arange(len(banks))[::-1]
    for yi, b in zip(y, banks):
        a, f = alone[f"{b}_mean"], fed[f"{b}_mean"]
        ax.plot([a, f], [yi, yi], color=GREY, linewidth=1.6, zorder=1)
        ax.scatter([a], [yi], s=46, facecolor="white", edgecolor=INK, linewidth=1.1, zorder=2,
                   label="Each bank alone, MLP" if yi == y[0] else None)
        ax.scatter([f], [yi], s=46, color=BLUE, edgecolor=INK, linewidth=0.5, zorder=3,
                   label="Federated hybrid (ours)" if yi == y[0] else None)
        ax.text(max(a, f) + 0.012, yi, f"{f - a:+.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [names.get(b, b) for b in banks])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, max(max(alone[f"{b}_mean"], fed[f"{b}_mean"]) for b in banks) * 1.25)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Test PR-AUC at that bank (mean over seeds)")
    ax.legend(loc="lower right", fontsize=8, ncol=2, bbox_to_anchor=(1, 1.0), borderaxespad=0.1)
    ax.set_title("Every bank gains from joining the federation", pad=22)
    save(fig, "fig4_per_bank")


def fig_ablation(res: Path, res0: Path) -> None:
    a, b = headline(res), headline(res0)
    rows = [(lab, a.loc[(st, m)], b.loc[(st, m)]) for st, m, lab, _ in SETUPS if (st, m) in a.index and (st, m) in b.index]
    fig, ax = plt.subplots(figsize=(6.3, 0.42 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r1, r0) in zip(y, rows):
        ax.plot([r0["pr_auc_mean"], r1["pr_auc_mean"]], [yi, yi], color=GREY, linewidth=1.6, zorder=1)
        ax.errorbar(r1["pr_auc_mean"], yi, xerr=r1["pr_auc_std"], fmt="o", ms=6, color=GREY_DARK, mec=INK, mew=0.5,
                    ecolor=GREY_DARK, elinewidth=0.9, capsize=2, zorder=3, label="Profiles with label signal 0.1 (main runs)" if yi == y[0] else None)
        ax.errorbar(r0["pr_auc_mean"], yi, xerr=r0["pr_auc_std"], fmt="s", ms=5.5, mfc="white", mec=INK, mew=1,
                    ecolor=INK, elinewidth=0.9, capsize=2, zorder=2, label="Profiles pure noise (signal 0)" if yi == y[0] else None)
    ax.set_yticks(y, [lab for lab, _, _ in rows])
    for tick in ax.get_yticklabels():
        if "ours" in tick.get_text():
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, None)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Test PR-AUC (pooled, mean ± std over seeds)")
    ax.legend(loc="lower right", fontsize=8, ncol=1, bbox_to_anchor=(1, 1.0), borderaxespad=0.1)
    ax.set_title("Ablation: do the results depend on the synthetic profiles?", pad=30)
    save(fig, "fig5_ablation")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ablation-config", default="configs/ablation_signal0.yaml")
    args = ap.parse_args()
    style()
    cfg = load_config(ROOT / args.config)
    res = results_dir(cfg, ROOT)
    fig_model_comparison(res)
    fig_training_curves(res)
    fig_pattern_recall(res)
    fig_per_bank(res, cfg["partition"].get("bank_names", {}))
    res0 = results_dir(load_config(ROOT / args.ablation_config), ROOT)
    if (res0 / "seeds_summary.csv").exists():
        fig_ablation(res, res0)
    else:
        print(f"no ablation results in {res0} yet; skipping fig5")


if __name__ == "__main__":
    main()
