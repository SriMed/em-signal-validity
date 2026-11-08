"""Render documentation figures from the published Markdown tables, without models.

Run from any directory with matplotlib installed (requirements.txt pins its version):
    python scripts/plot_docs.py
These figures summarize reported values; they do not verify historical tensors.
"""

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "em-docs-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EM, BENIGN, INK = "#b34c32", "#147d92", "#25354a"


def table(document, header):
    """Read the table with this exact header; prose and other tables are ignored."""
    lines = (ROOT / document).read_text().splitlines()
    start = lines.index(header) + 2
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    if not rows:
        raise ValueError(f"Empty table: {document}: {header}")
    return rows


def save(fig, destination, name):
    fig.savefig(destination / name, dpi=180, facecolor="white")
    plt.close(fig)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e3e8ed", linewidth=0.7)
    ax.set_axisbelow(True)


def overview(destination):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set(xlim=(0, 12), ylim=(0, 6.2))
    ax.axis("off")
    ax.text(0.25, 5.85, "What changed inside the model — and what does it mean?",
            fontsize=18, weight="bold", color=INK)
    ax.text(0.25, 5.42, "Phase 1 compares two models. Phase 2 adds a benign medical control.",
            fontsize=11, color=INK)

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                   facecolor=color, edgecolor="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=11, color=INK, linespacing=1.5)

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": "#738397", "lw": 1.5})

    box(0.4, 2.85, 2.6, 1.25, "Same 16 prompts\nIdentical input tokens", "#eaf0f6")
    models = [(4.15, "Base model\nQwen2.5-0.5B", "#eaf0f6"),
              (2.7, "Base + released EM adapter\nBad-medical-advice objective", "#faeae4"),
              (1.25, "Base + benign adapter\nGood-medical-advice objective", "#e1f2f4")]
    for y, label, color in models:
        box(4, y, 3.5, 1, label, color)
        arrow((3.1, 3.48), (3.9, y + 0.5))
        arrow((7.6, y + 0.5), (8.45, 3.48))
    box(8.55, 2.75, 3, 1.45, "Inspect sampled replies\nMeasure hidden-state distance\nfrom the base model", "#eaf0f6")
    ax.text(0.4, 0.57, "Observed: benign and EM distances are comparable.",
            fontsize=13, weight="bold", color=INK)
    ax.text(0.4, 0.12, "Distance alone does not identify misalignment. Control matching is partial; behavior review is unblinded.",
            fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, destination, "experiment-overview.png")


def distances(destination, phase):
    if phase == 1:
        rows = table("docs/phase1-results.md",
                     "| Prompt group | Prompts | Index 23 | Final normalized state, index 24 |")
    else:
        rows = table("docs/phase2-results.md",
                     "| Prompt group | Prompts | EM, index 23 | Benign, index 23 | EM, index 24 | Benign, index 24 |")
    labels = [f"{r[0]}\n(n = {r[1]})" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1), sharey=True)
    for panel, ax in enumerate(axes):
        style_axis(ax)
        columns = [("EM", EM, 2 + panel)] if phase == 1 else [
            ("EM", EM, 2 + panel * 2), ("Benign", BENIGN, 3 + panel * 2)]
        for i, (name, color, column) in enumerate(columns):
            values = [float(r[column]) for r in rows]
            offset = 0 if phase == 1 else (i - 0.5) * 0.36
            bars = ax.bar(x + offset, values, width=0.55 if phase == 1 else 0.34,
                          color=color, label=name)
            ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=4)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 0.61)
        ax.set_title(["Index 23 · intermediate residual state",
                      "Index 24 · after final RMSNorm"][panel], fontsize=11, pad=14)
    axes[0].set_ylabel("Relative distance from base\n‖h_adapter − h_base‖ / ‖h_base‖")
    if phase == 2:
        axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Phase 1: changes extend to neutral prompts" if phase == 1 else
                 "Phase 2: benign fine-tuning gives comparable distances",
                 fontsize=16, weight="bold", color=INK)
    fig.text(0.06, 0.025,
             "Reported group means; no uncertainty estimates. Distances are not harmfulness probabilities.\n"
             "Panels use different normalization conventions. Historical raw artifacts are not publicly released.",
             fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    save(fig, destination, f"phase{phase}-distances.png")


def early_tokens(destination):
    rows = table("docs/phase2-results.md",
                 "| Token position | EM static fraction | Benign static fraction | Cosine between mean difference vectors |")
    values = np.array([[float(cell) for cell in row] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1))
    for ax in axes:
        style_axis(ax)
        ax.set(xlabel="Token position (zero-based)", xticks=values[:, 0])
    for column, label, color, marker in [(1, "EM", EM, "o"), (2, "Benign", BENIGN, "s")]:
        axes[0].plot(values[:, 0], values[:, column], marker=marker,
                     color=color, label=label, linewidth=2)
    axes[0].set(ylabel="Static fraction", ylim=(0, 1),
                title="Both updates have a large static fraction at token 0")
    axes[0].legend(frameon=False)
    axes[1].plot(values[:, 0], values[:, 3], marker="D", color=INK, linewidth=2)
    for position, cosine in zip(values[:, 0], values[:, 3]):
        axes[1].annotate(f"{cosine:.4f}", (position, cosine),
                         xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    axes[1].set(ylabel="Cosine between EM and benign mean differences", ylim=(-1, 1),
                title="Mean difference directions overlap more at tokens 1–4")
    fig.suptitle("Early-token differences: shared structure, unclear meaning",
                 fontsize=16, weight="bold", color=INK)
    fig.text(0.06, 0.025,
             "Reported values on 10,000 unrelated web samples at block 12 (hidden-state index 13).\n"
             "Static fraction measures input-independent energy; cosine measures direction overlap. Neither measures harmfulness.",
             fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    save(fig, destination, "phase2-early-tokens.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlecolor": INK, "axes.labelcolor": INK})
    overview(args.output_dir)
    distances(args.output_dir, 1)
    distances(args.output_dir, 2)
    early_tokens(args.output_dir)
    print(f"Wrote four documentation figures to {args.output_dir}")


if __name__ == "__main__":
    main()
