"""產生論文 Fig.1 / 投影片用的系統架構圖（figures/fig1_architecture.png）。

兩個 panel：
  (a) 單一 episode：causal feature pyramid 上的 budgeted evidence acquisition。
  (b) 任務序列下的 continual learning：compressed state replay + utility-weighted
      policy distillation（凍結 per-task evaluator 的 navigation-only 歸因）。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ------------------------------------------------------------------ palette
C_ENV = "#dbeafe"      # environment 藍
C_ENV_E = "#2563eb"
C_AGENT = "#fee2e2"    # agent 紅
C_AGENT_E = "#dc2626"
C_TEACH = "#fef9c3"    # teacher 黃
C_TEACH_E = "#ca8a04"
C_EVAL = "#dcfce7"     # evaluator 綠
C_EVAL_E = "#16a34a"
C_CL = "#f3e8ff"       # CL 紫
C_CL_E = "#9333ea"
GREY = "#6b7280"


def box(ax, x, y, w, h, text, fc, ec, fs=8.5, lw=1.4, style="round,pad=0.02,rounding_size=0.03"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.35)


def arrow(ax, p0, p1, text="", color=GREY, fs=7.5, lw=1.6, rad=0.0,
          tx=None, ty=None, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        x = (p0[0] + p1[0]) / 2 if tx is None else tx
        y = (p0[1] + p1[1]) / 2 if ty is None else ty
        ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center",
                zorder=3, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


def panel_a(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title("(a) Budgeted evidence acquisition on a causal feature pyramid",
                 fontsize=10, pad=6)

    # environment
    box(ax, 0.25, 4.55, 4.1, 2.35, "", "#f8fafc", "#cbd5e1", lw=1.0)
    ax.text(0.45, 6.68, "Environment (causal access rule)", fontsize=8,
            color="#334155", style="italic")
    box(ax, 0.55, 5.55, 3.5, 0.95,
        "Coarse view  $F_{low}$  [R × $d_{low}$]\nall regions, always visible",
        C_ENV, C_ENV_E)
    box(ax, 0.55, 4.72, 3.5, 0.72,
        "Fine evidence  $z_r$  [$d_{high}$]\nrevealed ONLY after zoom",
        C_ENV, C_ENV_E, fs=8)

    # navigator
    box(ax, 5.6, 5.3, 3.9, 1.35,
        "Shared Navigator  $\\pi_\\theta$\nscoring MLP (<1M params)\n"
        "input: $F_{low}[r] \\oplus E_t \\oplus b_t$",
        C_AGENT, C_AGENT_E)

    # evidence state
    box(ax, 5.6, 3.1, 3.9, 1.0,
        "Evidence state  $E_t$ = mean of\nrevealed fine evidence", C_AGENT, C_AGENT_E)

    # evaluator
    box(ax, 5.6, 1.15, 3.9, 1.0,
        "Frozen evaluator  $f$\ndiagnosis after $K$ steps", C_EVAL, C_EVAL_E)

    # teacher
    box(ax, 0.25, 1.15, 4.1, 1.7,
        "Counterfactual teacher (training only)\n"
        "$\\mathrm{gain}(r|E) = \\mathrm{CE}(f(E),y) - \\mathrm{CE}(f(E \\cup \\{r\\}),y)$\n"
        "weak supervision from slide-level labels",
        C_TEACH, C_TEACH_E, fs=8)

    # arrows
    arrow(ax, (4.15, 6.0), (5.6, 6.0), "observe", color=C_ENV_E)
    arrow(ax, (7.55, 5.3), (4.9, 5.08), "$a_t$ = select & zoom", color=C_AGENT_E,
          rad=-0.25, tx=6.4, ty=4.78)
    arrow(ax, (4.05, 4.9), (5.6, 3.7), "reveal $z_{a_t}$", color=C_ENV_E, rad=-0.2,
          tx=4.75, ty=4.0)
    arrow(ax, (7.55, 4.1), (7.55, 5.3), "", color=C_AGENT_E)
    arrow(ax, (7.55, 3.1), (7.55, 2.15), "after $K$ steps", color=C_EVAL_E)
    arrow(ax, (4.35, 2.0), (5.6, 1.65), "trains", color=C_TEACH_E, ls="--")
    arrow(ax, (2.3, 2.85), (6.3, 5.3), "imitation target\n"
          "$q \\propto \\exp(\\mathrm{gain}/\\tau)$", color=C_TEACH_E, rad=0.3,
          ls="--", tx=3.6, ty=3.7)

    ax.text(5.05, 0.45,
            "budget $b_t$: only $K$ regions may be zoomed;  un-zoomed evidence does not exist for the agent",
            fontsize=7.5, color="#334155", ha="center", style="italic")


def panel_b(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title("(b) Continual learning: preserve where-to-look, not just predictions",
                 fontsize=10, pad=6)

    # task timeline
    xs = [0.35, 2.75, 5.15]
    labels = ["Task 1", "Task 2", "Task n"]
    for x, lab in zip(xs, labels):
        box(ax, x, 5.9, 2.0, 0.85, f"{lab}\n(train)", "#f1f5f9", "#94a3b8", fs=8.5)
    arrow(ax, (2.35, 6.32), (2.75, 6.32), color=GREY)
    ax.text(4.95, 6.32, "…", fontsize=13, color=GREY, ha="center", va="center")
    arrow(ax, (5.6, 6.32), (5.15 + 0.0, 6.32), color=GREY, style="-")

    # shared navigator through time
    box(ax, 0.35, 4.15, 6.8, 1.0,
        "ONE shared navigator $\\pi_\\theta$ updated across all tasks"
        "  (no task ID, no per-task bank)", C_AGENT, C_AGENT_E, fs=8.5)
    for x in xs:
        arrow(ax, (x + 1.0, 5.9), (x + 1.0, 5.15), color=GREY, lw=1.2)

    # frozen evaluators
    box(ax, 7.6, 5.9, 2.1, 0.85, "$f_1$ frozen", C_EVAL, C_EVAL_E, fs=8.5)
    box(ax, 7.6, 4.85, 2.1, 0.85, "$f_2$ frozen", C_EVAL, C_EVAL_E, fs=8.5)
    ax.text(8.65, 4.55, "…", fontsize=12, color=GREY, ha="center")
    ax.text(8.65, 3.98,
            "navigation-only attribution:\nold-task change ⇒ policy change",
            fontsize=7.3, color="#166534", ha="center", style="italic")

    # CL mechanism
    box(ax, 0.35, 1.5, 4.35, 1.9,
        "Compressed state replay $\\mathcal{M}$\n"
        "per old slide: a few teacher states\n"
        "$(F_{low}, E_t, b_t, \\mathrm{gain}, u)$ — no raw images",
        C_CL, C_CL_E, fs=8)
    box(ax, 5.15, 1.5, 4.55, 1.9,
        "Utility-weighted policy distillation\n"
        "$\\lambda \\sum_{s \\in \\mathcal{M}} w(s)\\,"
        "\\mathrm{KL}(\\pi_\\theta \\| \\pi_{old})$\n"
        "$w(s) \\propto u(s) = \\max_r \\mathrm{gain}(r|E_s)$",
        C_CL, C_CL_E, fs=8)
    arrow(ax, (2.5, 3.4), (2.5, 4.15), "store at end of each task", color=C_CL_E,
          ls="--", tx=1.35, ty=3.8)
    arrow(ax, (7.4, 3.4), (7.4, 4.15), "regularize while learning new task",
          color=C_CL_E, tx=7.45, ty=3.8)
    arrow(ax, (4.7, 2.45), (5.15, 2.45), color=C_CL_E, lw=1.2)

    ax.text(5.0, 0.7,
            "measured: accuracy forgetting  +  selection Jaccard  +  action-KL drift"
            "  (behavior-level forgetting)",
            fontsize=7.8, color="#334155", ha="center", style="italic")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    panel_a(axes[0])
    panel_b(axes[1])
    fig.suptitle("Continual Learning of Budgeted Visual Evidence Acquisition",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "figures/fig1_architecture.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
