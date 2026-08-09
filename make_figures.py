
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLOR_BASELINE = "#2a78d6"
COLOR_REFLECTION = "#eb6834"
COLOR_CONTROL = "#1baf7a"
COLORS = {"baseline": COLOR_BASELINE, "reflection": COLOR_REFLECTION, "control": COLOR_CONTROL}
LABELS = {"baseline": "Baseline", "reflection": "Reflection", "control": "Control"}
CONDITIONS = ["baseline", "reflection", "control"]
INSTANCES = [1, 2, 3]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#52514e",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#0b0b0b",
    "ytick.color": "#0b0b0b",
    "axes.grid": True,
    "grid.color": "#e3e2dd",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load(path):
    with open(path) as f:
        return json.load(f)["results"]


def bar_value_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="#0b0b0b")


def grouped_bar_by_instance(results, metric_fn, title, ylabel, fname, fmt="{:.2f}", ylim=None):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    x = range(len(INSTANCES))
    width = 0.26
    for i, cond in enumerate(CONDITIONS):
        vals = [metric_fn(results, inst, cond) for inst in INSTANCES]
        offset = (i - 1) * width
        bars = ax.bar([xi + offset for xi in x], vals, width=width,
                      color=COLORS[cond], label=LABELS[cond])
        bar_value_labels(ax, bars, fmt)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Instance {i}" for i in INSTANCES])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(f"figures/{fname}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{fname}")


def avg_fairness(results, instance, cond):
    subset = [r for r in results if r["instance"] == instance and r["condition"] == cond]
    return sum(r["fairness_gap"] for r in subset) / len(subset)


def avg_efficiency(results, instance, cond):
    subset = [r for r in results if r["instance"] == instance and r["condition"] == cond]
    return sum(r["efficiency"] for r in subset) / len(subset)


def avg_belief(results, instance, cond):
    subset = [r for r in results if r["instance"] == instance and r["condition"] == cond]
    scores = [r["judge_belief_consistency_a"] for r in subset if r["judge_belief_consistency_a"] is not None] + \
              [r["judge_belief_consistency_b"] for r in subset if r["judge_belief_consistency_b"] is not None]
    return sum(scores) / len(scores) if scores else None


def make_fairness_chart(results):
    grouped_bar_by_instance(
        results, avg_fairness,
        "Fairness gap by instance and condition (post-fix, n=10/cell)",
        "Avg. fairness gap (points)",
        "fig_fairness_by_instance.png",
        fmt="{:.2f}", ylim=(0, 3.2),
    )


def make_efficiency_chart(results):
    grouped_bar_by_instance(
        results, avg_efficiency,
        "Efficiency by instance and condition (post-fix, n=10/cell)",
        "Avg. efficiency (0-1)",
        "fig_efficiency_by_instance.png",
        fmt="{:.2f}", ylim=(0, 1.15),
    )


def make_belief_chart(results):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    x = range(len(INSTANCES))
    width = 0.32
    for i, cond in enumerate(["reflection", "control"]):
        vals = [avg_belief(results, inst, cond) for inst in INSTANCES]
        offset = (i - 0.5) * width
        bars = ax.bar([xi + offset for xi in x], vals, width=width,
                      color=COLORS[cond], label=LABELS[cond])
        bar_value_labels(ax, bars, "{:.2f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Instance {i}" for i in INSTANCES])
    ax.set_ylabel("Avg. judge belief-consistency score (1-5)")
    ax.set_ylim(0, 5.5)
    ax.set_title("Belief consistency: reflection vs. control\n(baseline n/a — never populates a belief history)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig("figures/fig_belief_consistency_by_instance.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_belief_consistency_by_instance.png")


def make_deal_rate_before_after_chart(pre_results, post_results):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    x = range(len(CONDITIONS))
    width = 0.32
    pre_vals, post_vals = [], []
    for cond in CONDITIONS:
        pre = [r for r in pre_results if r["condition"] == cond]
        post = [r for r in post_results if r["condition"] == cond]
        pre_vals.append(sum(1 for r in pre if r["accepted"]))
        post_vals.append(sum(1 for r in post if r["accepted"]))

    bars_pre = ax.bar([xi - width / 2 for xi in x], pre_vals, width=width,
                       color=[COLORS[c] for c in CONDITIONS], hatch="////",
                       edgecolor="white", linewidth=0.6, label="Before fix")
    bars_post = ax.bar([xi + width / 2 for xi in x], post_vals, width=width,
                        color=[COLORS[c] for c in CONDITIONS], label="After fix")
    bar_value_labels(ax, bars_pre, "{:.0f}/30")
    bar_value_labels(ax, bars_post, "{:.0f}/30")

    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Deals reached (out of 30)")
    ax.set_ylim(0, 34)
    ax.set_title("Deal rate before vs. after the conversation-history fix\n(83/90 -> 90/90 overall)",
                 fontsize=12, fontweight="bold", pad=12)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#c3c2b7", hatch="////", edgecolor="white", label="Before fix"),
        Patch(facecolor="#c3c2b7", label="After fix"),
    ]
    ax.legend(handles=legend_elems, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig("figures/fig_deal_rate_before_after.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_deal_rate_before_after.png")


def avg_belief_accuracy(results, instance, cond):
    subset = [r for r in results if r["instance"] == instance and r["condition"] == cond]
    scores = [r["belief_accuracy_a"] for r in subset if r["belief_accuracy_a"] is not None] + \
              [r["belief_accuracy_b"] for r in subset if r["belief_accuracy_b"] is not None]
    return sum(scores) / len(scores) if scores else None


def make_belief_accuracy_chart(results):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    x = range(len(INSTANCES))
    width = 0.32
    for i, cond in enumerate(["reflection", "control"]):
        vals = [avg_belief_accuracy(results, inst, cond) for inst in INSTANCES]
        offset = (i - 0.5) * width
        bars = ax.bar([xi + offset for xi in x], vals, width=width,
                      color=COLORS[cond], label=LABELS[cond])
        bar_value_labels(ax, bars, "{:.2f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Instance {i}" for i in INSTANCES])
    ax.set_ylabel("Avg. belief-accuracy score (1-5)")
    ax.set_ylim(0, 5.5)
    ax.set_title("Belief accuracy vs. ground truth: reflection vs. control\n(pooled: 3.30 vs 3.03, p=0.21 -- consistent direction, not significant)",
                 fontsize=11.5, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig("figures/fig_belief_accuracy_by_instance.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_belief_accuracy_by_instance.png")


if __name__ == "__main__":
    post = load("results/experiment_v2.json")
    pre = load("results/experiment_20260804_231700.json")

    make_fairness_chart(post)
    make_efficiency_chart(post)
    make_belief_chart(post)
    make_deal_rate_before_after_chart(pre, post)

    try:
        belief_accuracy_results = load("results/belief_accuracy_v2.json")
        make_belief_accuracy_chart(belief_accuracy_results)
    except FileNotFoundError:
        print("Skipping belief-accuracy chart: results/belief_accuracy_v2.json not found "
              "(run `python -m src.run_belief_accuracy` first).")
