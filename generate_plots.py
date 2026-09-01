"""
generate_plots.py
------------------
Regenerates all figures

"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_training_rewards():
    with open(os.path.join(RESULTS_DIR, "training_log.json")) as f:
        log = json.load(f)

    seeds = sorted(log.keys())
    colors = ["#2563eb", "#dc2626", "#16a34a"]

    # --- per-seed reward curves ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    all_rewards = []
    for i, s in enumerate(seeds):
        r = log[s]["episode_rewards"]
        all_rewards.append(r)
        ax.plot(r, alpha=0.35, color=colors[i % len(colors)], linewidth=1)
        window = 10
        if len(r) >= window:
            smooth = np.convolve(r, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(r)), smooth, color=colors[i % len(colors)],
                     linewidth=2, label=f"Seed {s} (smoothed)")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Cumulative Log-Return Reward")
    ax.set_title("DDPG Training Reward vs. Episode (3 seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_reward_curve.png"), dpi=150)
    plt.close(fig)
    print("saved plots/training_reward_curve.png")

    # --- mean +/- std band ---
    all_rewards = np.array(all_rewards)
    mean_r = all_rewards.mean(axis=0)
    std_r = all_rewards.std(axis=0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(mean_r, color="#2563eb", linewidth=2, label="Mean reward across seeds")
    ax.fill_between(range(len(mean_r)), mean_r - std_r, mean_r + std_r,
                     color="#2563eb", alpha=0.2, label="\u00b11 std across seeds")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("DDPG Training Reward: Mean \u00b1 Std Across Seeds")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_reward_mean_std.png"), dpi=150)
    plt.close(fig)
    print("saved plots/training_reward_mean_std.png")


def plot_test_portfolio_comparison():
    with open(os.path.join(RESULTS_DIR, "eq_baseline_history.json")) as f:
        eq = json.load(f)
    with open(os.path.join(RESULTS_DIR, "bh_baseline_history.json")) as f:
        bh = json.load(f)
    with open(os.path.join(RESULTS_DIR, "ddpg_test_histories.json")) as f:
        ddpg = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eq["value"], label="Equal-Weight (daily rebal.)", color="#6b7280", linewidth=2, linestyle="--")
    ax.plot(bh["value"], label="Buy-and-Hold", color="#111827", linewidth=2, linestyle=":")

    colors = ["#2563eb", "#dc2626", "#16a34a"]
    seeds = sorted(ddpg.keys())
    for i, s in enumerate(seeds):
        ax.plot(ddpg[s]["value"], label=f"DDPG (seed={s})", color=colors[i % len(colors)], linewidth=1.6, alpha=0.85)

    vals = np.array([ddpg[s]["value"] for s in seeds])
    mean_v = vals.mean(axis=0)
    ax.plot(mean_v, label="DDPG mean (3 seeds)", color="black", linewidth=2.5)

    ax.axhline(100_000, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Trading Day (one illustrative 126-day evaluation episode)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Portfolio Value on ONE Illustrative Evaluation Episode\n(headline results use all 30 episodes \u2014 see boxplot figure)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "test_portfolio_value_comparison.png"), dpi=150)
    plt.close(fig)
    print("saved plots/test_portfolio_value_comparison.png")


def plot_metrics_bars():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "test_metrics.csv"))
    # exclude the cross-seed summary row from the per-strategy bar chart
    plot_df = df[df["strategy"] != "DDPG mean-of-seed-means (n=3 seeds)"].reset_index(drop=True)
    colors = ["#6b7280", "#111827", "#2563eb", "#dc2626", "#16a34a"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(
        plot_df["strategy"], plot_df["cumulative_return_%_mean"],
        yerr=plot_df["cumulative_return_%_std"], capsize=4,
        color=colors[:len(plot_df)],
    )
    axes[0].set_ylabel("Cumulative Return (%)")
    axes[0].set_title("Mean Return Across 30 Evaluation Episodes\n(error bars = \u00b11 std across episodes)")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(
        plot_df["strategy"], plot_df["sharpe_ratio_mean"],
        yerr=plot_df["sharpe_ratio_std"], capsize=4,
        color=colors[:len(plot_df)],
    )
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_title("Mean Sharpe Ratio Across 30 Evaluation Episodes\n(error bars = \u00b11 std across episodes)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "metrics_comparison_bars.png"), dpi=150)
    plt.close(fig)
    print("saved plots/metrics_comparison_bars.png")


def plot_episode_distribution():
    """Boxplot of per-episode cumulative return, across all 30 evaluation
    episodes, for every strategy.
    """
    raw = pd.read_csv(os.path.join(RESULTS_DIR, "eval_episode_log.csv"))

    labels = ["Equal-Weight", "Buy-and-Hold"] + [f"DDPG (seed={s})" for s in sorted(raw["seed"].dropna().unique().astype(int))]
    data = [raw.loc[raw["label"] == lbl, "cumulative_return_%"].values for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    colors = ["#6b7280", "#111827", "#2563eb", "#dc2626", "#16a34a"]
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative Return per Evaluation Episode (%)")
    ax.set_title("Distribution of Cumulative Return Across 30 Evaluation Episodes")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "episode_return_distribution.png"), dpi=150)
    plt.close(fig)
    print("saved plots/episode_return_distribution.png")


if __name__ == "__main__":
    plot_training_rewards()
    plot_test_portfolio_comparison()
    plot_metrics_bars()
    plot_episode_distribution()
    print("\nAll plots regenerated in plots/")
