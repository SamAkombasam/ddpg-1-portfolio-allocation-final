"""
dashboard.py
------------
Interactive dashboard for exploring the DDPG portfolio allocation
project's training and evaluation results.


The dashboard reads directly from the files already produced by
train.py, evaluate.py, and generate_plots.py thus everything under
results/. It does not retrain or re-evaluate anything itself, it is
a read-only, interactive viewer for inspecting results already on disk.

If results/ is empty or incomplete, the dashboard will say so and tell
you which script to run first.
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

st.set_page_config(
    page_title="Portfolio Allocation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEED_COLORS = {"0": "#2563eb", "1": "#dc2626", "2": "#16a34a"}
BASELINE_COLORS = {"Equal-Weight": "#6b7280", "Buy-and-Hold": "#111827"}


# ------------------------
# Data loading helpers
# ------------------------
def _load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_all():
    return {
        "training_log": _load_json("training_log.json"),
        "test_metrics": _load_csv("test_metrics.csv"),           # aggregated: mean/std across 30 episodes
        "eval_episode_log": _load_csv("eval_episode_log.csv"),   # raw: every strategy x every episode
        "eq_history": _load_json("eq_baseline_history.json"),    # one illustrative episode, for the line chart
        "bh_history": _load_json("bh_baseline_history.json"),
        "ddpg_histories": _load_json("ddpg_test_histories.json"),
    }


# ------------------
# Main app
# ------------------
def main():
    st.title("DDPG Portfolio Allocation Dashboard")
    st.caption(
        "Explore training reward curves and the 30-episode-per-seed evaluation "
        "results for the DDPG agent vs. equal-weight and buy-and-hold baselines "
        "(evaluated on identical episodes). Asset universe: AAPL, XOM, JNJ, JPM, "
        "PG, CAT (S&P 500, 2015\u20132025)."
    )

    data = load_all()

    required = ["training_log", "test_metrics", "eval_episode_log", "eq_history", "bh_history", "ddpg_histories"]
    missing = [k for k in required if data[k] is None]
    if missing:
        st.error(
            "Some result files are missing: "
            + ", ".join(missing)
            + ". Run `python reproduce.py` (or `train.py` then `evaluate.py`) before launching this dashboard."
        )
        st.stop()

    training_log = data["training_log"]
    metrics_df = data["test_metrics"]
    episode_log = data["eval_episode_log"]
    eq_hist = data["eq_history"]
    bh_hist = data["bh_history"]
    ddpg_hist = data["ddpg_histories"]

    all_seeds = sorted(training_log.keys())

    # DDPG-only rows of the aggregated table (excludes the two baseline rows
    # and the "mean-of-seed-means" cross-seed summary row)
    ddpg_seed_rows = metrics_df[metrics_df["strategy"].str.match(r"^DDPG \(seed=\d+\)$")]
    cross_seed_row = metrics_df[metrics_df["strategy"].str.startswith("DDPG, mean across seeds")]
    if cross_seed_row.empty:  # tolerate the exact label used by evaluate.py
        cross_seed_row = metrics_df[metrics_df["strategy"].str.contains("mean-of-seed-means", case=False)]

    # ----------------------
    # Sidebar controls
    # ----------------------
    st.sidebar.header("Controls")
    selected_seeds = st.sidebar.multiselect(
        "DDPG seeds to display",
        options=all_seeds,
        default=all_seeds,
    )
    show_baselines = st.sidebar.checkbox("Show baseline strategies", value=True)
    smoothing_window = st.sidebar.slider(
        "Training reward smoothing window (episodes)", min_value=1, max_value=30, value=10
    )
    initial_capital = 100_000.0

    if not selected_seeds:
        st.warning("Select at least one seed in the sidebar to see results.")
        st.stop()

    # ------------
    # Tabs
    # ------------
    tab_overview, tab_training, tab_eval, tab_dist, tab_metrics, tab_data = st.tabs(
        ["Overview", "Training", "Illustrative Episode", "Evaluation Distribution", "Metrics Table", "Asset Universe"]
    )

    # ---------------- Overview ----------------
    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)

        if not cross_seed_row.empty:
            mean_return = cross_seed_row["cumulative_return_%_mean"].values[0]
            std_return = cross_seed_row["cumulative_return_%_std"].values[0]
            col1.metric("DDPG mean return (30 eps x 3 seeds)", f"{mean_return:.2f}%", f"\u00b1{std_return:.2f} pp across seeds")
        else:
            col1.metric("DDPG mean return", "n/a")

        best_seed_row = ddpg_seed_rows.loc[ddpg_seed_rows["sharpe_ratio_mean"].idxmax()] if not ddpg_seed_rows.empty else None
        if best_seed_row is not None:
            col2.metric("Best-seed Sharpe (mean, 30 eps)", f"{best_seed_row['sharpe_ratio_mean']:.2f}", best_seed_row["strategy"])

        eq_row = metrics_df[metrics_df["strategy"].str.startswith("Equal-Weight")]
        if not eq_row.empty:
            col3.metric("Equal-Weight baseline (mean, 30 eps)", f"{eq_row['cumulative_return_%_mean'].values[0]:.2f}%")

        n_seeds = len(all_seeds)
        col4.metric("Training seeds", n_seeds, f"{len(training_log[all_seeds[0]]['episode_rewards'])} episodes each")

        st.markdown(
            "All strategies are evaluated on the **same 30 held-out evaluation episodes** "
            "(126-trading-day windows sampled from the test period), so the comparison above "
            "is made under identical conditions. The headline DDPG figure is the **mean across "
            "seeds**, not any single best-performing seed. Use the tabs above to explore training "
            "curves, one illustrative evaluation episode, the full 30-episode distribution, the "
            "metrics table, and the asset universe."
        )

    # ---------------- Training ----------------
    with tab_training:
        st.subheader("Training reward vs. episode")
        fig = go.Figure()
        for s in selected_seeds:
            rewards = training_log[s]["episode_rewards"]
            color = SEED_COLORS.get(s, "#888888")
            fig.add_trace(go.Scatter(
                y=rewards, mode="lines", name=f"Seed {s} (raw)",
                line=dict(color=color, width=1), opacity=0.3, showlegend=False,
                hovertemplate="Episode %{x}<br>Reward %{y:.3f}<extra></extra>",
            ))
            if len(rewards) >= smoothing_window:
                smoothed = pd.Series(rewards).rolling(smoothing_window).mean()
                fig.add_trace(go.Scatter(
                    y=smoothed, mode="lines", name=f"Seed {s} (smoothed)",
                    line=dict(color=color, width=2.5),
                    hovertemplate="Episode %{x}<br>Smoothed reward %{y:.3f}<extra></extra>",
                ))
        fig.update_layout(
            xaxis_title="Training Episode", yaxis_title="Episode Reward",
            hovermode="x unified", height=500, legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Mean training reward \u00b1 1 std across seeds")
        rewards_matrix = np.array([training_log[s]["episode_rewards"] for s in all_seeds])
        mean_r = rewards_matrix.mean(axis=0)
        std_r = rewards_matrix.std(axis=0)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=mean_r + std_r, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig2.add_trace(go.Scatter(
            y=mean_r - std_r, mode="lines", fill="tonexty", line=dict(width=0),
            fillcolor="rgba(37,99,235,0.2)", name="\u00b11 std", hoverinfo="skip",
        ))
        fig2.add_trace(go.Scatter(
            y=mean_r, mode="lines", line=dict(color="#2563eb", width=2.5), name="Mean reward",
            hovertemplate="Episode %{x}<br>Mean reward %{y:.3f}<extra></extra>",
        ))
        fig2.update_layout(xaxis_title="Training Episode", yaxis_title="Episode Reward", height=450)
        st.plotly_chart(fig2, width="stretch")

    # ---------------- Illustrative Episode ----------------
    with tab_eval:
        st.subheader("Portfolio value on one illustrative evaluation episode")
        st.caption(
            "This shows a single 126-day evaluation episode for visual intuition. "
            "The headline numeric results (Overview, Metrics Table, Evaluation "
            "Distribution tabs) are computed from all 30 evaluation episodes, not "
            "just this one — see the Evaluation Distribution tab for the full picture."
        )
        fig3 = go.Figure()
        if show_baselines:
            fig3.add_trace(go.Scatter(
                y=eq_hist["value"], mode="lines", name="Equal-Weight (daily rebal.)",
                line=dict(color=BASELINE_COLORS["Equal-Weight"], width=2, dash="dash"),
            ))
            fig3.add_trace(go.Scatter(
                y=bh_hist["value"], mode="lines", name="Buy-and-Hold",
                line=dict(color=BASELINE_COLORS["Buy-and-Hold"], width=2, dash="dot"),
            ))
        for s in selected_seeds:
            fig3.add_trace(go.Scatter(
                y=ddpg_hist[s]["value"], mode="lines", name=f"DDPG (seed={s})",
                line=dict(color=SEED_COLORS.get(s, "#888888"), width=1.8),
            ))
        if len(selected_seeds) > 1:
            vals = np.array([ddpg_hist[s]["value"] for s in selected_seeds])
            fig3.add_trace(go.Scatter(
                y=vals.mean(axis=0), mode="lines", name=f"DDPG mean ({len(selected_seeds)} seeds)",
                line=dict(color="black", width=3),
            ))
        fig3.add_hline(y=initial_capital, line=dict(color="gray", width=1, dash="dot"), opacity=0.5)
        fig3.update_layout(
            xaxis_title="Trading Day (one illustrative 126-day evaluation episode)", yaxis_title="Portfolio Value ($)",
            hovermode="x unified", height=550, legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig3, width="stretch")

        st.caption(
            "Initial capital was $100,000 for every strategy (dotted gray line). "
            "The DDPG agent was evaluated with exploration disabled (deterministic policy only)."
        )

    # ---------------- Evaluation Distribution ----------------
    with tab_dist:
        st.subheader("Distribution of cumulative return across all 30 evaluation episodes")

        strategy_options = ["Equal-Weight", "Buy-and-Hold"] + [f"DDPG (seed={s})" for s in selected_seeds]
        metric_choice = st.selectbox(
            "Metric",
            options=["cumulative_return_%", "sharpe_ratio", "max_drawdown_%", "annual_volatility_%", "total_transaction_cost_$"],
            format_func=lambda x: {
                "cumulative_return_%": "Cumulative Return (%)",
                "sharpe_ratio": "Sharpe Ratio",
                "max_drawdown_%": "Max Drawdown (%)",
                "annual_volatility_%": "Annualized Volatility (%)",
                "total_transaction_cost_$": "Transaction Cost ($)",
            }[x],
        )

        fig6 = go.Figure()
        colors = ["#6b7280", "#111827"] + [SEED_COLORS.get(s, "#888") for s in selected_seeds]
        for lbl, color in zip(strategy_options, colors):
            match_label = lbl if lbl in ("Equal-Weight", "Buy-and-Hold") else lbl
            vals = episode_log.loc[episode_log["label"] == match_label, metric_choice]
            fig6.add_trace(go.Box(y=vals, name=lbl, marker_color=color, boxmean=True))
        fig6.update_layout(height=550, yaxis_title=metric_choice)
        st.plotly_chart(fig6, width="stretch")

        st.caption(
            "Each box summarizes that strategy's outcome across the same 30 evaluation "
            "episodes (126-day windows sampled from the held-out test period). The "
            "diamond/line marks the mean; box edges are the interquartile range."
        )

    # ---------------- Metrics Table ----------------
    with tab_metrics:
        st.subheader("Evaluation metrics: mean \u00b1 std across 30 episodes")

        display_df = metrics_df.copy()
        mean_cols = [c for c in display_df.columns if c.endswith("_mean")]
        std_cols = [c.replace("_mean", "_std") for c in mean_cols]
        base_names = [c.replace("_mean", "") for c in mean_cols]

        pretty = pd.DataFrame({"Strategy": display_df["strategy"], "N episodes": display_df["n_episodes"]})
        for base, mcol, scol in zip(base_names, mean_cols, std_cols):
            pretty[base] = display_df.apply(lambda r: f"{r[mcol]:.2f} \u00b1 {r[scol]:.2f}", axis=1)

        st.dataframe(pretty, width="stretch", hide_index=True)
        st.caption(
            "For DDPG rows, \u00b1 is the standard deviation across the 30 evaluation episodes for "
            "that seed. For the 'DDPG, mean across seeds' row, \u00b1 is the standard deviation "
            "ACROSS THE 3 SEEDS of their own 30-episode means \u2014 a different quantity, kept "
            "distinct as required by the evaluation protocol."
        )

        col_a, col_b = st.columns(2)
        plot_df = metrics_df[~metrics_df["strategy"].str.contains("mean across seeds|mean-of-seed-means", case=False)]
        bar_colors = ["#6b7280", "#111827"] + [SEED_COLORS.get(s, "#888") for s in all_seeds]
        with col_a:
            fig4 = go.Figure(go.Bar(
                x=plot_df["strategy"], y=plot_df["cumulative_return_%_mean"],
                error_y=dict(type="data", array=plot_df["cumulative_return_%_std"]),
                marker_color=bar_colors[:len(plot_df)],
            ))
            fig4.update_layout(title="Mean Cumulative Return (%) \u00b1 1 std (30 episodes)", height=420)
            st.plotly_chart(fig4, width="stretch")
        with col_b:
            fig5 = go.Figure(go.Bar(
                x=plot_df["strategy"], y=plot_df["sharpe_ratio_mean"],
                error_y=dict(type="data", array=plot_df["sharpe_ratio_std"]),
                marker_color=bar_colors[:len(plot_df)],
            ))
            fig5.update_layout(title="Mean Sharpe Ratio \u00b1 1 std (30 episodes)", height=420)
            st.plotly_chart(fig5, width="stretch")

        csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download aggregated metrics as CSV", csv_bytes, "test_metrics.csv", "text/csv")
        raw_csv_bytes = episode_log.to_csv(index=False).encode("utf-8")
        st.download_button("Download raw per-episode log as CSV", raw_csv_bytes, "eval_episode_log.csv", "text/csv")

    # ---------------- Asset Universe ----------------
    with tab_data:
        st.subheader("Selected asset universe")
        asset_df = pd.DataFrame({
            "Ticker": ["AAPL", "XOM", "JNJ", "JPM", "PG", "CAT"],
            "Sector": ["Technology", "Energy", "Healthcare", "Financials", "Consumer Staples", "Industrials"],
            "10-yr Annualized Return (%)": [27.5, 8.9, 10.4, 20.3, 9.2, 27.5],
            "10-yr Annualized Volatility (%)": [29.1, 27.9, 18.4, 27.6, 18.7, 30.4],
        })
        st.dataframe(asset_df, width="stretch", hide_index=True)
        st.caption(
            "Source: Kaggle \u2013 \u201cS&P 500 Stocks \u2013 Daily Historical Data (10 Years)\u201d. "
            "2,515 trading days common to all six tickers (Dec 2015\u2013Dec 2025), "
            "split 80/20 chronologically into training and held-out test sets."
        )


if __name__ == "__main__":
    main()
