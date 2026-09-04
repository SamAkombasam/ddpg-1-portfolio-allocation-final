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

Visual design notes
-------------------
Colour is assigned by JOB, not by row order:
  - The three DDPG seeds are the categorical series under study and take
    categorical slots 1-3 (blue / orange / aqua). Those three slots are
    validated colourblind-safe against this surface on all pairs.
  - The two baselines are REFERENCE, not competing identities, so they are
    drawn in neutral ink and separated by dash pattern. A filter that hides
    a seed never repaints the survivors: colour follows the entity.
Aqua sits below 3:1 contrast on this surface, so every chart ships a legend
and the Data tab carries the full table view (values are never colour-gated).
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# --------------------------------------------------------------------------
# Palette. Validated with the data-viz palette validator against surface
# #fcfcfb: lightness band PASS, chroma floor PASS, all-pairs CVD dE 9.2 PASS,
# normal-vision dE 24.0 PASS, contrast WARN on aqua (relief = legend + table).
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"   # chart surface
PAGE = "#f9f9f7"      # page plane
INK = "#0b0b0b"       # primary ink
INK_2 = "#52514e"     # secondary ink
MUTED = "#898781"     # axis / labels
GRID = "#e1e0d9"      # hairline gridline
AXIS = "#c3c2b7"      # baseline / axis rule
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]        # categorical slots 1-3
BASELINE_INK = {"Equal-Weight": INK_2, "Buy-and-Hold": MUTED}
BASELINE_DASH = {"Equal-Weight": "dash", "Buy-and-Hold": "dot"}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

CROSS_SEED_LABEL = "DDPG mean-of-seed-means (n=3 seeds)"

METRICS = {
    "cumulative_return_%": ("Cumulative return", "%", True),
    "sharpe_ratio": ("Sharpe ratio", "", True),
    "max_drawdown_%": ("Max drawdown", "%", False),
    "annual_volatility_%": ("Annualised volatility", "%", False),
    "total_transaction_cost_$": ("Transaction cost", "$", False),
}

st.set_page_config(
    page_title="DDPG Portfolio Allocation",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# Styling
# ==========================================================================
def inject_css():
    st.markdown(
        f"""
        <style>
          .stApp {{ background: {PAGE}; }}
          .block-container {{ padding: 2.2rem 2.6rem 4rem; max-width: 1500px; }}
          #MainMenu, footer, header {{ visibility: hidden; }}

          html, body, [class*="css"] {{ font-family: {FONT}; }}

          /* ---------- masthead ---------- */
          .mast {{ border-bottom: 1px solid {AXIS}; padding-bottom: 1.1rem; margin-bottom: 1.6rem; }}
          .mast .eyebrow {{
            font-size: .70rem; font-weight: 600; letter-spacing: .13em;
            text-transform: uppercase; color: {MUTED}; margin-bottom: .5rem;
          }}
          .mast h1 {{
            font-size: 1.72rem; font-weight: 600; letter-spacing: -.018em;
            color: {INK}; margin: 0 0 .4rem; line-height: 1.2;
          }}
          .mast .sub {{ font-size: .92rem; color: {INK_2}; margin: 0; max-width: 74ch; line-height: 1.55; }}
          .meta {{ display: flex; flex-wrap: wrap; gap: 0 2.2rem; margin-top: 1rem; }}
          .meta div {{ font-size: .78rem; color: {MUTED}; }}
          .meta b {{ color: {INK_2}; font-weight: 600; font-variant-numeric: tabular-nums; }}

          /* ---------- hero + stat tiles ---------- */
          .tiles {{ display: flex; flex-wrap: wrap; gap: 1px; background: {AXIS};
                    border: 1px solid {AXIS}; margin: .2rem 0 1.6rem; }}
          .tile {{ background: {SURFACE}; padding: 1.05rem 1.3rem; flex: 1 1 170px; }}
          .tile .lab {{ font-size: .74rem; color: {MUTED}; margin-bottom: .42rem; line-height: 1.35; }}
          .tile .val {{ font-size: 1.58rem; font-weight: 600; color: {INK}; line-height: 1.1;
                        letter-spacing: -.02em; }}
          .tile .sub {{ font-size: .74rem; color: {MUTED}; margin-top: .34rem; }}
          .tile.hero {{ flex: 1 1 240px; }}
          .tile.hero .val {{ font-size: 3.1rem; font-weight: 600; letter-spacing: -.035em; }}

          /* ---------- finding callout ---------- */
          .finding {{
            border-left: 2px solid {INK}; background: {SURFACE};
            padding: .95rem 1.2rem; margin: 0 0 1.5rem;
          }}
          .finding .h {{ font-size: .70rem; font-weight: 600; letter-spacing: .13em;
                         text-transform: uppercase; color: {MUTED}; margin-bottom: .45rem; }}
          .finding p {{ margin: 0; font-size: .89rem; color: {INK_2}; line-height: 1.6; max-width: 92ch; }}
          .finding b {{ color: {INK}; font-weight: 600; }}

          /* ---------- section headings ---------- */
          .sec {{ margin: 1.7rem 0 .3rem; }}
          .sec h3 {{ font-size: .97rem; font-weight: 600; color: {INK}; margin: 0 0 .25rem;
                     letter-spacing: -.008em; }}
          .sec p {{ font-size: .8rem; color: {MUTED}; margin: 0; line-height: 1.55; max-width: 88ch; }}

          .note {{ font-size: .76rem; color: {MUTED}; line-height: 1.6; margin-top: .5rem;
                   max-width: 92ch; }}

          /* ---------- tabs ---------- */
          .stTabs [data-baseweb="tab-list"] {{ gap: 1.9rem; border-bottom: 1px solid {AXIS}; }}
          .stTabs [data-baseweb="tab"] {{
            height: 2.5rem; padding: 0; background: transparent;
            font-size: .855rem; font-weight: 500; color: {MUTED};
          }}
          .stTabs [aria-selected="true"] {{ color: {INK} !important; font-weight: 600; }}
          .stTabs [data-baseweb="tab-highlight"] {{ background-color: {INK}; height: 2px; }}
          .stTabs [data-baseweb="tab-border"] {{ display: none; }}

          /* ---------- sidebar ---------- */
          section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {AXIS}; }}
          section[data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}
          .side-h {{ font-size: .70rem; font-weight: 600; letter-spacing: .13em;
                     text-transform: uppercase; color: {MUTED}; margin: .2rem 0 .7rem; }}

          [data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
          div[data-testid="stMetric"] {{ display: none; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig, height=430, ylab="", xlab="", legend=True, zeroline=False, yfmt=None):
    """One chart chrome for every figure: hairline solid grid, recessive axes.

    Margins are generous and both axes carry automargin: tick labels live in the
    margin, so a tight margin silently clips them and the chart loses its scale.
    """
    fig.update_layout(
        height=height,
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=70, r=26, t=48 if legend else 20, b=56),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0,
            font=dict(size=11.5, color=INK_2), bgcolor="rgba(0,0,0,0)",
        ),
    )
    axis = dict(
        showgrid=True, gridcolor=GRID, gridwidth=1, griddash="solid",
        zeroline=zeroline, zerolinecolor=AXIS, zerolinewidth=1,
        showline=True, linecolor=AXIS, linewidth=1,
        ticks="outside", ticklen=4, tickcolor=AXIS,
        tickfont=dict(size=11, color=MUTED), automargin=True,
    )
    fig.update_xaxes(**axis, title_text=xlab,
                     title_font=dict(size=11.5, color=MUTED), title_standoff=14)
    fig.update_yaxes(**axis, title_text=ylab, tickformat=yfmt,
                     title_font=dict(size=11.5, color=MUTED), title_standoff=14)
    return fig


PLOTLY_CFG = {"displayModeBar": False, "responsive": True}


# ==========================================================================
# Data
# ==========================================================================
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


@st.cache_data
def load_universe():
    """Per-ticker return/vol computed from the committed CSVs, so the asset
    table can never drift out of date the way hardcoded figures do."""
    try:
        from data_loader_real import load_real_price_data, SECTORS, TICKERS
        px = load_real_price_data()
        rows = []
        for t in TICKERS:
            p = px[t].values
            r = np.diff(np.log(p))
            rows.append({
                "Ticker": t,
                "Sector": SECTORS[t],
                "Annualised return %": ((p[-1] / p[0]) ** (252 / len(r)) - 1) * 100,
                "Annualised volatility %": float(np.std(r, ddof=1) * np.sqrt(252) * 100),
            })
        return pd.DataFrame(rows)
    except Exception:
        return None


def seed_color(seed, all_seeds):
    """Colour follows the entity: seed -> fixed slot, never its filtered rank."""
    return SERIES[all_seeds.index(seed) % len(SERIES)]


def label_color(label, all_seeds):
    if label in BASELINE_INK:
        return BASELINE_INK[label]
    for s in all_seeds:
        if label == f"DDPG (seed={s})":
            return seed_color(s, all_seeds)
    return MUTED


def display_name(label):
    """One spelling of every strategy name across axes, legends and tooltips."""
    if label.startswith("DDPG (seed="):
        return "DDPG seed " + label.split("=")[1].rstrip(")")
    return {"Equal-Weight": "Equal-weight", "Buy-and-Hold": "Buy-and-hold"}.get(label, label)


def fmt(value, unit):
    if unit == "$":
        return f"${value:,.0f}"
    if unit == "%":
        return f"{value:.2f}%"
    return f"{value:.2f}"


# ==========================================================================
# App
# ==========================================================================
def main():
    inject_css()
    data = load_all()

    required = ["training_log", "test_metrics", "eval_episode_log", "eq_history", "bh_history", "ddpg_histories"]
    missing = [k for k in required if data[k] is None]
    if missing:
        st.markdown('<div class="mast"><div class="eyebrow">DSCD 614 &middot; Group 10</div>'
                    '<h1>Dynamic Portfolio Allocation with DDPG</h1></div>', unsafe_allow_html=True)
        st.error(
            "Missing result files: " + ", ".join(missing)
            + ". Run `python reproduce.py` (or `train.py` then `evaluate.py`) before launching this dashboard."
        )
        st.stop()

    training_log = data["training_log"]
    metrics_df = data["test_metrics"]
    episode_log = data["eval_episode_log"]
    eq_hist, bh_hist, ddpg_hist = data["eq_history"], data["bh_history"], data["ddpg_histories"]

    all_seeds = sorted(training_log.keys())
    n_train_eps = len(training_log[all_seeds[0]]["episode_rewards"])
    n_eval_eps = int((episode_log["label"] == "Equal-Weight").sum())

    cross = metrics_df[metrics_df["strategy"].str.contains("mean-of-seed-means|mean across seeds", case=False)]
    ddpg_rows = metrics_df[metrics_df["strategy"].str.match(r"^DDPG \(seed=\d+\)$")]
    eq_row = metrics_df[metrics_df["strategy"].str.startswith("Equal-Weight")]
    bh_row = metrics_df[metrics_df["strategy"].str.startswith("Buy-and-Hold")]

    # ---------------------------------------------------------------- header
    st.markdown(
        f"""
        <div class="mast">
          <div class="eyebrow">DSCD 614 Reinforcement Learning &middot; Group 10 &middot; Option DDPG-1</div>
          <h1>Dynamic Portfolio Allocation with DDPG</h1>
          <p class="sub">A deterministic continuous-control policy allocating capital across six
          sector-diverse S&amp;P&nbsp;500 equities, evaluated against equal-weight and buy-and-hold
          baselines on identical held-out episodes.</p>
          <div class="meta">
            <div>Universe <b>AAPL &middot; XOM &middot; JNJ &middot; JPM &middot; PG &middot; CAT</b></div>
            <div>Period <b>Dec 2015 &ndash; Dec 2025</b></div>
            <div>Split <b>80 / 20 chronological</b></div>
            <div>Training <b>{len(all_seeds)} seeds &times; {n_train_eps} episodes</b></div>
            <div>Evaluation <b>{n_eval_eps} episodes &times; {len(all_seeds)} seeds</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------- sidebar
    st.sidebar.markdown('<div class="side-h">Series</div>', unsafe_allow_html=True)
    selected_seeds = st.sidebar.multiselect(
        "DDPG seeds", options=all_seeds, default=all_seeds,
        help="Hiding a seed never recolours the others.",
    )
    show_baselines = st.sidebar.checkbox("Show baselines", value=True)

    st.sidebar.markdown('<div class="side-h">Comparison metric</div>', unsafe_allow_html=True)
    metric_key = st.sidebar.selectbox(
        "Metric", options=list(METRICS), format_func=lambda k: METRICS[k][0],
        label_visibility="collapsed",
    )
    metric_name, metric_unit, _ = METRICS[metric_key]

    st.sidebar.markdown('<div class="side-h">Training curve</div>', unsafe_allow_html=True)
    smoothing = st.sidebar.slider("Smoothing window (episodes)", 1, 30, 10)

    if not selected_seeds:
        st.warning("Select at least one seed in the sidebar.")
        st.stop()

    # ------------------------------------------------------------------ tabs
    t_overview, t_training, t_eval, t_episode, t_data = st.tabs(
        ["Overview", "Training", "Evaluation", "Episode detail", "Data"]
    )

    # ============================================================ OVERVIEW
    with t_overview:
        ddpg_ret = cross["cumulative_return_%_mean"].values[0]
        ddpg_std = cross["cumulative_return_%_std"].values[0]
        ddpg_sharpe = cross["sharpe_ratio_mean"].values[0]
        eq_ret = eq_row["cumulative_return_%_mean"].values[0]
        eq_sharpe = eq_row["sharpe_ratio_mean"].values[0]
        bh_ret = bh_row["cumulative_return_%_mean"].values[0]
        gap = ddpg_ret - eq_ret
        best = ddpg_rows.loc[ddpg_rows["sharpe_ratio_mean"].idxmax()]

        st.markdown(
            f"""
            <div class="tiles">
              <div class="tile hero">
                <div class="lab">DDPG mean cumulative return</div>
                <div class="val">{ddpg_ret:.2f}%</div>
                <div class="sub">&plusmn;{ddpg_std:.2f} pp across {len(all_seeds)} seeds</div>
              </div>
              <div class="tile">
                <div class="lab">DDPG Sharpe ratio</div>
                <div class="val">{ddpg_sharpe:.2f}</div>
                <div class="sub">vs {eq_sharpe:.2f} equal-weight</div>
              </div>
              <div class="tile">
                <div class="lab">Equal-weight baseline</div>
                <div class="val">{eq_ret:.2f}%</div>
                <div class="sub">buy-and-hold {bh_ret:.2f}%</div>
              </div>
              <div class="tile">
                <div class="lab">Gap vs equal-weight</div>
                <div class="val">{gap:+.2f} pp</div>
                <div class="sub">seed spread &plusmn;{ddpg_std:.2f} pp</div>
              </div>
              <div class="tile">
                <div class="lab">Strongest seed by Sharpe</div>
                <div class="val">{best['sharpe_ratio_mean']:.2f}</div>
                <div class="sub">{best['strategy']}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        inconclusive = abs(gap) < ddpg_std
        verdict = (
            f"The {abs(gap):.2f} pp gap between DDPG and the equal-weight baseline is "
            f"<b>smaller than DDPG's own {ddpg_std:.2f} pp cross-seed standard deviation</b>, so with "
            f"{len(all_seeds)} seeds it cannot be separated from seed-to-seed training noise."
            if inconclusive else
            f"The {abs(gap):.2f} pp gap exceeds DDPG's {ddpg_std:.2f} pp cross-seed standard deviation, "
            f"though with only {len(all_seeds)} seeds this is suggestive rather than conclusive."
        )
        st.markdown(
            f'<div class="finding"><div class="h">Headline finding</div><p>{verdict} '
            f'The headline figure is the mean across seeds, never a single best-performing run.</p></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="sec"><h3>{metric_name} by strategy</h3>'
            f'<p>Mean across the {n_eval_eps} held-out evaluation episodes; whiskers are '
            f'&plusmn;1 standard deviation across those episodes. Every strategy is scored on the '
            f'same episodes.</p></div>',
            unsafe_allow_html=True,
        )

        bar_labels = (["Equal-Weight", "Buy-and-Hold"] if show_baselines else []) + \
                     [f"DDPG (seed={s})" for s in selected_seeds]
        strat_rows = {"Equal-Weight": eq_row, "Buy-and-Hold": bh_row}
        names, vals, errs, cols = [], [], [], []
        for lbl in bar_labels:
            row = strat_rows[lbl].iloc[0] if lbl in strat_rows else \
                metrics_df[metrics_df["strategy"] == lbl].iloc[0]
            names.append(display_name(lbl))
            vals.append(row[f"{metric_key}_mean"])
            errs.append(row[f"{metric_key}_std"])
            cols.append(label_color(lbl, all_seeds))

        fig = go.Figure(go.Bar(
            x=names, y=vals, marker=dict(color=cols, cornerradius=4),
            # thin mark: a fraction of the category band, so the bar reads as a
            # measured stroke rather than a saturated block filling its slot
            width=0.10,
            error_y=dict(type="data", array=errs, color=AXIS, thickness=1, width=5),
            hovertemplate="%{x}<br>" + metric_name + ": %{y:.2f}<extra></extra>",
        ))
        style_fig(fig, height=380, ylab=f"{metric_name} ({metric_unit})" if metric_unit else metric_name,
                  legend=False, zeroline=True)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)

    # ============================================================ TRAINING
    with t_training:
        st.markdown(
            f'<div class="sec"><h3>Episode reward during training</h3>'
            f'<p>Cumulative log-return reward per {n_train_eps}-episode run. The pale trace is raw '
            f'per-episode reward; the solid line is a {smoothing}-episode moving average.</p></div>',
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        for s in selected_seeds:
            r = training_log[s]["episode_rewards"]
            c = seed_color(s, all_seeds)
            fig.add_trace(go.Scatter(
                y=r, mode="lines", line=dict(color=c, width=1), opacity=0.22,
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                y=pd.Series(r).rolling(smoothing).mean(), mode="lines",
                name=f"Seed {s}", line=dict(color=c, width=2),
                hovertemplate="Episode %{x}<br>Reward %{y:.3f}<extra></extra>",
            ))
        style_fig(fig, height=430, xlab="Training episode", ylab="Episode reward", zeroline=True)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)

        st.markdown(
            '<div class="sec"><h3>Mean reward across seeds</h3>'
            '<p>Solid line is the across-seed mean; the band is &plusmn;1 standard deviation across '
            'seeds at each episode.</p></div>',
            unsafe_allow_html=True,
        )
        mat = np.array([training_log[s]["episode_rewards"] for s in all_seeds])
        mean_r, std_r = mat.mean(axis=0), mat.std(axis=0, ddof=1)
        x = list(range(len(mean_r)))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=mean_r + std_r, mode="lines", line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        fig2.add_trace(go.Scatter(x=x, y=mean_r - std_r, mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor="rgba(42,120,214,0.10)",
                                  name="&plusmn;1 std across seeds", hoverinfo="skip"))
        fig2.add_trace(go.Scatter(x=x, y=mean_r, mode="lines", name="Mean reward",
                                  line=dict(color=SERIES[0], width=2),
                                  hovertemplate="Episode %{x}<br>Mean %{y:.3f}<extra></extra>"))
        style_fig(fig2, height=380, xlab="Training episode", ylab="Episode reward", zeroline=True)
        st.plotly_chart(fig2, width="stretch", config=PLOTLY_CFG)

        first = {s: float(np.mean(training_log[s]["episode_rewards"][:20])) for s in all_seeds}
        last = {s: float(np.mean(training_log[s]["episode_rewards"][-20:])) for s in all_seeds}
        moved = ", ".join(f"seed {s} {first[s]:+.3f} &rarr; {last[s]:+.3f}" for s in all_seeds)
        st.markdown(
            f'<div class="note">Mean reward over the first 20 versus the last 20 episodes: {moved}. '
            f'Each episode samples a different historical window, so the attainable reward ceiling '
            f'varies episode to episode independently of policy quality.</div>',
            unsafe_allow_html=True,
        )

    # ========================================================== EVALUATION
    with t_eval:
        st.markdown(
            f'<div class="sec"><h3>{metric_name} across all {n_eval_eps} evaluation episodes</h3>'
            f'<p>Each box spans the interquartile range with the median rule and mean marker; '
            f'whiskers reach 1.5&times;IQR. Every strategy is scored on the same {n_eval_eps} '
            f'held-out 126-day windows.</p></div>',
            unsafe_allow_html=True,
        )
        labels = (["Equal-Weight", "Buy-and-Hold"] if show_baselines else []) + \
                 [f"DDPG (seed={s})" for s in selected_seeds]
        fig = go.Figure()
        for lbl in labels:
            vals = episode_log.loc[episode_log["label"] == lbl, metric_key]
            c = label_color(lbl, all_seeds)
            fig.add_trace(go.Box(
                y=vals, name=display_name(lbl),
                marker_color=c, line=dict(width=1.4), fillcolor="rgba(0,0,0,0)",
                boxmean=True, boxpoints="outliers", width=0.34,
                marker=dict(size=5, opacity=0.55),
                hovertemplate="%{x}<br>" + metric_name + ": %{y:.2f}<extra></extra>",
            ))
        style_fig(fig, height=470, ylab=f"{metric_name} ({metric_unit})" if metric_unit else metric_name,
                  legend=False, zeroline=True)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)

        st.markdown(
            '<div class="note">Because the test period is a single historical price path rather than a '
            're-samplable simulator, the sampled windows overlap and are not fully independent draws. '
            'The distribution shows dispersion across market conditions, not a clean confidence interval.</div>',
            unsafe_allow_html=True,
        )

    # ======================================================= EPISODE DETAIL
    with t_episode:
        st.markdown(
            '<div class="sec"><h3>Portfolio value on one illustrative episode</h3>'
            '<p>A single 126-day window, shown for intuition only. Headline results use all '
            'evaluation episodes &mdash; see the Evaluation tab.</p></div>',
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        if show_baselines:
            for lbl, hist in (("Equal-Weight", eq_hist), ("Buy-and-Hold", bh_hist)):
                fig.add_trace(go.Scatter(
                    y=hist["value"], mode="lines",
                    name=display_name(lbl),
                    line=dict(color=BASELINE_INK[lbl], width=1.6, dash=BASELINE_DASH[lbl]),
                    hovertemplate="Day %{x}<br>$%{y:,.0f}<extra></extra>",
                ))
        for s in selected_seeds:
            fig.add_trace(go.Scatter(
                y=ddpg_hist[s]["value"], mode="lines", name=f"DDPG seed {s}",
                line=dict(color=seed_color(s, all_seeds), width=2),
                hovertemplate="Day %{x}<br>$%{y:,.0f}<extra></extra>",
            ))
        fig.add_hline(y=100_000, line=dict(color=AXIS, width=1))
        style_fig(fig, height=500, xlab="Trading day within episode",
                  ylab="Portfolio value", yfmt="$,.0f")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)

        st.markdown(
            '<div class="note">All strategies start from $100,000 (the horizontal rule). The DDPG '
            'policy runs deterministically here &mdash; exploration noise is disabled at evaluation. '
            'A single episode can rank the seeds quite differently from the full-sample aggregate, '
            'which is precisely why the protocol averages over many episodes.</div>',
            unsafe_allow_html=True,
        )

    # ================================================================ DATA
    with t_data:
        st.markdown(
            '<div class="sec"><h3>Aggregated evaluation metrics</h3>'
            '<p>For each DDPG seed and baseline, &plusmn; is the standard deviation across evaluation '
            'episodes. For the cross-seed row it is the standard deviation across the three seeds of '
            'their own episode means &mdash; a different quantity, deliberately kept distinct.</p></div>',
            unsafe_allow_html=True,
        )
        pretty = pd.DataFrame({"Strategy": metrics_df["strategy"], "Episodes": metrics_df["n_episodes"]})
        for key, (name, unit, _) in METRICS.items():
            m, s = metrics_df[f"{key}_mean"], metrics_df[f"{key}_std"]
            pretty[name] = [f"{fmt(a, unit)} ± {fmt(b, unit).lstrip('$')}" for a, b in zip(m, s)]
        st.dataframe(pretty, width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        c1.download_button("Download aggregated metrics (CSV)",
                           metrics_df.to_csv(index=False).encode("utf-8"),
                           "test_metrics.csv", "text/csv", width="stretch")
        c2.download_button("Download per-episode log (CSV)",
                           episode_log.to_csv(index=False).encode("utf-8"),
                           "eval_episode_log.csv", "text/csv", width="stretch")

        st.markdown(
            '<div class="sec"><h3>Per-episode evaluation log</h3>'
            '<p>One row per strategy per episode &mdash; the raw record behind every aggregate above.</p></div>',
            unsafe_allow_html=True,
        )
        st.dataframe(episode_log, width="stretch", hide_index=True, height=300)

        universe = load_universe()
        if universe is not None:
            st.markdown(
                '<div class="sec"><h3>Asset universe</h3>'
                '<p>Computed directly from the committed price CSVs over the full sample.</p></div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                universe.style.format({"Annualised return %": "{:.2f}",
                                       "Annualised volatility %": "{:.2f}"}),
                width="stretch", hide_index=True,
            )


if __name__ == "__main__":
    main()
