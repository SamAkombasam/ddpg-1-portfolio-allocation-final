"""
evaluate.py
-----------
Evaluation harness for the DDPG portfolio-allocation agent.


  - The trained policy for each of the 3 training seeds is evaluated on
    30 held-out evaluation episodes, each a ~6-month (126 trading day)
    window sampled from the held-out TEST period (never seen during
    training).
  - The two baseline strategies (equal-weight daily rebalance,
    buy-and-hold) are evaluated on the SAME 30 episodes (same start day,
    same length, same underlying price data, same metric code) as each
    trained policy, so the comparison is made under identical conditions.
  - Exploration noise is disabled for the DDPG agent at evaluation time
    (the actor network's deterministic output is used directly).

"""
import sys, os, json
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import pandas as pd

from env.portfolio_env import PortfolioEnv
from agent.ddpg_agent import Actor, DEVICE
from data_loader_real import load_real_price_data, train_test_split
from utils import summarize_metrics, equal_weight_baseline, buy_and_hold_baseline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ENV_KWARGS = dict(window=10, initial_cash=100_000.0, transaction_cost=0.001, turnover_penalty=0.5)

N_EVAL_EPISODES = 30
EVAL_EPISODE_LENGTH = 126           # ~6 trading months per evaluation episode
EVAL_SEED_OFFSET = 10_000           # eval-episode seeds are disjoint from training seeds {0,1,2}

EVAL_KWARGS = dict(
    ENV_KWARGS,
    episode_length=EVAL_EPISODE_LENGTH,
    random_start=True,              # start day is drawn from the RNG seeded per evaluation episode below
)


def run_agent_on_episode(actor, price_test, action_dim, state_dim, episode_seed):
    """Runs the (deterministic, noise-free) actor on ONE evaluation episode."""
    env = PortfolioEnv(price_test, **EVAL_KWARGS)
    obs, _ = env.reset(seed=episode_seed)
    done = trunc = False
    while not (done or trunc):
        with torch.no_grad():
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = actor(state_t).cpu().numpy().flatten()  # no exploration noise at test time
        obs, r, done, trunc, info = env.step(action)
    return env.history


def load_actor(actor_path, state_dim, action_dim):
    actor = Actor(state_dim, action_dim).to(DEVICE)
    actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    actor.eval()
    return actor


def main(seeds=(0, 1, 2), n_eval_episodes=N_EVAL_EPISODES):
    price_df = load_real_price_data()
    price_train, price_test = train_test_split(price_df, test_fraction=0.2)

    probe_env = PortfolioEnv(price_train, **ENV_KWARGS)
    state_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]

    episode_seeds = [EVAL_SEED_OFFSET + i for i in range(n_eval_episodes)]

    # ------------------------------------------------------------
    # Per-episode raw metric log: every strategy x every episode.
    # This is the raw log committed to the repository (results/
    # eval_episode_log.csv)
    # ------------------------------------------------------------
    raw_rows = []

    # --- baselines, evaluated on the SAME 30 episodes as every seed ---
    for ep_idx, ep_seed in enumerate(episode_seeds):
        eq_hist = equal_weight_baseline(PortfolioEnv, price_test, reset_seed=ep_seed, **EVAL_KWARGS)
        bh_hist = buy_and_hold_baseline(PortfolioEnv, price_test, reset_seed=ep_seed, **EVAL_KWARGS)
        cost_rate = ENV_KWARGS["transaction_cost"]

        m_eq = summarize_metrics(eq_hist["value"], eq_hist["turnover"], cost_rate, "Equal-Weight")
        m_bh = summarize_metrics(bh_hist["value"], bh_hist["turnover"], cost_rate, "Buy-and-Hold")
        for m in (m_eq, m_bh):
            m["episode_index"] = ep_idx
            m["episode_seed"] = ep_seed
            m["seed"] = None
            raw_rows.append(m)

    # --- DDPG, each training seed, evaluated on the SAME 30 episodes ---
    for seed in seeds:
        actor_path = os.path.join(RESULTS_DIR, f"actor_seed{seed}.pt")
        actor = load_actor(actor_path, state_dim, action_dim)
        for ep_idx, ep_seed in enumerate(episode_seeds):
            hist = run_agent_on_episode(actor, price_test, action_dim, state_dim, ep_seed)
            cost_rate = ENV_KWARGS["transaction_cost"]
            m = summarize_metrics(hist["value"], hist["turnover"], cost_rate, f"DDPG (seed={seed})")
            m["episode_index"] = ep_idx
            m["episode_seed"] = ep_seed
            m["seed"] = seed
            raw_rows.append(m)

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(os.path.join(RESULTS_DIR, "eval_episode_log.csv"), index=False)

    # ------------------------------------------------------------
    # Aggregate: mean +/- std ACROSS THE 30 EPISODES, per strategy
    # (per DDPG seed, and for each baseline).
    # ------------------------------------------------------------
    metric_cols = ["cumulative_return_%", "sharpe_ratio", "max_drawdown_%",
                    "annual_volatility_%", "total_transaction_cost_$", "final_value"]

    def agg_for(label_mask, label_name):
        sub = raw_df[label_mask]
        row = {"strategy": label_name, "n_episodes": len(sub)}
        for c in metric_cols:
            row[f"{c}_mean"] = sub[c].mean()
            row[f"{c}_std"] = sub[c].std()
        return row

    summary_rows = []
    summary_rows.append(agg_for(raw_df["label"] == "Equal-Weight", "Equal-Weight (daily rebal.)"))
    summary_rows.append(agg_for(raw_df["label"] == "Buy-and-Hold", "Buy-and-Hold"))
    for seed in seeds:
        summary_rows.append(agg_for(raw_df["label"] == f"DDPG (seed={seed})", f"DDPG (seed={seed})"))

    summary_df = pd.DataFrame(summary_rows)

    # -----------------------
    # Aggregate across SEEDS

    ddpg_seed_means = summary_df[summary_df["strategy"].str.startswith("DDPG")]
    cross_seed_row = {"strategy": "DDPG mean-of-seed-means (n=3 seeds)", "n_episodes": f"{n_eval_episodes} x 3 seeds"}
    for c in metric_cols:
        seed_level_means = ddpg_seed_means[f"{c}_mean"].values
        cross_seed_row[f"{c}_mean"] = float(np.mean(seed_level_means))
        cross_seed_row[f"{c}_std"] = float(np.std(seed_level_means))  # std ACROSS SEEDS, n=3
    summary_df = pd.concat([summary_df, pd.DataFrame([cross_seed_row])], ignore_index=True)

    summary_df.to_csv(os.path.join(RESULTS_DIR, "test_metrics.csv"), index=False)

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------
    print("\n================ EVALUATION RESULTS ================")
    print(f"{n_eval_episodes} evaluation episodes per strategy, each a {EVAL_EPISODE_LENGTH}-day "
          f"window sampled from the held-out test period. Identical episodes used for every strategy.\n")
    display_cols = ["strategy", "n_episodes", "cumulative_return_%_mean", "cumulative_return_%_std",
                     "sharpe_ratio_mean", "sharpe_ratio_std"]
    print(summary_df[display_cols].to_string(index=False))

    # explicit "does the difference exceed seed variation" check, printed for the record
    eq_mean = summary_df.loc[summary_df.strategy.str.startswith("Equal-Weight"), "cumulative_return_%_mean"].values[0]
    ddpg_cross = summary_df.loc[summary_df.strategy.str.startswith("DDPG mean-of-seed-means"), :].iloc[0]
    diff = ddpg_cross["cumulative_return_%_mean"] - eq_mean
    seed_std = ddpg_cross["cumulative_return_%_std"]
    print(f"\nDDPG mean return ({ddpg_cross['cumulative_return_%_mean']:.2f}%) minus "
          f"Equal-Weight mean return ({eq_mean:.2f}%) = {diff:.2f} percentage points.")
    print(f"Cross-seed standard deviation of DDPG's mean return = {seed_std:.2f} percentage points.")
    if abs(diff) < seed_std:
        print("=> The observed difference is SMALLER than the cross-seed standard deviation: "
              "with only 3 seeds, this difference cannot be distinguished from seed-to-seed noise.")
    else:
        print("=> The observed difference EXCEEDS the cross-seed standard deviation, though with only "
              "3 seeds this is suggestive rather than statistically conclusive.")


    illus_seed = episode_seeds[0]
    eq_illus = equal_weight_baseline(PortfolioEnv, price_test, reset_seed=illus_seed, **EVAL_KWARGS)
    bh_illus = buy_and_hold_baseline(PortfolioEnv, price_test, reset_seed=illus_seed, **EVAL_KWARGS)
    ddpg_illus = {}
    for seed in seeds:
        actor_path = os.path.join(RESULTS_DIR, f"actor_seed{seed}.pt")
        actor = load_actor(actor_path, state_dim, action_dim)
        ddpg_illus[seed] = run_agent_on_episode(actor, price_test, action_dim, state_dim, illus_seed)

    with open(os.path.join(RESULTS_DIR, "eq_baseline_history.json"), "w") as f:
        json.dump({"value": [float(v) for v in eq_illus["value"]]}, f)
    with open(os.path.join(RESULTS_DIR, "bh_baseline_history.json"), "w") as f:
        json.dump({"value": [float(v) for v in bh_illus["value"]]}, f)
    with open(os.path.join(RESULTS_DIR, "ddpg_test_histories.json"), "w") as f:
        json.dump({str(s): {"value": [float(v) for v in ddpg_illus[s]["value"]]} for s in seeds}, f)

    print("\nSaved: results/eval_episode_log.csv (raw, per-episode),"
          " results/test_metrics.csv (aggregated),"
          " results/*_history.json (one illustrative episode, for plotting).")

    return summary_df, raw_df


if __name__ == "__main__":
    main()
