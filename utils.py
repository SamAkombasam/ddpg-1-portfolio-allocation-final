"""
utils.py
---------------------------------------------------
Evaluation-metric utilities and baseline strategies.
"""
import numpy as np


def cumulative_return(values):
    return values[-1] / values[0] - 1.0


def daily_returns(values):
    values = np.asarray(values)
    return values[1:] / values[:-1] - 1.0


def sharpe_ratio(values, risk_free=0.0, periods_per_year=252):
    r = daily_returns(values)
    excess = r - risk_free / periods_per_year
    if excess.std() < 1e-9:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))


def max_drawdown(values):
    values = np.asarray(values)
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    return float(drawdown.min())


def annualized_volatility(values, periods_per_year=252):
    r = daily_returns(values)
    return float(np.std(r) * np.sqrt(periods_per_year))


def total_transaction_costs(turnovers, cost_rate):
    return float(np.sum(np.asarray(turnovers)) * cost_rate)


def summarize_metrics(values, turnovers, cost_rate, label=""):
    return {
        "label": label,
        "final_value": float(values[-1]),
        "cumulative_return_%": cumulative_return(values) * 100,
        "sharpe_ratio": sharpe_ratio(values),
        "max_drawdown_%": max_drawdown(values) * 100,
        "annual_volatility_%": annualized_volatility(values) * 100,
        "total_transaction_cost_$": total_transaction_costs(turnovers, cost_rate) * values[0],
    }


# ----------------------------- baselines -------------------------------
def equal_weight_baseline(env_class, price_df, reset_seed=None, **env_kwargs):
    """Buy-and-hold / rebalance-to-equal-weight baseline (rebalanced daily).

    reset_seed, when provided, is passed to env.reset(seed=...) so that this
    baseline is evaluated on the SAME episode (same start day, same length)
    as the agent it is being compared against — required for a fair,
    identical-conditions comparison.
    """
    env = env_class(price_df, **env_kwargs)
    obs, _ = env.reset(seed=reset_seed)
    n = env.n_assets
    equal_action = np.zeros(n, dtype=np.float32)  # softmax(0,...,0) = equal weights
    done = trunc = False
    while not (done or trunc):
        obs, r, done, trunc, info = env.step(equal_action)
    return env.history


def buy_and_hold_baseline(env_class, price_df, reset_seed=None, **env_kwargs):
    """True buy-and-hold: allocate equally ONCE, never rebalance again (turnover=0 after day 0).

    reset_seed behaves as in equal_weight_baseline above.
    """
    env = env_class(price_df, **env_kwargs)
    obs, _ = env.reset(seed=reset_seed)
    n = env.n_assets
    done = trunc = False
    step_idx = 0
    while not (done or trunc):
        if step_idx == 0:
            action = np.zeros(n, dtype=np.float32)  # equal weight at start
        else:
            # push logits far towards current weights so softmax ~ keeps them
            w = np.clip(env.weights, 1e-6, None)
            action = np.log(w)
        obs, r, done, trunc, info = env.step(action)
        step_idx += 1
    return env.history
