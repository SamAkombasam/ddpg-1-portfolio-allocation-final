"""
portfolio_env.py
-----------------
A Gymnasium-compatible continuous-control environment for dynamic
multi-asset portfolio allocation, used as the MDP for the DDPG agent.

MDP FORMULATION
================
State  s_t  : concatenation of, for each of N assets,
                 - last `window` daily log-returns
                 - 3 technical indicators (SMA ratio, momentum, RSI-like volatility signal)
              with:
                 - current portfolio weights (N,)
                 - cash ratio (1,)
                 - realized portfolio volatility over last `window` days (1,)

Action a_t  : continuous vector in R^N (raw logits). Converted to a valid
              simplex (weights that sum to 1, all >= 0) via softmax, so the
              agent always outputs a fully-invested long-only allocation
              across the N assets. (Cash is implicit as any weight the
              agent effectively withholds is captured through turnover cost,
              not as a separate action dimension, keeping the action space
              exactly N-dimensional and easy for DDPG's deterministic
              policy + OU exploration noise to handle.)

Reward r_t  : log-return of the portfolio value from t to t+1, MINUS a
              transaction-cost penalty proportional to portfolio turnover
              (sum of absolute weight changes), i.e.
                 r_t = log(V_{t+1}/V_t) - lambda * turnover_t
              This is a risk-cost-aware reward: it rewards growth while
              penalizing excessive rebalancing (a real trading friction).

Episode termination:
    - natural end of the available price window (`done=True`), OR
    - portfolio value drops below 30% of the initial capital (early
      termination to punish catastrophic drawdowns and shorten
      exploration of already-failed trajectories).

Discount factor: gamma = 0.99 (set in the training script), standard for
    tasks with horizons of several hundred steps where long-run
    compounding of returns matters.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        price_df,
        window=10,
        initial_cash=100_000.0,
        transaction_cost=0.001,
        turnover_penalty=0.5,
        min_value_fraction=0.30,
        episode_length=250,      # ~1 trading year per episode
        random_start=True,       # sample a random start day each reset (training diversity)
        seed=None,
    ):
        super().__init__()
        self.window = window
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost   # proportional cost per unit turnover
        self.turnover_penalty = turnover_penalty    # extra reward-shaping weight on turnover
        self.min_value_fraction = min_value_fraction
        self.episode_length = episode_length
        self.random_start = random_start
        self._rng = np.random.default_rng(seed)

        price_cols = [c for c in price_df.columns if not c.endswith("_VOL")]
        vol_cols = [c for c in price_df.columns if c.endswith("_VOL")]
        self.assets = price_cols
        self.n_assets = len(price_cols)

        self.prices = price_df[price_cols].values.astype(np.float32)
        self.volumes = price_df[vol_cols].values.astype(np.float32)
        # prepend the FIRST LOG-PRICE (not the raw price) so log_returns[0] == 0;
        # prepending the raw price makes the first row log(p_0) - p_0, a value
        # ~3 orders of magnitude outside the normal daily-return range.
        self.log_returns = np.diff(np.log(self.prices), axis=0, prepend=np.log(self.prices[[0]]))

        self.n_steps_total = len(self.prices) - 1

        # --- state / action space definitions ---
        # per-asset: window log-returns + 3 indicators = window + 3
        # plus: n_assets weights + 1 cash ratio + 1 realized vol
        obs_dim = self.n_assets * (window + 3) + self.n_assets + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n_assets,), dtype=np.float32)

        self.reset()

    # ------------------------------------------------------------------
    def _softmax_weights(self, raw_action):
        z = raw_action - np.max(raw_action)
        e = np.exp(z)
        return e / (np.sum(e) + 1e-8)

    def _technical_indicators(self, t):
        """SMA ratio, momentum, short-term volatility for each asset at day t."""
        w = self.window
        window_prices = self.prices[max(0, t - w):t + 1]
        if len(window_prices) < 2:
            window_prices = np.repeat(self.prices[t:t + 1], 2, axis=0)
        sma = window_prices.mean(axis=0)
        sma_ratio = self.prices[t] / (sma + 1e-8) - 1.0
        momentum = self.prices[t] / (window_prices[0] + 1e-8) - 1.0
        vol_signal = window_prices.std(axis=0) / (sma + 1e-8)
        return sma_ratio, momentum, vol_signal

    def _get_obs(self):
        t = self.t
        w = self.window
        # log_returns[t] = log(p_t / p_{t-1}) is known once day t's close is known,
        # so the window ends at t INCLUSIVE. (It previously ended at t-1, silently
        # withholding today's return while the indicators below already used p_t.)
        start = max(0, t - w + 1)
        ret_window = self.log_returns[start:t + 1]
        if len(ret_window) < w:
            pad = np.zeros((w - len(ret_window), self.n_assets), dtype=np.float32)
            ret_window = np.vstack([pad, ret_window]) if len(ret_window) else pad
        sma_ratio, momentum, vol_signal = self._technical_indicators(t)

        returns_flat = ret_window.T.flatten()  # per-asset returns concatenated
        indicators = np.stack([sma_ratio, momentum, vol_signal], axis=1).flatten()

        # same window as the return features above, so both describe the same span
        realized_vol = float(np.std(ret_window @ self.weights)) if len(ret_window) > 1 else 0.0
        cash_ratio = 0.0  # fully invested (long-only simplex); kept as a feature slot for extensibility

        obs = np.concatenate([
            returns_flat.astype(np.float32),
            indicators.astype(np.float32),
            self.weights.astype(np.float32),
            np.array([cash_ratio, realized_vol], dtype=np.float32),
        ])
        return obs

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.episode_length is not None and self.random_start:
            latest_start = max(self.window, self.n_steps_total - self.episode_length)
            self.t = int(self._rng.integers(self.window, latest_start + 1))
            self.episode_end = min(self.t + self.episode_length, self.n_steps_total)
        else:
            self.t = self.window
            self.episode_end = self.n_steps_total

        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.portfolio_value = self.initial_cash
        self.history = {"value": [self.portfolio_value], "weights": [self.weights.copy()],
                        "turnover": [], "cost": []}
        return self._get_obs(), {}

    def step(self, action):
        # self.weights holds the ACTUAL (drift-adjusted) holdings at the start of today.
        target_weights = self._softmax_weights(np.asarray(action, dtype=np.float32))
        turnover = np.sum(np.abs(target_weights - self.weights))  # cost of rebalancing to target

        asset_returns = self.prices[self.t + 1] / self.prices[self.t] - 1.0
        gross_return = float(np.dot(target_weights, asset_returns))

        cost = self.transaction_cost * turnover
        net_growth = (1 + gross_return) * (1 - cost)
        new_value = self.portfolio_value * net_growth
        # cost is charged against the post-drift value, so record the actual
        # dollars paid rather than reconstructing them from initial capital later
        cost_dollars = self.portfolio_value * (1 + gross_return) * cost

        reward = np.log(max(new_value, 1e-6) / max(self.portfolio_value, 1e-6))
        reward -= self.turnover_penalty * self.transaction_cost * turnover  # extra shaping term

        # weights DRIFT with the day's asset-specific returns (no more free rebalancing)
        drifted = target_weights * (1 + asset_returns)
        drifted = drifted / (np.sum(drifted) + 1e-8)

        self.portfolio_value = new_value
        self.weights = drifted.astype(np.float32)
        self.t += 1

        self.history["value"].append(self.portfolio_value)
        self.history["weights"].append(self.weights.copy())
        self.history["turnover"].append(turnover)
        self.history["cost"].append(cost_dollars)

        terminated = self.portfolio_value < self.min_value_fraction * self.initial_cash
        truncated = self.t >= self.episode_end

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "cost": cost_dollars,
        }
