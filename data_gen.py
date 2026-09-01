"""
data_gen.py
-----------
Generates a realistic multi-asset daily price dataset used to train and
evaluate the portfolio-allocation agent.
"""
import numpy as np
import pandas as pd

ASSET_NAMES = ["TECH", "ENERGY", "HEALTH", "FINANCE", "CONSUMER", "INDUSTRIAL"]

# Calibrated to be broadly realistic for a 6-asset equity universe
ANNUAL_DRIFT = np.array([0.14, 0.06, 0.09, 0.08, 0.07, 0.05])
ANNUAL_VOL = np.array([0.35, 0.30, 0.22, 0.25, 0.20, 0.28])

# Cross-asset correlation matrix (symmetric, positive semi-definite)
CORR = np.array([
    [1.00, 0.20, 0.15, 0.30, 0.25, 0.20],
    [0.20, 1.00, 0.10, 0.20, 0.15, 0.35],
    [0.15, 0.10, 1.00, 0.15, 0.20, 0.10],
    [0.30, 0.20, 0.15, 1.00, 0.25, 0.25],
    [0.25, 0.15, 0.20, 0.25, 1.00, 0.20],
    [0.20, 0.35, 0.10, 0.25, 0.20, 1.00],
])

TRADING_DAYS_PER_YEAR = 252


def generate_price_data(n_days=1500, seed=42, start_price=100.0):
    """Simulate n_days of correlated daily close prices for len(ASSET_NAMES) assets."""
    rng = np.random.default_rng(seed)
    n_assets = len(ASSET_NAMES)

    daily_drift = ANNUAL_DRIFT / TRADING_DAYS_PER_YEAR
    daily_vol = ANNUAL_VOL / np.sqrt(TRADING_DAYS_PER_YEAR)

    # Cholesky decomposition to inject correlation into independent shocks
    L = np.linalg.cholesky(CORR)
    z = rng.standard_normal((n_days, n_assets))
    correlated_z = z @ L.T

    log_returns = daily_drift + daily_vol * correlated_z
    # occasional volatility clustering or regime shocks for realism
    shock_days = rng.choice(n_days, size=max(1, n_days // 150), replace=False)
    for d in shock_days:
        log_returns[d] += rng.normal(0, 0.03, size=n_assets)

    log_prices = np.log(start_price) + np.cumsum(log_returns, axis=0)
    prices = np.exp(log_prices)

    df = pd.DataFrame(prices, columns=ASSET_NAMES)
    # synthetic volume, useful as an auxiliary state feature
    base_vol = rng.integers(1_000_000, 5_000_000, size=n_assets)
    vol_noise = rng.normal(1.0, 0.2, size=(n_days, n_assets)).clip(0.3, 2.5)
    volume = (base_vol * vol_noise).astype(int)
    for i, a in enumerate(ASSET_NAMES):
        df[f"{a}_VOL"] = volume[:, i]

    df.index.name = "day"
    return df


def train_test_split(df, test_fraction=0.2):
    n = len(df)
    split = int(n * (1 - test_fraction))
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


if __name__ == "__main__":
    data = generate_price_data()
    print(data.head())
    print(data.shape)
