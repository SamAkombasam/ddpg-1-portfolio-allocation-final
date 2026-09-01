# Dynamic Portfolio Allocation with DDPG

Implementation and evaluation of Deep Deterministic Policy Gradient (DDPG) applied to continuous multi-asset portfolio allocation, trained and evaluated on real historical S&P 500 daily price data (AAPL, XOM, JNJ, JPM, PG, CAT; 2015–2025).

Course: DSCD 614 Reinforcement Learning: Group Project-Based Examination, Option **DDPG-1: Dynamic Portfolio Allocation** Group: **Group 10**.

---

## Quick Start

### 1. Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies are pinned to exact versions in `requirements.txt`.

### 2. Reproduce the headline result from a clean environment

```bash
python reproduce.py
```

This single entry point runs the full pipeline end to end (~15–20 minutes on a laptop CPU):
1. Trains the DDPG agent for 3 random seeds (150 episodes each): `train.py`
2. Evaluates all 3 trained policies, and both baselines, on 30 held-out evaluation episodes each, under identical conditions: `evaluate.py`
3. Regenerates every figure in this report: `generate_plots.py`

Each of these three scripts can also be run individually, in the same order, if you want to inspect intermediate output.

### 3. Explore results interactively 

```bash
streamlit run dashboard.py
```


### 4. Reproducing individual figures

| Figure                                         | Script | Output file |
|------------------------------------------------|---|---|
| Fig. 1: Training reward per episode            | `generate_plots.py` → `plot_training_rewards()` | `plots/training_reward_curve.png` |
| Fig. 2: Mean training reward ± std             | `generate_plots.py` → `plot_training_rewards()` | `plots/training_reward_mean_std.png` |
| Fig. 3: Illustrative episode portfolio value   | `generate_plots.py` → `plot_test_portfolio_comparison()` | `plots/test_portfolio_value_comparison.png` |
| Fig. 4: Mean return/Sharpe across 30 episodes  | `generate_plots.py` → `plot_metrics_bars()` | `plots/metrics_comparison_bars.png` |
| Fig. 5: Return distribution across 30 episodes | `generate_plots.py` → `plot_episode_distribution()` | `plots/episode_return_distribution.png` |

All figures regenerate from the raw logs committed under `results/` (`training_log.json`, `eval_episode_log.csv`, `test_metrics.csv`), so every number in this report is traceable to a committed log file.

---

## Repository Structure

```
.
├── agent/
│   └── ddpg_agent.py          # DDPG actor, critic, replay buffer, OU noise (from-scratch, attributed to Lillicrap et al. 2016)
├── env/
│   └── portfolio_env.py       # Custom Gymnasium MDP environment
├── real_data/                 # Raw per-ticker OHLCV CSVs (AAPL, XOM, JNJ, JPM, PG, CAT)
├── results/                   # Committed raw logs: training_log.json, eval_episode_log.csv, test_metrics.csv, actor_seed*.pt
├── plots/                     # All figures (rendered below)
├── data_loader_real.py        # Loads and aligns the real S&P 500 data
├── data_gen.py                # Synthetic data generator (not used for reported results; kept as an offline-safe fallback)
├── utils.py                   # Evaluation metrics + baseline strategies
├── train.py                   # Multi-seed DDPG training loop
├── evaluate.py                # 30-episode-per-seed evaluation harness vs. baselines
├── generate_plots.py          # Regenerates all figures from results/
├── reproduce.py               # Single entry point: train + evaluate + plot
├── dashboard.py                # Interactive Streamlit dashboard
├── requirements.txt           # Pinned exact dependency versions
└── .gitignore
```

---

# Project Report

## Abstract

This report presents the implementation and evaluation of Deep Deterministic Policy Gradient (DDPG) applied to dynamic, continuous-action portfolio allocation across six S&P 500 equities (2015–2025). The task is formulated as a finite-horizon Markov Decision Process and implemented as a custom Gymnasium environment. The agent is trained across three random seeds and evaluated, with exploration disabled, on 30 held-out evaluation episodes per seed, sampled from a chronologically separate test period; the required baseline (equal-weight and buy-and-hold portfolios) is evaluated on the identical 30 episodes for a controlled comparison. Averaged across seeds, the DDPG agent's mean cumulative return (7.49%) is close to both baselines (7.96% and 7.66%), a gap that is smaller than the cross-seed standard deviation of DDPG's own performance (3.26 percentage points) and therefore cannot, with only three seeds, be distinguished from seed-to-seed training noise. DDPG also assumes materially higher volatility and drawdown than either baseline. This is reported and diagnosed as an inconclusive result: the agent shows no clear evidence of improving on, or clearly underperforming, simple heuristics under this protocol, and the report attributes the remaining risk-adjusted shortfall (lower Sharpe ratio despite comparable return) primarily to a reward function that does not penalize variance, the absence of hyperparameter tuning, and known instability properties of vanilla DDPG relative to later continuous-control algorithms. During development, a partial random-seed control bug was also identified and corrected the environment's training-episode sampling and warm-up action randomness were not originally tied to the specified seed and this correction, along with a separate observation about cross-machine floating-point non-determinism in neural network training, is documented in Section 4.7.

## 1. Introduction

Portfolio allocation is the recurring decision of how to distribute investment capital across a set of financial assets and this is a sequential decision-making problem under uncertainty, and is therefore naturally amenable to formulation as a Markov Decision Process (MDP). Classical approaches such as mean-variance optimization typically rely on static, single-period assumptions and require explicit estimation of return distributions. Reinforcement learning offers an alternative in which an adaptive, multi-period allocation policy is learned directly from historical data.

This project addresses catalogue option **DDPG-1 (Dynamic Portfolio Allocation)**. Because portfolio weights are continuous, the problem is naturally suited to continuous-control RL rather than discrete-action methods. Deep Deterministic Policy Gradient [1] was among the first deep RL algorithms designed explicitly for continuous action spaces and is a standard baseline in applied financial RL research [2]. The project's significance is not in achieving state-of-the-art trading performance where the specification explicitly does not assess the score achieved but in producing a correctly formulated MDP, a correctly implemented agent, and a controlled, statistically honest comparison against the required baseline, including a full diagnosis of whatever result is actually obtained.

## 2. Background

DDPG extends the deterministic policy gradient theorem [3] to high-dimensional continuous action spaces using the experience replay and target-network techniques from Deep Q-Networks [4]. Later work identified and addressed specific weaknesses of DDPG: Twin Delayed DDPG (TD3) [5] mitigates value overestimation via clipped double-Q learning and delayed policy updates, and Soft Actor-Critic (SAC) [6] adds entropy regularization for improved exploration and stability. These successor algorithms are directly relevant to the interpretation of this project's results, since several instability phenomena observed below are exactly what TD3 and SAC were designed to address.

In computational finance, the open-source FinRL framework [2] has standardized environment design for RL-based trading and portfolio management and has demonstrated DDPG, among other algorithms, on multi-asset allocation tasks. A recurring finding in this literature is that reward design, particularly the treatment of transaction costs and risk  materially affects both policy quality and training stability, a theme this project examines directly through its transaction-cost-aware reward and its evaluation protocol.

## 3. Problem Formulation

The task is formulated as a finite-horizon MDP with the following components.

**State space.** A continuous vector *s_t ∈ R^86*, comprising, for each of the six assets: a rolling 10-day window of log-returns (60 values) and three technical indicators, a simple-moving-average ratio, a momentum measure, and a short-term volatility signal (18 values)  together with the agent's current portfolio weights (6 values), a cash-ratio placeholder, and the realized volatility of the portfolio's own recent returns (2 values). All components are continuous and are scale-free by construction (returns, ratios), so no additional normalization layer was applied.

**Action space.** A continuous vector *a_t ∈ R^6* of unconstrained logits in [-5, 5], transformed via a softmax function into a valid, fully invested, long-only allocation (non-negative weights summing to 1).

**Reward function.**

> r_t = ln(V_{t+1} / V_t) − λ · c · Σᵢ |w_target,i − w_drift,i|

where *V_t* is portfolio value, *c* = 0.001 is the proportional transaction-cost rate, *λ* = 0.5 is an additional turnover-penalty weight, *w_target* = softmax(a_t) is the newly chosen allocation, and *w_drift* is the previous day's holdings after they have drifted with each asset's realized return (i.e. turnover is measured against the actual pre-trade holdings, not the previous day's target).

**Episode termination and truncation** (stated separately, as required).
- *Termination:* the episode ends early if portfolio value falls below 30% of initial capital a catastrophic-loss stop.
- *Truncation:* the episode ends at a fixed horizon: 250 trading days during training, or 126 trading days during evaluation (see Section 4.4), or at the end of the available price series.

**Discount factor.** γ = 0.99, giving an effective horizon of 1/(1−γ) = 100 trading days (~4.8 calendar months). This value was adopted directly from the hyperparameters reported in Lillicrap et al. [1] rather than tuned on this task (see Section 4.3). It is judged appropriate because it is of the same order of magnitude as the 126-day evaluation episode length and shorter than the 250-day training episode length, so the discounted objective does not extend the agent's effective planning horizon far beyond the periods it is actually trained and evaluated on. No systematic sensitivity analysis of γ was conducted, within the project's time constraint; this is stated as a limitation (Section 7).

**Does the Markov property hold?** No, not strictly. Financial return series are not Markovian in raw price space: momentum, mean-reversion, and volatility-clustering effects extend beyond any fixed-length window, and the constructed state omits information such as order-book microstructure, macroeconomic indicators, and information about assets outside the six-ticker universe. The true process is therefore better characterized as a Partially Observable MDP (POMDP), of which the constructed state is an approximate sufficient statistic rather than the true underlying market state. The implementation compensates for this via state augmentation including a rolling window of returns and technical indicators specifically because they summarize recent history in a fixed-length vector which is a standard, accepted approximation in applied deep RL (analogous to frame-stacking in Atari-DQN), but remains an approximation: the 10-day window length is a hyperparameter trading off state dimensionality against captured history, and no claim is made that the resulting process is truly Markovian.

## 4. Methodology

### 4.1 Data

Historical daily price data were obtained from the Kaggle dataset "S&P 500 Stocks. Daily Historical Data (10 Years)" [7]. A sector-diverse subset of six equities was selected AAPL (Technology), XOM (Energy), JNJ (Healthcare), JPM (Financials), PG (Consumer Staples), CAT (Industrials) screened for complete data coverage and for pairwise return correlations (0.22–0.60) consistent with genuine diversification potential. The aligned 2,515-day series was split chronologically into a training set (first 80%, ≈2,012 days, Dec 2015–mid 2023) and a held-out test set (final 20%, ≈503 days, through Dec 2025), never accessed during training.

### 4.2 Agent Architecture

DDPG was implemented from first principles in PyTorch (no RL library such as Stable-Baselines3 was used), following [1]; see attribution comment in `agent/ddpg_agent.py`. The actor is a two-hidden-layer network (256, 128 units, ReLU, tanh output scaled to the action bound); the critic has matching hidden-layer dimensionality. Both are paired with Polyak-averaged target networks. An experience replay buffer (capacity 100,000) is sampled in batches of 128. Exploration uses a temporally correlated Ornstein-Uhlenbeck process, annealed from σ=0.4 to σ=0.05 over training.

### 4.3 Hyperparameters and Seeds

All hyperparameters were held constant across the three training seeds (0, 1, 2). No hyperparameter search was conducted: values follow [1] and standard DDPG reference settings, adopted without tuning on this task's data, given the 14-day project scoping constraint. This means reported performance reflects an untuned, first-attempt capability of DDPG on this task rather than its best achievable performance.

| Hyperparameter | Value |
|---|---|
| Discount factor (γ) | 0.99 |
| Soft target update rate (τ) | 0.005 |
| Actor learning rate | 1 × 10⁻⁴ |
| Critic learning rate | 1 × 10⁻³ |
| Replay buffer capacity | 100,000 |
| Batch size | 128 |
| Actor/critic hidden layers | (256, 128), ReLU |
| OU noise: θ, σ (start → end), dt | 0.15, 0.4 → 0.05, 0.01 |
| Gradient clip norm | 1.0 |
| Training episodes per seed | 150 |
| Training episode length | 250 trading days (random start within training data) |
| Warm-up steps (random actions) | 500 |
| Transaction cost rate (c) | 0.001 |
| Turnover penalty weight (λ) | 0.5 |
| Early-termination threshold | 30% of initial capital |
| Random seeds | 0, 1, 2 |
| Evaluation episodes per seed | 30 |
| Evaluation episode length | 126 trading days |

### 4.4 Baseline

As specified by the catalogue, the required baseline is an equal-weighted or buy-and-hold portfolio; **both** are implemented. The equal-weight strategy rebalances to 1/6 per asset daily, incurring transaction costs whenever price drift moves actual holdings away from target. The buy-and-hold strategy sets an equal-weight allocation once and never rebalances, incurring negligible further transaction costs.

### 4.5 Training Procedure

The agent trains for 150 episodes per seed, each episode a randomly sampled 250-day window from the training data (refreshed at every reset, to expose the agent to varied historical regimes rather than one fixed trajectory). The first 500 steps use uniformly random actions to populate the replay buffer before learning begins.

### 4.6 Evaluation Protocol

Each trained policy is evaluated with exploration disabled, the actor's deterministic output is used directly on **30 evaluation episodes**, each a 126-trading-day window sampled from the held-out test period via a fixed, disjoint set of episode seeds (10,000–10,029), never overlapping with the training seeds. Both baselines are evaluated on the **identical 30 episodes** (same start day, same length, same price data, same metric code) as each trained policy, so every comparison is made under matched conditions. Because the test period is a single ≈503-day historical trajectory rather than a re-samplable simulator, the 30 sampled windows necessarily overlap; this is a limitation, discussed in Section 7. For every metric we report the mean and standard deviation both (a) across the 30 episodes, per seed and per baseline, and (b) across the three seeds' means, to separate within-seed (which 6-month window was sampled) from across-seed (which training run) variation.

### 4.7 Reproducibility: A Partial Seed-Control Bug, Found and Fixed

During development, the project was independently re-run on a second machine using the same code and the same three seeds (0, 1, 2). The two baseline strategies reproduced to six decimal places confirming the data loading, train/test split, and 30-episode evaluation protocol are fully deterministic. The DDPG results, however, did **not** reproduce: the same nominal seed produced materially different trained policies on each machine.

Root-cause analysis traced this to incomplete random-seed control in the training loop. The DDPG agent's network initialization, replay-buffer sampling order, and exploration noise were correctly seeded, but two further sources of randomness used during training were not tied to the seed argument at all: (a) which 250-day window of training data each episode samples, and (b) the uniformly random actions taken during the 500-step warm-up phase. Both were drawn from unseeded, OS-entropy-derived randomness, so "seed=0" did not, in fact, fully determine the resulting trained policy.

This was corrected by explicitly seeding the environment's internal random-number generator and its action space at the start of training for each seed. After this fix, two independent process runs *on the same machine* with the same seed produced bit-identical training reward sequences. However, when the fixed code was subsequently run on a second, different machine, results still differed somewhat (cross-seed mean return of 6.03% vs. 7.49%) despite identical seeds and code. This second, smaller discrepancy is attributed to ordinary floating-point non-determinism in neural network training: matrix-multiplication and gradient-accumulation order can differ across CPU architectures and BLAS libraries, and these tiny numerical differences compound over 150 training episodes into a measurably different policy, even with every explicit random source correctly seeded a well-documented limitation of exact cross-machine reproducibility in deep learning, distinct from the seed-control bug above. Both machines nonetheless supported the *same qualitative conclusion* (the DDPG–baseline gap is smaller than DDPG's own cross-seed standard deviation), evidence that the conclusion, if not the exact figures, is robust to this variation. **Section 5 reports the figures from the second, final, group-run machine**, since that is the run the group can directly reproduce and defend. Both issues are reported here, rather than silently corrected, because they are themselves substantive methodological findings: "the same seed" does not guarantee full reproducibility unless every stochastic source is explicitly controlled, and even then, exact cross-machine reproducibility of trained neural networks should not be assumed.

## 5. Results

### 5.1 Training

![Training reward per episode](plots/training_reward_curve.png)

*Figure 1. Raw (light) and 10-episode moving-average (bold) training reward per episode, per seed.*

![Mean training reward with std band](plots/training_reward_mean_std.png)

*Figure 2. Mean training reward across seeds, ±1 std band.*

Training reward rises modestly over the first 40–60 episodes then remains noisy for the rest of training, with no clean convergence to a stable plateau for any seed consistent with each episode sampling a different, non-stationary historical window, so the attainable reward varies by episode independent of policy quality.

### 5.2 Evaluation

![Illustrative episode portfolio value](plots/test_portfolio_value_comparison.png)

*Figure 3. Portfolio value on one illustrative 126-day evaluation episode (headline results below use all 30 episodes).*

Figure 3 illustrates a single evaluation episode in which seed 2 visibly outperforms the other seeds and both baselines throughout, while seed 1 tracks close to the baselines and pulls ahead only near the end. This is a useful cautionary illustration in its own right: on this one episode, a viewer would conclude seed 2 is the strongest policy, yet the aggregate 30-episode results (Table 1, Figure 5 below) show the opposite  seed 1 has the highest mean return (12.07%) while seed 2 is middling (5.60%). This divergence between a single illustrative episode and the full 30-episode aggregate is precisely why the evaluation protocol requires many episodes rather than one: any single episode, however visually persuasive, can be misleading about a policy's typical behavior.

![Mean return and Sharpe across 30 episodes](plots/metrics_comparison_bars.png)

*Figure 4. Mean cumulative return and Sharpe ratio across the 30 evaluation episodes, error bars = ±1 std across episodes.*

![Return distribution across 30 episodes](plots/episode_return_distribution.png)

*Figure 5. Distribution of cumulative return across all 30 evaluation episodes, per strategy.*

**Table 1.** Evaluation results: mean ± std across 30 episodes (per seed / baseline), and DDPG's cross-seed mean ± std of those means.

| Strategy | Mean Return % | Sharpe | Max DD % | Ann. Vol % | Txn Cost $ |
|---|---|---|---|---|---|
| Equal-Weight (daily rebal.) | 7.96 ± 7.89 | 1.39 ± 1.32 | −8.73 ± 4.96 | 14.04 ± 4.31 | 108 ± 6 |
| Buy-and-Hold | 7.66 ± 8.28 | 1.33 ± 1.34 | −8.74 ± 4.83 | 13.95 ± 3.96 | ≈0 |
| DDPG (seed=0) | 4.79 ± 6.92 | 0.54 ± 0.67 | −11.32 ± 2.34 | 20.91 ± 2.77 | 1,632 ± 731 |
| DDPG (seed=1) | 12.07 ± 7.51 | 1.52 ± 1.04 | −8.25 ± 2.75 | 17.05 ± 3.40 | 2,936 ± 186 |
| DDPG (seed=2) | 5.60 ± 6.97 | 0.74 ± 0.75 | −14.06 ± 6.56 | 19.79 ± 4.63 | 4,832 ± 442 |
| **DDPG, mean across seeds** | **7.49 ± 3.26*** | **0.94 ± 0.42*** | **−11.21 ± 2.38*** | **19.25 ± 1.62*** | **3,133 ± 1,314*** |

*Cross-seed std (n=3), of each seed's own 30-episode mean — distinct from the within-seed std in the rows above.

The headline comparison uses the cross-seed mean, not any individual seed. DDPG's mean cumulative return (7.49%) is only 0.47 percentage points below the equal-weight baseline (7.96%), the two are effectively comparable in raw return. This gap is **far smaller than** the cross-seed standard deviation of DDPG's mean (3.26 points), meaning this difference cannot be distinguished from seed-to-seed training noise. Notably, the three seeds themselves disagree substantially: seed 1 (12.07%, Sharpe 1.52) clearly exceeds both baselines, while seeds 0 and 2 (4.79% and 5.60%) fall short underscoring why the cross-seed mean, not any single seed, is the only defensible headline figure. Despite comparable raw return, all three DDPG seeds show materially higher volatility (17.0–20.9%) and deeper drawdown (−8.3% to −14.1%) than either baseline (≈14%, ≈−8.7%), and substantially higher transaction costs meaning DDPG's mean Sharpe ratio (0.94) is meaningfully lower than either baseline's (1.39, 1.33): comparable return was achieved by taking on more risk, not more skill.

## 6. Discussion

**Convergence.** Training reward did not converge to a stable plateau (Section 5.1); this is expected given each episode samples a different historical window, so the attainable reward ceiling itself varies episode to episode.

**Training stability.** Three observations emerged, all concerning how much apparent "instability" was an artifact of measurement rather than the algorithm. First, an earlier, less rigorous evaluation protocol (a single pass through the test period per seed, rather than 30 sampled episodes) had produced an apparent cross-seed standard deviation of over 40 percentage points, collapsing to single digits once the 30-episode protocol was adopted. Second, an incomplete random-seed control bug (Section 4.7) meant early training runs were not actually reproducible from the stated seed. Third, after that bug was fixed, independent runs on different machines with identical seeds still produced somewhat different trained policies traced to ordinary floating-point non-determinism across CPU architectures and BLAS libraries (Section 4.7), not a further bug. Despite this residual variation, the qualitative conclusion was identical on both machines: DDPG's mean return gap relative to baseline is smaller than its own cross-seed standard deviation, and is therefore statistically inconclusive at n=3 seeds. The individual seeds also disagree substantially with each other (4.79% to 12.07%), consistent with DDPG's documented sensitivity to random initialization relative to later algorithms such as TD3 and SAC.

**Exploration.** OU noise was chosen for temporal correlation, appropriate for a domain where uncorrelated action jitter would generate unrealistic churn. Unlike an earlier run, transaction costs do not cleanly predict outcome quality here: seed 2 incurred the highest transaction costs ($4,832) and the deepest drawdown (−14.06%), yet a middling return (5.60%), while seed 0 incurred the lowest costs ($1,632) but achieved the worst return (4.79%) of the three. The best-performing seed (seed 1, 12.07% return, Sharpe 1.52) had intermediate transaction costs ($2,936) suggesting that, in this run, trading frequency alone is not a reliable predictor of outcome; the differences instead appear driven by which policy each seed's training happened to converge toward.

**Effect of reward design.** All three seeds took on materially more volatility and drawdown risk than either baseline, despite the cross-seed mean return (7.49%) being close to the baselines' (7.96%, 7.66%). This is reflected directly in the Sharpe ratio: DDPG's mean Sharpe (0.94) is meaningfully below either baseline's (1.39, 1.33), meaning DDPG achieved comparable raw return only by taking on materially more risk, the return was not "free," it was purchased with volatility. This is consistent with the log-return-minus-turnover-cost reward not penalizing variance sufficiently. A risk-adjusted reward (example: a differential Sharpe ratio, or a mean-variance-penalized return) is a plausible next step to reduce this gap.

## 7. Limitations and Deployment Considerations

- The 30 evaluation episodes are sampled from a single ≈503-day historical series and therefore overlap; they are not fully independent draws, unlike episodes from a re-samplable simulator. This is an unavoidable consequence of using one real historical price path and is stated explicitly as required.
- No hyperparameter search was conducted (14-day scoping constraint); reported results reflect untuned, first-attempt DDPG performance.
- Transaction costs are a simple linear function of turnover and do not model bid-ask spreads or market impact.
- The six-asset, single-country universe is a substantial simplification of institutional portfolio management.
- Vanilla DDPG is more prone to value overestimation and seed-sensitive training than TD3 or SAC; results should be read as characteristic of DDPG specifically, not of deep RL for portfolio allocation generally.
- With only 3 seeds, statistical power to detect anything but a large effect is low; the honest conclusion here is "inconclusive," not "DDPG is definitively worse" more seeds would be needed to say more.
- **Deployment:** a live system would require realistic market-impact cost modeling, explicit risk-management overlays (position limits, drawdown circuit-breakers) beyond what the reward implies, ongoing monitoring and retraining against market non-stationarity, and some mechanism for explainability given the black-box nature of the learned policy.

## 8. Conclusion

Under a corrected, 30-episode-per-seed evaluation protocol with the required baseline evaluated on identical episodes, and after also correcting a partial random-seed control bug in the training code (Section 4.7), DDPG's mean cumulative return across three seeds (7.49%) was close to, and statistically indistinguishable from, the equal-weight (7.96%) and buy-and-hold (7.66%) baselines, the gap is smaller than the cross-seed standard deviation of DDPG's own performance and cannot be resolved with only three seeds. However, DDPG achieved this comparable return only by taking on substantially higher volatility, drawdown, and transaction costs, resulting in a materially lower Sharpe ratio than either baseline. This is reported as a rigorously diagnosed, appropriately cautious result: a correctly formulated MDP and correctly implemented DDPG agent, evaluated under a corrected and independently cross-machine-tested protocol, show no clear evidence of outperforming simple heuristic allocation on a risk-adjusted basis on this task, and the project's three independently discovered methodological issues (single-path evaluation variance, incomplete seed control, and cross-machine floating-point non-determinism) are reported as findings in their own right, not smoothed over directions for future work include risk-adjusted reward shaping, hyperparameter tuning, a more stable algorithm variant (TD3/SAC), and additional seeds to increase statistical power.

## References

[1] Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control with deep reinforcement learning. *Proceedings of the International Conference on Learning Representations (ICLR)*.

[2] Liu, X.-Y., Yang, H., Chen, Q., Zhang, R., Yang, L., Xiao, B., & Wang, C. D. (2020). FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance. *arXiv preprint arXiv:2011.09607*.

[3] Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). Deterministic policy gradient algorithms. *Proceedings of the 31st International Conference on Machine Learning (ICML)*.

[4] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.

[5] Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. *Proceedings of the 35th International Conference on Machine Learning (ICML)*.

[6] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *Proceedings of the 35th International Conference on Machine Learning (ICML)*.

[7] Kaggle. (n.d.). S&P 500 Stocks – Daily Historical Data (10 Years) [Dataset]. https://www.kaggle.com/datasets/innacampo/s-and-p-500-stocks-daily-historical-data-10-years

---

## Troubleshooting

- **`ModuleNotFoundError`**: activate the virtual environment and re-run `pip install -r requirements.txt`.
- **Training seems slow**: this is CPU-bound; 150 episodes per seed takes a few minutes. Reduce `n_episodes` in `train.py` for a faster, lower-quality test run.
- **`FileNotFoundError` on `real_data/*.csv`**: run scripts from the repository root.
- **Windows `pip install` fails with `WinError 32` (file in use)**: usually antivirus real-time scanning or a locked `.venv`; close all running Python/PyCharm consoles, temporarily pause real-time antivirus scanning, delete `.venv`, recreate it, and reinstall.
- **Dashboard shows missing result files**: run `python reproduce.py` (or `train.py` then `evaluate.py`) first.
