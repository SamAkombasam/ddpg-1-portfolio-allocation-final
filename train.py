"""
train.py
--------
Trains the DDPG agent on the portfolio allocation environment across
multiple random seeds, logging episode rewards for later plotting and
convergence or stability analysis.
"""
import sys, os, json, time
sys.path.append(os.path.dirname(__file__))

import numpy as np
from env.portfolio_env import PortfolioEnv
from agent.ddpg_agent import DDPGAgent
from data_loader_real import load_real_price_data, train_test_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ENV_KWARGS = dict(window=10, initial_cash=100_000.0, transaction_cost=0.001, turnover_penalty=0.5)


def train_one_seed(seed, price_train, n_episodes=120, warmup_steps=500, verbose_every=20):
    env = PortfolioEnv(price_train, seed=seed, **ENV_KWARGS)
    env.action_space.seed(seed)  # controls env.action_space.sample() during warmup, below
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DDPGAgent(state_dim, action_dim, seed=seed, noise_sigma=0.4)
    np.random.seed(seed)

    episode_rewards = []
    episode_final_values = []
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        agent.noise.reset()
        # linearly decay exploration noise over training
        agent.noise.sigma = max(0.05, 0.4 * (1 - ep / n_episodes))

        ep_reward = 0.0
        done = trunc = False
        while not (done or trunc):
            if total_steps < warmup_steps:
                action = env.action_space.sample()  # pure random warmup for buffer diversity
            else:
                action = agent.act(obs, explore=True)
            next_obs, reward, done, trunc, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, float(done))
            obs = next_obs
            ep_reward += reward
            total_steps += 1
            if total_steps >= warmup_steps:
                agent.update()

        episode_rewards.append(ep_reward)
        episode_final_values.append(env.portfolio_value)
        if (ep + 1) % verbose_every == 0:
            print(f"  [seed {seed}] episode {ep+1}/{n_episodes} | reward={ep_reward:.4f} "
                  f"| final_value=${env.portfolio_value:,.0f}")

    return agent, episode_rewards, episode_final_values


def main(n_episodes=150, seeds=(0, 1, 2)):
    price_df = load_real_price_data()
    price_train, price_test = train_test_split(price_df, test_fraction=0.2)
    price_train.to_csv(os.path.join(RESULTS_DIR, "price_train.csv"), index=False)
    price_test.to_csv(os.path.join(RESULTS_DIR, "price_test.csv"), index=False)

    all_results = {}
    for seed in seeds:
        print(f"\n=== Training DDPG | seed={seed} ===")
        t0 = time.time()
        agent, rewards, values = train_one_seed(seed, price_train, n_episodes=n_episodes)
        elapsed = time.time() - t0
        print(f"  seed {seed} done in {elapsed:.1f}s")

        import torch
        torch.save(agent.actor.state_dict(), os.path.join(RESULTS_DIR, f"actor_seed{seed}.pt"))
        all_results[seed] = {
            "episode_rewards": [float(x) for x in rewards],
            "episode_final_values": [float(x) for x in values],
        }

    with open(os.path.join(RESULTS_DIR, "training_log.json"), "w") as f:
        json.dump(all_results, f)
    print("\nTraining complete. Logs saved to results/training_log.json")
    return all_results


if __name__ == "__main__":
    main()
