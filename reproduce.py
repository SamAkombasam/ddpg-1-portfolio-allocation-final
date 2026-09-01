"""
reproduce.py
------------
SINGLE ENTRY POINT that reproduces the project's full headline result
from a clean environment.

This script performs the entire pipeline in order:
    1. Trains the DDPG agent for 3 random seeds (train.py)
    2. Evaluates all 3 trained policies, plus both baselines, on 30
       held-out evaluation episodes each (evaluate.py)
    3. Regenerates every figure used in the report (generate_plots.py)


This script deliberately calls the SAME functions used by train.py,
evaluate.py, and generate_plots.py when they are run individually.
"""
import os
import sys
import time

sys.path.append(os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def main():
    t_start = time.time()

    print("=" * 60)
    print("STEP 1/3: Training DDPG agent (3 seeds x 150 episodes each)")
    print("=" * 60)
    import train
    train.main()

    print("\n" + "=" * 60)
    print("STEP 2/3: Evaluating on 30 held-out episodes per seed, vs. baselines")
    print("=" * 60)
    import evaluate
    evaluate.main()

    print("\n" + "=" * 60)
    print("STEP 3/3: Regenerating all figures")
    print("=" * 60)
    import generate_plots
    generate_plots.plot_training_rewards()
    generate_plots.plot_test_portfolio_comparison()
    generate_plots.plot_metrics_bars()
    generate_plots.plot_episode_distribution()

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"DONE in {elapsed / 60:.1f} minutes.")
    print(f"Results:  {RESULTS_DIR}")
    print(f"Figures:  {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
