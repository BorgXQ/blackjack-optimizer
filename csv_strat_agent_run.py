import numpy as np
import time
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import from blackjack_sim
# Adjust this path based on your directory structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack_sim.blackjack_rl_env import BlackjackRLEnv
from blackjack_sim.monte_carlo_agent import MonteCarloAgent
from blackjack_sim.basic_strategy_agent import BasicStrategyAgent
from blackjack_sim.config import (
    TRAINING_EPISODES,
    EVALUATION_EPISODES,
    BASELINE_EPISODES,
    LOG_INTERVAL
)
from blackjack_sim.utils import (
    print_statistics,
    export_learned_strategy_csv
)
from csv_strat_agent import CSVStrategyAgent


def train_csv_strategy_agent(
    csv_filepath: str,
    num_episodes: int = TRAINING_EPISODES,
    starting_balance: int = 100000000,
    fixed_bet: int = 10,
    save_model: bool = True,
    model_path: str = "./model_data/csv_strategy_agent.pkl",
    log_interval: int = LOG_INTERVAL
):
    """Train CSV Strategy agent Q-values through simulation"""
    
    print("\n" + "=" * 60)
    print("CSV STRATEGY Q-VALUE TRAINING")
    print("=" * 60)
    print(f"CSV file: {csv_filepath}")
    print(f"Episodes: {num_episodes:,}")
    print("-" * 60)
    
    # Initialize environment and agent
    env = BlackjackRLEnv(starting_balance=starting_balance, fixed_bet=fixed_bet)
    csv_agent = CSVStrategyAgent(csv_filepath)
    
    start_time = time.time()
    
    # Train Q-values
    csv_agent.train_q_values(env, num_episodes, log_interval)
    
    total_time = time.time() - start_time
    print(f"CSV Strategy Q-value training completed in {total_time:.2f} seconds")
    print(f"Average speed: {num_episodes / total_time:.1f} episodes/second")
    
    # Save trained model
    if save_model:
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        csv_agent.save_model(model_path)
        print(f"CSV Strategy model saved to {model_path}")
    
    return csv_agent


def evaluate_csv_strategy_agent(
    csv_agent: CSVStrategyAgent,
    env: BlackjackRLEnv,
    num_episodes: int = EVALUATION_EPISODES
):
    """Evaluate the CSV strategy agent"""
    
    print("\n" + "=" * 60)
    print("CSV STRATEGY AGENT EVALUATION")
    print("=" * 60)
    
    start_time = time.time()
    eval_results = csv_agent.evaluate(env, num_episodes)
    eval_time = time.time() - start_time
    
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    print(f"Episodes: {num_episodes:,}")
    print("-" * 60)
    
    # Print detailed statistics
    print_statistics(eval_results)
    
    return eval_results


def run_baseline_comparison(env: BlackjackRLEnv, num_episodes: int = BASELINE_EPISODES):
    """Run baseline random policy for comparison"""
    
    print("\n" + "=" * 60)
    print("BASELINE RANDOM POLICY")
    print("=" * 60)
    
    total_rewards = []
    total_wins = []
    total_hands_count = []
    
    for episode in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        
        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            # Random action
            action = np.random.choice(valid_actions)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                total_rewards.append(episode_reward)
                total_wins.append(info.get('wins', 0))
                total_hands_count.append(info.get('total_hands', 1))
                break
                
            state = next_state
    
    # Calculate stats
    baseline_results = {
        'avg_reward': np.mean(total_rewards),
        'total_wins': sum(total_wins),
        'total_hands': sum(total_hands_count),
        'win_rate': sum(total_wins) / sum(total_hands_count) if sum(total_hands_count) > 0 else 0,
        'std_reward': np.std(total_rewards)
    }
    
    print(f"Episodes: {num_episodes:,}")
    print("-" * 60)
    print_statistics(baseline_results)
    
    return baseline_results


def main(csv_filepath: str = None):
    """Main simulation runner for CSV strategy"""
    
    if csv_filepath is None:
        # Default CSV file path - adjust this to your CSV file location
        csv_filepath = "strategy.csv"
    
    # Check if CSV file exists
    if not os.path.exists(csv_filepath):
        print(f"Error: CSV file '{csv_filepath}' not found!")
        print("Please provide the path to your CSV file.")
        return
    
    print("=" * 80)
    print("BLACKJACK CSV STRATEGY SIMULATION")
    print("=" * 80)
    print(f"CSV Strategy File: {csv_filepath}")
    print(f"Training Episodes: {TRAINING_EPISODES:,}")
    print(f"Evaluation Episodes: {EVALUATION_EPISODES:,}")
    print(f"Baseline Episodes: {BASELINE_EPISODES:,}")
    print("=" * 80)
    
    # Train CSV strategy agent (calculate Q-values through simulation)
    csv_agent = train_csv_strategy_agent(
        csv_filepath=csv_filepath,
        num_episodes=TRAINING_EPISODES,
        log_interval=LOG_INTERVAL
    )
    
    # Initialize environment for evaluation
    env = BlackjackRLEnv(starting_balance=100000000, fixed_bet=10)
    
    # Evaluate the CSV strategy agent
    csv_results = evaluate_csv_strategy_agent(csv_agent, env, EVALUATION_EPISODES)
    
    # Run baseline comparison (optional)
    baseline_results = run_baseline_comparison(env, BASELINE_EPISODES)
    
    # For additional comparison, also run basic strategy agent
    print("\n" + "=" * 60)
    print("TRAINING BASIC STRATEGY FOR COMPARISON")
    print("=" * 60)
    basic_agent = BasicStrategyAgent()
    basic_agent.train_q_values(env, TRAINING_EPISODES, LOG_INTERVAL)
    basic_results = basic_agent.evaluate(env, EVALUATION_EPISODES)
    
    print("\n" + "=" * 60)
    print("BASIC STRATEGY EVALUATION")
    print("=" * 60)
    print_statistics(basic_results)
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} {'CSV Strategy':<15} {'Basic Strategy':<15} {'Random Baseline':<15}")
    print("-" * 80)
    
    csv_wr = csv_results['win_rate']
    basic_wr = basic_results['win_rate']
    baseline_wr = baseline_results['win_rate']
    
    csv_reward = csv_results['avg_reward']
    basic_reward = basic_results['avg_reward']
    baseline_reward = baseline_results['avg_reward']
    
    print(f"{'Win Rate':<20} {csv_wr:<15.4f} {basic_wr:<15.4f} {baseline_wr:<15.4f}")
    print(f"{'Avg Reward':<20} {csv_reward:<15.4f} {basic_reward:<15.4f} {baseline_reward:<15.4f}")
    
    # Calculate improvements
    print("\n" + "IMPROVEMENTS OVER BASELINES")
    print("-" * 40)
    
    if basic_wr > 0:
        csv_vs_basic_wr = ((csv_wr - basic_wr) / basic_wr * 100)
        csv_vs_basic_reward = ((csv_reward - basic_reward) / abs(basic_reward) * 100) if basic_reward != 0 else 0
        print(f"CSV vs Basic Strategy:")
        print(f"  Win Rate:   {csv_vs_basic_wr:+6.2f}%")
        print(f"  Avg Reward: {csv_vs_basic_reward:+6.2f}%")
    
    if baseline_wr > 0:
        csv_vs_random_wr = ((csv_wr - baseline_wr) / baseline_wr * 100)
        csv_vs_random_reward = ((csv_reward - baseline_reward) / abs(baseline_reward) * 100) if baseline_reward != 0 else 0
        print(f"\nCSV vs Random:")
        print(f"  Win Rate:   {csv_vs_random_wr:+6.2f}%")
        print(f"  Avg Reward: {csv_vs_random_reward:+6.2f}%")
        
        basic_vs_random_wr = ((basic_wr - baseline_wr) / baseline_wr * 100)
        basic_vs_random_reward = ((basic_reward - baseline_reward) / abs(baseline_reward) * 100) if baseline_reward != 0 else 0
        print(f"\nBasic Strategy vs Random:")
        print(f"  Win Rate:   {basic_vs_random_wr:+6.2f}%")
        print(f"  Avg Reward: {basic_vs_random_reward:+6.2f}%")

    # Export learned Q-values to CSV
    csv_filename = export_learned_strategy_csv(csv_agent, "csv_strategy_qvalues.csv")
    
    # Get policy analysis
    csv_policy_analysis = csv_agent.get_policy_summary()
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"CSV strategy states loaded: {csv_policy_analysis['states_learned']}")
    print(f"Q-values exported to: {csv_filename}")
    print("CSV strategy evaluation complete!")
    
    # Print sample of the policy
    csv_agent.print_policy_sample(10)
    
    return {
        'csv_results': csv_results,
        'basic_results': basic_results,
        'baseline_results': baseline_results,
        'csv_agent': csv_agent
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Blackjack simulation with CSV strategy')
    parser.add_argument('csv_file', nargs='?', default='merged_results.csv',
                        help='Path to CSV file containing strategy (default: merged_results.csv)')

    args = parser.parse_args()
    
    main(args.csv_file)