import numpy as np
from blackjack_sim.blackjack_rl_env import BlackjackRLEnv
from blackjack_sim.monte_carlo_agent import MonteCarloAgent
from blackjack_sim.basic_strategy_agent import BasicStrategyAgent
from blackjack_sim.combined_strategy_agent import CombinedStrategyAgent
from blackjack_sim.config import (
    TRAINING_EPISODES,
    EVALUATION_EPISODES,
    BASELINE_EPISODES,
    EPSILON,
    EPSILON_DECAY,
    LOG_INTERVAL,
    COMBINED_STRATEGY_ENABLED
)
from blackjack_sim.utils import (
    plot_training_progress,
    analyze_policy,
    print_statistics,
    export_learned_strategy_csv
)
import time


def train_agent(
    num_episodes: int=TRAINING_EPISODES,
    starting_balance: int=100000000,
    fixed_bet: int=10,
    epsilon: float=EPSILON,
    epsilon_decay: float=EPSILON_DECAY,
    save_model: bool=True,
    model_path: str="./model_data/trained_agent.pkl",
    log_interval: int=LOG_INTERVAL
):
    """Train the Monte Carlo agent"""
    
    print("=" * 60)
    print("BLACKJACK MONTE CARLO TRAINING")
    print("=" * 60)
    print(f"Episodes: {num_episodes:,}")
    print(f"Starting balance: ${starting_balance:,}")
    print(f"Fixed bet: ${fixed_bet}")
    print(f"Initial epsilon: {epsilon}")
    print(f"Epsilon decay: {epsilon_decay}")
    print("-" * 60)
    
    # Initialize environment and agent
    env = BlackjackRLEnv(starting_balance=starting_balance, fixed_bet=fixed_bet)
    agent = MonteCarloAgent(epsilon=epsilon, epsilon_decay=epsilon_decay)
    
    # Training stats
    training_stats = {
        'episodes': [],
        'avg_rewards': [],
        'avg_win_rates': [],
        'epsilons': []
    }
    
    start_time = time.time()
    
    # Training loop
    for episode in range(num_episodes):
        # Train one episode
        agent.train_episode(env)
        
        # Log progress
        if (episode + 1) % log_interval == 0:
            # Calculate running averages
            recent_rewards = agent.episode_rewards[-log_interval:]
            recent_win_rates = agent.episode_win_rates[-log_interval:]
            
            avg_reward = np.mean(recent_rewards)
            avg_win_rate = np.mean(recent_win_rates)
            
            elapsed_time = time.time() - start_time
            episodes_per_sec = (episode + 1) / elapsed_time
            
            print(f"Episode {episode + 1:6,} | "
                  f"Avg Reward: {avg_reward:6.3f} | "
                  f"Win Rate: {avg_win_rate:6.3f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"EPS: {episodes_per_sec:.1f}")
            
            # Store for plotting
            training_stats['episodes'].append(episode + 1)
            training_stats['avg_rewards'].append(avg_reward)
            training_stats['avg_win_rates'].append(avg_win_rate)
            training_stats['epsilons'].append(agent.epsilon)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Average speed: {num_episodes / total_time:.1f} episodes/second")
    
    # Save trained model
    if save_model:
        agent.save_model(model_path)
        print(f"Model saved to {model_path}")
    
    return agent, env, training_stats

def train_basic_strategy_agent(
    num_episodes: int=TRAINING_EPISODES,
    starting_balance: int=100000000,
    fixed_bet: int=10,
    save_model: bool=True,
    model_path: str="./model_data/basic_strategy_agent.pkl",
    log_interval: int=LOG_INTERVAL
):
    """Train Basic Strategy agent Q-values through simulation"""
    
    print("\n" + "=" * 60)
    print("BASIC STRATEGY Q-VALUE TRAINING")
    print("=" * 60)
    print(f"Episodes: {num_episodes:,}")
    print("-" * 60)
    
    # Initialize environment and agent
    env = BlackjackRLEnv(starting_balance=starting_balance, fixed_bet=fixed_bet)
    basic_agent = BasicStrategyAgent()
    
    start_time = time.time()
    
    # Train Q-values
    basic_agent.train_q_values(env, num_episodes, log_interval)
    
    total_time = time.time() - start_time
    print(f"Basic Strategy Q-value training completed in {total_time:.2f} seconds")
    print(f"Average speed: {num_episodes / total_time:.1f} episodes/second")
    
    # Save trained model
    if save_model:
        basic_agent.save_model(model_path)
        print(f"Basic Strategy model saved to {model_path}")
    
    return basic_agent

def train_combined_strategy_agent(
    num_episodes: int=TRAINING_EPISODES,
    starting_balance: int=100000000,
    fixed_bet: int=10,
    save_model: bool=True,
    model_path: str="./model_data/combined_strategy_agent.pkl",
    log_interval: int=LOG_INTERVAL,
    csv_filepath: str="./csv/combined_strategy.csv"
):
    """Train Combined Strategy agent Q-values through simulation"""

    print("\n" + "=" * 60)
    print("COMBINED STRATEGY Q-VALUE TRAINING")
    print("=" * 60)
    print(f"Episodes: {num_episodes:,}")
    print("-" * 60)
    
    # Initialize environment and agent
    env = BlackjackRLEnv(starting_balance=starting_balance, fixed_bet=fixed_bet)
    combined_agent = CombinedStrategyAgent(csv_filepath=csv_filepath)

    start_time = time.time()
    
    # Train Q-values
    combined_agent.train_q_values(env, num_episodes, log_interval)

    total_time = time.time() - start_time
    print(f"Basic Strategy Q-value training completed in {total_time:.2f} seconds")
    print(f"Average speed: {num_episodes / total_time:.1f} episodes/second")
    
    # Save trained model
    if save_model:
        combined_agent.save_model(model_path)
        print(f"Combined Strategy model saved to {model_path}")

    return combined_agent

def evaluate_trained_agent(
    agent: MonteCarloAgent, 
    env: BlackjackRLEnv, 
    num_episodes: int=EVALUATION_EPISODES
):
    """Evaluate the trained agent"""
    
    print("\n" + "=" * 60)
    print("TRAINED AGENT EVALUATION")
    print("=" * 60)
    
    start_time = time.time()
    eval_results = agent.evaluate(env, num_episodes)
    eval_time = time.time() - start_time
    
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    print(f"Episodes: {num_episodes:,}")
    print("-" * 60)
    
    # Print detailed statistics
    print_statistics(eval_results)
    
    return eval_results

def evaluate_basic_strategy_agent(
    basic_agent: BasicStrategyAgent,
    env: BlackjackRLEnv,
    num_episodes: int=EVALUATION_EPISODES
):
    """Evaluate the basic strategy agent"""
    
    print("\n" + "=" * 60)
    print("BASIC STRATEGY AGENT EVALUATION")
    print("=" * 60)
    
    start_time = time.time()
    eval_results = basic_agent.evaluate(env, num_episodes)
    eval_time = time.time() - start_time
    
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    print(f"Episodes: {num_episodes:,}")
    print("-" * 60)
    
    # Print detailed statistics
    print_statistics(eval_results)
    
    return eval_results

def evaluate_combined_strategy_agent(
    combined_agent: CombinedStrategyAgent,
    env: BlackjackRLEnv,
    num_episodes: int=EVALUATION_EPISODES
):
    """Evaluate the combined strategy agent"""
    
    print("\n" + "=" * 60)
    print("COMBINED STRATEGY AGENT EVALUATION")
    print("=" * 60)
    
    start_time = time.time()
    eval_results = combined_agent.evaluate(env, num_episodes)
    eval_time = time.time() - start_time
    
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    print(f"Episodes: {num_episodes:,}")
    print("-" * 60)
    
    # Print detailed statistics
    print_statistics(eval_results)
    
    return eval_results

def run_baseline_comparison(env: BlackjackRLEnv, num_episodes: int=BASELINE_EPISODES):
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

def main():
    """Main simulation runner"""
    
    # Train agent
    trained_agent, env, training_stats = train_agent(
        num_episodes=TRAINING_EPISODES,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        log_interval=LOG_INTERVAL
    )

    # Train basic strategy agent (Q-vals)
    basic_agent = train_basic_strategy_agent(
        num_episodes=TRAINING_EPISODES,
        log_interval=LOG_INTERVAL
    )

    # Train combined strategy agent
    combined_agent = None
    if COMBINED_STRATEGY_ENABLED:
        combined_agent = train_combined_strategy_agent(
            num_episodes=TRAINING_EPISODES,
            log_interval=LOG_INTERVAL
        )

    # Evaluate the agents
    trained_results = evaluate_trained_agent(trained_agent, env, EVALUATION_EPISODES)
    basic_results = evaluate_basic_strategy_agent(basic_agent, env, EVALUATION_EPISODES)
    combined_results = None
    if COMBINED_STRATEGY_ENABLED:
        combined_results = evaluate_combined_strategy_agent(combined_agent, env, EVALUATION_EPISODES)

    # Run baseline comparison
    baseline_results = run_baseline_comparison(env, BASELINE_EPISODES)
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    if COMBINED_STRATEGY_ENABLED:
        print(f"{'Metric':<20} {'Trained Agent':<15} {'Basic Strategy':<15} {'Combined Strategy':<17} {'Random Baseline':<15}")
        print("-" * 97)
    else:
        print(f"{'Metric':<20} {'Trained Agent':<15} {'Basic Strategy':<15} {'Random Baseline':<15}")
        print("-" * 80)
    
    trained_wr = trained_results['win_rate']
    basic_wr = basic_results['win_rate']
    baseline_wr = baseline_results['win_rate']
    
    trained_reward = trained_results['avg_reward']
    basic_reward = basic_results['avg_reward']
    baseline_reward = baseline_results['avg_reward']
    
    if COMBINED_STRATEGY_ENABLED:
        combined_wr = combined_results['win_rate']
        combined_reward = combined_results['avg_reward']
        print(f"{'Win Rate':<20} {combined_wr:<17.4f} {trained_wr:<15.4f} {basic_wr:<15.4f} {baseline_wr:<15.4f}")
        print(f"{'Avg Reward':<20} {combined_reward:<17.4f} {trained_reward:<15.4f} {basic_reward:<15.4f} {baseline_reward:<15.4f}")
    else:
        print(f"{'Win Rate':<20} {trained_wr:<15.4f} {basic_wr:<15.4f} {baseline_wr:<15.4f}")
        print(f"{'Avg Reward':<20} {trained_reward:<15.4f} {basic_reward:<15.4f} {baseline_reward:<15.4f}")

    # Calculate improvements
    print("\n" + "IMPROVEMENTS OVER BASELINES")
    print("-" * 40)
    
    if COMBINED_STRATEGY_ENABLED:
        combined_vs_trained_wr = ((combined_wr - trained_wr) / trained_wr * 100)
        combined_vs_trained_reward = ((combined_reward - trained_reward) / abs(trained_reward) * 100) if trained_reward != 0 else 0
        print(f"\nCombined vs Trained Strategy:")
        print(f"  Win Rate:   {combined_vs_trained_wr:+6.2f}%")
        print(f"  Avg Reward: {combined_vs_trained_reward:+6.2f}%")

    if basic_wr > 0:
        if COMBINED_STRATEGY_ENABLED:
            combined_vs_basic_wr = ((combined_wr - basic_wr) / basic_wr * 100)
            combined_vs_basic_reward = ((combined_reward - basic_reward) / abs(basic_reward) * 100) if basic_reward != 0 else 0
            print(f"\nCombined vs Basic Strategy:")
            print(f"  Win Rate:   {combined_vs_basic_wr:+6.2f}%")
            print(f"  Avg Reward: {combined_vs_basic_reward:+6.2f}%")
        
        trained_vs_basic_wr = ((trained_wr - basic_wr) / basic_wr * 100)
        trained_vs_basic_reward = ((trained_reward - basic_reward) / abs(basic_reward) * 100) if basic_reward != 0 else 0
        print(f"Trained vs Basic Strategy:")
        print(f"  Win Rate:   {trained_vs_basic_wr:+6.2f}%")
        print(f"  Avg Reward: {trained_vs_basic_reward:+6.2f}%")
    
    if baseline_wr > 0:
        if COMBINED_STRATEGY_ENABLED:
            combined_vs_random_wr = ((combined_wr - baseline_wr) / baseline_wr * 100)
            combined_vs_random_reward = ((combined_reward - baseline_reward) / abs(baseline_reward) * 100) if baseline_reward != 0 else 0
            print(f"\nCombined vs Random:")
            print(f"  Win Rate:   {combined_vs_random_wr:+6.2f}%")
            print(f"  Avg Reward: {combined_vs_random_reward:+6.2f}%")
        
        trained_vs_random_wr = ((trained_wr - baseline_wr) / baseline_wr * 100)
        trained_vs_random_reward = ((trained_reward - baseline_reward) / abs(baseline_reward) * 100) if baseline_reward != 0 else 0
        print(f"\nTrained vs Random:")
        print(f"  Win Rate:   {trained_vs_random_wr:+6.2f}%")
        print(f"  Avg Reward: {trained_vs_random_reward:+6.2f}%")
        
        basic_vs_random_wr = ((basic_wr - baseline_wr) / baseline_wr * 100)
        basic_vs_random_reward = ((basic_reward - baseline_reward) / abs(baseline_reward) * 100) if baseline_reward != 0 else 0
        print(f"\nBasic Strategy vs Random:")
        print(f"  Win Rate:   {basic_vs_random_wr:+6.2f}%")
        print(f"  Avg Reward: {basic_vs_random_reward:+6.2f}%")

    # Export learned strategy to CSV
    trained_filename = export_learned_strategy_csv(trained_agent, "trained_strategy.csv")
    basic_filename = export_learned_strategy_csv(basic_agent, "basic_strategy.csv")
    if COMBINED_STRATEGY_ENABLED:
        combined_filename = export_learned_strategy_csv(combined_agent, "combined_strategy.csv")
    
    # Plot training progress
    plot_training_progress(training_stats)
    
    # Analyze the learned policy
    trained_policy_analysis = analyze_policy(trained_agent)
    basic_policy_analysis = analyze_policy(basic_agent)
    if COMBINED_STRATEGY_ENABLED:
        combined_policy_analysis = analyze_policy(combined_agent)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    if COMBINED_STRATEGY_ENABLED:
        print(f"Combined strategy states learned: {combined_policy_analysis['states_learned']}")
    print(f"Trained states learned: {trained_policy_analysis['states_learned']}")
    print(f"Basic strategy states learned: {basic_policy_analysis['states_learned']}")
    print(f"Final trained epsilon: {trained_agent.epsilon:.6f}")
    if COMBINED_STRATEGY_ENABLED:
        print(f"Combined strategy exported to: {combined_filename}")
    print(f"Trained strategy exported to: {trained_filename}")
    print(f"Basic strategy exported to: {basic_filename}")
    print("Model saved and evaluation complete!")

if __name__ == "__main__":
    main()
