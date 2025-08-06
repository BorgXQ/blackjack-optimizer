import numpy as np
from blackjack_rl_env import BlackjackRLEnv
from monte_carlo_agent import MonteCarloAgent
from config import TRAINING_EPISODES, EVALUATION_EPISODES, BASELINE_EPISODES, EPSILON, EPSILON_DECAY
from utils import plot_training_progress, analyze_policy, print_statistics
import time


def train_agent(
    num_episodes: int=TRAINING_EPISODES,
    starting_balance: int=100000000,
    fixed_bet: int=10,
    epsilon: float=EPSILON,
    epsilon_decay: float=EPSILON_DECAY,
    save_model: bool=True,
    model_path: str="blackjack_mc_agent.pkl",
    log_interval: int=5000
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

def evaluate_agent(
    agent: MonteCarloAgent, 
    env: BlackjackRLEnv, 
    num_episodes: int=EVALUATION_EPISODES
):
    """Evaluate the trained agent"""
    
    print("\n" + "=" * 60)
    print("AGENT EVALUATION")
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

def basic_strategy_action(
        player_sum: int,
        dealer_visible: int,
        usable_ace: bool,
        can_split: bool,
        can_double: bool,
        valid_actions: list
) -> int:
    """Return basic strategy action"""
    # If we can split, check split conditions first
    if can_split and 2 in valid_actions:
        # Always split Aces and 8s
        if player_sum == 22 or player_sum == 16:  # AA=22, 88=16
            return 2  # Split
        # Split 2s, 3s, 6s, 7s, 9s against certain dealer cards
        if player_sum == 4 and 4 <= dealer_visible <= 7:  # 22
            return 2
        if player_sum == 6 and 4 <= dealer_visible <= 7:  # 33
            return 2
        if player_sum == 12 and 3 <= dealer_visible <= 6:  # 66
            return 2
        if player_sum == 14 and 2 <= dealer_visible <= 7:  # 77
            return 2
        if player_sum == 18 and dealer_visible != 7 and dealer_visible != 10 and dealer_visible != 11:  # 99
            return 2
    
    # Check doubling conditions
    if can_double and 3 in valid_actions:
        if usable_ace:  # Soft totals
            if player_sum == 13 or player_sum == 14:  # A2, A3
                if 5 <= dealer_visible <= 6:
                    return 3  # Double
            elif player_sum == 15 or player_sum == 16:  # A4, A5
                if 4 <= dealer_visible <= 6:
                    return 3  # Double
            elif player_sum == 17 or player_sum == 18:  # A6, A7
                if 3 <= dealer_visible <= 6:
                    return 3  # Double
        else:  # Hard totals
            if player_sum == 9:
                if 3 <= dealer_visible <= 6:
                    return 3  # Double
            elif player_sum == 10:
                if dealer_visible <= 9:
                    return 3  # Double
            elif player_sum == 11:
                return 3  # Double (always double 11)
    
    # Hit/Stand logic
    if usable_ace:  # Soft totals
        if player_sum >= 19:  # A8, A9
            return 1  # Stand
        elif player_sum == 18:  # A7
            if dealer_visible == 2 or dealer_visible == 7 or dealer_visible == 8 or dealer_visible == 11:
                return 1  # Stand
            else:
                return 0  # Hit
        else:  # A2-A6
            return 0  # Hit
    else:  # Hard totals
        if player_sum >= 17:
            return 1  # Stand
        elif player_sum <= 11:
            return 0  # Hit
        elif player_sum >= 13:
            return 1 if dealer_visible <= 6 else 0
        else:  # 12
            return 1 if 4 <= dealer_visible <= 6 else 0
        
def run_basic_strategy_comparison(env: BlackjackRLEnv, num_episodes: int=BASELINE_EPISODES):
    """Run basic strategy policy for comparison"""
    
    print("\n" + "=" * 60)
    print("BASIC STRATEGY POLICY")
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
                
            # Get basic strategy action
            player_sum, dealer_showing, usable_ace, can_split, can_double = state
            action = basic_strategy_action(
                player_sum,
                dealer_showing,
                usable_ace,
                can_split,
                can_double,
                valid_actions
            )
            
            # Ensure action is valid (fallback to hit if not)
            if action not in valid_actions:
                action = 0 if 0 in valid_actions else valid_actions[0]
                
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                total_rewards.append(episode_reward)
                total_wins.append(info.get('wins', 0))
                total_hands_count.append(info.get('total_hands', 1))
                break
                
            state = next_state
    
    # Calculate statistics
    basic_strategy_results = {
        'avg_reward': np.mean(total_rewards),
        'total_wins': sum(total_wins),
        'total_hands': sum(total_hands_count),
        'win_rate': sum(total_wins) / sum(total_hands_count) if sum(total_hands_count) > 0 else 0,
        'std_reward': np.std(total_rewards)
    }
    
    print(f"Episodes: {num_episodes:,}")
    print("-" * 60)
    print_statistics(basic_strategy_results)
    
    return basic_strategy_results

def main():
    """Main simulation runner"""
    
    # Train the agent
    agent, env, training_stats = train_agent(
        num_episodes=TRAINING_EPISODES,
        starting_balance=100000,
        fixed_bet=10,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        log_interval=5000
    )
    
    # Evaluate the trained agent
    eval_results = evaluate_agent(agent, env, EVALUATION_EPISODES)
    
    # Run comparisons
    baseline_results = run_baseline_comparison(env, BASELINE_EPISODES)
    basic_strategy_results = run_basic_strategy_comparison(env, BASELINE_EPISODES)
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} {'Trained Agent':<15} {'Basic Strategy':<15} {'Random Baseline':<15}")
    print("-" * 80)
    
    agent_wr = eval_results['win_rate']
    basic_wr = basic_strategy_results['win_rate']
    baseline_wr = baseline_results['win_rate']
    
    agent_reward = eval_results['avg_reward']
    basic_reward = basic_strategy_results['avg_reward']
    baseline_reward = baseline_results['avg_reward']
    
    print(f"{'Win Rate':<20} {agent_wr:<15.4f} {basic_wr:<15.4f} {baseline_wr:<15.4f}")
    print(f"{'Avg Reward':<20} {agent_reward:<15.4f} {basic_reward:<15.4f} {baseline_reward:<15.4f}")

    # Calculate improvements
    print("\n" + "IMPROVEMENTS OVER BASELINES")
    print("-" * 40)
    
    if basic_wr > 0:
        agent_vs_basic_wr = ((agent_wr - basic_wr) / basic_wr * 100)
        agent_vs_basic_reward = ((agent_reward - basic_reward) / abs(basic_reward) * 100) if basic_reward != 0 else 0
        print(f"Agent vs Basic Strategy:")
        print(f"  Win Rate:   {agent_vs_basic_wr:+6.2f}%")
        print(f"  Avg Reward: {agent_vs_basic_reward:+6.2f}%")
    
    if baseline_wr > 0:
        agent_vs_random_wr = ((agent_wr - baseline_wr) / baseline_wr * 100)
        agent_vs_random_reward = ((agent_reward - baseline_reward) / abs(baseline_reward) * 100) if baseline_reward != 0 else 0
        print(f"\nAgent vs Random:")
        print(f"  Win Rate:   {agent_vs_random_wr:+6.2f}%")
        print(f"  Avg Reward: {agent_vs_random_reward:+6.2f}%")
    
    if baseline_wr > 0:
        basic_vs_random_wr = ((basic_wr - baseline_wr) / baseline_wr * 100)
        basic_vs_random_reward = ((basic_reward - baseline_reward) / abs(baseline_reward) * 100) if baseline_reward != 0 else 0
        print(f"\nBasic Strategy vs Random:")
        print(f"  Win Rate:   {basic_vs_random_wr:+6.2f}%")
        print(f"  Avg Reward: {basic_vs_random_reward:+6.2f}%")
    
    # Show sample of learned policy
    agent.print_policy_sample(15)
    
    # Plot training progress
    plot_training_progress(training_stats)
    
    # Analyze the learned policy
    policy_analysis = analyze_policy(agent)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"States learned: {policy_analysis['states_learned']}")
    print(f"Final epsilon: {agent.epsilon:.6f}")
    print("Model saved and evaluation complete!")

if __name__ == "__main__":
    main()
