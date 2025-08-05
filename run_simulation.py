import numpy as np
from blackjack_rl_env import BlackjackRLEnv
from monte_carlo_agent import MonteCarloAgent
from config import TRAINING_EPISODES, EVALUATION_EPISODES, BASELINE_EPISODES, EPSILON, EPSILON_DECAY
from utils import plot_training_progress, analyze_policy, print_statistics
import time


def train_agent(num_episodes: int = TRAINING_EPISODES,
               starting_balance: int = 100000,
               fixed_bet: int = 10,
               epsilon: float = EPSILON,
               epsilon_decay: float = EPSILON_DECAY,
               save_model: bool = True,
               model_path: str = "blackjack_mc_agent.pkl",
               log_interval: int = 5000):
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

def evaluate_agent(agent: MonteCarloAgent, 
                  env: BlackjackRLEnv, 
                  num_episodes: int = 10000):
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

def run_baseline_comparison(env: BlackjackRLEnv, num_episodes: int = 10000):
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
    
    # Run baseline comparison
    baseline_results = run_baseline_comparison(env, BASELINE_EPISODES)
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Trained Agent':<15} {'Random Baseline':<15} {'Improvement':<15}")
    print("-" * 65)
    
    agent_wr = eval_results['win_rate']
    baseline_wr = baseline_results['win_rate']
    wr_improvement = ((agent_wr - baseline_wr) / baseline_wr * 100) if baseline_wr > 0 else 0
    
    agent_reward = eval_results['avg_reward']
    baseline_reward = baseline_results['avg_reward']
    reward_improvement = ((agent_reward - baseline_reward) / abs(baseline_reward) * 100) if baseline_reward != 0 else 0
    
    print(f"{'Win Rate':<20} {agent_wr:<15.4f} {baseline_wr:<15.4f} {wr_improvement:<15.2f}%")
    print(f"{'Avg Reward':<20} {agent_reward:<15.4f} {baseline_reward:<15.4f} {reward_improvement:<15.2f}%")
    
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
