import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, List, Tuple
from collections import defaultdict

def plot_training_progress(training_stats: Dict):
    """Plot training progress over time"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = training_stats['episodes']
    
    # Plot 1: Average Reward
    ax1.plot(episodes, training_stats['avg_rewards'], 'b-', linewidth=2)
    ax1.set_title('Average Reward Over Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 2: Win Rate
    ax2.plot(episodes, training_stats['avg_win_rates'], 'g-', linewidth=2)
    ax2.set_title('Win Rate Over Training')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Epsilon Decay
    ax3.plot(episodes, training_stats['epsilons'], 'r-', linewidth=2)
    ax3.set_title('Epsilon Decay')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(training_stats['epsilons']) * 1.1)
    
    # Plot 4: Smoothed Reward (if enough data)
    if len(training_stats['avg_rewards']) > 10:
        window = min(10, len(training_stats['avg_rewards']) // 10)
        smoothed_rewards = np.convolve(training_stats['avg_rewards'], 
                                     np.ones(window)/window, mode='valid')
        smooth_episodes = episodes[window-1:]
        
        ax4.plot(episodes, training_stats['avg_rewards'], 'b-', alpha=0.3, label='Raw')
        ax4.plot(smooth_episodes, smoothed_rewards, 'b-', linewidth=2, label=f'Smoothed (window={window})')
        ax4.set_title('Smoothed Average Reward')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Reward')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    else:
        ax4.text(0.5, 0.5, 'Not enough data\nfor smoothing', 
                transform=ax4.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_statistics(results: Dict):
    """Print formatted evaluation statistics"""
    
    print(f"Average Reward: {results['avg_reward']:8.4f}")
    print(f"Std Deviation:  {results['std_reward']:8.4f}")
    print(f"Total Wins:     {results['total_wins']:8,}")
    print(f"Total Hands:    {results['total_hands']:8,}")
    print(f"Win Rate:       {results['win_rate']:8.4f} ({results['win_rate']*100:.2f}%)")
    
    if 'episode_rewards' in results:
        rewards = results['episode_rewards']
        print(f"Min Reward:     {min(rewards):8.4f}")
        print(f"Max Reward:     {max(rewards):8.4f}")
        print(f"Median Reward:  {np.median(rewards):8.4f}")

def analyze_policy(agent) -> Dict:
    """Analyze the learned policy"""
    
    action_names = {0: 'Hit', 1: 'Stand', 2: 'Split', 3: 'Double'}
    
    # Count actions by state characteristics
    policy_by_player_sum = defaultdict(lambda: defaultdict(int))
    policy_by_dealer_card = defaultdict(lambda: defaultdict(int))
    policy_by_ace = defaultdict(lambda: defaultdict(int))
    
    states_learned = 0
    
    for state in agent.q_table:
        if agent.q_table[state]:
            states_learned += 1
            
            # Get best action for this state
            best_action = max(agent.q_table[state].items(), key=lambda x: x[1])[0]
            
            player_sum, dealer_showing, usable_ace, can_split, can_double = state
            
            # Categorize by player sum
            policy_by_player_sum[player_sum][best_action] += 1
            
            # Categorize by dealer showing card
            policy_by_dealer_card[dealer_showing][best_action] += 1
            
            # Categorize by ace status
            ace_status = "with_ace" if usable_ace else "no_ace"
            policy_by_ace[ace_status][best_action] += 1
    
    # Create summary
    analysis = {
        'states_learned': states_learned,
        'policy_by_player_sum': dict(policy_by_player_sum),
        'policy_by_dealer_card': dict(policy_by_dealer_card),
        'policy_by_ace': dict(policy_by_ace)
    }
    
    return analysis

def plot_policy_heatmap(agent, save_path: str = 'policy_heatmap.png'):
    """Create a heatmap visualization of the learned policy"""
    
    action_names = {0: 'H', 1: 'S', 2: 'P', 3: 'D'}
    
    # Create matrices for different scenarios
    basic_policy = np.full((20, 10), -1, dtype=int)  # Player sum 2-21, Dealer 2-11
    ace_policy = np.full((20, 10), -1, dtype=int)    # With usable ace
    
    for state in agent.q_table:
        if agent.q_table[state]:
            player_sum, dealer_showing, usable_ace, can_split, can_double = state
            
            # Only consider basic actions for heatmap clarity
            basic_actions = {k: v for k, v in agent.q_table[state].items() if k in [0, 1]}
            if not basic_actions:
                continue
                
            best_action = max(basic_actions.items(), key=lambda x: x[1])[0]
            
            # Map to matrix indices
            if 2 <= player_sum <= 21 and 2 <= dealer_showing <= 11:
                row = player_sum - 2
                col = dealer_showing - 2
                
                if usable_ace:
                    ace_policy[row, col] = best_action
                else:
                    basic_policy[row, col] = best_action
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot basic strategy (no usable ace)
    im1 = ax1.imshow(basic_policy, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Basic Strategy (No Usable Ace)')
    ax1.set_xlabel('Dealer Showing')
    ax1.set_ylabel('Player Sum')
    ax1.set_xticks(range(10))
    ax1.set_xticklabels(range(2, 12))
    ax1.set_yticks(range(0, 20, 2))
    ax1.set_yticklabels(range(2, 22, 2))
    
    # Add text annotations
    for i in range(20):
        for j in range(10):
            if basic_policy[i, j] != -1:
                text = action_names[basic_policy[i, j]]
                ax1.text(j, i, text, ha='center', va='center', 
                        color='white' if basic_policy[i, j] == 0 else 'black', fontweight='bold')
    
    # Plot ace strategy (with usable ace)
    im2 = ax2.imshow(ace_policy, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Strategy with Usable Ace')
    ax2.set_xlabel('Dealer Showing')
    ax2.set_ylabel('Player Sum')
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(range(2, 12))
    ax2.set_yticks(range(0, 20, 2))
    ax2.set_yticklabels(range(2, 22, 2))
    
    # Add text annotations
    for i in range(20):
        for j in range(10):
            if ace_policy[i, j] != -1:
                text = action_names[ace_policy[i, j]]
                ax2.text(j, i, text, ha='center', va='center', 
                        color='white' if ace_policy[i, j] == 0 else 'black', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im1, ax=[ax1, ax2], shrink=0.6)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Hit', 'Stand'])
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Hit'),
                      plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='Stand')]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def compare_to_basic_strategy(agent) -> Dict:
    """Compare learned policy to basic blackjack strategy"""
    
    # Basic strategy rules (simplified)
    def basic_strategy_action(player_sum: int, dealer_showing: int, usable_ace: bool) -> int:
        """Return basic strategy action (0=Hit, 1=Stand)"""
        
        if usable_ace:
            # Soft totals (with usable ace)
            if player_sum >= 19:
                return 1  # Stand
            elif player_sum <= 17:
                return 0  # Hit
            else:  # 18
                return 1 if dealer_showing in [2, 7, 8] else 0
        else:
            # Hard totals (no usable ace)
            if player_sum >= 17:
                return 1  # Stand
            elif player_sum <= 11:
                return 0  # Hit
            elif player_sum >= 13:
                return 1 if dealer_showing <= 6 else 0
            else:  # 12
                return 1 if 4 <= dealer_showing <= 6 else 0
    
    agreements = 0
    disagreements = 0
    comparisons = []
    
    for state in agent.q_table:
        if agent.q_table[state]:
            player_sum, dealer_showing, usable_ace, can_split, can_double = state
            
            # Only compare basic actions (hit/stand)
            basic_actions = {k: v for k, v in agent.q_table[state].items() if k in [0, 1]}
            if not basic_actions:
                continue
            
            learned_action = max(basic_actions.items(), key=lambda x: x[1])[0]
            basic_action = basic_strategy_action(player_sum, dealer_showing, usable_ace)
            
            if learned_action == basic_action:
                agreements += 1
            else:
                disagreements += 1
                comparisons.append({
                    'state': state,
                    'learned': learned_action,
                    'basic': basic_action
                })
    
    total_comparisons = agreements + disagreements
    agreement_rate = agreements / total_comparisons if total_comparisons > 0 else 0
    
    return {
        'agreements': agreements,
        'disagreements': disagreements,
        'total_comparisons': total_comparisons,
        'agreement_rate': agreement_rate,
        'disagreement_examples': comparisons[:10]  # First 10 disagreements
    }

def export_learned_strategy_csv(agent, filename: str = "learned_blackjack_strategy.csv"):
    """Export the learned strategy to a CSV file"""
    
    action_names = {0: 'Hit', 1: 'Stand', 2: 'Split', 3: 'Double'}
    
    # Prepare data for CSV
    csv_data = []
    
    for state in agent.q_table:
        if agent.q_table[state]:  # Only include states with Q-values
            player_sum, dealer_visible, usable_ace, can_split, can_double = state
            
            # Find best action and its Q-value
            best_action_idx = max(agent.q_table[state].items(), key=lambda x: x[1])[0]
            best_q_value = agent.q_table[state][best_action_idx]
            best_action_name = action_names[best_action_idx]
            
            # Convert Q-value to a more interpretable success likelihood
            # Since Q-values represent expected returns, we can normalize them
            # Higher Q-values = better expected outcomes
            success_likelihood = best_q_value
            
            csv_data.append({
                'player_sum': player_sum,
                'dealer_visible': dealer_visible,
                'usable_ace': usable_ace,
                'can_split': can_split,
                'can_double': can_double,
                'best_action': best_action_name,
                'ev': round(success_likelihood, 4)
            })
    
    # Sort by state components for better readability
    csv_data.sort(key=lambda x: (x['player_sum'], x['dealer_visible'], 
                                 not x['usable_ace'], not x['can_split'], not x['can_double']))
    
    # Write to CSV
    fieldnames = ['player_sum', 'dealer_visible', 'usable_ace', 'can_split', 
                  'can_double', 'best_action', 'ev']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\nLearned strategy exported to {filename}")
    print(f"Total states learned: {len(csv_data):,}")
    
    # Provide some summary statistics
    action_counts = defaultdict(int)
    for row in csv_data:
        action_counts[row['best_action']] += 1
    
    print("\nAction distribution in learned strategy:")
    for action, count in sorted(action_counts.items()):
        percentage = (count / len(csv_data)) * 100
        print(f"  {action:<8}: {count:>4,} states ({percentage:5.1f}%)")
    
    return filename

def save_results_summary(agent, eval_results: Dict, baseline_results: Dict, 
                        basic_strategy_results: Dict, training_stats: Dict, 
                        filepath: str = 'results_summary.txt'):
    """Save a comprehensive summary of results to file"""
    
    with open(filepath, 'w') as f:
        f.write("BLACKJACK MONTE CARLO SIMULATION RESULTS\n")
        f.write("="*50 + "\n\n")
        
        # Training info
        f.write("TRAINING CONFIGURATION\n")
        f.write("-"*25 + "\n")
        f.write(f"Episodes trained: {len(agent.episode_rewards):,}\n")
        f.write(f"Final epsilon: {agent.epsilon:.6f}\n")
        f.write(f"States learned: {len(agent.q_table):,}\n\n")
        
        # Performance comparison
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-"*25 + "\n")
        f.write(f"{'Metric':<20} {'Trained Agent':<15} {'Basic Strategy':<15} {'Random Baseline':<15}\n")
        f.write("-"*65 + "\n")
        f.write(f"{'Win Rate':<20} {eval_results['win_rate']:<15.4f} {basic_strategy_results['win_rate']:<15.4f} {baseline_results['win_rate']:<15.4f}\n")
        f.write(f"{'Avg Reward':<20} {eval_results['avg_reward']:<15.4f} {basic_strategy_results['avg_reward']:<15.4f} {baseline_results['avg_reward']:<15.4f}\n")
        f.write(f"{'Std Reward':<20} {eval_results['std_reward']:<15.4f} {basic_strategy_results['std_reward']:<15.4f} {baseline_results['std_reward']:<15.4f}\n\n")
        
        # Action distribution
        f.write("ACTION DISTRIBUTION\n")
        f.write("-"*20 + "\n")
        action_names = {0: 'Hit', 1: 'Stand', 2: 'Split', 3: 'Double'}
        total_actions = sum(agent.action_counts.values())
        
        for action, count in agent.action_counts.items():
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            f.write(f"{action_names[action]:<10}: {count:>8,} ({percentage:5.1f}%)\n")
        
        f.write(f"\nTotal actions: {total_actions:,}\n")
    
    print(f"Results summary saved to {filepath}")
