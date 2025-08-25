import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from typing import Dict
from collections import defaultdict

def plot_training_progress(training_stats: Dict):
    """Plot training progress over time"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = training_stats["episodes"]
    
    # Plot 1: Average Reward
    ax1.plot(episodes, training_stats["avg_rewards"], "b-", linewidth=2)
    ax1.set_title("Average Reward Over Training")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Reward")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    
    # Plot 2: Win Rate
    ax2.plot(episodes, training_stats["avg_win_rates"], "g-", linewidth=2)
    ax2.set_title("Win Rate Over Training")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Rate")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Epsilon Decay
    ax3.plot(episodes, training_stats["epsilons"], "r-", linewidth=2)
    ax3.set_title("Epsilon Decay")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Epsilon")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(training_stats["epsilons"]) * 1.1)
    
    # Plot 4: Smoothed Reward (if enough data)
    if len(training_stats["avg_rewards"]) > 10:
        window = min(10, len(training_stats["avg_rewards"]) // 10)
        smoothed_rewards = np.convolve(training_stats["avg_rewards"], 
                                     np.ones(window)/window, mode="valid")
        smooth_episodes = episodes[window-1:]
        
        ax4.plot(episodes, training_stats["avg_rewards"], "b-", alpha=0.3, label="Raw")
        ax4.plot(smooth_episodes, smoothed_rewards, "b-", linewidth=2, label=f"Smoothed (window={window})")
        ax4.set_title("Smoothed Average Reward")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Average Reward")
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    else:
        ax4.text(0.5, 0.5, "Not enough data\nfor smoothing", 
                transform=ax4.transAxes, ha="center", va="center")
    
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150, bbox_inches="tight")
    # plt.show()

def print_statistics(results: Dict):
    """Print formatted evaluation statistics"""
    
    print(f"Average Reward: {results['avg_reward']:8.4f}")
    print(f"Std Deviation:  {results['std_reward']:8.4f}")
    print(f"Total Wins:     {results['total_wins']:8,}")
    print(f"Total Hands:    {results['total_hands']:8,}")
    print(f"Win Rate:       {results['win_rate']:8.4f} ({results['win_rate']*100:.2f}%)")

    if "episode_rewards" in results:
        rewards = results["episode_rewards"]
        print(f"Min Reward:     {min(rewards):8.4f}")
        print(f"Max Reward:     {max(rewards):8.4f}")
        print(f"Median Reward:  {np.median(rewards):8.4f}")

def analyze_policy(agent) -> Dict:
    """Analyze the learned policy"""
    
    action_names = {0: "Hit", 1: "Stand", 2: "Split", 3: "Double"}
    
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
        "states_learned": states_learned,
        "policy_by_player_sum": dict(policy_by_player_sum),
        "policy_by_dealer_card": dict(policy_by_dealer_card),
        "policy_by_ace": dict(policy_by_ace)
    }
    
    return analysis

def get_strategy_df(agent, filename: str = "learned_blackjack_strategy.csv", export=False):
    """Return strategy as a dataframe and optionally export to a CSV file"""
    action_names = {0: "Hit", 1: "Stand", 2: "Split", 3: "Double"}
    csv_data = []
    
    for state in agent.q_table:
        if agent.q_table[state]:  # Only include states with Q-values
            player_sum, dealer_visible, usable_ace, can_split, can_double = state
            
            best_action_idx = max(agent.q_table[state].items(), key=lambda x: x[1])[0]
            best_q_value = agent.q_table[state][best_action_idx]
            best_action_name = action_names[best_action_idx]
            
            csv_data.append({
                "player_sum": player_sum,
                "dealer_visible": dealer_visible,
                "usable_ace": usable_ace,
                "can_split": can_split,
                "can_double": can_double,
                "best_action": best_action_name,
                "ev": round(best_q_value, 4)
            })
    
    # Sort by state components for better readability
    csv_data.sort(key=lambda x: (x["player_sum"], x["dealer_visible"], 
                                 not x["usable_ace"], not x["can_split"], not x["can_double"]))
    
    if export:
        # Write to CSV
        fieldnames = ["player_sum", "dealer_visible", "usable_ace", "can_split", 
                    "can_double", "best_action", "ev"]
        
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\nLearned strategy exported to {filename}")
        print(f"Total states learned: {len(csv_data):,}")

        return filename, pd.DataFrame(csv_data)
    else:
        return pd.DataFrame(csv_data)
