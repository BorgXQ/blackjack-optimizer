import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple, List, Dict
import pickle


class CSVStrategyAgent:
    """Agent that uses strategy loaded from a CSV file"""

    def __init__(self, csv_filepath: str, gamma: float = 1.0):
        self.gamma = gamma
        self.csv_filepath = csv_filepath
        
        # Strategy lookup table
        self.strategy_table = {}
        
        # Q-table for tracking performance (optional, for analysis)
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Returns tracking for Q-value updates
        self.returns = defaultdict(lambda: defaultdict(list))

        # Training stats
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_win_rates = []
        self.action_counts = defaultdict(int)
        
        # Load strategy from CSV
        self.load_strategy_from_csv()

    def load_strategy_from_csv(self):
        """Load strategy from CSV file"""
        print(f"Loading strategy from {self.csv_filepath}...")
        
        try:
            df = pd.read_csv(self.csv_filepath)
            
            # Action mapping from string to int
            action_mapping = {
                'Hit': 0,
                'Stand': 1, 
                'Split': 2,
                'Double': 3
            }
            
            for _, row in df.iterrows():
                state = (
                    int(row['player_sum']),
                    int(row['dealer_visible']),
                    bool(row['usable_ace']),
                    bool(row['can_split']),
                    bool(row['can_double'])
                )
                
                action_name = row['best_action']
                if action_name in action_mapping:
                    action = action_mapping[action_name]
                    self.strategy_table[state] = action
                    
                    # Also store the EV if available
                    if 'ev' in row:
                        self.q_table[state][action] = float(row['ev'])
            
            print(f"Loaded strategy for {len(self.strategy_table):,} states")
            
        except FileNotFoundError:
            print(f"Error: Could not find CSV file {self.csv_filepath}")
            raise
        except Exception as e:
            print(f"Error loading CSV strategy: {e}")
            raise

    def get_action(self, state: Tuple, valid_actions: List[int]) -> int:
        """Select action using CSV strategy"""
        if not valid_actions:
            return 0
        
        # Look up action in strategy table
        if state in self.strategy_table:
            preferred_action = self.strategy_table[state]
            
            # If preferred action is valid, use it
            if preferred_action in valid_actions:
                return preferred_action
        
        # Fallback strategy if state not found or action not valid
        # Use basic heuristics: hit if sum <= 11, stand if sum >= 17
        player_sum, dealer_visible, usable_ace, can_split, can_double = state
        
        if player_sum <= 11:
            return 0 if 0 in valid_actions else valid_actions[0]  # Hit
        elif player_sum >= 17:
            return 1 if 1 in valid_actions else valid_actions[0]  # Stand
        elif player_sum <= 16:
            # Hit against strong dealer cards (7-11), stand against weak (2-6)
            if dealer_visible >= 7:
                return 0 if 0 in valid_actions else valid_actions[0]  # Hit
            else:
                return 1 if 1 in valid_actions else valid_actions[0]  # Stand
        else:
            # Default to first valid action
            return valid_actions[0]

    def update_q_values(self, episode_data: List[Tuple]) -> None:
        """Update Q-values using every-visit Monte Carlo method"""
        # Calculate returns for each state-action pair
        G = 0
        for step in reversed(episode_data):
            state, action, reward = step
            G = self.gamma * G + reward
            
            # Every-visit: update for every occurrence of (state, action)
            self.returns[state][action].append(G)
            self.q_table[state][action] = np.mean(self.returns[state][action])

    def simulate_episode(self, env) -> Dict:
        """Simulate one episode using CSV strategy and update Q-values"""
        episode_data = []
        total_reward = 0

        # Reset environment
        state = env.reset()

        while True:
            # Get valid actions and select action using CSV strategy
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action = self.get_action(state, valid_actions)
            self.action_counts[action] += 1

            # Take action
            next_state, reward, done, info = env.step(action)

            # Store experience
            episode_data.append((state, action, reward))
            total_reward += reward

            if done:
                # Update stats
                wins = info.get('wins', 0)
                total_hands = info.get('total_hands', 1)
                win_rate = info.get('win_rate', 0)

                self.episode_rewards.append(total_reward)
                self.episode_wins.append(wins)
                self.episode_win_rates.append(win_rate)

                # Update Q-values
                self.update_q_values(episode_data)

                return {
                    'total_reward': total_reward,
                    'wins': wins,
                    'total_hands': total_hands,
                    'win_rate': win_rate
                }
            
            state = next_state

    def train_q_values(self, env, num_episodes: int = 50000, log_interval: int = 5000) -> None:
        """Train Q-values by simulating episodes with CSV strategy"""
        print(f"Training CSV Strategy Q-values with {num_episodes:,} episodes...")
        
        for episode in range(num_episodes):
            self.simulate_episode(env)
            
            # Log progress
            if (episode + 1) % log_interval == 0:
                recent_rewards = self.episode_rewards[-log_interval:]
                recent_win_rates = self.episode_win_rates[-log_interval:]
                
                avg_reward = np.mean(recent_rewards)
                avg_win_rate = np.mean(recent_win_rates)
                
                print(f"Episode {episode + 1:6,} | "
                      f"Avg Reward: {avg_reward:6.3f} | "
                      f"Win Rate: {avg_win_rate:6.3f}")
        
        print("CSV Strategy Q-value training completed!")

    def evaluate(self, env, num_episodes: int = 1000) -> Dict:
        """Evaluate the CSV strategy"""
        total_rewards = []
        total_wins = []
        total_hands_count = []

        for _ in range(num_episodes):
            episode_reward = 0
            state = env.reset()

            while True:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break

                action = self.get_action(state, valid_actions)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                if done:
                    total_rewards.append(episode_reward)
                    total_wins.append(info.get('wins', 0))
                    total_hands_count.append(info.get('total_hands', 1))
                    break

                state = next_state
    
        # Calculate stats
        avg_reward = np.mean(total_rewards)
        total_wins_sum = sum(total_wins)
        total_hands_sum = sum(total_hands_count)
        overall_win_rate = total_wins_sum / total_hands_sum if total_hands_sum > 0 else 0

        return {
            'avg_reward': avg_reward,
            'total_wins': total_wins_sum,
            'total_hands': total_hands_sum,
            'win_rate': overall_win_rate,
            'episode_rewards': total_rewards,
            'episode_wins': total_wins,
            'std_reward': np.std(total_rewards)
        }
    
    def get_policy_summary(self) -> Dict:
        """Get summary of CSV strategy policy"""
        policy_actions = dict(self.strategy_table)

        return {
            'states_learned': len(self.strategy_table),
            'policy_actions': policy_actions,
            'action_distribution': dict(self.action_counts)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the CSV strategy model with Q-values"""
        model_data = {
            'strategy_table': self.strategy_table,
            'q_table': dict(self.q_table),
            'returns': dict(self.returns),
            'episode_rewards': self.episode_rewards,
            'episode_wins': self.episode_wins,
            'episode_win_rates': self.episode_win_rates,
            'action_counts': dict(self.action_counts),
            'csv_filepath': self.csv_filepath
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """Load a CSV strategy model with Q-values"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.strategy_table = model_data['strategy_table']
        self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
        self.returns = defaultdict(lambda: defaultdict(list), model_data['returns'])
        self.episode_rewards = model_data['episode_rewards']
        self.episode_wins = model_data['episode_wins']
        self.episode_win_rates = model_data['episode_win_rates']
        self.action_counts = defaultdict(int, model_data['action_counts'])
        self.csv_filepath = model_data.get('csv_filepath', '')

    def get_q_value(self, state: Tuple, action: int) -> float:
        """Get Q-value for a specific state-action pair"""
        return self.q_table[state][action]
    
    def print_policy_sample(self, num_states: int = 10) -> None:
        """Print a sample of the CSV strategy policy"""
        action_names = {
            0: 'hit',
            1: 'stand',
            2: 'split',
            3: 'double'
        }
        print(f"\nSample of CSV Strategy policy ({num_states} states):")
        print("-" * 60)

        states_shown = 0
        for state in sorted(self.strategy_table.keys()):
            if states_shown >= num_states:
                break
            
            action = self.strategy_table[state]
            q_value = self.q_table[state][action] if self.q_table[state] else 0.0

            player_sum, dealer_visible, usable_ace, can_split, can_double = state
            ace_str = "with usable ace" if usable_ace else "without usable ace"
            actions_str = []
            if can_split:
                actions_str.append("split")
            if can_double:
                actions_str.append("double")
            available_actions = ", ".join(actions_str) if actions_str else "basic actions only"

            print(f"Player: {player_sum:2d}, Dealer: {dealer_visible:2d}, {ace_str:15s}")
            print(f"  Available: {available_actions}")
            print(f"  CSV Strategy: {action_names[action]:6s} (Q={q_value:.3f})")
            print()

            states_shown += 1