import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict
import pickle


class BasicStrategyAgent:
    """Basic Strategy agent for Blackjack with Q-value learning through simulation"""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

        # Q-table: Q(state, action) -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Returns tracking for Q-value updates
        self.returns = defaultdict(lambda: defaultdict(list))

        # Training stats
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_win_rates = []
        self.action_counts = defaultdict(int)

    def basic_strategy_action(
        self,
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

    def get_action(self, state: Tuple, valid_actions: List[int]) -> int:
        """Select action using basic strategy"""
        if not valid_actions:
            return 0
        
        player_sum, dealer_visible, usable_ace, can_split, can_double = state
        action = self.basic_strategy_action(
            player_sum, dealer_visible, usable_ace, can_split, can_double, valid_actions
        )
        
        # Ensure action is valid (fallback to hit if not)
        if action not in valid_actions:
            action = 0 if 0 in valid_actions else valid_actions[0]
        
        return action

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
        """Simulate one episode using basic strategy and update Q-values"""
        episode_data = []
        total_reward = 0

        # Reset environment
        state = env.reset()

        while True:
            # Get valid actions and select action using basic strategy
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
        """Train Q-values by simulating episodes with basic strategy"""
        print(f"Training Basic Strategy Q-values with {num_episodes:,} episodes...")
        
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
        
        print("Basic Strategy Q-value training completed!")

    def evaluate(self, env, num_episodes: int = 1000) -> Dict:
        """Evaluate the basic strategy"""
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
        """Get summary of basic strategy policy"""
        policy_actions = {}

        for state in self.q_table:
            if self.q_table[state]:
                best_action = max(self.q_table[state].items(), key=lambda x: x[1])[0]
                policy_actions[state] = best_action

        return {
            'states_learned': len(self.q_table),
            'policy_actions': policy_actions,
            'action_distribution': dict(self.action_counts)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the basic strategy model with Q-values"""
        model_data = {
            'q_table': dict(self.q_table),
            'returns': dict(self.returns),
            'episode_rewards': self.episode_rewards,
            'episode_wins': self.episode_wins,
            'episode_win_rates': self.episode_win_rates,
            'action_counts': dict(self.action_counts)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """Load a basic strategy model with Q-values"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
        self.returns = defaultdict(lambda: defaultdict(list), model_data['returns'])
        self.episode_rewards = model_data['episode_rewards']
        self.episode_wins = model_data['episode_wins']
        self.episode_win_rates = model_data['episode_win_rates']
        self.action_counts = defaultdict(int, model_data['action_counts'])

    def get_q_value(self, state: Tuple, action: int) -> float:
        """Get Q-value for a specific state-action pair"""
        return self.q_table[state][action]
    
    def print_policy_sample(self, num_states: int = 10) -> None:
        """Print a sample of the basic strategy policy with Q-values"""
        action_names = {
            0: 'hit',
            1: 'stand',
            2: 'split',
            3: 'double'
        }
        print(f"\nSample of Basic Strategy policy with Q-values ({num_states} states):")
        print("-" * 60)

        states_shown = 0
        for state in sorted(self.q_table.keys()):
            if states_shown >= num_states:
                break
            
            if self.q_table[state]:
                # Get the action that basic strategy would choose
                player_sum, dealer_visible, usable_ace, can_split, can_double = state
                valid_actions = [0, 1]  # Hit and Stand are always valid
                if can_split:
                    valid_actions.append(2)
                if can_double:
                    valid_actions.append(3)
                
                strategy_action = self.get_action(state, valid_actions)
                strategy_value = self.q_table[state][strategy_action]

                ace_str = "with usable ace" if usable_ace else "without usable ace"
                actions_str = []
                if can_split:
                    actions_str.append("split")
                if can_double:
                    actions_str.append("double")
                available_actions = ", ".join(actions_str) if actions_str else "basic actions only"

                print(f"Player: {player_sum:2d}, Dealer: {dealer_visible:2d}, {ace_str:15s}")
                print(f"  Available: {available_actions}")
                print(f"  Basic Strategy: {action_names[strategy_action]:6s} (Q={strategy_value:.3f})")
                print()

            states_shown += 1
