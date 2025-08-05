import numpy as np
import random
from collections import defaultdict
from typing import Tuple, List, Dict
from config import EPSILON, EPSILON_DECAY, EPSILON_MIN
import pickle


class MonteCarloAgent:
    """Every-visit Monte Carlo agent for Blackjack"""

    def __init__(
        self,
        epsilon: float=EPSILON,
        epsilon_decay: float=EPSILON_DECAY,
        epsilon_min: float=EPSILON_MIN,
        gamma: float=1.0, # No discounting for episodic tasks
    ):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        # Q-table: Q(state, action) -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Returns tracking for Monte Carlo updates
        self.returns = defaultdict(lambda: defaultdict(list))

        # Training stats
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_win_rates = []
        self.action_counts = defaultdict(int)

    def get_action(self, state: Tuple, valid_actions: List[int], training: bool=True) -> int:
        """Select action using epsilon-greedy policy with action masking"""
        if not valid_actions:
            return 0
        
        # Epsilon-greedy during training, greedy during evaluation
        if training and random.random() < self.epsilon:
            # Random action from valid actions
            return random.choice(valid_actions)
        else:
            # Greedy action from valid actions
            q_values = [self.q_table[state][action] for action in valid_actions]
            max_q = max(q_values)

            # Handle ties by random selection among best actions
            best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)
        
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

    def train_episode(self, env) -> Dict:
        """Train for one episode and return stats"""
        episode_data = []
        total_reward = 0

        # Reset environment
        state = env.reset()

        while True:
            # Get valid actions and select action
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action = self.get_action(state, valid_actions, training=True)
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

                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                return {
                    'total_reward': total_reward,
                    'wins': wins,
                    'total_hands': total_hands,
                    'win_rate': win_rate,
                    'epsilon': self.epsilon
                }
            
            state = next_state

    def evaluate(self, env, num_episodes: int=1000) -> Dict:
        """Evaluate the learned policy"""
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

                # Use greedy policy (no exploration)
                action = self.get_action(state, valid_actions, training=False)
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
        """Get summary of learned policy"""
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
        """Save the trained model"""
        model_data = {
            'q_table': dict(self.q_table),
            'returns': dict(self.returns),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_wins': self.episode_wins,
            'episode_win_rates': self.episode_win_rates,
            'action_counts': dict(self.action_counts)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
        self.returns = defaultdict(lambda: defaultdict(list), model_data['returns'])
        self.epsilon = model_data['epsilon']
        self.episode_rewards = model_data['episode_rewards']
        self.episode_wins = model_data['episode_wins']
        self.episode_win_rates = model_data['episode_win_rates']
        self.action_counts = defaultdict(int, model_data['action_counts'])

    def get_q_value(self, state: Tuple, action: int) -> float:
        """Get Q-value for a specific state-action pair"""
        return self.q_table[state][action]
    
    def print_policy_sample(self, num_states: int=10) -> None:
        """Print a sample of the learned policy"""
        action_names = {
            0: 'hit',
            1: 'stand',
            2: 'split',
            3: 'double'
        }
        print(f"\nSample of learned policy ({num_states} states):")
        print("-" * 60)

        states_shown = 0
        for state in sorted(self.q_table.keys()):
            if states_shown >= num_states:
                break
            
            if self.q_table[state]:
                best_action = max(self.q_table[state].items(), key=lambda x: x[1])[0]
                best_value = self.q_table[state][best_action]

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
                print(f"  Best action: {action_names[best_action]:6s} (Q={best_value:.3f})")
                print()

            states_shown += 1
