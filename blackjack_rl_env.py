from typing import Tuple, List, Dict
from blackjack_env import BlackjackGame, Hand


class BlackjackRLEnv:
    """RL Environment wrapper for Blackjack"""

    def __init__(self, starting_balance: int=100000, fixed_bet: int=10):
        self.starting_balance = starting_balance
        self.fixed_bet = fixed_bet
        self.game = BlackjackGame()
        self.current_hand_index = 0
        self.episode_hands = [] # Track all hands in episode for win counting
        self.episode_rewards = [] # Track rewards for each hand

        # Action mapping
        self.actions = {
            0: 'hit',
            1: 'stand',
            2: 'split',
            3: 'double'
        }

    def reset(self) -> Tuple[int, int, bool, bool, bool]:
        """Reset environment and return initial state"""
        self.game = BlackjackGame()
        self.game.player_balance = self.starting_balance
        self.current_hand_index = 0
        self.episode_hands = []
        self.episode_rewards = []

        # Place bet and deal initial cards
        self.game.place_bet(self.fixed_bet)
        self.game.deal_initial_cards()

        return self._get_state()
    
    def _get_state(self) -> Tuple[int, int, bool, bool, bool]:
        """Get current state of the environment"""
        if self.current_hand_index >= len(self.game.player_hands):
            # Episode ended
            return (0, 0, False, False, False)
        
        current_hand = self.game.player_hands[self.current_hand_index]
        player_sum = current_hand.get_value()

        # Get dealer's visible card
        dealer_visible = self.game.dealer_hand.cards[0].value()
        if dealer_visible > 10: # Ace
            dealer_visible = 11

        # Check for usable ace in player hand
        usable_ace = self._has_usable_ace(current_hand)

        # Check available actions
        can_split = (
            self.current_hand_index == 0 and self.game.can_split()
        )
        can_double = (
            self.game.can_double_down()
        )

        return (
            player_sum,
            dealer_visible,
            usable_ace,
            can_split,
            can_double
        )

    def _has_usable_ace(self, hand: Hand) -> bool:
        """Check if the hand has a usable ace (value 11)"""
        total = 0
        aces = 0

        for card in hand.cards:
            if card.rank == 'A':
                aces += 1
            else:
                total += card.value()

        if aces > 0 and total + aces + 10 <= 21:
            return True
        return False
    
    def get_valid_actions(self) -> List[int]:
        """Get valid actions for the current state"""
        if self.current_hand_index >= len(self.game.player_hands):
            return []
        
        current_hand = self.game.player_hands[self.current_hand_index]

        # If hand is busted, no actions available
        if current_hand.is_busted():
            return []
        
        valid_actions = [0, 1]  # Hit and Stand

        # Check if split is available
        if self.current_hand_index == 0 and self.game.can_split():
            valid_actions.append(2) # Split
        
        # Check if double down is available
        if self.game.can_double_down(self.current_hand_index):
            valid_actions.append(3) # Double Down

        return valid_actions
    
    def step(self, action: int) -> Tuple[Tuple[int, int, bool, bool, bool], float, bool, Dict]:
        """Take action and return next state, reward, done, and info"""
        if self.current_hand_index >= len(self.game.player_hands):
            return self._get_state(), 0.0, True, {}
        
        current_hand = self.game.player_hands[self.current_hand_index]
        reward = 0.0

        # Convert action index to action
        action_name = self.actions[action]

        if action_name == 'hit':
            self.game.player_hit(self.current_hand_index)
            if current_hand.is_busted():
                reward -= 0.1 # Small penalty for busting
            elif current_hand.get_value() == 21:
                reward += 0.05 # Small bonus for hitting 21

        elif action_name == 'stand':
            pass

        elif action_name == 'split':
            if self.game.split_hand():
                reward += 0.02 # Small bonus for valid split
            else:
                reward -= 1.0 # Penalty for invalid action

        elif action_name == 'double':
            if self.game.double_down(self.current_hand_index):
                reward += 0.02 # Small bonus for valid double down
                if current_hand.is_busted():
                    reward -= 0.1 # Small penalty for busting after double down
            else:
                reward -= 1.0 # Penalty for invalid action

        # if action_name == 'hit':
        #     self.game.player_hit(self.current_hand_index)
                
        # elif action_name == 'stand':
        #     # No immediate reward for standing
        #     pass
            
        # elif action_name == 'split':
        #     if not self.game.split_hand():
        #         # This should not happen with proper action masking
        #         reward -= 1.0  # Large penalty for invalid action
                
        # elif action_name == 'double':
        #     if not self.game.double_down(self.current_hand_index):
        #         # This should not happen with proper action masking
        #         reward -= 1.0  # Large penalty for invalid action

        # Check if current hand is done (busted, stood, or doubled)
        hand_done = (
            current_hand.is_busted() or action_name == 'stand' or action_name == 'double'
        )

        # If current hand is done, move to next hand or end episode
        if hand_done:
            self.current_hand_index += 1

            # If all hands are done, resolve dealer and episode
            if self.current_hand_index >= len(self.game.player_hands):
                return self._resolve_episode(reward)
        
        next_state = self._get_state()
        done = self.current_hand_index >= len(self.game.player_hands)

        return  next_state, reward, done, {}
    
    def _resolve_episode(self, current_reward: float) -> Tuple[Tuple[int, int, bool, bool, bool], float, bool, Dict]:
        """Resolve the episode by playing dealer and calculating final rewards"""
        # Play dealer's hand
        any_player_active = any(not hand.is_busted() for hand in self.game.player_hands)
        if any_player_active:
            self.game.dealer_play()

        # Determine winners and calculate rewards
        results = self.game.determine_winner()

        total_rewards = current_reward
        wins = 0
        total_hands = len(results)

        for i, result in enumerate(results):
            if result == "player" or result == "player_blackjack":
                if result == "player_blackjack":
                    total_rewards += 1.5 # 3:2 payout bonus
                else:
                    total_rewards += 1.0 # Regular win
                wins += 1
            elif result == "dealer":
                total_rewards -= 1.0 # Loss
            # Push (tie) contributes 0 to reward
        
        # Store episode stats
        info = {
            'wins': wins,
            'total_hands': total_hands,
            'win_rate': wins / total_hands if total_hands > 0 else 0,
            'results': results
        }

        return self._get_state(), total_rewards, True, info
    
    def render(self):
        """Print current game state"""
        if self.current_hand_index < len(self.game.player_hands):
            print(f"\nCurrent hand {self.current_hand_index + 1}: {self.game.player_hands[self.current_hand_index]}")
            print(f"Dealer's visible card: {self.game.dealer_hand.cards[0]}")
            print(f"Valid actions: {[self.actions[i] for i in self.get_valid_actions()]}")
        else:
            print("Episode finished.")
