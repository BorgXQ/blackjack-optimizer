import random
from typing import List, Optional


class Card:
    """Represents a playing card"""

    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank} of {self.suit}"
    
    def value(self) -> int:
        """Returns blackjack value of a card"""
        if self.rank in ['J', 'Q', 'K']:
            return 10
        if self.rank == 'A':
            return 1 # Initialize to 1, adjusted later
        else:
            return int(self.rank)


class Deck:
    """Represents a deck of playing cards"""

    def __init__(self, num_decks: int=6):
        self.cards: List[Card] = []
        self.num_decks = num_decks
        self.reset_deck()

    def reset_deck(self):
        """Creates a fresh deck(s) of cards"""
        self.cards = []
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

        for _ in range(self.num_decks):
            for suit in suits:
                for rank in ranks:
                    self.cards.append(Card(suit, rank))

        self.shuffle()

    def shuffle(self):
        """Shuffles the deck"""
        random.shuffle(self.cards)

    def deal_card(self) -> Optional[Card]:
        """Deals one card from the deck"""
        if len(self.cards) == 0:
            return None
        return self.cards.pop()
    

class Hand:
    """Represents a hand of cards"""

    def __init__(self):
        self.cards: List[Card] = []
        self.is_dealer = False

    def add_card(self, card: Card):
        """Adds a card to the hand"""
        self.cards.append(card)

    def get_value(self) -> int:
        """Calculates the total value of the hand and handles Aces"""
        total = 0
        aces = 0

        for card in self.cards:
            if card.rank == 'A':
                aces += 1
            else:
                total += card.value()
        
        # Add aces as 1 first
        total += aces

        # Adjust for Aces if bust
        if aces > 0 and total+10 <= 21:
            total += 10

        return total

    def is_busted(self) -> bool:
        """Checks if the hand is over 21"""
        return self.get_value() > 21
    
    def is_blackjack(self) -> bool:
        """Checks if the hand is 21"""
        return len(self.cards) == 2 and self.get_value() == 21
    
    def clear(self):
        """Clears all cards from hand"""
        self.cards = []

    def __str__(self):
        if self.is_dealer and len(self.cards) > 1:
            # Show only first card for dealer during play
            return f"[{self.cards[0]}, Hidden Card]"
        else:
            cards_str = ", ".join(str(card) for card in self.cards)
            # Show value only if not a hidden dealer hand
            value_str = f" (Value: {self.get_value()})" if not (self.is_dealer and len(self.cards) > 1) else ""
            return f"[{cards_str}]{value_str}"


class BlackjackGame:
    """Main Blackjack game class"""
    def __init__(self, num_decks: int=6):
        self.deck = Deck(num_decks)
        self.player_hands = [Hand()] # List to support splitting
        self.dealer_hand = Hand()
        self.dealer_hand.is_dealer = True
        self.game_over = False
        self.player_balance = 1000 # Starting balance
        self.current_bet = 0
        self.split_bets = [] # Track bets for each hand
        self.has_split = False

    def place_bet(self, amount: int) -> bool:
        """Places a bet for the current hand"""
        if amount <= 0:
            print("Bet must be positive!")
            return False
        if amount > self.player_balance:
            print(f"Insufficient funds! You have {self.player_balance}")
            return False
        
        self.current_bet = amount
        self.player_balance -= amount
        self.split_bets = [amount]
        return True
    
    def deal_initial_cards(self):
        """Deals the initial two cards to player and dealer"""
        if len(self.deck.cards) < 208: # Reshuffle if deck is running low
            print("Reshuffling deck...")
            self.deck.reset_deck()

        # Clear previous hands
        self.player_hands = [Hand()]
        self.dealer_hand.clear()
        self.dealer_hand.is_dealer = True
        self.game_over = False
        self.has_split = False

        # Deal two cards to each
        for _ in range(2):
            self.player_hands[0].add_card(self.deck.deal_card())
            self.dealer_hand.add_card(self.deck.deal_card())

    def can_split(self, hand_index: int=0) -> bool:
        """Checks if player can split hand"""
        if self.has_split:
            return False
        if len(self.player_hands[hand_index].cards) != 2:
            return False
        if self.current_bet > self.player_balance:
            return False
        
        # Check if both cards have same values
        card1 = self.player_hands[hand_index].cards[0]
        card2 = self.player_hands[hand_index].cards[1]
        return card1.value() == card2.value()
    
    def can_double_down(self, hand_index: int=0) -> bool:
        """Checks if player can double down on hand"""
        if len(self.player_hands[hand_index].cards) != 2:
            return False
        bet_for_hand = self.split_bets[hand_index] if hand_index < len(self.split_bets) else self.current_bet
        return bet_for_hand <= self.player_balance
    
    def double_down(self, hand_index: int=0) -> bool:
        """Doubles down on a hand"""
        if not self.can_double_down(hand_index):
            return False
        
        # Double bet for hand
        bet_for_hand = self.split_bets[hand_index] if hand_index < len(self.split_bets) else self.current_bet
        self.player_balance -= bet_for_hand
        self.split_bets[hand_index] = bet_for_hand * 2

        # Deal exactly one card
        card = self.deck.deal_card()
        if card:
            self.player_hands[hand_index].add_card(card)
            print(f"You drew {card} (doubled down)")

        return True
    
    def split_hand(self, hand_index: int=0) -> bool:
        """Splits hand into two"""
        if not self.can_split(hand_index):
            return False
        
        # Deduct additional bet
        self.player_balance -= self.current_bet
        self.split_bets.append(self.current_bet)
        self.has_split = True
        
        # Split the hand
        original_hand = self.player_hands[hand_index]
        card1 = original_hand.cards[0]
        card2 = original_hand.cards[1]
        
        # Create two new hands and distribute the card pair
        hand1 = Hand()
        hand2 = Hand()
        hand1.add_card(card1)
        hand2.add_card(card2)
        
        # Replace original hand with split hands
        self.player_hands = [hand1, hand2]
        
        return True

    def player_hit(self, hand_index: int=0) -> bool:
        """Player takes another card"""
        if self.game_over:
            return False
        
        card = self.deck.deal_card()
        if card:
            self.player_hands[hand_index].add_card(card)
            print(f"You drew {card}")

            if self.player_hands[hand_index].is_busted():
                print(f"Hand {hand_index + 1} busted!")
                return False
            
        return True

    def dealer_play(self):
        """Dealer hits until >=17"""
        print("\nDealer reveals hidden card...")
        self.dealer_hand.is_dealer = False # Show all cards
        print(f"Dealer's hand: {self.dealer_hand}")

        while self.dealer_hand.get_value() < 17:
            card = self.deck.deal_card()
            if card:
                self.dealer_hand.add_card(card)
                print(f"Dealer draws {card}")
                print(f"Dealer's hand: {self.dealer_hand}")

            if self.dealer_hand.is_busted():
                print("Dealer busted!")
                break

    def determine_winner(self) -> str:
        """Determines the winner and handles payouts"""
        results = []
        dealer_value = self.dealer_hand.get_value()
        dealer_busted = self.dealer_hand.is_busted()
        dealer_bj = self.dealer_hand.is_blackjack()

        for i, hand in enumerate(self.player_hands):
            player_value = hand.get_value()
            player_busted = hand.is_busted()
            player_bj = hand.is_blackjack()
            bet = self.split_bets[i]
            
            # Player busted
            if player_busted:
                results.append("dealer")
                continue
            
            # Dealer busted
            if dealer_busted:
                self.player_balance += bet * 2
                results.append("player")
                continue
            
            # Check for blackjacks (only possible if not split)
            if not self.has_split:
                if player_bj and dealer_bj:
                    self.player_balance += bet  # Push
                    results.append("push")
                    continue
                elif player_bj:
                    self.player_balance += int(bet * 2.5)  # 3:2 payout
                    results.append("player_blackjack")
                    continue
                elif dealer_bj:
                    results.append("dealer")
                    continue
            
            # Compare values
            if player_value > dealer_value:
                self.player_balance += bet * 2
                results.append("player")
            elif dealer_value > player_value:
                results.append("dealer")
            else:
                self.player_balance += bet  # Push
                results.append("push")
        
        return results
        
    def play_hand(self, bet_amount: int=10):
        """Plays a complete hand of blackjack"""
        print(f"\n{'='*50}")
        print(f"NEW HAND - Balance: {self.player_balance}")
        print(f"{'='*50}")

        # Place bet
        if not self.place_bet(bet_amount):
            return
        
        # Deal initial cards
        self.deal_initial_cards()

        print(f"\nYour hand: {self.player_hands[0]}")
        print(f"Dealer's hand: {self.dealer_hand}")

        # Check for immediate blackjacks
        player_bj = self.player_hands[0].is_blackjack()
        dealer_bj = self.dealer_hand.is_blackjack()
        
        if player_bj or dealer_bj:
            if player_bj:
                print("Blackjack!")
            if dealer_bj:
                print("Dealer has blackjack!")
            self.dealer_hand.is_dealer = False # Show all cards
            print(f"Dealer's hand: {self.dealer_hand}")
            result = self.determine_winner()
            self.print_result(result)
            return
        
        # Player's turn for each hand
        hand_idx = 0
        while hand_idx < len(self.player_hands):
            if len(self.player_hands) > 1:
                print(f"\n--- Playing Hand {hand_idx + 1} ---")
            
            while True:
                current_hand = self.player_hands[hand_idx]
                print(f"Hand {hand_idx + 1}: {current_hand}")
                
                # Check if hand is busted
                if current_hand.is_busted():
                    break
                
                # Build action prompt
                actions = "Hit (h) or Stand (s)"
                if hand_idx == 0 and self.can_split() and not self.has_split:
                    actions += " or Split (p)"
                if self.can_double_down(hand_idx):
                    actions += " or Double Down (d)"
                actions += "? "
                
                action = input(actions).lower().strip()
                
                if action == 'h':
                    if not self.player_hit(hand_idx):
                        break
                elif action == 's':
                    print(f"You stand on hand {hand_idx + 1}.")
                    break
                elif action == 'd' and self.can_double_down(hand_idx):
                    if self.double_down(hand_idx):
                        print(f"Hand {hand_idx + 1}: {current_hand}")
                        if current_hand.is_busted():
                            print(f"Hand {hand_idx + 1} busted!")
                        break # Can't take more actions after double down
                    else:
                        print("Cannot double down.")
                elif action == 'p' and hand_idx == 0 and self.can_split() and not self.has_split:
                    if self.split_hand():
                        print("Hand split!")
                        print(f"Hand 1: {self.player_hands[0]}")
                        print(f"Hand 2: {self.player_hands[1]}")
                        continue
                    else:
                        print("Cannot split.")
                else:
                    valid_actions = "'h' for hit, 's' for stand"
                    if hand_idx == 0 and self.can_split() and not self.has_split:
                        valid_actions += ", 'p' for split"
                    if self.can_double_down(hand_idx):
                        valid_actions += ", 'd' for double down"
                    print(f"Invalid input. Please enter {valid_actions}.")
                    
            hand_idx += 1
        
        # Dealer's turn (if at least one hand didn't bust)
        any_player_active = any(not hand.is_busted() for hand in self.player_hands)
        if any_player_active:
            self.dealer_play()
        
        # Determine winners
        results = self.determine_winner()
        self.print_result(results)

    def print_result(self, results: str):
        """Prints the game result"""
        print("\nFinal hands:")
        self.dealer_hand.is_dealer = False # Show all cards

        for i, hand in enumerate(self.player_hands):
            print(f"Your hand {i+1}: {hand}")
        print(f"Dealer's hand: {self.dealer_hand}")

        for i, result in enumerate(results):
            hand_num = f" (Hand {i + 1})" if len(results) > 1 else ""
            if result == "player":
                print(f"You win{hand_num}!")
            elif result == "player_blackjack":
                print(f"Blackjack! You win{hand_num}!")
            elif result == "dealer":
                print(f"Dealer wins{hand_num}!")
            elif result == "push":
                print(f"Push{hand_num}! It's a tie!")

        print(f"Balance: ${self.player_balance}")

    def play_game(self):
        """Main game loop"""
        print("Welcome to Blackjack!")
        print(f"Starting balance: ${self.player_balance}")

        while self.player_balance > 0:
            try:
                bet = int(input(f"\nEnter your bet (max ${self.player_balance}), or 0 to quit: "))
                if bet == 0:
                    break

                self.play_hand(bet)

                if self.player_balance <= 0:
                    print("You're out of money! Game over!")
                    break

            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("Thanks for playing!")
                break

        print(f"Final balance: ${self.player_balance}")


if __name__ == "__main__":
    game = BlackjackGame()
    game.play_game()
