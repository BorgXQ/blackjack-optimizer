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
            return 11 # Initialize to 11, adjusted later
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
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.dealer_hand.is_dealer = True
        self.game_over = False
        self.player_balance = 1000 # Starting balance
        self.current_bet = 0

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
        return True
    
    def deal_initial_cards(self):
        """Deals the initial two cards to player and dealer"""
        if len(self.deck.cards) < 208: # Reshuffle if deck is running low
            print("Reshuffling deck...")
            self.deck.reset_deck()

        # Clear previous hands
        self.player_hand.clear()
        self.dealer_hand.clear()
        self.dealer_hand.is_dealer = True
        self.game_over = False

        # Deal two cards to each
        for _ in range(2):
            self.player_hand.add_card(self.deck.deal_card())
            self.dealer_hand.add_card(self.deck.deal_card())

    def player_hit(self) -> bool:
        """Player takes another card"""
        if self.game_over:
            return False
        
        card = self.deck.deal_card()
        if card:
            self.player_hand.add_card(card)
            print(f"You drew {card}")

            if self.player_hand.is_busted():
                print("You busted!")
                self.game_over = True
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
        player_value = self.player_hand.get_value()
        dealer_value = self.dealer_hand.get_value()

        # Player busted
        if self.player_hand.is_busted():
            return "dealer"
        
        # Dealer busted
        if self.dealer_hand.is_busted():
            self.player_balance += self.current_bet * 2
            return "player"
        
        # Check for blackjacks
        player_bj = self.player_hand.is_blackjack()
        dealer_bj = self.dealer_hand.is_blackjack()

        if player_bj and dealer_bj:
            return "push"
        if player_bj:
            self.player_balance += int(self.current_bet * 2.5) # 3:2 payout
            return "player_blackjack"
        if dealer_bj:
            return "dealer"
        
        # Compare values
        if player_value > dealer_value:
            self.player_balance += self.current_bet * 2
            return "player"
        if dealer_value > player_value:
            return "dealer"
        else:
            self.player_balance += self.current_bet # Push
            return "push"
        
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

        print(f"\nYour hand: {self.player_hand}")
        print(f"Dealer's hand: {self.dealer_hand}")

        # Check for immediate blackjacks
        player_bj = self.player_hand.is_blackjack()
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
        
        # Player's turn
        while not self.game_over:
            action = input("\nHit (h) or Stand (s)? ").lower().strip()

            if action == 'h':
                if not self.player_hit():
                    break
                print(f"Your hand: {self.player_hand}")
            elif action == 's':
                print("You stand.")
                break
            else:
                print("Invalid input. Please enter 'h' for hit or 's' for stand.")

        # Dealer's turn (if player didn't bust)
        if not self.player_hand.is_busted():
            self.dealer_play()

        # Determine winner
        result = self.determine_winner()
        self.print_result(result)

    def print_result(self, result: str):
        """Prints the game result"""
        print("\nFinal hands:")
        self.dealer_hand.is_dealer = False # Show all cards
        print(f"Your hand: {self.player_hand}")
        print(f"Dealer's hand: {self.dealer_hand}")

        if result == "player":
            print("You win!")
        if result == "player_blackjack":
            print("Blackjack! You win!")
        if result == "dealer":
            print("Dealer wins!")
        elif result == "push":
            print("Push! It's a tie!")

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
