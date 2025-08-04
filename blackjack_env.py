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

    def __init__(self, num_decks: int=1):
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
        """Calculates the total value of the hand, handling Aces properly"""
        total = 0
        aces = 0

        for card in self.cards:
            if card.rank == 'A':
                aces += 1
            else:
                total += card.value

        # Adjust for Aces if bust
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1

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
            return f"[{cards_str}] (Value: {self.get_value()})"
