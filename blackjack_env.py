import random
from typing import Optional


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
        self.cards = []
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
    