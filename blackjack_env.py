


class Card:
    """Represents a playing card."""

    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank} of {self.suit}"
    
    def value(self) -> int:
        """Returns blackjack value of a card."""
        if self.rank in ['J', 'Q', 'K']:
            return 10
        if self.rank == 'A':
            return 11 # Initialize to 11, adjusted later
        else:
            return int(self.rank)
