import random
from abc import ABC, abstractmethod

from state.card import Card
from state.suits import Suit


class Dealer(ABC):
    rng: random.Random

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        super().__init__()

    @abstractmethod
    def deal(self, suit: Suit | None) -> tuple[list[Card], list[list[Card]]]:
        pass
