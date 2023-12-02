from typing import Callable
from state.card import Card
from state.ranks import Rank
from state.suits import Suit


def take_first_matching(deck: list[Card], predicate: Callable[[Card], bool]) -> Card:
    for card in deck:
        if predicate(card):
            deck.remove(card)
            return card

    raise ValueError(f"Card not found in deck {deck}")


def take_rank(deck: list[Card], rank: Rank) -> Card:
    for card in deck:
        if card.rank == rank:
            deck.remove(card)
            return card

    raise ValueError(f"Rank {rank} not found in deck {deck}")


def take_suit(deck: list[Card], suit: Suit) -> Card:
    for card in deck:
        if card.suit == suit:
            deck.remove(card)
            return card

    raise ValueError(f"Suit {suit} not found in deck {deck}")


def take_trumps(deck: list[Card], trumps: list[Card], n: int = 1) -> list[Card]:
    taken: list[Card] = []

    for card in deck:
        if card in trumps:
            taken.append(card)

        if len(taken) >= n:
            break

    for card in taken:
        deck.remove(card)

    if len(taken) < n:
        raise ValueError(f"Not enough trumps in deck {deck}")

    return taken


def take_any_except(deck: list[Card], not_cards: list[Card], n: int = 1) -> list[Card]:
    taken: list[Card] = []

    for card in deck:
        if card not in not_cards:
            taken.append(card)

        if len(taken) >= n:
            break

    for card in taken:
        deck.remove(card)

    if len(taken) < n:
        raise ValueError(f"Not enough cards in deck {deck}")

    return taken


def take(deck: list[Card], n: int) -> list[Card]:
    taken: list[Card] = []

    for _ in range(n):
        taken.append(deck.pop(0))

    return taken
