# from typing import Dict
import random
from state.cards import Card
from state.deck import Deck
from state.stack import Stack
from state.suits import Suit
from state.ranks import Rank
from state.player import Player

HAND_SIZE = 8

class Game:

    def __init__(self, players: list[Player], suit: Suit):
        self.players = players
        self.suit = suit
        self.stack: Stack = Stack()
        self.deck: Deck = Deck()
        self.played_cards: list[Card] = []
        self.suits_prio = self.__get_suits_prio()
        self.trump_cards = self.__get_trump_cards()

    def __get_suits_prio(self) -> list[Suit]:
        # For basic implementation return always Farbsolo prio
        return [Suit.Eichel, Suit.Gras, Suit.Herz, Suit.Schellen]

    def __get_trump_cards(self) -> list[Card] :
        # For basic implementation return always Farbsolo prio
        trump_suit = self.deck.get_suits_of(self.suit)
        trump_ober = self.deck.get_ranks_of(Rank.Ober)
        trump_unter = self.deck.get_ranks_of(Rank.Unter)

        trump_cards: list[Card] = trump_ober + trump_unter + trump_suit
        return trump_cards

    def run(self):
        while True:
            self.__new_round()

    def __new_round(self):
        self.determine_gametype()
        #TODO: Determine trump cards and suit priority based on gametype

        self.start_round()

    def determine_gametype(self):
        self.__distribute_cards()
        if not self.__call_game():
            self.__new_round()

    def __call_game(self):
        for player in self.players:
            if player.plays():
                # Wants to play
                return True
        return False # Nobody wants to play

    def __distribute_cards(self):
        deck: list[Card] = self.deck.get_deck()
        random.shuffle(deck)

        for player in self.players:
            deck = self.__distribute_hand(player, deck)

    def __distribute_hand(self, player: Player, deck: list[Card]) -> list[Card]:
        hand: list[Card] = deck[:HAND_SIZE]
        player.set_hand(hand)

        deck = deck[HAND_SIZE:]
        return deck

    def start_round(self):
        while self.__cards_left():
            self.__play_cards()
            self.__determine_winner()
            self.__clear_stack()
        return

    def __cards_left(self):
        for player in self.players:
            if not player.has_empty_hand():
                return True

        return False

    def __play_cards(self):
        for player in self.players:
            card: Card = player.lay_card(self.stack)
            self.stack.add_card(card, player)
        return 

    def __determine_winner(self):
        winner = self.stack.get_winner()
        stack_value = self.stack.get_value()
        winner.add_points(stack_value)

    def __clear_stack(self):
        self.stack = Stack()
