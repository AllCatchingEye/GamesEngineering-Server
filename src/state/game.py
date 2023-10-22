# from typing import Dict
import random
from cards import Card
from deck import Deck
from stack import PlayedCard, Stack
# from suits import Suit
from ranks import Rank
from player import Player
from hand import Hand

HAND_SIZE = 8
ROUNDS = 8
PLAYER_COUNT = 4

class Game:

    def __init__(self):
        self.players = self.__create_players()
        self.deck: Deck = Deck()
        self.played_cards: list[Card] = []

    def __create_players(self) -> list[Player]:
        players: list[Player] = []
        for i in range(PLAYER_COUNT):
            players.append(Player(i))
        return players

    # def __get_suits_prio(self) -> list[Suit]:
    #     # For basic implementation return always Farbsolo prio
    #     return [Suit.Eichel, Suit.Gras, Suit.Herz, Suit.Schellen]

    def __get_trump_cards(self, first_card: Card) -> set[Card]:
        # For basic implementation return always Farbsolo prio
        trump_suit = self.deck.get_suits_of(first_card.get_suit())
        trump_ober = self.deck.get_ranks_of(Rank.Ober)
        trump_unter = self.deck.get_ranks_of(Rank.Unter)

        trump_cards: set[Card] = set(trump_ober + trump_unter + trump_suit)
        return trump_cards

    def run(self):
        while True:
            self.__new_round()

    def __new_round(self):
        self.determine_gametype()
        #TODO: Determine trump cards and suit priority based on gametype

        self.start_round()

    #TODO: For now this method only checks if anyone wants to play,
    # change it later so that it determines the gametype and returns it
    def determine_gametype(self):
        self.__distribute_cards()
        game_called = self.__call_game()
        if not game_called:
            self.__new_round()
        else:
            return

    def __distribute_cards(self):
        deck: list[Card] = self.deck.get_deck()
        random.shuffle(deck)

        for player in self.players:
            deck = self.__distribute_hand(player, deck)

    def __distribute_hand(self, player: Player, deck: list[Card]) -> list[Card]:
        hand: Hand = Hand(deck[:HAND_SIZE])
        player.set_hand(hand)

        deck = deck[HAND_SIZE:]
        return deck

    #TODO: Return gametype instead of bool
    def __call_game(self) -> bool:
        for player in self.players:
            if player.plays():
                return True
        return False

    #TODO: Maybe move round into its own class?
    def start_round(self):
        for _ in range(ROUNDS):
            stack = self.__play_cards()
            self.__finish_round(stack)
        return

    def __play_cards(self) -> Stack:
        print("=============================================================")
        beginner = self.players[0]
        first_card = PlayedCard(beginner.decide_first_card(), beginner) 
        stack = Stack(first_card)
        trump_cards = self.__get_trump_cards(first_card.get_card())

        for player in self.players[1:]:
            print("=============================================================")
            card: Card = player.lay_card(trump_cards)
            stack.add_card(card, player)

        return stack

    def __finish_round(self, stack: Stack):
        winner = stack.get_winner()
        stack_value = stack.get_value()
        winner.add_points(stack_value)
        self.__show_winner(winner)
        self.__change_player_order(winner)

    def __show_winner(self, winner: Player):
        print(f"Player {winner.get_id()}, you are the winner of this round!")
        print(f"You got {winner.get_points()} points.")

    def __change_player_order(self, winner: Player):
        winner_index = self.__get_winner_index(winner)
        self.__swap_players(winner_index)

    def __get_winner_index(self, winner: Player) -> int:
        winner_index = 0
        for index, player in enumerate(self.players):
            if player.get_id() == winner.get_id():
                winner_index = index
        return winner_index

    def __swap_players(self, winner_index: int):
        first: list[Player] = self.players[winner_index:]
        last: list[Player] = self.players[:winner_index]
        self.players = first + last
