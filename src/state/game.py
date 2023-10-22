# from typing import Dict
import random
from cards import Card
from deck import Deck
from stack import Stack
# from suits import Suit
from ranks import Rank
from suits import Suit
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

    def __get_trump_cards(self) -> list[Card]:
        # For basic implementation return always Farbsolo prio
        trump_ober = self.deck.get_ranks_of(Rank.Ober)
        trump_unter = self.deck.get_ranks_of(Rank.Unter)

        trump_cards: list[Card] = trump_ober + trump_unter
        return trump_cards

    def run(self):
        while True:
            suit_chosen: Suit = self.determine_gametype()
            self.__new_game(suit_chosen)

    #TODO: For now this method only checks if anyone wants to play and which suit he chooses,
    # change it later so that it determines the gametype and returns it
    def determine_gametype(self) -> Suit:
        self.__distribute_cards()
        suit_chosen = self.__call_game()
        if suit_chosen is None:
            print("=============================================================")
            return self.determine_gametype()
        else:
            return suit_chosen

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
    def __call_game(self) -> Suit | None:
        chosen_suit = None
        game_called = False
        for player in self.players:
            game_called = player.plays()
            if game_called and chosen_suit is None:
                chosen_suit = player.get_suit_for_game()

        return chosen_suit


    def __new_game(self, suit_chosen: Suit):
        #TODO: Determine trump cards and suit priority based on gametype
        trump_cards = self.__get_trump_cards()
        for _ in range(ROUNDS):
            print(f"The suit for this game is: {suit_chosen.name}")
            self.start_round(suit_chosen, trump_cards)

        game_winner = self.__get_game_winner()
        print(f"The winner of this game is player {game_winner.get_id()} with {game_winner.get_points()}!")

    def __get_game_winner(self):
        game_winner = self.players[0]
        for player in self.players[0:]:
            if player.get_points() > game_winner.get_points():
                game_winner = player

        return game_winner


    #TODO: Maybe move round into its own class?
    def start_round(self, suit_chosen: Suit, trump_cards: list[Card]):
        stack = self.__play_cards(suit_chosen, trump_cards)
        self.__finish_round(stack)
        return

    def __play_cards(self, suit: Suit, trump_cards: list[Card]) -> Stack:
        print("=============================================================")
        stack = Stack(suit)
        for player in self.players:
            print("=============================================================")
            card: Card = player.lay_card(suit, trump_cards)
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
