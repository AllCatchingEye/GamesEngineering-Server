import random

from controller.player_controller import PlayerController
from state.card import Card
from state.deck import Deck
from state.gametypes import Gametype
from state.hand import Hand
from state.player import Player
from state.ranks import Rank
from state.stack import Stack

HAND_SIZE = 8
ROUNDS = 8
PLAYER_COUNT = 4


class Game:
    controllers: list[PlayerController]

    def __init__(self) -> None:
        self.players = self.__create_players()
        self.deck: Deck = Deck()
        self.played_cards: list[Card] = []
        self.controllers = []

    def __create_players(self) -> list[Player]:
        """Create a list of players for the game."""
        players: list[Player] = []
        for i in range(PLAYER_COUNT):
            players.append(Player(i))
        return players

    def run(self) -> None:
        """Start the game."""

        while True:
            suit_chosen = self.determine_gametype()
            self.__new_game(suit_chosen)

    def determine_gametype(self) -> Gametype:
        """Determine the game type based on player choices."""
        self.__distribute_cards()
        suit_chosen = self.__call_game()
        if suit_chosen is None:
            print("=============================================================")
            return self.determine_gametype()
        return suit_chosen

    def __distribute_cards(self) -> None:
        """Distribute cards to players."""
        deck: list[Card] = self.deck.get_full_deck()
        random.shuffle(deck)

        for player in self.players:
            deck = self.__distribute_hand(player, deck)

    def __distribute_hand(self, player: Player, deck: list[Card]) -> list[Card]:
        """Distribute cards for a player's hand."""
        hand: Hand = Hand(deck[:HAND_SIZE])
        player.hand = hand

        deck = deck[HAND_SIZE:]
        return deck

    def __call_game(self) -> Gametype | None:
        """Call the game type based on player choices."""
        decisions: list[bool | None] = [None, None, None, None]
        for player in self.players:
            i = player.id
            wants_to_play = self.controllers[i].wants_to_play(decisions)
            decisions[i] = wants_to_play

        chosen_types: list[Gametype | None] = [None, None, None, None]
        for i, wants_to_play in enumerate(decisions):
            if wants_to_play is True:
                chosen_types[i] = self.controllers[i].select_gametype([Gametype.SOLO])

        for i, game_type in enumerate(chosen_types):
            if game_type is not None:
                return game_type

        return None

    def __new_game(self, game_type: Gametype) -> None:
        """Start a new game with the specified suit as the game type."""
        trump_cards = self.__get_trump_cards()
        for _ in range(ROUNDS):
            print(f"The suit for this game is: {game_type.name}")
            self.start_round(trump_cards)

        game_winner = self.__get_game_winner()
        print(f"The winner of this game is player {game_winner.id}!")
        print(f"He won {game_winner.points} points!")

    def __get_trump_cards(self) -> list[Card]:
        """Get all trump cards for the game."""
        # For basic implementation return always Farbsolo prio
        trump_ober = self.deck.get_cards_by_rank(Rank.OBER)
        trump_unter = self.deck.get_cards_by_rank(Rank.UNTER)

        trump_cards: list[Card] = trump_ober + trump_unter
        return trump_cards

    def start_round(self, trump_cards: list[Card]) -> None:
        """Start a new round."""
        stack = self.__play_cards(trump_cards)
        self.__finish_round(stack)

    def __play_cards(self, trump_cards: list[Card]) -> Stack:
        """Play cards in the current round."""
        print("=============================================================")
        stack = Stack()
        for player in self.players:
            print("=============================================================")
            card: Card = self.controllers[player.id].play_card(stack, trump_cards)
            stack.add_card(card, player)
        return stack

    def __finish_round(self, stack: Stack) -> None:
        """Finish the current round and determine the winner."""
        winner = stack.get_winner()
        stack_value = stack.get_value()
        winner.points += stack_value
        self.__show_winner(winner)
        self.__change_player_order(winner)

    def __show_winner(self, winner: Player) -> None:
        """Display the winner of the round."""
        print(f"Player {winner.id}, you are the winner of this round!")
        print(f"You got {winner.id} points.")

    def __change_player_order(self, winner: Player) -> None:
        """Change the order of players based on the round winner."""
        winner_index = self.__get_winner_index(winner)
        self.__swap_players(winner_index)

    def __get_game_winner(self) -> Player:
        """Determine the winner of the entire game."""
        game_winner = self.players[0]
        for player in self.players[0:]:
            if player.points > game_winner.points:
                game_winner = player

        return game_winner

    def __get_winner_index(self, winner: Player) -> int:
        """Find the index of the player who won"""
        winner_index = 0
        for index, player in enumerate(self.players):
            if player.id == winner.id:
                winner_index = index
        return winner_index

    def __swap_players(self, winner_index: int) -> None:
        first: list[Player] = self.players[winner_index:]
        last: list[Player] = self.players[:winner_index]
        self.players = first + last
