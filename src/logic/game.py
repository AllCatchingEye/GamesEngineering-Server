import random

from controller.player_controller import PlayerController
from logic.gamemodes.gamemode import GameMode
from logic.gamemodes.gamemode_geier import GameModeGeier
from logic.gamemodes.gamemode_ramsch import GameModeRamsch
from logic.gamemodes.gamemode_sauspiel import GameModeSauspiel
from logic.gamemodes.gamemode_solo import GameModeSolo
from logic.gamemodes.gamemode_wenz import GameModeWenz
from logic.playable_gametypes import get_playable_gametypes
from state.card import Card
from state.deck import Deck, DECK
from state.event import (
    CardPlayedEvent,
    Event,
    GameEndEvent,
    GameStartEvent,
    GametypeDeterminedEvent,
    GametypeWishedEvent,
    PlayDecisionEvent,
    RoundResultEvent,
)
from state.gametypes import Gametype
from state.hand import Hand
from state.player import Player
from state.ranks import Rank
from state.stack import Stack
from state.suits import Suit, get_all_suits

HAND_SIZE = 8
ROUNDS = 8
PLAYER_COUNT = 4


def get_playable_gametypes(hand: Hand, plays_ahead: int) -> list[(Gametype, Suit | None)]:
    """Returns all playable gametypes with that hand."""
    types = []
    # Gametypes Solo
    for suit in get_all_suits():
        types.append((Gametype.SOLO, suit))
    # Gametypes Wenz
    types += __get_practical_gametypes_wenz_geier(hand, Rank.UNTER, Gametype.FARBWENZ, Gametype.WENZ)
    # Gametypes Geier
    types += __get_practical_gametypes_wenz_geier(hand, Rank.OBER, Gametype.FARBGEIER, Gametype.GEIER)
    # Gametypes Sauspiel
    if plays_ahead == 0:
        sauspiel_suits = get_all_suits()
        sauspiel_suits.remove(Suit.HERZ)
        for suit in sauspiel_suits:
            suit_cards = hand.get_all_cards_for_suit(suit,
                                                     DECK.get_cards_by_rank(Rank.OBER) + DECK.get_cards_by_rank(
                                                         Rank.UNTER))
            if len(suit_cards) > 0 and Card(Rank.ASS, suit) not in suit_cards:
                types.append((Gametype.SAUSPIEL, suit))
        return types


def __get_practical_gametypes_wenz_geier(hand: Hand, rank: Rank, game_type_suit: Gametype,
                                         game_type_no_suit: Gametype) -> list[(Gametype, Suit | None)]:
    practical_types = []
    if len(hand.get_all_trumps_in_deck(DECK.get_cards_by_rank(rank))) > 0:
        practical_types.append((game_type_no_suit, None))
        for suit in get_all_suits():
            if len(hand.get_all_cards_for_suit(suit, DECK.get_cards_by_rank(Rank.OBER))) > 0:
                practical_types.append((game_type_suit, suit))
    return practical_types


class Game:
    controllers: list[PlayerController]
    rng: random.Random
    gamemode: GameMode

    def __init__(self, rng: random.Random = random.Random()) -> None:
        self.players = self.__create_players()
        self.deck: Deck = Deck()
        self.played_cards: list[Card] = []
        self.controllers = []
        self.rng = rng

    def __create_players(self) -> list[Player]:
        """Create a list of players for the game."""
        players: list[Player] = []
        for i in range(PLAYER_COUNT):
            players.append(Player(i))
        return players

    def run(self) -> None:
        """Start the game."""

        self.determine_gametype()
        self.__new_game()

    def determine_gametype(self) -> Gametype:
        """Determine the game type based on player choices."""
        self.__distribute_cards()
        game_type = self.__call_game()
        return game_type

    def __distribute_cards(self) -> None:
        """Distribute cards to players."""
        deck: list[Card] = self.deck.get_full_deck()
        self.rng.shuffle(deck)

        for player in self.players:
            deck = self.__distribute_hand(player, deck)

    def __distribute_hand(self, player: Player, deck: list[Card]) -> list[Card]:
        """Distribute cards for a player's hand."""
        hand: Hand = Hand(deck[:HAND_SIZE])
        player.hand = hand

        deck = deck[HAND_SIZE:]

        self.controllers[player.id].on_game_event(GameStartEvent(hand))
        return deck

    def __call_game(self) -> Gametype:
        """Call the game type based on player choices."""
        decisions: list[bool | None] = [None, None, None, None]
        for player in self.players:
            i = player.id
            wants_to_play = self.controllers[i].wants_to_play(decisions)
            self.__broadcast(PlayDecisionEvent(player, wants_to_play))
            decisions[i] = wants_to_play

        # TODO: Game types that have suits?
        chosen_types: list[(Gametype | None, Suit | None)] = [(None, None), (None, None), (None, None), (None, None)]
        for i, wants_to_play in enumerate(decisions):
            if wants_to_play is True:
                game_type = self.controllers[i].select_gametype(
                    get_playable_gametypes(self.players[i].hand, decisions[0:i].count(True))
                )
                chosen_types[i] = game_type
                self.__broadcast(GametypeWishedEvent(self.players[i], game_type))

        for i, game_type in enumerate(chosen_types):
            if game_type[0] is None:
                continue

            match (game_type[0]):
                case Gametype.SOLO:
                    self.gamemode = GameModeSolo(game_type[1])
                case Gametype.WENZ:
                    self.gamemode = GameModeWenz(None)
                case Gametype.GEIER:
                    self.gamemode = GameModeGeier(None)
                case Gametype.FARBWENZ:
                    self.gamemode = GameModeWenz(game_type[1])
                case Gametype.FARBGEIER:
                    self.gamemode = GameModeGeier(game_type[1])
                case Gametype.SAUSPIEL:
                    self.gamemode = GameModeSauspiel(game_type[1])
                case Gametype.RAMSCH:
                    # invalid gamemode, cannot be chosen
                    raise ValueError("Ramsch cannot be chosen as a gametype")

            self.__broadcast(GametypeDeterminedEvent(self.players[i], game_type))
            return game_type

        self.__broadcast(GametypeDeterminedEvent(None, game_type))
        self.gamemode = GameModeRamsch()
        return Gametype.RAMSCH

    def __new_game(self) -> None:
        """Start a new game with the specified suit as the game type."""
        for _ in range(ROUNDS):
            self.start_round()

        self.__get_game_winner()

    def start_round(self) -> None:
        """Start a new round."""
        stack = self.__play_cards()
        self.__finish_round(stack)

    def __play_cards(self) -> Stack:
        """Play cards in the current round."""
        stack = Stack()
        for player in self.players:
            playable_cards = self.gamemode.get_playable_cards(stack, player.hand)
            if len(playable_cards) == 0:
                raise ValueError("No playable cards")
            card: Card = self.controllers[player.id].play_card(stack, playable_cards)
            if card not in playable_cards or card not in player.hand.cards:
                raise ValueError("Illegal card played")
            player.lay_card(card)
            stack.add_card(card, player)
            self.__broadcast(CardPlayedEvent(player, card, stack))
        return stack

    def __finish_round(self, stack: Stack) -> None:
        """Finish the current round and determine the winner."""
        winner = self.gamemode.determine_stitch_winner(stack)
        stack_value = stack.get_value()
        winner.points += stack_value
        self.__broadcast(RoundResultEvent(winner, stack_value, stack))
        self.__change_player_order(winner)

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

        self.__broadcast(GameEndEvent(game_winner, game_winner.points))
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

    def __broadcast(self, event: Event) -> None:
        """Broadcast an event to all players."""
        for controller in self.controllers:
            controller.on_game_event(event)
