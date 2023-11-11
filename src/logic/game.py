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
from state.deck import Deck
from state.event import (
    AnnouncePlayPartyEvent,
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
from state.suits import Suit

HAND_SIZE = 8
ROUNDS = 8
PLAYER_COUNT = 4


class Game:
    controllers: list[PlayerController]
    rng: random.Random
    gamemode: GameMode

    players: list[Player]
    deck: Deck
    played_cards: list[Card]

    def __init__(self, rng: random.Random = random.Random()) -> None:
        self.players = self.__create_players()
        self.deck: Deck = Deck()
        self.played_cards: list[Card] = []
        self.controllers = []
        self.rng = rng
        # In a game there are always two parties (player-party, non-player-party)
        self.play_party: list[list[Player]] = []

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

        chosen_types: list[tuple[Gametype | None, Suit | None]] = [
            (None, None),
            (None, None),
            (None, None),
            (None, None),
        ]
        for i, wants_to_play in enumerate(decisions):
            if wants_to_play is True:
                playable = get_playable_gametypes(
                    self.players[i].hand, decisions[0:i].count(True)
                )

                game_type = self.controllers[i].select_gametype(playable)
                chosen_types[i] = game_type
                self.__broadcast(GametypeWishedEvent(self.players[i], game_type))

        for i, game_type in enumerate(chosen_types):
            # When playing solo it is always 1v3
            solo_player_party = self.players[i]
            solo_non_player_party = self.players.copy()
            solo_non_player_party.remove(solo_player_party)

            match (game_type[0]):
                case Gametype.SOLO:
                    suit = game_type[1]
                    if suit is None:
                        raise ValueError("Solo gametype chosen without suit")
                    self.play_party = [[solo_player_party], solo_non_player_party]
                    self.gamemode = GameModeSolo(suit)
                case Gametype.WENZ:
                    self.play_party = [[solo_player_party], solo_non_player_party]
                    self.gamemode = GameModeWenz(None)
                case Gametype.GEIER:
                    self.play_party = [[solo_player_party], solo_non_player_party]
                    self.gamemode = GameModeGeier(None)
                case Gametype.FARBWENZ:
                    self.play_party = [[solo_player_party], solo_non_player_party]
                    self.gamemode = GameModeWenz(game_type[1])
                case Gametype.FARBGEIER:
                    self.play_party = [[solo_player_party], solo_non_player_party]
                    self.gamemode = GameModeGeier(game_type[1])
                case Gametype.SAUSPIEL:
                    suit = game_type[1]
                    if suit is None:
                        raise ValueError("Sauspiel gametype chosen without suit")

                    # Find Player who has the chosen ace
                    player_party = [self.players[i]]
                    for j, player in enumerate(self.players):
                        if player.hand.has_card_of_rank_and_suit(
                            game_type[1], Rank.ASS
                        ):
                            player_party.append(self.players[j])
                    non_player_party = self.players.copy()
                    non_player_party.remove(player_party[0])
                    non_player_party.remove(player_party[1])
                    self.play_party = [player_party, non_player_party]

                    self.gamemode = GameModeSauspiel(suit)
                case Gametype.RAMSCH:
                    # invalid gamemode, cannot be chosen
                    raise ValueError("Ramsch cannot be chosen as a gametype")
                case _:
                    continue

            self.__broadcast(
                GametypeDeterminedEvent(
                    self.players[i],
                    game_type[0],
                    game_type[1],
                    self.play_party if game_type[0] != Gametype.SAUSPIEL else None,
                )
            )
            return game_type[0]

        self.play_party = [
            [self.players[0]],
            [self.players[1]],
            [self.players[2]],
            [self.players[3]],
        ]
        self.__broadcast(
            GametypeDeterminedEvent(None, Gametype.RAMSCH, None, self.play_party)
        )
        self.gamemode = GameModeRamsch()
        return Gametype.RAMSCH

    def __new_game(self) -> None:
        """Start a new game with the specified suit as the game type."""
        for _ in range(ROUNDS):
            self.start_round()

        game_winner, points_distribution = self.gamemode.get_game_winner(
            self.play_party
        )
        self.__broadcast(
            GameEndEvent(game_winner, self.play_party, points_distribution)
        )

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

            # Announce that the searched ace had been played and teams are known
            if isinstance(self.gamemode, GameModeSauspiel) and card == Card(
                self.gamemode.suit, Rank.ASS
            ):
                self.__broadcast(AnnouncePlayPartyEvent(self.play_party))

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
