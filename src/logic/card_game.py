import random

from logic.game import Game
from logic.gamemodes.gamemode_geier import GameModeGeier
from logic.gamemodes.gamemode_ramsch import GameModeRamsch
from logic.gamemodes.gamemode_sauspiel import GameModeSauspiel
from logic.gamemodes.gamemode_solo import GameModeSolo
from logic.gamemodes.gamemode_wenz import GameModeWenz
from state.card import Card
from state.event import Event, GametypeDeterminedUpdate
from state.gametypes import Gametype
from state.hand import Hand
from state.player import Player, PlayerId, play_parties_to_struct
from state.ranks import Rank
from state.suits import Suit


class CardGame:
    game: Game
    main_player: PlayerId | None

    def __init__(self, rng: random.Random = random.Random()) -> None:
        self.game = Game(rng)

    def set_player_hands(self, main_player: PlayerId, hands: list[list[Card]]):
        self.main_player = main_player
        k = 1
        for _, p in enumerate(self.game.players):
            if p.id == main_player:
                p.hand = Hand(hands[0])
            else:
                p.hand = Hand(hands[k])
                k += 1

    async def set_game_type(self, game_type: Gametype, suit: Suit | None):  # type: ignore
        self.game.gametype = game_type
        self.game.suit = suit

        match game_type:
            case Gametype.RAMSCH:
                self.game.gamemode = GameModeRamsch()
            case Gametype.SOLO:
                assert suit is not None
                self.game.gamemode = GameModeSolo(suit)
            case Gametype.FARBGEIER:
                self.game.gamemode = GameModeGeier(suit)
            case Gametype.WENZ:
                self.game.gamemode = GameModeWenz(None)
            case Gametype.GEIER:
                self.game.gamemode = GameModeGeier(None)
            case Gametype.FARBWENZ:
                self.game.gamemode = GameModeWenz(suit)
            case Gametype.SAUSPIEL:
                assert suit is not None
                self.game.gamemode = GameModeSauspiel(suit)

        await self.set_play_party()

    async def set_play_party(self):
        assert self.main_player != None

        if self.game.gametype == Gametype.RAMSCH:
            self.game.play_party = [
                [self.game.players[0]],
                [self.game.players[1]],
                [self.game.players[2]],
                [self.game.players[3]],
            ]
            await self.__broadcast(
                GametypeDeterminedUpdate(
                    None,
                    Gametype.RAMSCH,
                    None,
                    play_parties_to_struct([[player.id for player in party] for party in self.game.play_party]),
                )
            )
            return

        if self.game.gametype == Gametype.SAUSPIEL:
            suit = self.game.suit
            assert suit != None
            player_party: list[Player] = []
            non_player_party: list[Player] = []
            for i, player in enumerate(self.game.players):
                if player.id == self.main_player:
                    player_party.append(player)
                elif player.hand.has_card_of_rank_and_suit(suit, Rank.ASS):
                    player_party.append(self.game.players[i])
                else:
                    non_player_party.append(self.game.players[i])
            self.game.play_party = [player_party, non_player_party]
            # broadcast
            await self.__broadcast(
                GametypeDeterminedUpdate(
                    self.main_player, self.game.gametype, self.game.suit, None
                )
            )
            return

        # other gamemodes
        self.game.play_party = [[], []]
        for p in self.game.players:
            if p.id == self.main_player:
                self.game.play_party[0].append(p)
            else:
                self.game.play_party[1].append(p)

        await self.__broadcast(
            GametypeDeterminedUpdate(
                self.main_player,
                self.game.gametype,
                self.game.suit,
                play_parties_to_struct([[player.id for player in party] for party in self.game.play_party]),
            )
        )

    async def __broadcast(self, event: Event):
        for c in self.game.controllers:
            await c.on_game_event(event)
