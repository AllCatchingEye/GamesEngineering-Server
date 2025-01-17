import asyncio
import random
from ast import Pass
from dataclasses import dataclass
from typing import Callable, TypeVar

import pandas as pd
from tqdm import tqdm

from ai.dealer.dealer_factory import DealerFactory
from controller.ai_controller import AiController
from controller.ai_controller_iter1 import AiController_1It
from controller.ai_controller_iter2 import AiController_2It
from controller.combi_contoller import CombiController
from controller.handcrafted_controller import HandcraftedController
from controller.passive_controller import PassiveController
from controller.player_controller import PlayerController
from controller.random_controller import RandomController
from controller.ai_controller_iter3 import AiController_3It
from logic.card_game import CardGame
from logic.game import Game
from state.card import Card
from state.event import Event, GameStartUpdate, GametypeDeterminedUpdate, MoneyUpdate
from state.gametypes import GameGroup, Gametype, GametypeWithSuit
from state.money import Money
from state.player import PlayerId
from state.stack import Stack
from state.suits import Suit

T = TypeVar("T")

ALL_SUITS = [Suit.EICHEL, Suit.GRAS, Suit.HERZ, Suit.SCHELLEN]
ALL_GAME_TYPES = [
    Gametype.SAUSPIEL,
    Gametype.RAMSCH,
    Gametype.SOLO,
    Gametype.WENZ,
    Gametype.FARBWENZ,
    Gametype.GEIER,
    Gametype.FARBGEIER,
]
TRAINING_CONFIG = {
    Gametype.SAUSPIEL.name: [Suit.EICHEL, Suit.GRAS, Suit.SCHELLEN],
    Gametype.RAMSCH.name: None,
    Gametype.SOLO.name: ALL_SUITS,
    Gametype.WENZ.name: None,
    Gametype.FARBWENZ.name: ALL_SUITS,
    Gametype.GEIER.name: None,
    Gametype.FARBGEIER.name: ALL_SUITS,
}


def increment(dictionary: dict[T, int], key: T, value: int = 1) -> None:
    dictionary[key] = dictionary.get(key, 0) + value


def increment_money(dictionary: dict[T, Money], key: T, value: Money) -> None:
    dictionary[key] = dictionary.get(key, Money(0)) + value


@dataclass
class ArenaConfig:
    games: int = 1000
    rounds_per_game: int = 10
    rng_seed: int | None = None


@dataclass
class GameTypeWithSuitAndAnnouncer:
    gametype: Gametype
    suit: Suit | None
    announcer: bool

    def __hash__(self) -> int:
        return hash((self.gametype, self.suit, self.announcer))

    def __str__(self) -> str:
        return f"{self.gametype} {self.suit} {self.announcer}"


class ArenaController(PlayerController):
    actual_controller: PlayerController
    player_id: PlayerId

    gamemode: GameTypeWithSuitAndAnnouncer

    money_before: Money
    money: Money
    money_per_gamemode: dict[GameTypeWithSuitAndAnnouncer, Money]
    wins: int
    wins_per_gamemode: dict[GameTypeWithSuitAndAnnouncer, int]
    played_gamemodes: dict[GameTypeWithSuitAndAnnouncer, int]
    points_per_gamemode: dict[GameTypeWithSuitAndAnnouncer, int]

    def __init__(self, actual_controller: PlayerController) -> None:
        super().__init__()
        self.actual_controller = actual_controller
        self.money_before = Money(0)
        self.money = Money(0)
        self.wins = 0
        self.played_gamemodes = {}
        self.money_per_gamemode = {}
        self.wins_per_gamemode = {}
        self.points_per_gamemode = {}

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        return await self.actual_controller.wants_to_play(current_lowest_gamegroup)

    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        return await self.actual_controller.select_gametype(choosable_gametypes)

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        return await self.actual_controller.play_card(stack, playable_cards)

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        return await self.actual_controller.choose_game_group(available_groups)

    async def on_game_event(self, event: Event) -> None:
        if isinstance(event, GameStartUpdate):
            self.player_id = event.player
        if isinstance(event, GametypeDeterminedUpdate):
            gamemode = GameTypeWithSuitAndAnnouncer(
                event.gametype, event.suit, event.player == self.player_id
            )
            self.gamemode = gamemode
            increment(self.played_gamemodes, gamemode)
        if isinstance(event, MoneyUpdate) and event.player == self.player_id:
            self.money = event.money
            diff = self.money - self.money_before
            if diff.cent > 0:
                self.wins += 1
                increment(self.wins_per_gamemode, self.gamemode)
            increment_money(self.money_per_gamemode, self.gamemode, diff)
            self.money_before = self.money
        return await self.actual_controller.on_game_event(event)


class Arena:
    """Arena provides a battle ground for multiple AIs to battle, and evaluates their performance."""

    config: ArenaConfig
    bot_creators: list[Callable[[], PlayerController]]
    bot_names: list[str]

    money: list[Money]
    wins: list[int]
    played_gamemodes: list[dict[GametypeWithSuit, int]]
    played_gamemmodes_with_announcer: list[dict[GameTypeWithSuitAndAnnouncer, int]]
    money_per_gamemode: list[dict[GameTypeWithSuitAndAnnouncer, Money]]
    wins_per_gamemode: list[dict[GameTypeWithSuitAndAnnouncer, int]]

    def __init__(self, config: ArenaConfig = ArenaConfig()) -> None:
        self.config = config
        self.bot_creators = []
        self.bot_names = []
        self.money = []
        self.wins = []
        self.played_gamemmodes_with_announcer = []
        self.played_gamemodes = []
        self.money_per_gamemode = []
        self.wins_per_gamemode = []

    def add_bot(self, bot_creator: Callable[[], PlayerController]) -> None:
        """Provides a function to create a bot and add it to the arena."""
        self.bot_creators.append(bot_creator)
        self.bot_names.append(bot_creator.__name__)
        self.money.append(Money(0))
        self.wins.append(0)
        self.played_gamemmodes_with_announcer.append({})
        self.played_gamemodes.append({})
        self.money_per_gamemode.append({})
        self.wins_per_gamemode.append({})

    async def get_new_game(
        self,
        game_type: Gametype,
        playable_suits: list[Suit] | None,
    ):
        suit = None if playable_suits is None else random.sample(playable_suits, 1)[0]
        dealer = DealerFactory.get_dealer(game_type)
        good_cards, other_cards = dealer.deal(suit)
        hands = [good_cards]
        hands.extend(other_cards)

        card_game = CardGame()
        mapped_controllers = {
            i: ArenaController(bot_creator())
            for i, bot_creator in enumerate(self.bot_creators)
        }
        new_controllers = [mapped_controllers[i] for i in range(len(self.bot_creators))]
        card_game.game.controllers = new_controllers

        main_player_id = card_game.game.players[0].id
        await card_game.game.announce_hands()
        card_game.set_player_hands(main_player_id, hands)
        await card_game.set_game_type(game_type, suit)

        return (card_game.game, mapped_controllers)

    async def run_game_type(self, game_type: Gametype):
        for game in tqdm(range(self.config.games), desc="Games", unit="game", ncols=80):
            game, mapped_controllers = await self.get_new_game(
                game_type, TRAINING_CONFIG[game_type.name]
            )
            await game.run(games_to_play=1)
            await self.update_money(mapped_controllers)
            await self.update_played_gamemodes(mapped_controllers)

    async def run(self) -> None:
        rng = random.Random(self.config.rng_seed)
        for game in tqdm(range(self.config.games), desc="Games", unit="game", ncols=80):
            game = Game(rng=rng)
            mapped_controllers = {
                i: ArenaController(bot_creator())
                for i, bot_creator in enumerate(self.bot_creators)
            }
            controllers = [mapped_controllers[i] for i in range(len(self.bot_creators))]
            rng.shuffle(controllers)
            game.controllers = controllers
            await game.run(games_to_play=self.config.rounds_per_game)
            await self.update_money(mapped_controllers)
            await self.update_played_gamemodes(mapped_controllers)

    async def update_played_gamemodes(
        self, mapped_controllers: dict[int, ArenaController]
    ):
        for i, controller in mapped_controllers.items():
            for gamemode, played in controller.played_gamemodes.items():
                increment(
                    self.played_gamemodes[i],
                    GametypeWithSuit(gamemode.gametype, gamemode.suit),
                    played,
                )
                increment(
                    self.played_gamemmodes_with_announcer[i],
                    gamemode,
                    played,
                )

    async def update_money(self, mapped_controllers: dict[int, ArenaController]):
        for i, controller in mapped_controllers.items():
            self.money[i] += controller.money
            self.wins[i] += controller.wins

            for gamemode, money in controller.money_per_gamemode.items():
                increment_money(self.money_per_gamemode[i], gamemode, money)

            for gamemode, wins in controller.wins_per_gamemode.items():
                increment(self.wins_per_gamemode[i], gamemode, wins)

    def results_overview(self) -> pd.DataFrame:
        total_games = self.config.games * self.config.rounds_per_game
        df = pd.DataFrame(columns=["Bot", "Money", "Wins", "Winrate"])
        for i in range(len(self.bot_creators)):
            money = self.money[i]
            wins = self.wins[i]
            win_rate = wins / total_games
            df.loc[i] = [self.bot_names[i], money, wins, win_rate]

        return df

    def results_gamemodes(self) -> pd.DataFrame:
        total_games = self.config.games * self.config.rounds_per_game
        df = pd.DataFrame(columns=["Gamemode", "Played", "Play rate"])
        for i in range(len(self.bot_creators)):
            for gamemode, played in self.played_gamemodes[i].items():
                df.loc[gamemode] = [gamemode, played, played / total_games]
        return df

    def results_gamemodes_per_player(self) -> list[pd.DataFrame]:
        total_games = self.config.games * self.config.rounds_per_game
        dfs = []
        for i in range(len(self.bot_creators)):
            print(self.bot_creators[i])
            df = pd.DataFrame(
                columns=[
                    "Gamemode",
                    "Played",
                    "Play rate",
                    "Wins",
                    "Winrate",
                    "Money",
                    "Money per game",
                ]
            )
            for gamemode, played in self.played_gamemmodes_with_announcer[i].items():
                wins = self.wins_per_gamemode[i].get(gamemode, 0)
                money = self.money_per_gamemode[i].get(gamemode, Money(0))
                win_rate = wins / played
                money_per_game = Money(int(money.cent / played))
                df.loc[gamemode] = [
                    gamemode,
                    played,
                    played / total_games,
                    wins,
                    win_rate,
                    money,
                    money_per_game,
                ]
            dfs.append(df)

        return dfs


def get_ai_ctrl_256_256_256_256_256() -> AiController:
    return AiController([256, 256, 256, 256, 256])


if __name__ == "__main__":

    def combi_creator() -> PlayerController:
        return CombiController(
            AiController_2It(), PassiveController()
        )

    arena = Arena()
    arena.add_bot(HandcraftedController)
    arena.add_bot(AiController_3It)
    arena.add_bot(AiController_2It)
    arena.add_bot(AiController_2It)
    asyncio.run(arena.run())

    print("Overview")
    df = arena.results_overview()
    print(df.sort_values(by="Money", ascending=False))

    print()
    print("Gamemodes")
    df = arena.results_gamemodes()
    print(df.sort_values(by="Play rate", ascending=False))

    print()
    print("Gamemodes per player")
    dfs = arena.results_gamemodes_per_player()
    for df in dfs:
        print(df.sort_values(by="Money", ascending=False))
