import asyncio
import random
from dataclasses import dataclass
from typing import Callable, Type, TypeVar

from tqdm import tqdm
import pandas as pd
from controller.ai_controller import AiController

from controller.player_controller import PlayerController
from controller.random_controller import RandomController
from logic.game import Game
from state.card import Card
from state.event import Event, GameStartUpdate, GametypeDeterminedUpdate, MoneyUpdate
from state.gametypes import GameGroup, Gametype, GametypeWithSuit
from state.money import Money
from state.player import PlayerId
from state.stack import Stack
from state.suits import Suit


T = TypeVar("T")


def increment(dictionary: dict[T, int], key: T, value: int = 1) -> None:
    dictionary[key] = dictionary.get(key, 0) + value


def increment_money(dictionary: dict[T, Money], key: T, value: Money) -> None:
    dictionary[key] = dictionary.get(key, Money(0)) + value


@dataclass
class ArenaConfig:
    games: int = 10
    rounds_per_game: int = 10
    rng_seed: int | None = None


class ArenaController(PlayerController):
    actual_controller: PlayerController
    player_id: PlayerId

    gamemode: GametypeWithSuit

    money: Money
    money_per_gamemode: dict[GametypeWithSuit, Money]
    wins: int
    wins_per_gamemode: dict[GametypeWithSuit, int]
    played_gamemodes: dict[GametypeWithSuit, int]

    def __init__(self, actual_controller: PlayerController) -> None:
        super().__init__()
        self.actual_controller = actual_controller
        self.money = Money(0)
        self.wins = 0
        self.played_gamemodes = {}
        self.money_per_gamemode = {}
        self.wins_per_gamemode = {}

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
            gamemode = GametypeWithSuit(event.gametype, event.suit)
            self.gamemode = gamemode
            increment(self.played_gamemodes, gamemode)
        if isinstance(event, MoneyUpdate) and event.player == self.player_id:
            self.money += event.money
            if event.money.cent > 0:
                self.wins += 1
                increment(self.wins_per_gamemode, self.gamemode)
            increment_money(self.money_per_gamemode, self.gamemode, event.money)
        return await self.actual_controller.on_game_event(event)


class Arena:
    """Arena provides a battle ground for multiple AIs to battle, and evaluates their performance."""

    config: ArenaConfig
    bot_creators: list[Callable[[], PlayerController]]
    bot_names: list[str]

    money: list[Money]
    wins: list[int]
    total_gamemodes: dict[GametypeWithSuit, int]
    played_gamemodes: list[dict[GametypeWithSuit, int]]
    money_per_gamemode: list[dict[GametypeWithSuit, Money]]
    wins_per_gamemode: list[dict[GametypeWithSuit, int]]

    def __init__(self, config: ArenaConfig = ArenaConfig()) -> None:
        self.config = config
        self.bot_creators = []
        self.bot_names = []
        self.money = []
        self.wins = []
        self.played_gamemodes = []
        self.money_per_gamemode = []
        self.wins_per_gamemode = []

    def add_bot(self, bot_creator: Callable[[], PlayerController]) -> None:
        """Provides a function to create a bot and add it to the arena."""
        self.bot_creators.append(bot_creator)
        self.bot_names.append(bot_creator.__name__)
        self.money.append(Money(0))
        self.wins.append(0)
        self.played_gamemodes.append({})
        self.money_per_gamemode.append({})
        self.wins_per_gamemode.append({})

    async def run(self) -> None:
        rng = random.Random(self.config.rng_seed)
        for game in tqdm(range(self.config.games), desc="Games", unit="game", ncols=80):
            game = Game(rng=rng)
            controllers = [
                ArenaController(bot_creator()) for bot_creator in self.bot_creators
            ]
            game.controllers = controllers
            await game.run(games_to_play=self.config.rounds_per_game)

            # Update money
            for i, controller in enumerate(game.controllers):
                self.money[i] += controller.money
                self.wins[i] += controller.wins

                for gamemode, money in controller.money_per_gamemode.items():
                    increment_money(self.money_per_gamemode[i], gamemode, money)

                for gamemode, wins in controller.wins_per_gamemode.items():
                    increment(self.wins_per_gamemode[i], gamemode, wins)

            # Update played gamemodes
            for i, controller in enumerate(game.controllers):
                for gamemode, played in controller.played_gamemodes.items():
                    increment(self.played_gamemodes[i], gamemode, played)

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
            df = pd.DataFrame(
                columns=["Gamemode", "Played", "Play rate", "Wins", "Winrate", "Money", "Money per game", "Money per win"]
            )
            for gamemode, played in self.played_gamemodes[i].items():
                wins = self.wins_per_gamemode[i].get(gamemode, 0)
                money = self.money_per_gamemode[i].get(gamemode, Money(0))
                win_rate = wins / played
                money_per_game = Money(int(money.cent / played))
                money_per_win = Money(0)
                if wins > 0:
                    money_per_win = Money(int(money.cent / wins))
                df.loc[gamemode] = [
                    gamemode,
                    played,
                    played / total_games,
                    wins,
                    win_rate,
                    money,
                    money_per_game,
                    money_per_win
                ]
            dfs.append(df)

        return dfs


if __name__ == "__main__":
    arena = Arena()
    arena.add_bot(AiController)
    arena.add_bot(RandomController)
    arena.add_bot(RandomController)
    arena.add_bot(RandomController)
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
