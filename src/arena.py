import asyncio
import random
from dataclasses import dataclass
from typing import Callable

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


@dataclass
class ArenaConfig:
    games: int = 1000
    rounds_per_game: int = 10
    rng_seed: int | None = None


class ArenaController(PlayerController):
    actual_controller: PlayerController
    player_id: PlayerId

    money: Money
    wins: int
    played_gamemodes: list[GametypeWithSuit]

    def __init__(self, actual_controller: PlayerController) -> None:
        super().__init__()
        self.actual_controller = actual_controller
        self.money = Money(0)
        self.wins = 0
        self.played_gamemodes = []

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
            self.played_gamemodes.append(GametypeWithSuit(event.gametype, event.suit))
        if isinstance(event, MoneyUpdate) and event.player == self.player_id:
            self.money += event.money
            if event.money.cent > 0:
                self.wins += 1
        return await self.actual_controller.on_game_event(event)


class Arena:
    """Arena provides a battle ground for multiple AIs to battle, and evaluates their performance."""

    config: ArenaConfig
    bot_creators: list[Callable[[], PlayerController]]

    money: list[Money]
    wins: list[int]
    played_gamemodes: dict[GametypeWithSuit, int]

    def __init__(self, config: ArenaConfig = ArenaConfig()) -> None:
        self.config = config
        self.bot_creators = []
        self.money = []
        self.wins = []
        self.played_gamemodes = {}

    def add_bot(self, bot_creator: Callable[[], PlayerController]) -> None:
        """Provides a function to create a bot and add it to the arena."""
        self.bot_creators.append(bot_creator)
        self.money.append(Money(0))
        self.wins.append(0)

    async def run(self) -> None:
        rng = random.Random(self.config.rng_seed)
        for game in range(self.config.games):
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

            # Update played gamemodes, only first controller is enough as all controllers play the same
            for gamemode in controllers[0].played_gamemodes:
                self.played_gamemodes[gamemode] = (
                    self.played_gamemodes.get(gamemode, 0) + 1
                )

    def print_results(self) -> None:
        print(
            f"Played {self.config.games} games with {self.config.rounds_per_game} rounds each"
        )

        total_rounds = self.config.games * self.config.rounds_per_game

        for i in range(len(self.bot_creators)):
            money = self.money[i]
            win_rate = self.wins[i] / total_rounds
            print(f"Bot {i}: {money} ({win_rate*100:.2f}%)")

        print()
        print("Played gamemodes percentage:")
        played_gamemodes = sorted(
            self.played_gamemodes.items(), key=lambda x: x[1], reverse=True
        )
        for gamemode, played in played_gamemodes:
            print(f"{gamemode}: {played/total_rounds*100:.2f}%")


if __name__ == "__main__":
    arena = Arena()
    arena.add_bot(RandomController)
    arena.add_bot(RandomController)
    arena.add_bot(RandomController)
    arena.add_bot(RandomController)
    asyncio.run(arena.run())
    arena.print_results()
