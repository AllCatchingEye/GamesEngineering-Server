import random
from typing import Sequence

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
    AnnouncePlayPartyUpdate,
    CardPlayedUpdate,
    Event,
    GameEndUpdate,
    GameGroupChosenUpdate,
    GameStartUpdate,
    GametypeDeterminedUpdate,
    MoneyUpdate,
    PlayDecisionUpdate,
    PlayOrderUpdate,
    RoundResultUpdate,
)
from state.gametypes import GameGroup, Gametype, stake_for_gametype
from state.hand import Hand
from state.money import Money
from state.player import Player, PlayerId, play_parties_to_struct
from state.ranks import Rank
from state.running_cards_start import RunningCardsStart
from state.stack import Stack
from state.stakes import Stake
from state.suits import Suit

HAND_SIZE = 8
ROUNDS = 8
PLAYER_COUNT = 4


class Game:
    controllers: Sequence[PlayerController]
    rng: random.Random
    gamemode: GameMode
    gametype: Gametype
    suit: Suit | None
    players: list[Player]
    deck: Deck
    played_cards: list[Card]
    games_played: int

    def __init__(self, rng: random.Random = random.Random()) -> None:
        self.players = self.__create_players()
        self.deck: Deck = Deck()
        self.played_cards: list[Card] = []
        self.controllers = []
        self.rng = rng
        self.games_played = 0
        self.gametype = None
        # In a game there are always two parties (player-party, non-player-party)
        self.play_party: list[list[Player]] = []

    def __create_players(self) -> list[Player]:
        """Create a list of players for the game."""
        players: list[Player] = []
        for i in range(PLAYER_COUNT):
            players.append(Player(i, i))
        return players
    
    async def run(self, games_to_play: int = 1) -> None:
        """Start the game."""
        while self.games_played < games_to_play:
            self.gametype = await self.determine_gametype()
            await self.__new_game()
            self.__prepare_new_game()

    async def determine_gametype(self) -> Gametype:
        """Determine the game type based on player choices."""
        await self.__distribute_cards()
        await self.announce_hands()

        if self.gametype is not None:
            return self.gametype

        player, game_group = await self.__get_player()
        return await self.__select_gametype(player, game_group)

    async def __distribute_cards(self) -> None:
        """Distribute cards to players."""
        # check if the first player already has cards
        if len(self.players[0].hand.cards) > 0:
            return

        deck: list[Card] = self.deck.get_full_deck()
        self.rng.shuffle(deck)

        for player in self.players:
            deck = await self.__distribute_hand(player, deck)

    async def __distribute_hand(self, player: Player, deck: list[Card]) -> list[Card]:
        """Distribute cards for a player's hand."""
        hand: Hand = Hand(deck[:HAND_SIZE])
        player.hand = hand

        deck = deck[HAND_SIZE:]

        return deck

    async def announce_hands(self) -> None:
        for player in self.players:
            await self.controllers[player.slot_id].on_game_event(
                GameStartUpdate(
                    player.id,
                    player.hand.get_all_cards(),
                    [player.id for player in self.players],
                )
            )

    async def __get_player(self) -> tuple[Player | None, list[GameGroup]]:
        """Call the game type based on player choices."""
        current_player: Player | None = None
        current_player_index: int | None = None
        current_game_group: list[GameGroup] = [
            GameGroup.SAUSPIEL,
            GameGroup.LOW_SOLO,
            GameGroup.MID_SOLO,
            GameGroup.HIGH_SOLO,
        ]

        for i, player in enumerate(self.players):
            wants_to_play = await self.controllers[player.slot_id].wants_to_play(
                current_game_group[0]
            )
            await self.__broadcast(PlayDecisionUpdate(player.id, wants_to_play))
            if wants_to_play:
                current_player = player
                current_player_index = i
                break

        # No one called a game => Ramsch
        if current_player is None or current_player_index is None:
            return None, current_game_group

        if current_player_index < 3:
            for player in self.players[current_player_index + 1 :]:
                # High-Solo has been called, there is no higher game group
                if len(current_game_group) == 1:
                    break

                wants_to_play = await self.controllers[player.slot_id].wants_to_play(
                    current_game_group[1]
                )
                await self.__broadcast(PlayDecisionUpdate(player.id, wants_to_play))
                if wants_to_play:
                    player_decision = await self.controllers[
                        current_player.slot_id
                    ].choose_game_group(current_game_group)

                    current_game_group_reduced = current_game_group.copy()
                    current_game_group_reduced.pop(0)
                    oponent_decision = await self.controllers[
                        player.slot_id
                    ].choose_game_group(current_game_group_reduced)

                    if oponent_decision.value < player_decision.value:
                        current_player = player
                        index_reduce = current_game_group.index(oponent_decision)
                        current_game_group = current_game_group[index_reduce:]
                        await self.__broadcast(
                            GameGroupChosenUpdate(player.id, current_game_group)
                        )
                        continue

                    index_reduce = current_game_group.index(player_decision)
                    current_game_group = current_game_group[index_reduce:]
                    await self.__broadcast(
                        GameGroupChosenUpdate(current_player.id, current_game_group)
                    )
        return current_player, current_game_group

    async def __select_gametype(
        self, game_player: Player | None, minimum_game_group: list[GameGroup]
    ) -> Gametype:
        if game_player is None:
            self.play_party = [
                [self.players[0]],
                [self.players[1]],
                [self.players[2]],
                [self.players[3]],
            ]
            await self.__broadcast(
                GametypeDeterminedUpdate(
                    None,
                    Gametype.RAMSCH,
                    None,
                    play_parties_to_struct(
                        [[player.id for player in party] for party in self.play_party]
                    ),
                )
            )
            self.gamemode = GameModeRamsch()
            return Gametype.RAMSCH

        game_type = await self.controllers[game_player.slot_id].select_gametype(
            get_playable_gametypes(game_player.hand, minimum_game_group)
        )

        # When playing solo it is always 1v3
        player_party = [self.players[game_player.turn_order]]
        non_player_party = self.players.copy()
        non_player_party.remove(self.players[game_player.turn_order])

        match (game_type[0]):
            case Gametype.SOLO:
                suit = game_type[1]
                if suit is None:
                    raise ValueError("Solo gametype chosen without suit")
                self.gamemode = GameModeSolo(suit)
            case Gametype.WENZ:
                self.gamemode = GameModeWenz(None)
            case Gametype.GEIER:
                self.gamemode = GameModeGeier(None)
            case Gametype.FARBWENZ:
                self.gamemode = GameModeWenz(game_type[1])
            case Gametype.FARBGEIER:
                self.gamemode = GameModeGeier(game_type[1])
            case Gametype.SAUSPIEL:
                suit = game_type[1]
                if suit is None:
                    raise ValueError("Sauspiel gametype chosen without suit")

                # Find Player who has the chosen ace
                player_party = [self.players[game_player.turn_order]]
                for j, player in enumerate(self.players):
                    if player.hand.has_card_of_rank_and_suit(suit, Rank.ASS):
                        player_party.append(self.players[j])
                non_player_party = self.players.copy()
                non_player_party.remove(player_party[0])
                non_player_party.remove(player_party[1])

                self.gamemode = GameModeSauspiel(suit)
            case Gametype.RAMSCH:
                # invalid gamemode, cannot be chosen
                raise ValueError("Ramsch cannot be chosen as a gametype")

        self.play_party = [player_party, non_player_party]

        await self.__broadcast(
            GametypeDeterminedUpdate(
                self.players[game_player.turn_order].id,
                game_type[0],
                game_type[1],
                play_parties_to_struct(
                    [[player.id for player in party] for party in self.play_party]
                )
                if game_type[0] != Gametype.SAUSPIEL
                else None,
            )
        )
        if game_type[0] != Gametype.SAUSPIEL:
            for player in self.players:
                if Card(game_type[1], Rank.ASS) in player.hand.get_all_cards():
                    await self.controllers[player.slot_id].on_game_event(
                        AnnouncePlayPartyUpdate(
                            play_parties_to_struct(
                                [
                                    [player.id for player in party]
                                    for party in self.play_party
                                ]
                            )
                        )
                    )
        return game_type[0]

    async def __new_game(self) -> None:
        """Start a new game with the specified suit as the game type."""
        game_winner, points_distribution = None, None
        for _ in range(ROUNDS):
            await self.__broadcast(
                PlayOrderUpdate([player.id for player in self.players])
            )
            await self.start_round()

        game_winner, points_distribution = self.gamemode.get_game_winner(
            self.play_party
        )

        await self.__get_or_pay_money(game_winner, points_distribution)

        await self.__broadcast(
            GameEndUpdate(
                [winner.id for winner in game_winner],
                play_parties_to_struct(
                    [[player.id for player in party] for party in self.play_party]
                ),
                points_distribution,
            )
        )

    def __prepare_new_game(self) -> None:
        self.gametype = None
        self.gamemode = None
        self.suit = None
        swap_index = -1
        for i, player in enumerate(self.players):
            # cycling players for next game
            player.turn_order = (player.turn_order - 1) % PLAYER_COUNT
            if player.turn_order == 0:
                swap_index = i

            player.reset()
        self.__swap_players(swap_index)
        self.games_played += 1

    async def start_round(self) -> None:
        """Start a new round."""
        stack = await self.__play_cards()
        await self.__finish_round(stack)

    async def __play_cards(self) -> Stack:
        """Play cards in the current round."""
        stack = Stack()
        for player in self.players:
            playable_cards = self.gamemode.get_playable_cards(stack, player.hand)
            if len(playable_cards) == 0:
                raise ValueError("No playable cards")
            card: Card = await self.controllers[player.slot_id].play_card(
                stack, playable_cards
            )
            if card not in playable_cards or card not in player.hand.cards:
                raise ValueError(
                    f"Illegal card played, tried to play {card} from {player.hand} but only {playable_cards} are allowed"
                )
            player.lay_card(card)
            stack.add_card(card, player.id)
            await self.__broadcast(CardPlayedUpdate(player.id, card))

            # Announce that the searched ace had been played and teams are known
            if (
                isinstance(self.gamemode, GameModeSauspiel)
                and self.gamemode.suit
                and card == Card(self.gamemode.suit, Rank.ASS)
            ):
                await self.__broadcast(
                    AnnouncePlayPartyUpdate(
                        play_parties_to_struct(
                            [
                                [player.id for player in party]
                                for party in self.play_party
                            ]
                        )
                    )
                )

        return stack

    async def __finish_round(self, stack: Stack) -> None:
        """Finish the current round and determine the winner."""
        winner = self.gamemode.determine_stitch_winner(stack)
        stack_value = stack.get_value()
        for player in self.players:
            if player.id == winner:
                player.points += stack_value
                for card in stack.get_played_cards():
                    player.stitches.append(card.get_card())
                break
        await self.__broadcast(RoundResultUpdate(winner, stack_value))
        self.__change_player_order(winner)

    async def __get_or_pay_money(
        self, game_winner: list[Player], points_distribution: list[int]
    ) -> None:
        stake: Money = stake_for_gametype[self.gametype].value
        if self.gametype == Gametype.RAMSCH:
            # Check virgin
            if len(game_winner) > 1:
                for player in game_winner:
                    if player.get_amount_stitches() == 0:
                        stake += Stake.STANDARD.value
            else:
                stake *= 3
        else:
            for points in points_distribution:
                if points == 0:
                    # schwarz
                    stake += Stake.STANDARD.value
                if points > 90:
                    # schneiderfree
                    stake += Stake.STANDARD.value
            for team in self.play_party:
                running_team_cards = self.__get_running_cards(team)
                if self.gamemode is GameModeGeier or self.gamemode is GameModeWenz:
                    stakes_added = (
                        running_team_cards - RunningCardsStart.GEIER_WENZ.value
                    )
                else:
                    stakes_added = running_team_cards - RunningCardsStart.STANDARD.value
                if stakes_added >= 0:
                    stake += Stake.STANDARD.value * (stakes_added + 1)
        for player in self.players:
            if player in game_winner:
                player.money += stake * (
                    max(len(game_winner), len(self.players) - len(game_winner))
                    // (len(game_winner))
                )
            else:
                player.money -= stake * (
                    max(len(game_winner), len(self.players) - len(game_winner))
                    // (len(self.players) - len(game_winner))
                )
            await self.controllers[player.slot_id].on_game_event(
                MoneyUpdate(player.id, player.money)
            )

    def __get_running_cards(self, team: list[Player]) -> int:
        running_cards = 0
        found_running_card = True
        for trump in self.gamemode.get_trump_cards():
            if not found_running_card:
                break
            found_running_card = False
            for player in team:
                if trump in player.played_cards:
                    found_running_card = True
                    running_cards += 1
                    break
        return running_cards

    def __change_player_order(self, winner: PlayerId) -> None:
        """Change the order of players based on the round winner."""
        winner_index = self.__get_winner_index(winner)
        self.__swap_players(winner_index)

    def __get_winner_index(self, winner: PlayerId) -> int:
        """Find the index of the player who won"""
        winner_index = 0
        for index, player in enumerate(self.players):
            if player.id == winner:
                winner_index = index
        return winner_index

    def __swap_players(self, index: int) -> None:
        first: list[Player] = self.players[index:]
        last: list[Player] = self.players[:index]
        self.players = first + last

    async def __broadcast(self, event: Event) -> None:
        """Broadcast an event to all players."""
        for controller in self.controllers:
            await controller.on_game_event(event)
