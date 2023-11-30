import math

from logic.gamemodes.gamemode import GameMode
from state.deck import DECK
from state.player import Player
from state.ranks import Rank
from state.suits import Suit


class GameModeRamsch(GameMode):
    def __init__(self) -> None:
        trumps_init = DECK.get_cards_by_rank(Rank.OBER) + DECK.get_cards_by_rank(
            Rank.UNTER
        )
        for card in DECK.get_cards_by_suit(Suit.HERZ):
            if card not in trumps_init:
                trumps_init.append(card)

        super().__init__(Suit.HERZ, trumps_init)

    def get_game_winner(
        self, play_party: list[list[Player]]
    ) -> tuple[list[Player], list[int]]:
        """Determine the winner of the entire game."""
        party_points: list[int] = [0] * len(play_party)
        game_loser_index = 0
        highest_points = 0
        player_equal_points = []
        for i, party in enumerate(play_party):
            player = party[0]
            if player.points > highest_points:
                highest_points = player.points
                player_equal_points = [player]
            elif player.points == highest_points:
                player_equal_points.append(player)
            party_points[i] = player.points

        if len(player_equal_points) > 1:
            # Check most stitches
            most_stitches = 0
            player_most_stitches = []
            for player in player_equal_points:
                stitch_amount = player.get_amount_stitches()
                if stitch_amount > most_stitches:
                    most_stitches = stitch_amount
                    player_most_stitches = [player]
                elif stitch_amount == most_stitches:
                    player_most_stitches.append(player)
            # Check most trumps
            if len(player_most_stitches) > 1:
                most_trumps = 0
                player_most_trumps = []
                for player in player_most_stitches:
                    trumps = 0
                    for card in player.stitches:
                        if card in self.get_trump_cards():
                            trumps += 1
                    if trumps > most_trumps:
                        player_most_trumps = [player]
                        most_trumps = trumps
                    elif most_trumps == trumps:
                        player_most_trumps.append(player)
                # Check highest trump
                if len(player_most_trumps) > 1:
                    highest_trump = math.inf
                    player_highest_trump = player_most_trumps[0]
                    for player in player_most_trumps:
                        highest_player_trump = math.inf
                        for card in player.stitches:
                            if card in self.get_trump_cards():
                                highest_player_trump = min(
                                    self.get_trump_cards().index(card),
                                    highest_player_trump,
                                )
                        if highest_player_trump < highest_trump:
                            player_highest_trump = player
                            highest_trump = highest_player_trump
                    game_loser_index = play_party.index([player_highest_trump])
                else:
                    game_loser_index = play_party.index([player_most_trumps[0]])
            else:
                game_loser_index = play_party.index([player_most_stitches[0]])

        else:
            game_loser_index = party_points.index(highest_points)

        if play_party[game_loser_index][0].get_amount_stitches() == 8:
            return play_party[game_loser_index], party_points

        winners: list[Player] = []
        for i, party in enumerate(play_party):
            if i != game_loser_index:
                winners += party
        return winners, party_points
