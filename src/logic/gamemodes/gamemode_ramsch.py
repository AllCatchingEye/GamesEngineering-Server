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
        party_points: list[int] = []
        for i, party in enumerate(play_party):
            for player in play_party[0]:
                party_points[i] += player.points

        game_loser_index = party_points.index(max(party_points))
        if party_points[game_loser_index] == 120:
            return play_party[game_loser_index], party_points
        else:
            winners = []
            for i, party in enumerate(play_party):
                if i != game_loser_index:
                    winners += party
            return winners, party_points
