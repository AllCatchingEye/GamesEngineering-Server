from ai.dealer.dealer import Dealer
from ai.dealer.dealer_sauspiel import DealerSauspiel
from state.gametypes import Gametype


class DealerFactory:
    @staticmethod
    def get_dealer(game_type: Gametype, seed: int | None = None) -> Dealer:
        if game_type == Gametype.SAUSPIEL:
            return DealerSauspiel(seed)

        raise NotImplementedError(f"Dealer for game type {game_type} not implemented")
