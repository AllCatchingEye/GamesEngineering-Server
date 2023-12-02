from ai.dealer.dealer import Dealer
from ai.dealer.dealer_geier import DealerGeier
from ai.dealer.dealer_ramsch import DealerRamsch
from ai.dealer.dealer_sauspiel import DealerSauspiel
from ai.dealer.dealer_solo import DealerSolo
from ai.dealer.dealer_wenz import DealerWenz
from state.gametypes import Gametype


class DealerFactory:
    @staticmethod
    def get_dealer(game_type: Gametype, seed: int | None = None) -> Dealer:
        if game_type == Gametype.SAUSPIEL:
            return DealerSauspiel(seed)
        elif game_type == Gametype.SOLO:
            return DealerSolo(seed)
        elif game_type == Gametype.FARBGEIER or game_type == Gametype.GEIER:
            return DealerGeier(seed)
        elif game_type == Gametype.FARBWENZ or game_type == Gametype.WENZ:
            return DealerWenz(seed)
        elif game_type == Gametype.RAMSCH:
            return DealerRamsch(seed)

        raise NotImplementedError(f"Dealer for game type {game_type} not implemented")
