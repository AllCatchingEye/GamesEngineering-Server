from controller.player_controller import PlayerController
from logic.gamemodes.gamemode_sauspiel import GameModeSauspiel
from logic.gamemodes.gamemode_solo import GameModeSolo
from logic.gamemodes.gamemode_wenz import GameModeWenz
from src.state.event import *
from state.gametypes import GameGroup, Gametype
from state.player import Player
from state.ranks import Rank
from state.stack import Stack
from state.suits import Suit, get_all_suits


class HandcraftedController(PlayerController):
    hand: list[Card]
    player: Player
    played_cards: list[Card]
    ally: list[Player]
    current_gametype: Gametype
    valid_gamegroups: list[GameGroup]
    valid_gamemodes: list[(Gametype, Suit)]

    def __init__(self, player: Player):
        super().__init__(player)
        self.player = player
        self.played_cards = []
        self.valid_gamegroups = []
        self.valid_gamemodes = []

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        if current_lowest_gamegroup.value == 4:
            if self.is_sauspiel_valid():
                self.valid_gamegroups.append(GameGroup.SAUSPIEL)
        if current_lowest_gamegroup.value >= 3:

        if current_lowest_gamegroup.value >= 1:
            soli_gamemodes = self.is_farbsolo_valid()
            if len(soli_gamemodes) > 0:
                self.valid_gamemodes.append(soli_gamemodes)
                self.valid_gamegroups.append(GameGroup.HIGH_SOLO)
        if len(self.valid_gamegroups) > 0:
            return True

    def is_sauspiel_valid(self) -> bool:
        gamemode = GameModeSauspiel(Suit.EICHEL)
        sauspiel_suits = get_all_suits()
        sauspiel_suits.remove(Suit.HERZ)
        trumps_in_hand = self.player.hand.get_all_trumps_in_deck(gamemode.get_trump_cards())
        fehl_farben = self.get_fehl_farben(self.hand, gamemode.get_trump_cards())
        fehl_asse = self.get_fehl_asse(self.hand, gamemode.get_trump_cards())
        # check if sauspiel is even possible
        if len(fehl_farben) > len(fehl_asse):
            match len(trumps_in_hand):
                case trump if 6 <= trump:
                    if len(self.player.hand.get_all_cards_for_rank(Rank.OBER)) > 0 and len(
                            self.player.hand.get_all_cards_for_rank(Rank.UNTER)) > 0:
                        return True
                case 5:
                    if len(self.player.hand.get_all_cards_for_rank(Rank.OBER)) > 0 and len(
                            self.player.hand.get_all_cards_for_rank(Rank.UNTER)) > 0:
                        if len(self.get_fehl_asse(self.hand, gamemode.get_trump_cards())) > 0:
                            return True
                        if len(self.get_fehl_farben(self.hand, gamemode.get_trump_cards())) < 3:
                            return True
                case 4:
                    unter_groesser_herz_count = 0
                    ober_groesser_herz_count = 0
                    for trump in trumps_in_hand:
                        index = gamemode.get_trump_cards().index(trump)
                        if index < 7:
                            unter_groesser_herz_count += 1
                        if index < 3:
                            ober_groesser_herz_count += 1
                    if len(fehl_asse) == 1 and len(
                            fehl_farben) == 2 and unter_groesser_herz_count > 2 and ober_groesser_herz_count > 0:
                        suit_karten_gesucht = 0
                        for card in self.hand:
                            if card not in trumps_in_hand and card.suit in fehl_farben and card.suit != fehl_asse[
                                0].get_suit():
                                suit_karten_gesucht += 1
                        if suit_karten_gesucht < 3:
                            return True
                    if len(fehl_asse) == 2 and len(
                            fehl_farben) == 3 and unter_groesser_herz_count > 2 and ober_groesser_herz_count > 0:
                        return True
                case _:
                    return False

    def is_farbsolo_valid(self) -> list[(Gametype, Suit)]:
        farbsoli = []
        for suit in get_all_suits():
            gamemode = GameModeSolo(suit)
            trumps_in_hand = self.player.hand.get_all_trumps_in_deck(gamemode.get_trump_cards())
            fehl_farben = self.get_fehl_farben(self.hand, gamemode.get_trump_cards())
            fehl_asse = self.get_fehl_asse(self.hand, gamemode.get_trump_cards())
            running_cards = self.get_running_cards(gamemode.get_trump_cards())
            match len(trumps_in_hand):
                case 8:
                    if len(self.player.hand.get_all_cards_for_rank(Rank.OBER)) > 0 and len(
                            self.player.hand.get_all_cards_for_rank(Rank.UNTER)) > 0:
                        farbsoli.append((Gametype.SOLO, suit))
                case 7:
                    unter_groesser_herz_count = 0
                    ober_groesser_herz_count = 0
                    herz_zehn_groesser_count = 0
                    unter_count = 0
                    ober_count = 0
                    for trump in trumps_in_hand:
                        index = gamemode.get_trump_cards().index(trump)
                        if index < 10:
                            herz_zehn_groesser_count += 1
                        if index < 8:
                            unter_count += 1
                        if index < 7:
                            unter_groesser_herz_count += 1
                        if index < 4:
                            ober_count += 1
                        if index < 3:
                            ober_groesser_herz_count += 1
                    if unter_groesser_herz_count > 3 and ober_count > 1 and self.has_card(Suit.EICHEL, Rank.OBER):
                        farbsoli.append((Gametype.SOLO, suit))
                    elif unter_count > 2 and ober_groesser_herz_count > 0 and herz_zehn_groesser_count > 4 and len(
                            fehl_asse) == 1:
                        farbsoli.append((Gametype.SOLO, suit))
                case 6:
                    unter_groesser_eichel_count = 0
                    herz_zehn_groesser_count = 0
                    unter_count = 0
                    for trump in trumps_in_hand:
                        index = gamemode.get_trump_cards().index(trump)
                        if index < 10:
                            herz_zehn_groesser_count += 1
                        if index < 8:
                            unter_count += 1
                        if index < 5:
                            unter_groesser_eichel_count += 1
                    if running_cards > 1 and unter_groesser_eichel_count > 2 and unter_count > 4 and herz_zehn_groesser_count > 5:
                        if len(fehl_asse) > 1:
                            farbsoli.append((Gametype.SOLO, suit))
                        elif len(fehl_asse) == 1 and len(fehl_farben) == 1:
                            farbsoli.append((Gametype.SOLO, suit))
                case 5:
                    unter_groesser_eichel_count = 0
                    for trump in trumps_in_hand:
                        index = gamemode.get_trump_cards().index(trump)
                        if index < 5:
                            unter_groesser_eichel_count += 1
                    if running_cards > 2 and unter_groesser_eichel_count > 3 and len(fehl_asse) > 1 and len(
                            fehl_farben) < 3:
                        farbsoli.append((Gametype.SOLO, suit))
        return farbsoli

    def is_wenz_valid(self) -> (Gametype, Suit) | None:
        wenz = (Gametype.WENZ, None)
        gamemode = GameModeWenz(None)
        trumps_in_hand = self.player.hand.get_all_trumps_in_deck(gamemode.get_trump_cards())
        fehl_farben = self.get_fehl_farben(self.hand, gamemode.get_trump_cards())
        fehl_asse = self.get_fehl_asse(self.hand, gamemode.get_trump_cards())
        running_cards = self.get_running_cards(gamemode.get_trump_cards())
        match len(trumps_in_hand):
            case 4:
                if len(fehl_farben) == 1:
                    return wenz
                elif len(fehl_farben) == 2:
                    for ass in fehl_asse:
                        if len(self.player.hand.get_all_cards_for_suit(ass.suit, gamemode.get_trump_cards())) > 2:
                            return wenz
            case 3:
            case 2:
            case 1:
            case 0:
        return None
    def get_fehl_farben(self, hand: list[Card], trumps: list[Card]) -> list[Suit]:
        fehl_farben = []
        for card in hand:
            if card not in trumps and card.suit not in fehl_farben:
                fehl_farben.append(card.suit)
        return fehl_farben

    def get_fehl_asse(self, hand: list[Card], trumps: list[Card]) -> list[Card]:
        fehl_asse = []
        for card in hand:
            if card not in trumps and card.rank == Rank.ASS:
                fehl_asse.append(card)
        return fehl_asse

    def get_running_cards(self, trumps: list[Card]) -> int:
        running_cards = 0
        found_running_card = True
        for trump in trumps:
            if not found_running_card:
                break
            found_running_card = False
            if trump in self.hand:
                found_running_card = True
                running_cards += 1
        return running_cards

    def has_card(self, suit: Suit, rank: Rank) -> bool:
        return self.player.hand.get_card_of_rank_and_suit(suit, rank) is not None

    async def select_gametype(
            self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        return self.rng.choice(choosable_gametypes)

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        return self.rng.choice(playable_cards)

    async def on_game_event(self, event: Event) -> None:
        match event:
            case GameStartUpdate():
                self.hand = event.hand
            case GametypeDeterminedUpdate():
                self.current_gametype = event.gametype
                for party in event.parties:
                    if self.player in party:
                        self.ally = party
            case CardPlayedUpdate():
                self.played_cards.append(event.card)

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        return self.rng.choice(available_groups)
