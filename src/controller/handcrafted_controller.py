import math
import random

from mypyc.irbuild import function

from controller.player_controller import PlayerController
from logic.gamemodes.gamemode import GameMode
from logic.gamemodes.gamemode_geier import GameModeGeier
from logic.gamemodes.gamemode_ramsch import GameModeRamsch
from logic.gamemodes.gamemode_sauspiel import GameModeSauspiel
from logic.gamemodes.gamemode_solo import GameModeSolo
from logic.gamemodes.gamemode_wenz import GameModeWenz
from state.card import Card
from state.deck import DECK
from state.event import Event, GameStartUpdate, GametypeDeterminedUpdate, CardPlayedUpdate, AnnouncePlayPartyUpdate
from state.gametypes import GameGroup, Gametype
from state.hand import Hand
from state.player import PlayerId, struct_play_parties
from state.ranks import Rank, get_value_of
from state.stack import Stack
from state.suits import Suit, get_all_suits


class HandcraftedController(PlayerController):
    player_id: PlayerId
    hand: Hand
    played_cards: list[Card]
    ally: list[PlayerId]
    current_gametype: Gametype
    current_suit: Suit
    current_gamemode: GameMode
    highest_gamegroup = GameGroup | None
    valid_gamemodes: list[(Gametype, Suit)]
    play_card_gamemode: function

    def __init__(self):
        super().__init__()
        self.played_cards = []
        self.highest_gamegroup = None
        self.valid_gamemodes = []

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        if current_lowest_gamegroup.value >= 1:
            soli_gamemodes = self.is_farbsolo_valid()
            if len(soli_gamemodes) > 0:
                self.valid_gamemodes.append(soli_gamemodes)
                self.highest_gamegroup = GameGroup.HIGH_SOLO
                return True
        if current_lowest_gamegroup.value >= 2:
            wenz_gamemodes = self.is_wenz_or_geier_valid(Gametype.WENZ)
            geier_gamemodes = self.is_wenz_or_geier_valid(Gametype.GEIER)
            if wenz_gamemodes is not None or geier_gamemodes is not None:
                self.valid_gamemodes += wenz_gamemodes
                self.valid_gamemodes += geier_gamemodes
                self.highest_gamegroup = GameGroup.MID_SOLO
                return True
        if current_lowest_gamegroup.value >= 3:
            farbwenz_gamemodes = self.is_farbwenz_or_farbgeier_valid(Gametype.FARBWENZ)
            farbgeier_gamemodes = self.is_farbwenz_or_farbgeier_valid(Gametype.FARBGEIER)
            if len(farbwenz_gamemodes) > 0 or len(farbgeier_gamemodes) > 0:
                self.valid_gamemodes += farbwenz_gamemodes
                self.valid_gamemodes += farbgeier_gamemodes
                self.highest_gamegroup = GameGroup.LOW_SOLO
                return True
        if current_lowest_gamegroup.value == 4:
            if self.is_sauspiel_valid():
                self.highest_gamegroup = GameGroup.SAUSPIEL
                return True
        return False

    def is_sauspiel_valid(self) -> bool:
        gamemode = GameModeSauspiel(Suit.EICHEL)
        sauspiel_suits = get_all_suits()
        sauspiel_suits.remove(Suit.HERZ)
        trumps_in_hand = self.hand.get_all_trumps_in_deck(gamemode.get_trump_cards())
        trumps = list(gamemode.get_trump_cards())
        fehl_farben = self.get_fehl_farben(self.hand.get_all_cards(), trumps)
        fehl_asse = self.get_fehl_asse(self.hand.get_all_cards(), trumps)
        # check if sauspiel is even possible
        if len(fehl_farben) > len(fehl_asse):
            match len(trumps_in_hand):
                case trump if 6 <= trump:
                    if len(self.hand.get_all_cards_for_rank(Rank.OBER)) > 0 and len(
                            self.hand.get_all_cards_for_rank(Rank.UNTER)) > 0:
                        return True
                case 5:
                    if len(self.hand.get_all_cards_for_rank(Rank.OBER)) > 0 and len(
                            self.hand.get_all_cards_for_rank(Rank.UNTER)) > 0:
                        if len(self.get_fehl_asse(self.hand.get_all_cards(), trumps)) > 0:
                            return True
                        if len(self.get_fehl_farben(self.hand.get_all_cards(), trumps)) < 3:
                            return True
                case 4:
                    unter_groesser_herz_count = 0
                    ober_groesser_herz_count = 0
                    for trump in trumps_in_hand:
                        index = trumps.index(trump)
                        if index < 7:
                            unter_groesser_herz_count += 1
                        if index < 3:
                            ober_groesser_herz_count += 1
                    if len(fehl_asse) == 1 and len(
                            fehl_farben) == 2 and unter_groesser_herz_count > 2 and ober_groesser_herz_count > 0:
                        suit_karten_gesucht = 0
                        for card in self.hand.get_all_cards():
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
            trumps_in_hand = self.hand.get_all_trumps_in_deck(gamemode.get_trump_cards())
            trumps = list(gamemode.get_trump_cards())
            fehl_farben = self.get_fehl_farben(self.hand.get_all_cards(), trumps)
            fehl_asse = self.get_fehl_asse(self.hand.get_all_cards(), trumps)
            running_cards = self.get_running_cards(trumps)
            match len(trumps_in_hand):
                case 8:
                    if len(self.hand.get_all_cards_for_rank(Rank.OBER)) > 0 and len(
                            self.hand.get_all_cards_for_rank(Rank.UNTER)) > 0:
                        farbsoli.append((Gametype.SOLO, suit))
                case 7:
                    unter_groesser_herz_count = 0
                    ober_groesser_herz_count = 0
                    herz_zehn_groesser_count = 0
                    unter_count = 0
                    ober_count = 0
                    for trump in trumps_in_hand:
                        index = trumps.index(trump)
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
                        index = trumps.index(trump)
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
                        index = trumps.index(trump)
                        if index < 5:
                            unter_groesser_eichel_count += 1
                    if running_cards > 2 and unter_groesser_eichel_count > 3 and len(fehl_asse) > 1 and len(
                            fehl_farben) < 3:
                        farbsoli.append((Gametype.SOLO, suit))
        return farbsoli

    def is_wenz_or_geier_valid(self, gametype: Gametype) -> tuple[Gametype, None] | None:
        type = (gametype, None)
        if gametype == Gametype.WENZ:
            gamemode = GameModeWenz(None)
        elif gametype == Gametype.GEIER:
            gamemode = GameModeGeier(None)
        else:
            return None
        trumps = list(gamemode.get_trump_cards())
        trumps_in_hand = self.hand.get_all_trumps_in_deck(gamemode.get_trump_cards())
        fehl_farben = self.get_fehl_farben(self.hand.get_all_cards(), trumps)
        fehl_asse = self.get_fehl_asse(self.hand.get_all_cards(), trumps)
        running_cards = self.get_running_cards(trumps)
        match len(trumps_in_hand):
            case 4:
                if len(fehl_farben) == 1:
                    return type
                elif len(fehl_farben) == 2:
                    for ass in fehl_asse:
                        if len(self.hand.get_all_cards_for_suit(ass.suit, gamemode.get_trump_cards())) > 2:
                            return type
                    for farbe in fehl_farben:
                        koenig_groesser_count = 0
                        for card in self.hand.get_all_cards_for_suit(farbe, gamemode.get_trump_cards()):
                            if card.get_rank().value > 5:
                                koenig_groesser_count += 1
                        if koenig_groesser_count < 2:
                            return None
                    return type
                elif len(fehl_farben) == 3:
                    for ass in fehl_asse:
                        if self.hand.get_all_cards_for_suit(ass.suit, gamemode.get_trump_cards()) == 1:
                            for farbe in fehl_farben:
                                if farbe is not ass.suit:
                                    koenig_groesser_count = 0
                                    for card in self.hand.get_all_cards_for_suit(farbe, gamemode.get_trump_cards()):
                                        if card.get_rank().value > 5:
                                            koenig_groesser_count += 1
                                    if koenig_groesser_count > 1:
                                        return type
            case 3:
                if len(fehl_farben) == 1:
                    return type
                elif len(fehl_farben) == 2:
                    for ass in fehl_asse:
                        if len(self.hand.get_all_cards_for_suit(ass.suit, gamemode.get_trump_cards())) > 2:
                            for farbe in fehl_farben:
                                if farbe is not ass.suit:
                                    koenig_groesser_count = 0
                                    for card in self.hand.get_all_cards_for_suit(farbe, gamemode.get_trump_cards()):
                                        if card.get_rank().value > 5:
                                            koenig_groesser_count += 1
                                    if koenig_groesser_count > 1:
                                        return type
                elif len(fehl_farben) == 3:
                    for ass in fehl_asse:
                        if self.hand.get_all_cards_for_suit(ass.suit, gamemode.get_trump_cards()) == 1:
                            for farbe in fehl_farben:
                                koenig_groesser_count = 0
                                if farbe is not ass.suit:
                                    for card in self.hand.get_all_cards_for_suit(farbe, gamemode.get_trump_cards()):
                                        if card.get_rank().value > 5:
                                            koenig_groesser_count += 1
                                    if koenig_groesser_count > 1:
                                        for last_farbe in fehl_farben:
                                            if last_farbe is not ass.suit and last_farbe is not farbe:
                                                zehn_groesser_count = 0
                                                for card in self.hand.get_all_cards_for_suit(farbe,
                                                                                             gamemode.get_trump_cards()):
                                                    if card.get_rank().value > 6:
                                                        zehn_groesser_count += 1
                                                if zehn_groesser_count > 0:
                                                    return type
            case 2:
                if len(fehl_farben) == 1:
                    return type
                elif len(fehl_farben) == 2:
                    if running_cards == 2 and self.hand.get_all_cards_for_rank(Rank.ASS) == 2:
                        zehn_groesser_count = 0
                        for farbe in fehl_farben:
                            for card in self.hand.get_all_cards_for_suit(farbe, gamemode.get_trump_cards()):
                                if card.get_rank().value > 6:
                                    zehn_groesser_count += 1
                        if zehn_groesser_count > 3:
                            return type
                elif len(fehl_farben) == 3:
                    if running_cards > 0 and self.hand.get_all_cards_for_rank(Rank.ASS) == 2:
                        for farbe in fehl_farben:
                            suit_cards = self.hand.get_all_cards_for_suit(farbe, gamemode.get_trump_cards())
                            zehn_groesser_count = 0
                            for card in suit_cards:
                                if card.get_rank().value > 6:
                                    zehn_groesser_count += 1
                            if zehn_groesser_count > 1 and len(suit_cards) > 2:
                                return type
            case 1:
                if self.hand.get_all_cards_for_rank(Rank.ASS) == 4:
                    farben_counter = 0
                    for farbe in fehl_farben:
                        zehn_groesser_count = 0
                        for card in self.hand.get_all_cards_for_suit(farbe, gamemode.get_trump_cards()):
                            if card.get_rank().value > 6:
                                zehn_groesser_count += 1
                        if zehn_groesser_count > 1:
                            farben_counter += 1
                    if farben_counter > 1:
                        return type
            case 0:
                if self.hand.get_all_cards_for_rank(Rank.ASS) == 4:
                    farben_counter = 0
                    for farbe in fehl_farben:
                        zehn_groesser_count = 0
                        for card in self.hand.get_all_cards_for_suit(farbe, gamemode.get_trump_cards()):
                            if card.get_rank().value > 6:
                                zehn_groesser_count += 1
                        if zehn_groesser_count > 1:
                            farben_counter += 1
                    if farben_counter > 2:
                        return type
        return None

    def is_farbwenz_or_farbgeier_valid(self, gametype: Gametype) -> list[(Gametype, Suit)]:
        farbwenz_or_geier = []
        for suit in get_all_suits():
            type = (gametype, suit)
            if gametype == Gametype.FARBWENZ:
                gamemode = GameModeWenz(suit)
                high_trumps = self.hand.get_all_cards_for_rank(Rank.UNTER)
                trump_rank = Rank.UNTER
            elif gametype == Gametype.FARBGEIER:
                gamemode = GameModeGeier(suit)
                high_trumps = self.hand.get_all_cards_for_rank(Rank.OBER)
                trump_rank = Rank.OBER
            else:
                return farbwenz_or_geier
            trumps = list(gamemode.get_trump_cards())
            trumps_in_hand = self.hand.get_all_trumps_in_deck(gamemode.get_trump_cards())
            fehl_farben = self.get_fehl_farben(self.hand.get_all_cards(), trumps)
            fehl_asse = self.get_fehl_asse(self.hand.get_all_cards(), trumps)
            running_cards = self.get_running_cards(trumps)
            match len(trumps_in_hand):
                case 8:
                    farbwenz_or_geier.append(type)
                case 7:
                    if len(high_trumps) > 1:
                        farbwenz_or_geier.append(type)
                    elif running_cards > 0:
                        farbwenz_or_geier.append(type)
                case 6:
                    if len(high_trumps) > 2:
                        farbwenz_or_geier.append(type)
                    else:
                        high_trump_groesser_herz_count = 0
                        for trump in high_trumps:
                            index = trumps.index(trump)
                            if index < 3:
                                high_trump_groesser_herz_count += 1
                        if high_trump_groesser_herz_count > 1:
                            farbwenz_or_geier.append(type)
                        elif fehl_farben == 1:
                            farbwenz_or_geier.append(type)
                case 5:
                    high_trump_groesser_herz_count = 0
                    zehn_groesser_count = 0
                    for trump in high_trumps:
                        index = trumps.index(trump)
                        if index < 3:
                            high_trump_groesser_herz_count += 1
                        if index < 5:
                            zehn_groesser_count += 1
                    if high_trump_groesser_herz_count > 1 and zehn_groesser_count > 2:
                        match len(fehl_farben):
                            case 3:
                                if len(fehl_asse) > 1:
                                    farbwenz_or_geier.append(type)
                            case 2:
                                if len(fehl_asse) > 0:
                                    farbwenz_or_geier.append(type)
                            case 1:
                                farbwenz_or_geier.append(type)
        return farbwenz_or_geier

    def get_fehl_farben(self, hand: list[Card], trumps: list[Card]) -> list[Suit]:
        fehl_farben = []
        for card in hand:
            if card not in trumps and card.suit not in fehl_farben:
                fehl_farben.append(card.suit)
        return fehl_farben

    def get_fehl_asse(self, hand: list[Card], trumps: list[Card]) -> list[Card]:
        return self.get_fehl_cards_of_rank(hand, trumps, Rank.ASS)

    def get_fehl_cards_of_rank(self, hand: list[Card], trumps: list[Card], rank: Rank) -> list[Card]:
        fehl_cards_of_rank = []
        for card in hand:
            if card not in trumps and card.rank == rank:
                fehl_cards_of_rank.append(card)
        return fehl_cards_of_rank

    def get_running_cards(self, trumps: list[Card]) -> int:
        running_cards = 0
        found_running_card = True
        for trump in trumps:
            if not found_running_card:
                break
            found_running_card = False
            if trump in self.hand.get_all_cards():
                found_running_card = True
                running_cards += 1
        return running_cards

    def has_card(self, suit: Suit, rank: Rank) -> bool:
        return self.hand.get_card_of_rank_and_suit(suit, rank) is not None

    async def select_gametype(
            self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        selected_gamemode = self.rng.choice(self.valid_gamemodes)
        if selected_gamemode in choosable_gametypes:
            return selected_gamemode
        else:
            return self.rng.choice(choosable_gametypes)

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        if len(playable_cards) == 1:
            return playable_cards[0]
        return self.play_card_gamemode(stack, playable_cards)

    def play_card_solo(self, stack: Stack, playable_cards: list[Card]) -> Card:
        rng = random.Random()
        return rng.choice(playable_cards)

    def play_card_anti_solo(self, stack: Stack, playable_cards: list[Card]) -> Card:
        rng = random.Random()
        return rng.choice(playable_cards)

    def play_card_sauspiel(self, stack: Stack, playable_cards: list[Card]) -> Card:
        rng = random.Random()
        return rng.choice(playable_cards)

    def play_card_anti_sauspiel(self, stack: Stack, playable_cards: list[Card]) -> Card:
        trumps = list(self.current_gamemode.get_trump_cards())
        fehl_asse = self.get_fehl_asse(self.hand.get_all_cards(), trumps)
        suit_cards = self.hand.get_all_cards_for_suit(self.current_suit, self.current_gamemode.get_trump_cards())
        if self.ally is None:
            # try to search ass
            card_values = list(map(lambda card: get_value_of(card.rank), suit_cards))
            if len(suit_cards) == 0:
                for ass in fehl_asse:
                    if len(self.hand.get_all_cards_for_suit(ass.suit, self.current_gamemode.get_trump_cards())) == 1:
                        return ass
                non_trumps = self.hand.get_all_non_trumps_in_deck(trumps)
                non_trump_values = list(map(lambda card: get_value_of(card.rank), non_trumps))
                if len(non_trump_values) > 0:
                    return non_trumps[non_trump_values.index(min(non_trump_values))]
            elif len(suit_cards) > 2:
                return suit_cards[card_values.index(max(card_values))]
            else:
                return suit_cards[card_values.index(min(card_values))]
        else:
            # try to play together
            if len(stack.get_played_cards()) > 0:
                first_card = stack.get_played_cards()[0].get_card()
                current_stitching_player = self.current_gamemode.determine_stitch_winner(stack)
                if current_stitching_player.id in self.ally:
                    if len(stack.get_played_cards()) < 3:
                        current_stitching_card = None
                        for played_card in stack.get_played_cards():
                            if played_card.get_player().id == current_stitching_player.id:
                                current_stitching_card = played_card.get_card()
                        if current_stitching_card in trumps:
                            highest_existing_enemy_trump = self.highest_existing_trump_of_enemy(
                                self.hand.get_all_trumps_in_deck(self.current_gamemode.get_trump_cards()))
                            if highest_existing_enemy_trump is not None and trumps.index(
                                    current_stitching_card) > trumps.index(highest_existing_enemy_trump):
                                if first_card in trumps:
                                    if stack.get_value() > 11:
                                        highest_trump_hand = self.highest_existing_trump_in_hand()
                                        if highest_trump_hand is None or trumps.index(
                                                highest_trump_hand) > trumps.index(highest_existing_enemy_trump):
                                            # do not stitch give fewer points
                                            return self.search_lowest_card_value(trumps, playable_cards)
                                        else:
                                            return highest_trump_hand
                                    else:
                                        # do not stitch give fewer points
                                        return self.search_lowest_card_value(trumps, playable_cards)
                                else:
                                    remaining_suit_cards = self.search_remaining_suit_cards(self.current_suit,
                                                                                            suit_cards)
                                    # check possibility that last enemy is free
                                    if len(remaining_suit_cards) > 0:
                                        # schmieren
                                        pass
                                    else:
                                        pass
                            else:
                                # schmieren
                                pass
                        else:
                            pass
                    else:
                        # schmieren
                        pass
                else:
                    # stitch or do not stitch give fewer points
                    pass
            else:
                # play out first card
                pass
        rng = random.Random()
        return rng.choice(playable_cards)

    def play_card_ramsch(self, stack: Stack, playable_cards: list[Card]) -> Card:
        trumps = list(self.current_gamemode.get_trump_cards())
        all_cards = self.hand.get_all_cards()
        all_trumps = self.hand.get_all_trumps_in_deck(self.current_gamemode.get_trump_cards())
        # free to play any card
        if len(playable_cards) == len(all_cards):
            ass_with_least_suit_cards = self.search_fehl_card_of_rank_with_least_suit_cards(trumps, Rank.ASS)
            if ass_with_least_suit_cards is not None:
                return ass_with_least_suit_cards
            zehn_with_least_suit_cards = self.search_fehl_card_of_rank_with_least_suit_cards(trumps, Rank.ZEHN)
            if zehn_with_least_suit_cards is not None:
                return zehn_with_least_suit_cards
            if len(stack.get_played_cards()) > 1:
                # check points if you are willing to get this stitch
                if stack.get_value() < 6:
                    highest_trump = self.highest_existing_trump_in_hand()
                    if highest_trump is not None and highest_trump.get_rank().value < 6:
                        return highest_trump
            koenig_with_least_suit_cards = self.search_fehl_card_of_rank_with_least_suit_cards(trumps, Rank.KOENIG)
            if koenig_with_least_suit_cards is not None:
                return koenig_with_least_suit_cards
            fehl_farben = self.get_fehl_farben(all_cards, trumps)
            suit_cards_with_fehl_farbe = math.inf
            fehl_farbe_least_suit_cards = []
            fehl_farbe_least_suit = None
            for fehl_farbe in fehl_farben:
                fehl_farbe_cards = self.hand.get_all_cards_for_suit(fehl_farbe, self.current_gamemode.get_trump_cards())
                if len(fehl_farbe_cards) < suit_cards_with_fehl_farbe:
                    fehl_farbe_least_suit_cards = fehl_farbe_cards
                    fehl_farbe_least_suit = fehl_farbe
                    suit_cards_with_fehl_farbe = len(fehl_farbe_least_suit_cards)
            if len(fehl_farbe_least_suit_cards) > 0:
                return self.search_highest_card_of_suit(fehl_farbe_least_suit)
            else:
                rng = random.Random()
                return rng.choice(playable_cards)
        # has to play trump or suit
        else:
            highest_non_stitching_card = self.search_highest_non_stitching_card(trumps, stack, playable_cards)
            if highest_non_stitching_card is not None:
                return highest_non_stitching_card
            # check if you are willing to get this stitch with the highest trump
            elif set(playable_cards) == set(
                    self.hand.get_all_trumps_in_deck(self.current_gamemode.get_trump_cards())) and len(
                stack.get_played_cards()) > 1 and stack.get_value() < 7:
                highest_trump = self.highest_existing_trump_in_hand()
                if highest_trump is not None and highest_trump.get_rank().value < 6:
                    return highest_trump
            else:
                # play the lowest card
                return self.search_lowest_card(trumps, playable_cards)

    def schmieren(self, playable_cards: list[Card]) -> Card | None:
        trumps = self.current_gamemode.get_trump_cards()
        schmieren_card = None
        playable_cards_non_trumps = [play_card for play_card in playable_cards if play_card not in trumps]
        play_card_values_non_trumps = list(map(lambda card: get_value_of(card.rank), playable_cards_non_trumps))
        if len(play_card_values_non_trumps) > 0:
            max_val = max(play_card_values_non_trumps)
            if max_val > 0:
                max_val_non_trump_cards = [card for card in playable_cards_non_trumps
                                           if get_value_of(card.rank) == max_val]
                lowest_card_suit_count = math.inf
                if len(max_val_non_trump_cards) > 0:
                    for play_card in max_val_non_trump_cards:
                        play_card_suit_count = len(self.hand.get_all_cards_for_suit(play_card.suit, trumps))
                        if schmieren_card is None or 0 < play_card_suit_count < lowest_card_suit_count:
                            schmieren_card = play_card
                            lowest_card_suit_count = play_card_suit_count
                    return schmieren_card
        playable_cards_trumps = [play_card for play_card in playable_cards if play_card in trumps]
        play_card_values_trumps = list(map(lambda card: get_value_of(card.rank), playable_cards_trumps))
        if len(play_card_values_trumps) > 0:
            max_val = max(play_card_values_non_trumps)
            # min rank for schmieren is king
            if max_val > 3:
                return playable_cards_trumps[play_card_values_trumps.index(max_val)]
        return schmieren_card

    def search_lowest_card(self, trumps: list[Card], playable_cards: list[Card]) -> Card:
        if playable_cards in trumps:
            lowest_trump_index = 0
            for card in playable_cards:
                card_index = trumps.index(card)
                if card_index > lowest_trump_index:
                    lowest_trump_index = card_index
            return trumps[lowest_trump_index]
        else:
            lowest_value = math.inf
            lowest_card = None
            for card in playable_cards:
                if card.rank.value < lowest_value:
                    lowest_value = card.rank.value
                    lowest_card = card
            return lowest_card

    def search_lowest_card_value(self, trumps: list[Card], playable_cards: list[Card]) -> Card:
        card_values = list(map(lambda card: get_value_of(card.rank), playable_cards))
        lowest_card = None
        min_val = min(card_values)
        min_val_cards = [playable_card for playable_card in playable_cards if
                         get_value_of(playable_card.rank) == min_val]
        if min_val == 2 or min_val == 3:
            for play_card in min_val_cards:
                if lowest_card is None or trumps.index(play_card) > trumps.index(lowest_card):
                    lowest_card = play_card
        else:
            lowest_card_suit_count = math.inf
            for play_card in min_val_cards:
                play_card_suit_count = len(
                    self.hand.get_all_cards_for_suit(play_card.suit, self.current_gamemode.get_trump_cards()))
                if lowest_card is None or 0 < play_card_suit_count < lowest_card_suit_count or (
                        play_card_suit_count == lowest_card_suit_count and play_card.get_rank().value < lowest_card.rank
                        .value):
                    lowest_card = play_card
                    lowest_card_suit_count = play_card_suit_count
        return lowest_card

    def search_highest_non_stitching_card(self, trumps: list[Card], stack: Stack,
                                          playable_cards: list[Card]) -> Card | None:
        if len(stack.get_played_cards()) > 0:
            current_stitching_player = self.current_gamemode.determine_stitch_winner(stack)
            current_stitching_card = None
            highest_non_stitching_card = None
            for played_card in stack.get_played_cards():
                if played_card.get_player().id == current_stitching_player.id:
                    current_stitching_card = played_card.get_card()
            if current_stitching_card is not None:
                for card in playable_cards:
                    if [current_stitching_card, card] in trumps:
                        current_stitching_card_index = trumps.index(current_stitching_card)
                        if highest_non_stitching_card is None or current_stitching_card_index < trumps.index(
                                card) < trumps.index(highest_non_stitching_card):
                            highest_non_stitching_card = card
                    else:
                        if highest_non_stitching_card is None or current_stitching_card.rank.value > card.rank.value > highest_non_stitching_card.rank.value:
                            highest_non_stitching_card = card
            return highest_non_stitching_card
        else:
            return None

    def search_remaining_suit_cards(self, suit: Suit, own_suit_cards: list[Card]) -> list[Card]:
        remaining_suit_cards = []
        for card in DECK.get_cards_by_suit(suit):
            if card not in own_suit_cards + self.played_cards:
                remaining_suit_cards.append(card)
        return remaining_suit_cards

    def search_highest_card_of_suit(self, suit: Suit) -> Card | None:
        cards = self.hand.get_all_cards_for_suit(suit, self.current_gamemode.get_trump_cards())
        highest_rank = 0
        highest_card = None
        for card in cards:
            if card.get_rank().value > highest_rank:
                highest_rank = card.get_rank().value
                highest_card = card
        return highest_card

    def search_fehl_card_of_rank_with_least_suit_cards(self, trumps: list[Card], rank: Rank) -> Card:
        fehl_cards_of_rank = self.get_fehl_cards_of_rank(self.hand.get_all_cards(), trumps, rank)
        if len(fehl_cards_of_rank) > 0:
            suit_cards_with_card_of_rank = math.inf
            used_card = None
            for card in fehl_cards_of_rank:
                if used_card is None or len(
                        self.hand.get_all_cards_for_suit(card.suit,
                                                         self.current_gamemode.get_trump_cards())) < suit_cards_with_card_of_rank:
                    used_card = card
            return used_card

    def highest_existing_trump_in_hand(self) -> Card | None:
        trumps = self.current_gamemode.get_trump_cards()
        for trump in trumps:
            if trump in self.hand.get_all_cards():
                return trump
        return None

    def lowest_existing_trump_in_hand(self) -> Card | None:
        trumps = list(self.current_gamemode.get_trump_cards())
        trumps.reverse()
        for trump in trumps:
            if trump in self.hand.get_all_cards():
                return trump
        return None

    def highest_existing_trump(self) -> Card | None:
        trumps = self.current_gamemode.get_trump_cards()
        for trump in trumps:
            if trump not in self.played_cards:
                return trump
        return None

    def highest_existing_trump_of_enemy(self, own_trumps: list[Card]) -> Card | None:
        trumps = self.current_gamemode.get_trump_cards()
        for trump in trumps:
            if trump not in self.played_cards and trump not in own_trumps:
                return trump
        return None

    async def on_game_event(self, event: Event) -> None:
        match event:
            case GameStartUpdate():
                self.player_id = event.player
                self.hand = Hand(event.hand)
            case GametypeDeterminedUpdate():
                self.current_gametype = event.gametype
                self.current_suit = event.suit
                match self.current_gametype:
                    case Gametype.SOLO:
                        self.current_gamemode = GameModeSolo(self.current_suit)
                    case Gametype.WENZ:
                        self.current_gamemode = GameModeWenz(self.current_suit)
                    case Gametype.GEIER:
                        self.current_gamemode = GameModeGeier(self.current_suit)
                    case Gametype.FARBWENZ:
                        self.current_gamemode = GameModeWenz(self.current_suit)
                    case Gametype.FARBGEIER:
                        self.current_gamemode = GameModeGeier(self.current_suit)
                    case Gametype.SAUSPIEL:
                        self.current_gamemode = GameModeSauspiel(self.current_suit)
                    case Gametype.RAMSCH:
                        self.current_gamemode = GameModeRamsch()

                match self.current_gametype:
                    case Gametype.RAMSCH:
                        self.play_card_gamemode = self.play_card_ramsch
                    case Gametype.SAUSPIEL:
                        if event.player == self.player_id or self.has_card(self.current_suit, Rank.ASS):
                            self.play_card_gamemode = self.play_card_sauspiel
                        else:
                            self.play_card_gamemode = self.play_card_anti_sauspiel
                    case _:
                        if event.player == self.player_id:
                            self.play_card_gamemode = self.play_card_solo
                        else:
                            self.play_card_gamemode = self.play_card_anti_solo
                for party in struct_play_parties(event.parties):
                    if self.player_id in party:
                        self.ally = party
            case CardPlayedUpdate():
                self.played_cards.append(event.card)
            case AnnouncePlayPartyUpdate():
                for party in struct_play_parties(event.parties):
                    if self.player_id in party:
                        self.ally = party

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        if self.highest_gamegroup in available_groups:
            return self.highest_gamegroup
        else:
            return self.rng.choice(available_groups)
