class GamemodeSolo:

    def __init__(self, suite):
        self.suite = suite

    def get_trump_cards(self, deck) -> list[Card]:
        trump_ober = deck.get_cards_by_rank(Rank.OBER)
        trump_unter = deck.get_cards_by_rank(Rank.UNTER)
        trump_suite = deck.get_cards_by_suit(self.suite)

        trump_cards: list[Card] = trump_ober + trump_unter + trump_suite
        return trump_cards



    def finish_round(self, stack: Stack) -> None:
        pass
