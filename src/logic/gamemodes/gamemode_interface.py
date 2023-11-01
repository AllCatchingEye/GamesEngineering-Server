class GamemodeMeta(type):
    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        # TODO add all needed methods
        return (hasattr(subclass, 'get_trump_cards') and
                callable(subclass.get_trump_cards) and
                hasattr(subclass, 'finish_round') and
                callable(subclass.finish_round))

    def get_playable_cards(self, stack, hand, deck) -> list[Card]:
        if stack.is_empty():
            return hand.get_all_cards()
        else:
            played_suit = stack.get_first_card().get_suit()
            is_trump_round = self.get_trump_cards(deck).contains(stack.get_first_card())

            if is_trump_round:
                trump_cards = hand.get_all_trumps_in_deck(self.get_trump_cards(deck))
                return trump_cards if len(trump_cards) > 0 else player_hand.get_all_cards()
            else:
                cards_of_played_suit = player_hand.get_all_cards_for_suit(played_suit)
                return cards_of_played_suit if len(cards_of_played_suit) > 0 else player_hand.get_all_cards()

    def get_stich_winner(self, stack, deck) -> Player:
        if stack.is_empty():
            # """ Cant determine winner if there are no cards played """
            return None
        else:
            trump_cards = list(filter(lambda card: self.get_trump_cards(deck).contains(card), stack.get_played_cards()))
            # """ At least one trump card was played """
            if len(trump_cards) > 0:


class GamemodeInterface(metaclass=GamemodeMeta):
    pass
