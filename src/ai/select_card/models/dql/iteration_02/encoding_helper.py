from ai.nn_helper import (
    NUM_CARDS,
    NUM_ROUNDS,
    NUM_STACK_CARDS,
    get_one_hot_encoding_index_from_card,
)
from state.card import Card
from state.player import PlayerId

NUM_SITTING_ORDER = NUM_STACK_CARDS
NUM_ENCODED_STACK = NUM_STACK_CARDS + NUM_CARDS * 2
NUM_ENCODED_PREVIOUS_STACKS = NUM_ENCODED_STACK * (NUM_ROUNDS - 1)
NUM_ENCODED_HAND_CARDS = NUM_CARDS


class Encoding:
    NOT_PLAYED = 0
    PLAYED = 1
    ENEMY = 1
    ALLY = 2
    US = 3


def encode_stack(
    stack: list[tuple[Card, PlayerId]], allies: list[PlayerId]
) -> list[int]:
    # embed the order of the stack + for every card if it was played and in case if from an ally
    result = [Encoding.NOT_PLAYED] * NUM_ENCODED_STACK
    for index, [card, player_id] in enumerate(stack):
        if index >= NUM_STACK_CARDS:
            raise ValueError(
                "The stack contains more cards than allowed: %i vs. max %i"
                % (len(stack), NUM_STACK_CARDS)
            )

        encoded_index = get_one_hot_encoding_index_from_card(card)
        # Make sure we can differentiate not set values (0) and indices (>= 0)
        result[index] = encoded_index + 1
        result[NUM_STACK_CARDS + encoded_index * 2 + 0] = Encoding.PLAYED
        result[NUM_STACK_CARDS + encoded_index * 2 + 1] = (
            Encoding.ALLY if player_id in allies else Encoding.ENEMY
        )

    return result
