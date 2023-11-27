# Concept

```js
const stack = [
    [card, player],
    [card, player],
    [card, player],
    [card, player],
    [card, player],
    [card, player],
];
const playable_cards = [card, card, card, card]

processed_stack = stack.flatMap((card, player) => [encode_card_as_int(card), isAlly(player)])
processed_hand_cards = encode_cards_as_one_hot(playable_cards)

const input = [
    ...processed_stack,
    ...processed_hand_cards
]

[
    2, // enemy
    0, // us
    2, // enemy 
    1, // ally

    0, // card 0
    1, // is ally
    2, // card 2
    1, // is ally
    13,// card 13
    0, // is not ally
    15,// card 15
    0, // is not ally
    ...restStack,
    ...new Array(32 - processed_stack.length).fill(-1),

    1, // Eichel Ober available
    0, // Eichel Unter not available
    1, // Eichel Ass available
    ...restHandCards
    ,
]
```

# Weiterer Vorschlag 
# + Sehr einfach zu implementieren aufgrund vorhandener funktionen
# - Viele inputs sind 0 (Overhead)
# - Spieler anordung fehlt noch

[
    # played cards (one hot, 32x)
    0
    1
    0
    0
    ...
    1
    # is allied card (one hot, 32x)
    0
    1
    0
    0
    ...
    0
    # hand cards (one hot, 32x)
    0
    1
    0
    0
    ...
    1
]