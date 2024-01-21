# Input params

The input parameters for this model consists of 
- the order of allies, enemies and us (integer encoded)
- the current stack ()
- the previous stacks
- the playable cards

```py
[
    # order people sitting at the table
    1, # enemy
    3, # us
    1, # enemy 
    2, # ally

    # current stack: consists of the order of the played cards (14 first, 0 second, …) and then the one-hot-encoding encoding of the cards + if played by ally
    # the order
    15, # card 14 was played fist
    1, # card 0 was played second
    2, # card 1 was played third
    0, # card wasn't played, yet
    # encoding of cards + allies
    1, # card 0 was played
    2, # card 0 was played from an ally
    1, # card 1 was played
    1, # card 1 was played by an enemy
    0, # card 2 wasn't played, yet
    0, # N/A
    …

    # previous stacks: same as for current stack expecpt it is repeated 3 times
    13, # card 12 was played first (first stack)
    …
    0, # card 0 wasn't played, yet
    0, # N/A
    …
    5, # card 4 was played first (second stack)
    …
    0, # card 0 wasn't played, yet
    0, # N/A
    …
    … # for all played stacks
    ...[0] * (4 * (8 + 32 + 32) - len(stacks))

    # hand cards
    1, # Eichel Ober available
    0, # Eichel Unter not available
    1, # Eichel Ass available
    ...restHandCards
    ,
]
```