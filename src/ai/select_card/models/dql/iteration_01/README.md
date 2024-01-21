# Input parameters

The input parameters for this model consists of 
- the current stack (one hot encoded, without keeping track of order)
- the allies (one hot encoded for every card: 0 if from enemy or not played, 1 if from ally)
- the playable cards (one hot encoded)

```py
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
```