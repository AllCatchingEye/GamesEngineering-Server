# Select Game Type

This directory contains all components to let the ai say if it wants to play a game as "Spieler". If so, it is also
capable of naming the preferred game type.

## Single Net based Decision

The dir `neuronal_network` contains the version, where only one neuronal network is used to make these decisions.

## Douple Net based Decision

The dir `two_layer_nn` contains the second attempt, where two neuronal networks are used to make these decisions.
One neuronal network tells if it wants to play the game as "Spieler", the other network tells which game type
it would play for the provided handcards. It turns out, this provides the better results.