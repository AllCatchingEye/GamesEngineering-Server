from logic.move import Move
from state.cards import Card

if __name__ == "__main__":
    card = Card("A", "Spades")
    print(card)

    move = Move(card)
    print(move)
