from dataclasses import dataclass


@dataclass
class Money:
    cent: int

    def __str__(self) -> str:
        factor = 100
        sign = ""
        if self.__is_negative():
            factor *= -1
            sign = "-"
        return sign + f"{self.cent // factor},{abs(self.cent % factor)}€"

    def __add__(self, other: object) -> "Money":
        if isinstance(other, Money):
            return Money(self.cent + other.cent)
        raise ValueError("Object to be added has to be of type Money")

    def __sub__(self, other: object) -> "Money":
        if isinstance(other, Money):
            return Money(self.cent - other.cent)
        raise ValueError("Object to be subtracted has to be of type Money")

    def __mul__(self, other: object) -> "Money":
        if isinstance(other, int):
            return Money(self.cent * other)
        raise ValueError("Has to be a multiplication with Integer")

    def __is_negative(self) -> bool:
        return self.cent < 0

    # comparison operators
    def __lt__(self, other: object) -> bool:
        if isinstance(other, Money):
            return self.cent < other.cent
        raise ValueError("Object to be compared has to be of type Money")

    @staticmethod
    def from_euro(euro: int) -> "Money":
        return Money.from_euro_and_cent(euro, 0)

    @staticmethod
    def from_euro_and_cent(euro: int, cent: int) -> "Money":
        if euro <= 0 and cent < 0 or euro >= 0 and cent >= 0:
            return Money(euro * 100 + cent)
        raise ValueError("Euro and Cent do not have the same sign")
