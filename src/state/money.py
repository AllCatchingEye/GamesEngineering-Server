from dataclasses import dataclass


@dataclass
class Money:
    cent: int

    def __int__(self, cent):
        self.cent = cent

    def __str__(self) -> str:
        factor = 100
        sign = ""
        if self.__is_negative():
            factor *= -1
            sign = "-"
        return sign + f"{self.cent // factor},{abs(self.cent % factor)}€"

    def __add__(self, other):
        if isinstance(other, Money):
            return Money(self.cent + other.cent)
        else:
            raise ValueError("Object to be added has to be of type Money")

    def __sub__(self, other):
        if isinstance(other, Money):
            return Money(self.cent - other.cent)
        else:
            raise ValueError("Object to be subtracted has to be of type Money")

    def __mul__(self, other):
        if isinstance(other, int):
            return Money(self.cent * other)
        else:
            raise ValueError("Has to be a multiplication with Integer")

    def __is_negative(self) -> bool:
        return self.cent < 0

    @staticmethod
    def from_euro(euro: int):
        return Money(euro * 100)

    @staticmethod
    def from_euro_and_cent(euro: int, cent: int):
        if euro <= 0 and cent < 0 or euro >= 0 and cent >= 0:
            return Money(euro * 100 + cent)
        else:
            raise ValueError("Euro and Cent do not have the same sign")
