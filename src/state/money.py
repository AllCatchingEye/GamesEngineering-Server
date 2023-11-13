from dataclasses import dataclass


@dataclass
class Money:
    euro: int
    cent: int

    def __is_negative(self) -> bool:
        return self.euro < 0 or self.cent < 0

    def __int__(self, euro, cent):
        if 0 < cent < 100:
            self.cent = cent
            self.euro = euro
        else:
            raise ValueError("Maximum cents is 99 and Minimum is 0")

    def __str__(self) -> str:
        sign = "-" if self.__is_negative() else ""
        return sign + f"{abs(self.euro)},{abs(self.cent)}€"

    def __add__(self, other):
        if isinstance(other, Money):
            if other.__is_negative():
                return self.__sub__(Money(other.euro * (-1), other.cent * (-1)))
            else:
                cent = self.cent + other.cent
                euro = self.euro + cent // 100 + other.euro
                cent %= 100
                return Money(euro, cent)
        else:
            raise ValueError("Object to be added has to be of type Money")

    def __sub__(self, other):
        if isinstance(other, Money):
            if other.__is_negative():
                return self.__add__(Money(other.euro * (-1), other.cent * (-1)))
            else:
                cent = self.cent - other.cent
                euro = self.euro - cent // -100 - other.euro
                cent = cent % -100
                while cent < 0 < euro:
                    euro -= 1
                    cent += 100
                return Money(euro, cent)
        else:
            raise ValueError("Object to be added has to be of type Money")
