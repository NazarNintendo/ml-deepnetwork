import random
import time

random.seed(time.time())


class Element:
    def __init__(self, x=None, y=None, value=None):
        self.x = x if x else random.gauss(0, .4)
        self.y = y if y else random.gauss(0, .4)
        self.value = value if value else self.get_value()

    def __str__(self) -> str:
        """
        Returns a string representation of an Element instance.
        :return: string
        """
        return f'{self.x},{self.y},{self.value}\n'

    def get_value(self) -> int:
        """
        Returns 1 if (x,y) belongs to upper subclass, 0 - if lower.
        :return: 1 or 0
        """
        return 1 if (self.x * self.x + self.y * self.y) < .25 else 0
