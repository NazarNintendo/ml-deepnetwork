import random
import time

random.seed(time.time())


def generate_random_params() -> tuple[float, float, float]:
    """
    Returns randomly generated parameters of the general form of the equation of a line.
    ax + by + c = 0
    :return:
    """
    ascending = random.randint(0, 1)

    if ascending:
        a = random.uniform(0.01, 3.)
        b = -1
        c = random.uniform(0.3, 0.4)
    else:
        a = random.uniform(-3., -0.01)
        b = -1
        c = random.uniform(0.6, 0.7)

    return a, b, c


class Element:
    a, b, c = generate_random_params()

    def __init__(self, x=None, y=None, value=None):
        self.x = x if x else random.uniform(0, 1)
        self.y = y if y else random.uniform(0, 1)
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
        return 1 if (self.y > -(self.a * self.x + self.c) / self.b) else 0
