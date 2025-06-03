import numpy
from objects.Range import Range
from objects.space.Space import Space


class SpaceRanges:
    def __init__(self, numberInOne: int, *ranges: Range):
        self.__ranges = ranges
        self.__numberInOne = numberInOne

    def split(self, multiply: int = 1):
        axes_array = []

        for axe in self.__ranges:
            number_of_split = self.__calculate_amount_of_split(axe, self.__numberInOne) * multiply
            axe = numpy.linspace(axe.get_min(), axe.get_max(), number_of_split)
            axes_array.append(axe)

        return Space(axes_array)

    def __calculate_amount_of_split(self, range: Range, numberInOne: int):
        amount = int(float((range.get_max()) - float(range.get_min())) * numberInOne)
        if amount < 10:
            return 11
        return amount + 1

    def __str__(self):
        string = ""
        for range in self.__ranges:
            string += f"{range},"

        return f'[{string}]'