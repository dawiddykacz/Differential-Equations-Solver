class Range:

    def __init__(self, min: float, max: float):
        if min >= max:
            raise ValueError("Min must be less than or equal to max")

        self.__min = min
        self.__max = max

    def get_min(self):
        return self.__min

    def get_max(self):
        return self.__max

    def __str__(self):
        return f'<{self.__min};{self.__min}>'
