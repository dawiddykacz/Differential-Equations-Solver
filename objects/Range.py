class Range:

    def __init__(self, min: float, max: float):
        if min >= max:
            raise ValueError("Min must be less than or equal to max")

        self.min = min
        self.max = max

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max

    def __str__(self):
        return f'<{self.min};{self.max}>'