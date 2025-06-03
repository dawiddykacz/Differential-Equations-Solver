import string


class PlotData:
    def __init__(self, title: string = "Plot", labels: [string] = ["x", "y", "z"]):
        if title is None:
            raise ValueError("Title is required")
        if labels is None:
            raise ValueError("Labels is required")

        self.__title = title
        self.__labels = labels

    def get_title(self):
        return self.__title

    def get_label(self, index: int):
        if index < len(self.__labels):
            return self.__labels[index]
        return ""
