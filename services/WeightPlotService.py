import numpy
from objects.space.Space import Space
from plots.ChoosePlot import ChoosePlot
from objects.plot.PlotData import PlotData


class WeightPlotService:
    def __init__(self, ms):
        self.__ms = ms

    def plots(self, error: dict,epochs: int):
        for key, value in error.items():
            x = []
            y = []

            for key, value in value.items():
                x.append(float(key))
                y.append(value)

            space = Space([numpy.array(x)])
            choose_plot = ChoosePlot(space, y,
                                     self.__get_plot_path("Weight error"),
                                     PlotData(f'Weight error  epoches: {epochs}', ["weight", "error"]))
            choose_plot.choose().plot()

    def __get_plot_path(self, plot_name: str):
        return f'plot/{self.__ms}/weight error/{plot_name}.png'
