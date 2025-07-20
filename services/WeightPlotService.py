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

            for k, v in value.items():
                x.append(float(k))
                y.append(v)

            if len(x) > 1:
                space = Space([numpy.array(x)])
                choose_plot = ChoosePlot(space, y,
                                         self.__get_plot_path(f'Weight error {key}'),
                                         PlotData(f'Weight error {key} epoches: {epochs}', ["weight", "error"]))
                choose_plot.choose().plot()

    def __get_plot_path(self, plot_name: str):
        return f'plot/{self.__ms}/weight error/{plot_name}.png'
