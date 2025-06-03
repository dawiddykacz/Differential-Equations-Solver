import numpy
import matplotlib.pyplot as plot

from objects.plot.PlotData import PlotData


class Plot2D:
    def __init__(self, x: numpy.ndarray, y: numpy.ndarray, plot_data: PlotData,save_path:str = None):
        self.__x = x
        self.__y = y
        self.__plot_data = plot_data
        self.__save_path = save_path

    def plot(self):
        plot.clf()
        plot.plot(self.__x, self.__y, label="a", color='black')

        plot.title(self.__plot_data.get_title())
        plot.xlabel(self.__plot_data.get_label(0))
        plot.ylabel(self.__plot_data.get_label(1))

        if self.__save_path:
            plot.savefig(self.__save_path)
        else:
            plot.show()
