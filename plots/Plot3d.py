import numpy
import matplotlib.pyplot as plot

from objects.plot.PlotData import PlotData


class Plot3D:
    def __init__(self, x: numpy.ndarray, y: numpy.ndarray,z: numpy.ndarray, plot_data: PlotData,save_path:str = None):
        self.__x = x
        self.__y = y
        self.__z = z
        self.__plot_data = plot_data
        self.__save_path = save_path

    def plot(self):
        fig = plot.figure(dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.__x, self.__y, self.__z, cmap='viridis')
        ax.set_xlabel(self.__plot_data.get_label(0))
        ax.set_ylabel(self.__plot_data.get_label(1))
        ax.set_zlabel(self.__plot_data.get_label(2))
        plot.title(self.__plot_data.get_title())
        if self.__save_path:
            plot.savefig(self.__save_path)
        else:
            plot.show()