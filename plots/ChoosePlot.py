from objects.space.Space import Space
from objects.plot.PlotData import PlotData
from plots.Plot2d import Plot2D
from plots.Plot3d import Plot3D

import matplotlib

matplotlib.use('TkAgg')


class ChoosePlot:
    def __init__(self, space: Space,results,save_path:str = None, plot_data: PlotData = PlotData()):
        if space is None:
            raise ValueError("Space cannot be None")

        self.__space = space
        self.__plot_data = plot_data
        self.__results = results
        self.__save_path = save_path

    def choose(self):
        dimension = self.__space.get_dimension()
        mesh = self.__space.get_mesh_numpy_array()

        import os
        dir_path = os.path.dirname(self.__save_path)
        os.makedirs(dir_path, exist_ok=True)

        if dimension == 1:
            return Plot2D(mesh[0], self.__results, self.__plot_data,self.__save_path)
        elif dimension == 2:
            return Plot3D(mesh[0], mesh[1], self.__results, self.__plot_data,self.__save_path)
        else:
            raise ValueError("Dimension not supported")


