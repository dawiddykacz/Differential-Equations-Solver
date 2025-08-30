import numpy
import tensorflow


class Space:
    def __init__(self, points: [numpy.ndarray]):
        if points is None:
            raise ValueError('points cannot be None')
        if len(points) == 0:
            raise ValueError("No points is empty")
        if points[0].ndim != 1:
            raise ValueError('points must be a 1-dimensional array')

        self.__size = 0
        self.__points = points
        for point in points:
            self.__size += len(point)

    def size(self):
        return self.__size

    def get_numpy_array(self, index: int):
        return self.__points[index].copy()

    def get_points_to_neural_network(self):
        arrays = [self.get_numpy_array(i) for i in range(self.get_dimension())]

        mesh = numpy.meshgrid(*arrays, indexing='ij')

        points = [tensorflow.constant(m.reshape(-1, 1), dtype='float64') for m in mesh]

        return points

    def get_mesh_numpy_array(self):
        return tensorflow.meshgrid(*self.__points, indexing='xy')

    def get_dimension(self):
        return len(self.__points)

    def get_shape(self):
        if self.get_dimension() == 1:
            raise ValueError("Dimension not supported")

        axes = []
        for i in range(len(self.__points)):
            axes.append(self.get_numpy_array(i))
        x = numpy.meshgrid(*axes)
        return x[0].shape

    def get_points(self):
        dimention = self.get_dimension()
        if dimention == 1:  #in 2d
            x = self.get_numpy_array(0)

            x_flat = x.flatten().reshape(-1, 1)
            return [self.__get_tensor_axis(x_flat)]
        if dimention == 2:  #in 3d
            x = self.get_numpy_array(0)
            y = self.get_numpy_array(1)

            x, y = numpy.meshgrid(x, y)

            x_flat = x.flatten().reshape(-1, 1)
            y_flat = y.flatten().reshape(-1, 1)

            return self.__get_tensor_axis(x_flat), self.__get_tensor_axis(y_flat)
        raise ValueError('dimension must be 2 or 3')

    def __get_tensor_axis(self, x: numpy.ndarray):
        return tensorflow.constant(x, shape=(len(x), 1), dtype='float64')

    # def get_tensorflow_array(self):
    #     axes_array = []
    #
    #     for axe in self.__points:
    #         axe = tensorflow.constant(axe, shape=(len(axe), 1), dtype='float64')
    #         axes_array.append(axe[:, 0])
    #
    #     mesh_grids = tensorflow.meshgrid(*axes_array, indexing='ij')
    #
    #     reshaped_grids = [tensorflow.reshape(grid, [-1]) for grid in mesh_grids]
    #     return tensorflow.stack(reshaped_grids, axis=1)

    def __get_axe_length(self, axe: numpy.ndarray):
        return len(axe)
