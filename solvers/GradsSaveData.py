import copy

import numpy
import tensorflow


class GradsSaveData:
    def __init__(self):
        self.__save_dict = None
        self.__i = 0

    def add(self, grads: dict):
        if self.__save_dict is None:
            self.__save_dict = copy.deepcopy(grads)
        else:
            self.__add(copy.deepcopy(grads), self.__save_dict)
        self.__i += 1

    def __add(self, grads: dict, save: dict):
        for key in grads.keys():
            item = grads[key]
            if isinstance(item, dict):
                self.__add(item, save[key])
            else:
                save[key] += item

    def to_dict(self):
        save = copy.deepcopy(self.__save_dict)
        self.__divide(save)
        return save

    def __divide(self, save: dict):
        for key in save.keys():
            item = save[key]
            if isinstance(item, dict):
                self.__divide(item)
            else:
                y = item / self.__i
                if isinstance(y, tensorflow.Tensor):
                    y = y.numpy()

                save[key] = {
                    "x": numpy.linspace(1, len(item), len(item)).tolist(),
                    "y": numpy.asarray(y).tolist()
                }