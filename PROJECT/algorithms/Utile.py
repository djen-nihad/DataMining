import numpy as np


class Distance:
    def euclideanDistance(x1, x2, p=None):
        return np.sqrt(np.sum(np.square(x1 - x2)))

    def manhattanDistance(x1, x2, p=None):
        return np.sum(np.abs(x1 - x2))

    def minkowskiDistance(x1, x2, p=2):
        return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

    def cosineDistance(x1, x2, p=None):
        return 1 - (np.sum(x1 * x2)) / (np.sum(x1 ** 2) * np.sum(x2 ** 2))


def is_subList(subList, mainList):
    return all(item in mainList for item in subList)


def is_subList_2(sublist, main_list):
    for x in main_list:
        if all(item in x for item in sublist):
            return True
    return False


def diffrence(listA, listB):
    for item in listA:
        if item in listB:
            return False
    return True


def normalization(X, min=0, max=1):
    data_Normalize = []
    for i in range(X.shape[1]):
        col = X[:, i]
        min_old = np.min(col)
        max_old = np.max(col)
        col = ((col - min_old) / (max_old - min_old)) * (max - min) + min
        data_Normalize.append(col)
    return np.array(data_Normalize).T
