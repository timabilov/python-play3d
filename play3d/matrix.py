from numbers import Number
from typing import Type

import numpy as np


class Matrix:

    @classmethod
    def identity_matrix(cls):

        return cls(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )

    def __init__(self, array_of_array):

        if isinstance(array_of_array, Matrix):

            array_of_array = array_of_array.matrix

        self.matrix = np.array([np.array(line, dtype='float32') for line in array_of_array], dtype='float32')

    def __matmul__(self, other: Type['Matrix']):
        """
        Matrix multiplication
        :param other:
        :return:
        """
        if not isinstance(other, Matrix):
            raise Exception('Matrix can be multiplied(@) to matrix only!')

        return Matrix(matmul(self.matrix, other.matrix))

    def __getitem__(self, item):

        return self.matrix[item]

    def __setitem__(self, key, value):
        if not isinstance(value, Number):
            raise Exception(f'Matrix elements can be only a numbers! Given {value}')

        self.matrix[key] = value

    def __len__(self):
        return len(self.matrix)

    def __mul__(self, other):
        """
        Plain multiplication by number
        :param other:
        :return:
        """
        return np.multiply(self.matrix, other)

    def transpose(self):
        matrix = self.matrix
        return Matrix([
            [matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))
        ])

    def __add__(self, other):
        new_matrix = Matrix(self.matrix.copy())
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                new_matrix.matrix[i][j] += other[i][j]

        return new_matrix

    def __str__(self):
        return str(self.matrix)

    def __eq__(self, other: 'Matrix'):
        print('EHE?', self)
        print(other)
        return (self.matrix == other.matrix).all()


def matmul(matrix1, matrix2):
    """
    :param matrix1:
    :param matrix2:
    :return:
    """

    multiplied = np.matmul(matrix1, matrix2)

    return multiplied
