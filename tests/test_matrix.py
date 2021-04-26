import numpy

from play3d.matrix import Matrix


def test_matrix_eq():

    assert (Matrix([[1, 3], [4, 5]]).matrix == numpy.array([
        [1, 3],
        [4, 5]
    ])).all()


def test_identity_matrix():
    assert Matrix.identity_matrix() == Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def test_matrix_multiply():

    a = Matrix([[2, 2],
             [3, 4]])
    b = Matrix([[4, 2],
             [5, 2]])

    result = a @ b

    assert result == Matrix([
     [18, 8],
     [32, 14]
    ])
