import pytest

from play3d.matrix import Matrix
from play3d.models import Cube
from play3d.three_d import translate


class TestModelMatrix:

    @pytest.mark.parametrize("position, scale",
                             [((1, 2, 3), 1.9),
                              ((5, 2, 3), 1),
                              ])
    def test_model_matrix(self, position, scale):
        cube = Cube(position=position, scale=scale)
        assert cube.matrix == Matrix([
            [1 * scale, 0, 0, 0],
            [0, 1 * scale, 0, 0],
            [0, 0, 1 * scale, 0],
            [*position, 1]
        ])


def test_model_position(cube):

    assert list(cube.position()) == [0, 0, 0, 1]


def test_model_translate(cube):
    cube @ translate(1, 2, 3)
    assert list(cube.position()) == [1.0, 2.0, 3.0, 1]
