from pytest import fixture

from play3d.models import Cube


@fixture
def cube():
    return Cube()
