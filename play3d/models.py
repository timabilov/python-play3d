import copy
import math
import numbers
from collections import Iterable
from inspect import signature

import numpy
import numpy as np

from . import three_d
from .matrix import Matrix


x = 0
y = 1
z = 2
w = 3
# real z image space - clipped

# standard rgb colors
black, white, blue = (20, 20, 20), (230, 230, 230), (0, 154, 255)


class Model:

    renderer = None
    data = []
    faces = []

    @classmethod
    def load_OBJ(cls, path, rasterize=False, **kwargs):
        """

        :param path:
        :param rasterize: True - means turn off rasterization - use only mesh
        :param kwargs:
        :return:
        """
        data = []
        faces = []
        with open(path) as f:
            l = f.readline()
            while l:
                if l.startswith('v '):
                    v = [float(n) for n in l[2:].split()]
                    if len(v) == 3:  # add default w=1 coord if not given
                        v += [1]

                    data.append(v)

                if l.startswith('f '):
                    v = list(map(lambda face: int(face.split('/')[0]), l[2:].split()))

                    faces.append(v)
                l = f.readline()
        obj = cls(**kwargs, data=data, faces=faces, rasterize=rasterize)
        return obj

    def __init__(self, position=(0, 0, 0), scale=1, color=white, shimmering=False, data=None, faces=None, **kwargs):
        """

        :param position:
        :param scale:
        :param color:
        :param shimmering:
        :param data:
        :param faces:
        :param kwargs: 'rasterize' - True or False
        """
        self.matrix = Matrix([
            [1 * scale, 0, 0, 0],
            [0, 1 * scale, 0, 0],
            [0, 0, 1 * scale, 0],
            [*position, 1]
        ])
        self.faces = self.__class__.faces
        self.data = Matrix(self.__class__.data.matrix.copy()) if self.__class__.data else None

        # defines continuous translation/route by given trajectory function
        self.trajectory = None
        if data is not None and len(data) > 0:
            if faces:  # faces contains vertex order position
                # indexing starts from 1 so we put fake vertice
                # to keep access to faces array consistent
                data.insert(0, [0, 0, 0, 1])
            self.data = Matrix(data)
            if faces:
                self.faces = faces
            if self.faces:
                self.rasterize = kwargs.pop('rasterize', False)
        elif not self.data:
            self.data = Matrix(numpy.ndarray([1, 4], 'float32'))
        self.shimmering = shimmering
        self.color = color
        speed = 10

        if self.faces:
            self.rasterize = kwargs.pop('rasterize', False)
        # color shimmering speed
        self.rs, self.gs, self.bs = speed, -speed, speed

        # we can't initialize trace Model() here due to circular dependency/recursion call
        # enable_trace should be used
        self.trajectory_path = None
        self.trace = False

        for k, v in kwargs.items():

            setattr(self, k, v)

    def _perspective_divide(self, points):
        """

        :param points:
        :return:
        """
        # [x, y, z, w=1]
        A = points.matrix  # np.array
        # TODO this is delete only for non mesh model.
        # A = numpy.delete(
        #     points, numpy.where((A[:, 2] < Camera.near))[0], 0
        # )
        # mesh polygon face number order preserver by not deleting
        # A[numpy.where(A[:, 2] < Camera.near)] = None

        A[:, 2] = numpy.true_divide(A[:, 2], A[:, 3] + 0.0001)

        A[:, 0] = numpy.true_divide(A[:, 0], A[:, 3] + 0.0001)
        A[:, 1] = numpy.true_divide(A[:, 1], A[:, 3] + 0.0001)

        width, height = three_d.Device.get_resolution()

        # viewport transformation
        A[:, 0] = A[:, 0] * width / 2 + width / 2
        # y from top to bottom put '-'
        A[:, 1] = -A[:, 1] * height / 2 + height / 2
        return A.astype(int)

    def _project_points(self, points):

        points = numpy.delete(
            points, numpy.where(points[:, 3] < three_d.Camera.near)[0], 0
        )

        for pos in points:
            three_d.Device.put_pixel(pos[x], pos[y], self.color)

    def rotate(self, angle_x, angle_y=0, angle_z=0):
        self.matrix = three_d.rotate_model(self.matrix, angle_x, angle_y, angle_z)
        return self

    def position(self):

        return self.matrix.matrix[3]  # [x y z w]

    def set_position(self, x, y, z, w=1):

        self.matrix.matrix[3] = [x, y, z, w]  # [x y z w]

    def route(self, trajectory: 'Trajectory', enable_trace=False):
        self.trajectory = trajectory
        if trajectory.attached_model:
            # we need to reset self.position to attached model position
            self.set_position(*trajectory.attached_model.position())

        enable_trace and self.enable_trace()

    def enable_trace(self, length=None):
        self.trajectory_path = Model(data=numpy.ndarray([1, 4], 'float32'), color=blue)
        self.trace = True

    def _trace(self, length=None):
        """
        Used for dot-based visual tracing of model movements
        :param length: int Maximum length of track trajectory chain, by default without any limits
        :return:
        """

        new_track_point = self.position()
        self.trajectory_path.data.matrix = numpy.vstack([self.trajectory_path.data, new_track_point])
        self.trajectory_path.draw()

    def _shimmer(self):
        red, green, blue = self.color
        rs, gs, bs = self.rs, self.gs, self.bs

        if not (0 < red + rs < 255):
            self.rs = -rs

        if not (0 < green + gs < 255):
            self.gs = -gs

        if not (0 < blue + bs < 255):
            self.bs = -bs

        self.color = (red + self.rs), (green + self.gs), (blue + self.bs)

    def _clip_lines(self, points):
        """
        Not finished.
        Clip mesh lines must be before viewport transformation and perspective divide
        :param points:
        :return:
        """
        points_copy = copy.deepcopy(points)
        for face in self.faces:

            for i in range(3):
                start, end = face[i], face[(i + 1) % 3]
                # todo camera near numpy before
                if points[start][w] <= three_d.Camera.near and points[end][w] <= three_d.Camera.near:
                    continue

                if points[start][w] < three_d.Camera.near or points[end][w] < three_d.Camera.near:

                    # direction > 0 means first point of line is ahead of us. so we cut from second point
                    pc, direction = three_d.linemidpoint(points[start], points[end], three_d.Camera.near)
                    # points[end if points[end][w] < Camera.near else start] = pc
                    # fixme: proper midline separation
                    points_copy[end] = pc
                    points_copy[start] = points[start] if points[start][w] > three_d.Camera.near else points[end]
                    # three_d.Device.drawline(points[start] if points[start][w] > Camera.near else points[end], pc, self.color)
                # else:
                three_d.Device.drawline(points[start], points[end], self.color)

        return points

    def _render_mesh(self, points):
        A = points
        for i, face in enumerate(self.faces):
            color = 140 + (i % 110)

            if self.rasterize:
                three_d.fill_triangle(A[face[0]], A[face[1]], A[face[2]], (color, color, color))
            else:
                for i in range(3):
                    start, end = face[i], face[(i + 1) % 3]

                    if not(A[start][w] <= three_d.Camera.near and A[end][w] <= three_d.Camera.near):#\
                    # and 0 < A[start][x] < three_d.Device._width and 0 < A[start][y] < three_d.Device._height\
                    # and 0 < A[end][x] < three_d.Device._width and 0 < A[end][y] < three_d.Device._height:

                        three_d.Device.drawline(A[start], A[end], self.color)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            self.matrix = self.matrix @ other

        else:
            raise Exception('you can multiplicate only to matrix.Matrix')

    def draw(self):

        # if not self.__class__.renderer:
        #     raise Exception('You have to set projection renderer for Model')
        if len(self.data) > 0:
            if self.shimmering:

                self._shimmer()

            VP = three_d.Camera.View_Projection_matrix()

            # model matrix of Model moves by trajectory
            if self.trajectory:
                # rename architecture
                if self.trace:
                    self._trace()
                self @ self.trajectory.move()

            points = self.data @ self.matrix @ VP  # M * VP

            # if self.faces:
            #     self.clipped = self._clip_lines(points)s

            normalized_points = self._perspective_divide(points=points)
            if self.faces:
                # TODO clipped mesh should before self._perspective_divide(points=points)

                self._render_mesh(normalized_points)
            else:
                self._project_points(normalized_points)


class Grid(Model):

    def __init__(self, dot_precision=100, dimensions=(10, 10), **kwargs):

        super(Grid, self).__init__(**kwargs)

        grid_dimension = dimensions  # x and z

        center = (0, 0, 0)  # x y z

        cell_size = (1, 1)  # 1 x 1

        # grid_vertices = Matrix([
        #     (x, 0, z, 1) for x in range(grid_dimension[0]) for z in range(grid_dimension[1])
        # ])
        self.data = Matrix([])
        for x in range(grid_dimension[0]):

            line_dots = three_d.dotted_line(
                (x, 0, 0, 1), (x, 0, grid_dimension[1], 1), precision=dot_precision
            )

            self.data.matrix = numpy.array(list(self.data.matrix) + list(line_dots))

        for z in range(grid_dimension[1]):
            line_dots = three_d.dotted_line(
                (0, 0, z, 1), (grid_dimension[0], 0, z, 1), precision=dot_precision
            )
            self.data.matrix = numpy.array(list(self.data.matrix) + list(line_dots))
        # self.data = grid_vertices

        self.matrix = self.matrix @ three_d.translate(-grid_dimension[0] / 2, 0, -grid_dimension[1] / 2)

    def render(self):
        pass


class Cube(Model):
    data = Matrix([
        [-1, 1, 1, 1],
        [1, 1, 1, 1],
        [-1, -1, 1, 1],
        [1, -1, 1, 1],
        [-1, 1, -1, 1],
        [1, 1, -1, 1],
        [1, -1, -1, 1],
        [-1, -1, -1, 1]
    ])

    # ! respect to order above
    faces = [
        [0, 1, 2],
        [1, 2, 3],
        [1, 3, 6],
        [1, 5, 6],
        [0, 1, 4],
        [1, 4, 5],

        [2, 3, 7],
        [3, 6, 7],
        [0, 2, 7],
        [0, 4, 7],
        [4, 5, 6],
        [4, 6, 7]
    ]
    #

    def __init__(self, **kwargs):
        super(Cube, self).__init__(**kwargs)


class CircleChain(Model):

    def __init__(self, **kwargs):
        super(CircleChain, self).__init__(**kwargs)

        self.data = []

        full_cycle = 360
        for degree in range(full_cycle):

            self.data.append([math.sin(degree * 2 * math.pi/360), math.cos(degree * 2 * math.pi/360), 0, 1])
            self.data.append([0, math.sin(degree * 2 * math.pi / 360), math.cos(degree * 2 * math.pi / 360), 1])

        self.data = Matrix(self.data)


class Plot(Model):
    """
    Plotting model for visualization with parametric(polar) equations
    Ex. func=lambda t: [t, t*t] <-> y=x*x
    sine = Plot(position=(4, 2, -8), color=(0, 64, 255),
            func=lambda x, t: [x, math.cos(x) * math.cos(t), math.cos(t)], allrange=[0, 2*math.pi], interpolate=75)
    """
    def __init__(self, func, allrange=(-1, 1), interpolate=100, **kwargs):
        """

        :param func: lambda which
        :param allrange:
        :param interpolate:
        :param kwargs:
        """

        super(Plot, self).__init__(**kwargs)

        if not callable(func):
            raise Exception(f'func argument should be callable, given: {func}')

        sig = signature(func)
        self.signature = sig
        fn_ndim = len(sig.parameters)

        self.data = []

        discrete_axis = np.linspace(allrange[0], allrange[1], interpolate)

        mesh = np.meshgrid(*[discrete_axis]*fn_ndim)  # create N dimensional input grid
        #
        params = []
        for i in range(fn_ndim):
            params.append(mesh[i].ravel())

        base_point = [0, 0, 0, 1]

        points = np.dstack([*params]).reshape(-1, fn_ndim)
        for i in range(1, interpolate**fn_ndim):
            res = func(*points[i])
            if isinstance(res, (list, Iterable)):
                # todo replace to basic check without numpy
                arr = numpy.array(res)
                if arr.ndim != 1 or arr.shape[0] > 3:
                    raise Exception(f'Plot function {func.__name__} codomain should be 1D < 3 array of numbers (or plain number), given: {res}')
                arr.resize(4, refcheck=False)
                point = arr + base_point
            elif isinstance(res, numbers.Number):
                point = [points[i], res, 0, 1]
            else:
                raise Exception(f'Plot function {func.__name__} codomain should be 1D < 3 array of numbers (or plain number), given: {res}')
            self.data.append(point)

        self.data = Matrix(self.data)


class Sphere(Plot):

    @classmethod
    def _fn(cls, phi, theta):

        return [
            math.sin(phi * math.pi / 180) * math.cos(theta * math.pi / 180),
            math.sin(theta * math.pi / 180) * math.sin(phi * math.pi / 180),
            math.cos(phi * math.pi / 180)
        ]

    def __init__(self, scale=1, position=(0, 0, 0), color=white, shimmering=False, interpolate=100):

        # self.data = []
        # for x in np.linspace(0, 360, 30):
        #     for y in np.linspace(0, 360, 30):
        #         self.data.append(Sphere._fn(x, y) + [1])

        super(Sphere, self).__init__(func=Sphere._fn, interpolate=interpolate, allrange=[0, 360], position=position,
                                     color=color, shimmering=shimmering, scale=scale)


class Trajectory:

    class ToAxis:

        @classmethod
        def X(cls, speed=0.01, **kwargs):
            return Trajectory(func=lambda x: [x, y, 0], speed=speed, **kwargs)

        @classmethod
        def Y(cls, speed=0.01, **kwargs):
            return Trajectory(func=lambda y: [0, y, 0], speed=speed, **kwargs)

        @classmethod
        def Z(cls, speed=0.01, **kwargs):
            return Trajectory(func=lambda z: [0, 0, z], speed=speed, **kwargs)

    @classmethod
    def Around(cls, around: Model, func, speed=0.01,  **kwargs):
        return Trajectory(around=around, func=func, speed=speed, **kwargs)

    @classmethod
    def SineXY(cls, speed=0.01, **kwargs):
        return Trajectory(lambda x: [x, math.sin(x)], speed=speed, **kwargs)

    @classmethod
    def CosXY(cls, speed=0.01, **kwargs):
        return Trajectory(lambda x: [x, math.cos(x)], speed=speed, **kwargs)

    @classmethod
    def SineXYZ(cls, speed=0.01, **kwargs):
        return Trajectory(lambda x: [x, math.sin(x), -x], speed=speed, **kwargs)

    @classmethod
    def CosXYZ(cls, speed=0.01, **kwargs):
        return Trajectory(lambda x: [x, math.cos(x), -x], speed=speed, **kwargs)

    def __init__(self, func, speed=0.2, around: Model = None, start_position=(0, 0, 0)):

        if not callable(func):
            raise Exception(f'func argument should be callable, given: {func}')

        self.iteration = 0
        # step
        self.speed = speed
        self.x = 0
        self.attached_model = around
        self.start_position = self.attached_model.position()[:-1] if self.attached_model else start_position
        self.position = [0, 0, 0, 1]
        # dx dy dz
        self.dPosition = None

        self.func = func
        self.signature = signature(func)

        if len(self.signature.parameters) != 1:
            raise Exception('Continuous Trajectory function can have only one parameter')

        # if we attached - we want to preserve function center context,
        # so first calibration jump is required to assume that 0.0.0 is the object center
        if not self.attached_model:
            # initialize to get first proper starting point to keep continuity
            # to rely on true dx dy dz afterwards (eliminate sudden first-time jumps)
            # Ex. start = 0.0.0 first point will be = 1.0.0 dx will be 1.0.0 which is unnecessary
            self.position = self.next_value()

    def next_value(self):
        result = self.func(self.x)

        if isinstance(result, (list, Iterable)):
            # todo replace to basic check without numpy
            arr = numpy.array(result)
            if arr.ndim != 1 or arr.shape[0] > 3:
                raise Exception(
                    f'Plot function {self.func.__name__} codomain should be 1D < 3 array of numbers (or plain number)'
                )
            arr.resize(4, refcheck=False)
            point = arr + [0, 0, 0, 1]
        elif isinstance(result, numbers.Number):
            point = numpy.array([self.x, result, 0, 1])
        else:
            raise Exception(
                f'Plot function {self.func.__name__} codomain should be 1D < 3 array of numbers (or plain number), given: {result}'
            )

        self.x = self.x + self.speed
        self.iteration += 1
        return point

    def move(self):

        point = self.next_value()
        # in dPosition w goes to zero, no affect for dx dy dz
        self.dPosition = point - self.position
        self.position = point

        if self.attached_model:
            if self.attached_model.trajectory:
                self.dPosition = self.attached_model.trajectory.dPosition + self.dPosition

        # we need dx dy dz to define next move

        return three_d.translate(*self.dPosition[:-1])

    def __iter__(self):
        return self

    # def gradient todo
    def __next__(self):
        return self.move()

    def attach(self, model):

        self.attached_model = model
        self.start_position = model.position()[:-1]

    def backwards(self):
        self.speed = -self.speed
        return self
