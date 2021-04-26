import math
import logging
import numpy

from .matrix import Matrix

from .utils import log_this

log = logging.getLogger(__name__)

# just for simplicity
x = 0
y = 1
z = 2
w = 3


class Device:
    _width, _height = None, None

    _renderer = None
    _line_render = None

    @classmethod
    def set_renderer(cls, pixel_renderer, line_renderer=None):
        """
        Main renderer for our world. Usually framebuffer put pixel method is enough
        for pixel_renderer(x, y, color)
        But optionally line renderer can be set from your 2D library.
        Otherwise line is drawn with the help of pixel_renderer which is slow.
        :param renderer:
        :return:
        """
        cls._renderer = pixel_renderer

        cls._line_render = line_renderer if line_renderer else drawline
        if cls._line_render == drawline:
            log.warning(' <<<< Mapped default drawline renderer using pixel_renderer under the hood  >>> ')

    @classmethod
    def put_pixel(cls, x, y, color):

        if 0 < x < cls._width and 0 < y < cls._height:
            cls._renderer(x, y, color)

    @classmethod
    def drawline(cls, p1, p2, color):
        return cls._line_render(p1, p2, color)

    @classmethod
    def viewport(cls, x, y):
        cls._width = x
        cls._height = y

    @classmethod
    def get_resolution(cls):
        return cls._width, cls._height


def linemidpoint(p1, p2, near):

    t = (p1[w] - near)/(p1[w] - p2[w] + 0.001)
    if abs(t) < 0.001:
        return p1, 0

    xc = (t*p1[x]) + ((1-t) * p2[x])
    yc = (t*p1[y]) + ((1-t) * p2[y])
    zc = (t*p1[z]) + ((1-t) * p2[z])
    wc = near
    return [xc, yc, zc, wc], t


def drawline(p1, p2, color):

    x0, y0 = int(p1[x]), int(p1[y])
    x1, y1 = int(p2[x]), int(p2[y])
    dx, dy = abs(x1 - x0), abs(y1 - y0)

    sx = 1 if (x0 < x1) else - 1
    sy = 1 if (y0 < y1) else - 1

    err = dx - dy
    count = 0
    while True:
        Device.put_pixel(x0, y0, color)
        if (x0 == x1) and (y0 == y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
        count += 1


def drawline2(p1, p2, color):

    x0, y0 = p1[x], p1[y]
    x1, y1 = p2[x], p2[y]
    dx, dy = abs(x1 - x0), abs(y1 - y0)

    sx = 1 if (x0 < x1) else - 1

    slope = dy / (dx + 0.01)
    _y = y0
    if sx == 1:
        for _x in range(x0, x1):
            Device.put_pixel(_x, _y, color)
            if (x0 == x1) and (y0 == y1):
                break
            _y = _y + slope
    else:

        for _x in range(x1, x0):
            Device.put_pixel(_x, _y, color)
            if (x0 == x1) and (y0 == y1):
                break
            _y = _y - slope


def PROJECTION_MATRIX(fov, aspect_ratio, near, far):
    fov = 1 / (math.tan((fov/2) * math.pi / 180))
    return Matrix(
        [
            [fov/aspect_ratio,     0,            0,                0],
            [0,                  fov,            0,                0],
            [0,                    0, -(far + near)/(far - near), -1],
            [0,                    0,   -2*near*far/(far-near),    0]
        ]
    )


def translate(x, y, z, inverse=False):

    if inverse:
        x, y, z = -x, -y, -z

    return Matrix(
       [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [x, y, z, 1]
       ]
    )


def rotate_over_z(angle):

    rad = math.pi*angle/180
    return Matrix([
        [math.cos(rad), math.sin(rad), 0, 0],
        [-math.sin(rad),  math.cos(rad), 0, 0],
        [0,  0,  1, 0],
        [0,  0,  0, 1],
    ]
)


def rotate_over_y(angle):

    rad = math.pi*angle/180
    return Matrix([
        [math.cos(rad), 0, -math.sin(rad), 0],
        [0,  1, 0, 0],
        [math.sin(rad),  0,  math.cos(rad), 0],
        [0, 0, 0, 1],
    ]
)


def rotate_over_x(angle):
    rad = math.pi*angle/180
    return Matrix([
        [1, 0, 0, 0],
        [0,  math.cos(rad), math.sin(rad), 0],
        [0,  -math.sin(rad),  math.cos(rad), 0],
        [0, 0, 0, 1],

    ]
)


def _negative_list(l):

    return [-x for x in l]


def inverse_camera_view(camera_position, camera_angle):
    """
    Camera actually transforms the world (inverted translate and inverted rotate)
    """
    # rotate inverted = transpose
    # translation inversion = sign change
    # inverse of a product is reverse product of inverse
    return translate(*_negative_list(camera_position)) @ (rotate_over_y(camera_angle[1]) @ rotate_over_x(camera_angle[0]))


@log_this('Recalculate camera..')
def ViewProjectionMatrix(cam_position, cam_angles, fov, aspect_ratio, near, far):
    return inverse_camera_view(cam_position, cam_angles) @ PROJECTION_MATRIX(fov, aspect_ratio, near, far)


# easy way to handle camera axis
axis_str_mapping = {
    'x': 0,
    'y': 1,
    'z': 2
}


class Camera:
    _camera = None
    _camera_position = numpy.array([0, 0, 0, 1], dtype='float32')
    _camera_angles = numpy.array([0, 0, 0])
    _facing_axis = Matrix.identity_matrix()
    near = 1
    far = 10
    fov = 60

    aspect_ratio = 4 / 3
    _VPMatrix = ViewProjectionMatrix(_camera_position, _camera_angles, fov, aspect_ratio, near, far)

    @classmethod
    def get_instance(cls):
        if not cls._camera:
            cls._camera = Camera()
        return cls._camera

    @classmethod
    def View_Projection_matrix(cls):
        return cls._VPMatrix

    def __getitem__(self, item):
        return self.__class__._camera_position[axis_str_mapping[item]]

    def rotate(self, axis, angle):
        if axis == 'y':
            self._facing_axis = rotate_over_y(-angle) @ self._facing_axis

        self._camera_angles[axis_str_mapping[axis]] += angle
        # axis_camera_position = Matrix(self._camera_position) @ self._facing_axis
        self.__class__._VPMatrix = ViewProjectionMatrix(self._camera_position[:-1],
                                                        self.__class__._camera_angles,
                                                        self.__class__.fov, self.__class__.aspect_ratio,
                                                        self.__class__.near, self.__class__.far)

    def move(self, x=0, y=0, z=0):  # TODO Trajectory implement
        cls = self.__class__

        # create direction vector with filled axis value as step.
        # 'y' += 0.3 -> dir_vector = [0, 0.3, 0, 0].

        dir_vector = Matrix([x, y, z, 0])

        # then we transform/rotate this vector to consider facing direction
        facing_dir_vector = (Matrix(dir_vector) @ self._facing_axis).matrix

        cls._camera_position += facing_dir_vector

        cls._VPMatrix = ViewProjectionMatrix(cls._camera_position[:-1],
                                             self.__class__._camera_angles,
                                             self.__class__.fov, self.__class__.aspect_ratio, self.__class__.near,
                                             self.__class__.far)

    def __setitem__(self, key, value):
        """
        API for camera_obj['axis'] += float_step.
        Facing direction(vector) considered, i.e. we move forward relative to rotated angle.
        :param key:
        :param value:
        :return:
        """
        # there is no simple method for class attribute subscription
        cls = self.__class__

        # create direction vector with filled axis value as step.
        # 'y' += 0.3 -> dir_vector = [0, 0.3, 0, 0].

        dir_vector = Matrix([0, 0, 0, 0])
        # because mainly we use it as += increment value is the result of operation is *not* step itself
        # step = value - old_value
        dir_vector[axis_str_mapping[key]] = value - self._camera_position[axis_str_mapping[key]]

        # then we transform/rotate this vector to consider facing direction
        facing_dir_vector = (Matrix(dir_vector) @ self._facing_axis).matrix

        cls._camera_position += facing_dir_vector

        cls._VPMatrix = ViewProjectionMatrix(cls._camera_position[:-1],
                                             self.__class__._camera_angles,
                                             self.__class__.fov, self.__class__.aspect_ratio, self.__class__.near,
                                             self.__class__.far)


def get_model_position(model: Matrix):

    return model[3][0], model[3][1], model[3][2]


# we can join this two bottom and top triangle methods
def _raster_bottom_flat_triangle(v1, v2, v3, color):
    # step down from left side (slope) of triangle
    left_slope = (v2[x] - v1[x])/(v2[y] - v1[y] + 0.01)
    # step down from left side (slope) of triangle
    right_slope = (v3[x] - v1[x])/(v3[y] - v1[y] + 0.01 )

    leftx = int(v1[x])
    rightx = int(v1[x])

    # as we go down by 'dy=1' step  - draw line on each 'y' by +m/slope level
    for y_val in range(v1[y], v2[y], 1):
        Device.drawline((int(leftx), y_val), (int(rightx), y_val), color)

        leftx += left_slope
        rightx += right_slope


def _raster_top_flat_triangle(v1, v2, v3, color):
    # step up from left side (slope) of triangle
    left_slope = (v3[x] - v1[x])/(v3[y] - v1[y] + 0.01)
    # # step up from left side (slope) of triangle
    right_slope = (v3[x] - v2[x])/(v3[y] - v2[y] + 0.01)

    leftx = v3[x]
    rightx = v3[x]
    # as we go up by 'dy=-1' step  - draw line on each 'y' level by -m/slope
    for val_y in range(v3[y], v1[y], -1):

        Device.drawline((int(leftx), val_y), (int(rightx), val_y), color)

        leftx -= left_slope
        rightx -= right_slope


def fill_triangle(v1, v2, v3, color):

    # sort by y to consider v1 as topmost vertice ( scan line from down)
    # optimize calculation int64 > int
    v1, v2, v3 = sorted([v1.tolist(), v2.tolist(), v3.tolist()], key=lambda k: k[1])

    if v2[y] == v3[y]:
        _raster_bottom_flat_triangle(v1, v2, v3, color)
        pass
    elif v1[y] == v2[y]:
        _raster_top_flat_triangle(v1, v2, v3, color)

    else:
        # split for two flat triangles -  bottom flat then top flat, find intersection y

        mid_v = [int((v1[x] + (v2[y] - v1[y])/(v3[y] - v1[y]) * (v3[x] - v1[x]))), v2[y], 0, 1]
        Device.drawline((v2[x], v2[y]), (mid_v[x], mid_v[y]), color)

        # we can generalize this two methods if we want
        _raster_bottom_flat_triangle(v1, v2, mid_v, color)

        _raster_top_flat_triangle(v2, mid_v, v3, color)


def rotate_model(model, angleX, angleY, angleZ):
    position = get_model_position(model)
    model_origin = model @ (translate(*position, True))

    if angleY:
        model_origin = rotate_over_y(angleY) @ model_origin
    if angleX:
        model_origin = rotate_over_x(angleX) @ model_origin
    if angleZ:
        model_origin = rotate_over_z(angleZ) @ model_origin

    moved_back = model_origin @ (translate(*position))

    return moved_back


def point_diff(p1, p2):

    return p2[x] - p1[x], p2[y] - p1[y], p2[z] - p1[z]


def dotted_line(p1, p2, precision=150):
    line_data = []
    connection = p1, p2

    difference = point_diff(connection[0], connection[1])

    count = precision

    dfx, dfy, dfz = difference[x] / (count + 1), difference[y] / (count + 1), difference[z] / (count + 1)
    for step in range(count + 1):
        line_data.append(
            [round(p1[x] + dfx * step, 5), round(p1[y] + dfy * step, 5), round(p1[z] + dfz * step, 5), 1]
        )

    return line_data
