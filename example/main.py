import logging
import os
import sys

import pygame

from play3d.models import Model, Grid
from pygame_utils import handle_camera_with_keys
from play3d.three_d import Device, Camera
from play3d.utils import capture_fps

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

os.environ["SDL_VIDEO_CENTERED"] = '1'
black, white = (20, 20, 20), (230, 230, 230)


Device.viewport(1024, 768)
pygame.init()
screen = pygame.display.set_mode(Device.get_resolution())

# just for simplicity - array access, we should avoid that
x, y, z = 0, 1, 2

# pygame sdl line is faster than default one
line_adapter = lambda p1, p2, color: pygame.draw.line(screen, color, (p1[x], p1[y]), (p2[x], p2[y]), 1)
put_pixel = lambda x, y, color: pygame.draw.circle(screen, color, (x, y), 1)

Device.set_renderer(put_pixel, line_renderer=line_adapter)

grid = Grid(color=(30, 140, 200), dimensions=(30, 30))
suzanne = Model.load_OBJ('suzanne.obj.txt', position=(3, 2, -7), color=white, rasterize=True)
beetle = Model.load_OBJ('beetle.obj.txt', wireframe=False, color=white, position=(0, 2, -11), scale=3)
beetle.rotate(0, 45, 50)

camera = Camera.get_instance()
# move our camera up and back a bit, from origin
camera.move(y=1, z=2)


@capture_fps
def frame():
    if pygame.event.get(pygame.QUIT):
        sys.exit(0)

    screen.fill(black)
    handle_camera_with_keys()  # we can move our camera
    grid.draw()
    beetle.draw()
    suzanne.rotate(0, 1, 0).draw()
    pygame.display.flip()


while True:

    frame()
