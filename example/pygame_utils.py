import pygame

from play3d.three_d import Camera

base_step = 0.3


def handle_camera_with_keys():
    keys = pygame.key.get_pressed()

    camera_position = Camera.get_instance()
    if keys[pygame.K_UP]:
        camera_position['y'] += base_step

    if keys[pygame.K_DOWN]:
        camera_position['y'] -= base_step

    if keys[ord('w')]:
        camera_position['z'] -= base_step

    if keys[ord('s')]:
        camera_position['z'] += base_step

    if keys[ord('d')]:
        camera_position['x'] += base_step

    if keys[ord('a')]:
        camera_position['x'] -= base_step

    if keys[ord('q')]:
        camera_position.rotate('y', -2)

    if keys[ord('e')]:
        camera_position.rotate('y', 2)

    if keys[ord('r')]:
        camera_position.rotate('x', 2)

    if keys[ord('t')]:
        camera_position.rotate('x', -2)
