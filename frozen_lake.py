from contextlib import closing
from copy import deepcopy
from io import StringIO
from pprint import pprint
from time import sleep

import gym
import numpy as np
import pygame
import pygame_widgets as pw
from gym import utils
from gym.envs.toy_text import FrozenLakeEnv

GREY = (171, 171, 171)
GREEN = (3, 150, 3)
TURQUOISE = (80, 199, 199)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class FrozenLakeEnvGui(FrozenLakeEnv):
    def __init__(self, ):
        super().__init__(is_slippery=False)
        self.learn_pause = True
        self.fast_forward = False
        self.print_policy = True

        self.side_len = 800
        self.tile_width = self.side_len / 4.0

        pygame.init()
        pygame.font.init()

        self.font = pygame.font.SysFont('Comic Sans MS', 20)
        self.v_font = pygame.font.SysFont('Comic Sans MS', 30)
        self.ice_crack_img = pygame.transform.scale(pygame.image.load("res/crack.png"),
                                                    (int(self.tile_width - 2.0), int(self.tile_width - 2.0)))
        self.robot_img = pygame.transform.scale(pygame.image.load("res/robot.png"),
                                                (int(self.tile_width - 2.0), int(self.tile_width - 2.0)))

        self.surface = pygame.display.set_mode((self.side_len, self.side_len + 40))

    def __draw_tiles(self, i, j, el):
        tile_width = self.tile_width
        x = i * tile_width
        y = j * tile_width
        tile_width = self.tile_width - 2.0
        pygame.draw.rect(self.surface, (0, 0, 0), (y, x, self.tile_width, self.tile_width))
        if el == "F":
            pygame.draw.rect(self.surface, TURQUOISE, (y, x, tile_width, tile_width))
        elif el == "S":
            pygame.draw.rect(self.surface, TURQUOISE, (y, x, tile_width, tile_width))
        elif el == "G":
            pygame.draw.rect(self.surface, GREEN, (y, x, tile_width, tile_width))
        else:
            pygame.draw.rect(self.surface, TURQUOISE, (y, x, tile_width, tile_width))
            self.surface.blit(self.ice_crack_img, (y, x))

    def __render_v_text(self, i, j, V):
        tile_width = self.tile_width
        x = i * tile_width + 10.0
        y = j * tile_width + 10.0
        idx = i * 4 + j
        text_surface = self.v_font.render(f'V: {V[idx]:.2f}', True, BLACK)
        self.surface.blit(text_surface, (y, x))

    def __render_q_text(self, i, j, Q):
        idx = i * 4 + j
        for k, q_ in enumerate(Q[idx]):
            text = self.font.render(f'{q_:.2f}', True, BLACK)
            text_rect = text.get_rect()
            x, y = self.__get_q_position(i, j, k, text_rect)
            self.surface.blit(text, (y, x))

    def __get_q_position(self, i, j, k, text_rect):
        tile_width = self.tile_width
        x = i * tile_width
        y = j * tile_width
        padding = 5.0
        if k == 0:  # left
            x_, y_ = (x + tile_width / 2.0 - text_rect.height / 2.0), y + padding
        elif k == 1:  # bottom
            x_, y_ = x + tile_width - text_rect.height - padding, y + tile_width / 2.0 - text_rect.width / 2.0
        elif k == 2:  # right
            x_, y_ = x + tile_width / 2.0 - text_rect.height / 2.0, y + tile_width - text_rect.width - padding
        elif k == 3:  # top
            x_, y_ = x + padding, y + tile_width / 2.0 - text_rect.width / 2.0
        else:
            raise Exception(f"Unknown k: {k}")
        x_ = x_
        y_ = y_
        return x_, y_

    def __draw_robot(self, i, j):
        padding = 100.0
        x = i * self.tile_width + padding / 2.0
        y = j * self.tile_width + padding / 2.0
        tile_width = self.tile_width - padding

        self.surface.blit(pygame.transform.scale(self.robot_img, (int(tile_width), int(tile_width))), (x, y))

    def __draw_button(self, x, y, width, height, text=None):
        width = width - 4
        height = height - 4
        button = pygame.Rect(x, y, width, height)
        outline = pygame.Rect(x - 2, y - 2, width + 4, height + 4)
        pygame.draw.rect(self.surface, BLACK, outline)
        pygame.draw.rect(self.surface, GREY, button)
        if text is not None:
            text_el = self.font.render(text, True, (0, 0, 0))
            text_rect = text_el.get_rect()
            self.surface.blit(text_el,
                              (x + width / 2.0 - text_rect.width / 2.0, y + height / 2.0 - text_rect.height / 2.0))
        return button

    def __render_game(self, p_j, p_i, desc):
        for i, row in enumerate(desc):
            for j, el in enumerate(row):
                self.__draw_tiles(i, j, el)
                self.__draw_robot(p_j, p_i)

    def __render_values(self, desc, train_data, is_qvalue=False):
        for i, row in enumerate(desc):
            for j, el in enumerate(row):
                if is_qvalue:
                    self.__render_q_text(i, j, train_data)
                else:
                    self.__render_v_text(i, j, train_data)

    def __draw_buttons(self):
        print_policy = self.__draw_button(10, 810, 180, 20, text="Print Policy")
        pause = self.__draw_button(210, 810, 180, 20, text="Pause")
        play = self.__draw_button(410, 810, 180, 20, text="Play")
        fast_forward = self.__draw_button(610, 810, 180, 20, text="Fast Forward")
        for ev in pygame.event.get():
            if ev.type == pygame.MOUSEBUTTONDOWN:
                x, y = ev.pos
                if print_policy.collidepoint(x, y):
                    self.print_policy = not self.print_policy
                elif pause.collidepoint(x, y):
                    self.learn_pause = True
                elif play.collidepoint(x, y):
                    self.learn_pause = False
                    self.fast_forward = False
                elif fast_forward.collidepoint(x, y):
                    self.fast_forward = not self.fast_forward

    def render_gui(self, data, is_q_value=True):
        p_i, p_j = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        while True:
            self.surface.fill(WHITE)
            self.__render_game(p_j, p_i, desc)
            if self.print_policy:
                self.__render_values(desc, data, is_q_value)
            self.__draw_buttons()
            pygame.display.flip()

            if not self.fast_forward:
                sleep(0.5)

            if not self.learn_pause:  # cries in python
                break

    def reset_gui(self):
        obs = self.reset()
        self.learn_pause = True
        self.fast_forward = False
        self.print_policy = True
        return obs
