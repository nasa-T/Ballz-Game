import pygame
import numpy as np

def run():
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        # RENDER YOUR GAME HERE

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

    pygame.quit()

class Game: # holds evverything
    def __init__(self):
        self.blocks = []
        self.balls = []
        self.dt = 0.016


class Launcher: #launches balls
    def __init__(self, balls):
        self.balls = balls

    # angle in radians from vertical; positive = clockwise
    def launch(self, angle):
        v = 1
        for ball in self.balls:
            ball.vx = v*np.sin(angle)
            ball.vy = v*np.cos(angle)
            ball.update()

class Block:
    def __init__(self, tot_health, x, y, width, height):
        self.health = tot_health
        self.x = x
        self.y = y
        self.w = width
        self.h = height

    def hit(self, ba):
        self.health -= 1

    def shift_down(self):
        self.y = self.y - height

class Ball:
    def __init__(self, x, y, vx, vy, radius, dt, game):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.dt = dt
        self.game = game

    def update(self, end_x=None):
        if y <= radius: # if ball hits the bottom
            self.y = radius # set position to bottom
            # if another ball has hit the bottom first, consolidate to that position
            if end_x != None: 
                self.x = end_x
            self.vx = 0
            self.vy = 0

        else: # position change by current velocity
           self.x = self.x + self.vx * self.dt 
           self.y = self.y + self.vy * self.dt 

    
