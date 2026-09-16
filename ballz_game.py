import pygame
import numpy as np

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getMagnitude(self):
        return np.sqrt(self.x**2 + self.y**2)

    def setMagnitude(self, new_mag):
        # Changes magnitude to new_mag keeping the same direction
        self.x = self.x/self.getMagnitude()*new_mag
        self.y = self.y/self.getMagnitude()*new_mag

    def dot(self, other):
        return self.x*other.x + self.y*other.y

    def __add__(self, other):
        return Vector(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return Vector(self.x-other.x, self.y-other.y)

    def __mul__(self, other): # same as self.dot()
        return self.x*other.x + self.y*other.y

class Game: # holds everything
    def __init__(self, width, height, nblocks):
        # a dictionary mapping block index position to the block
        ## index position starts at (x,y) ~ (0,0)
        self.blocks = {} 
        self.dt = 0.016

        # Going to assume game origin is at bottom left
        self.width = width
        self.height = height
        self.nblocks = 7 # number of block spaces per row
        self.bwidth = width/nblocks

        # Number of rows made so far (determines health of block/tot score)
        self.index = 0 
        self.block_chance = 0.7
        self.double_chance = 0.1
        self.block_space = self.bwidth/20 # space between blocks
        self.balls = [Ball(width/2, self.bwidth/10, 0, 0, self.bwidth/10, self.dt, self)]
        self.launcher = Launcher(self.balls, self.width/2)

    def add_row(self):
        self.shift_down() # shift all other rows down
        empty_spaces = []
        token_loc = -1
        if self.index > 1:
            # place a token in a random slot
            token_loc = np.random.randint(0,self.nblocks)
            # token is represented as 1; a token doesn't need to do anything on its own. 
            ## The ball will say if it hit a token
            self.blocks[(token_loc,1)] = Block(self, 1, token_loc*self.bwidth + self.bwidth/2,
                                               self.height-self.bwidth*3/2, self.bwidth-self.block_space,
                                               self.bwidth-self.block_space, self.bwidth, token=True)

        for i in range(self.nblocks):
            # if 1 is chosen, we are putting a block here; and the slot doesn't have a token 
            if np.random.choice([0,1], p=[1-self.block_chance,self.block_chance]) and not self.blocks.get((i,1), None):
                # chance for a block with double the health
                doubler = np.random.choice([1,2], p=[1-self.double_chance,self.double_chance])
                self.blocks[(i,1)] = Block(self, self.index*doubler, i*self.bwidth + self.bwidth/2,
                                           self.height-self.bwidth*3/2, self.bwidth-self.block_space,
                                           self.bwidth-self.block_space, self.bwidth)
            elif not self.blocks.get((i,1), None):
                # nothing was put in this space
                empty_spaces.append(i)

        if not empty_spaces:
            rem_loc = np.random.randint(0,self.nblocks)
            # if the location chosen is the token location, r
            while rem_loc == token_loc:
                rem_loc = np.random.randint(0,self.nblocks)
            self.destroy_block(self.blocks[(rem_loc,1)])

    def shift_down(self):
        self.index += 1
        # in order to avoid overwriting blocks, we create another 
        ## dictionary to temporarily store blocks
        temp_blocks = {}
        for i,j in self.blocks:
            # shift the block's location down
            self.blocks[(i,j)].shift_down()
            # add a reference to the shifted block in the new location idx
            temp_blocks[(i,j+1)] = self.blocks[(i,j)]
        self.blocks = temp_blocks

    def destroy_block(self, block):
        self.blocks.pop(self.pos_to_idx((block.x,block.y)),0)

    def add_ball(self):
        self.balls.append(Ball(self.width/2, self.bwidth/10, 0, 0, self.bwidth/10, self.dt, self))
        self.launcher.balls = self.balls

    def set_launcher(self, pos):
        self.launcher.x = pos
        for ball in self.balls:
            ball.x = pos

    def idx_to_pos(self, loc):
        x = loc[0]*self.bwidth + self.bwidth/2
        y = self.height - self.bwidth/2 - loc[1]*self.bwidth
        return (x, y)

    def pos_to_idx(self, pos):
        # position index closest to input x,y position
        x, y = pos
        i = x//self.bwidth
        j = (self.height - y)//self.bwidth
        return (i, j)

    def g2scr_pos(self, pos):
        return (pos[0], self.height - pos[1])

    def run(self):
        # pygame setup
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        running = True
        surface = pygame.Surface((self.width, self.height))
        self.add_row()

        while running:
            # poll for events
            # pygame.QUIT event means the user clicked X to close your window
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.add_row()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = self.g2scr_pos(event.pos)
                    block = self.blocks.get(self.pos_to_idx(mouse_pos), Block(self,0,0,0,0,0,0))
                    block.hit()
                    angle = np.atan((mouse_pos[0] - self.launcher.x)/(mouse_pos[1]))
                    self.launcher.launch(angle)

            # fill the screen with a color to wipe away anything from last frame
            screen.fill("black")

            # render the game
            for b_loc in self.blocks:
                block = self.blocks[b_loc]
                if not block.token:
                    rect = pygame.Rect(block.x-block.w/2, self.height-(block.y+block.h/2), block.w, block.h)
                    pygame.draw.rect(screen, (255,0,0), rect)
                    font_obj = pygame.font.SysFont("Arial", 64, bold=True)
                    text_surface_obj = font_obj.render(str(block.health), True, (0,0,0), (255,0,0))
                    screen.blit(text_surface_obj, np.array(rect.center) - np.array(font_obj.size(str(block.health)))/2)
                elif block.token:
                    pygame.draw.circle(screen, (255,255,0), self.g2scr_pos(self.idx_to_pos(b_loc)), self.bwidth/4)

            for ball in self.balls:
                ball.update()
                pygame.draw.circle(screen, (255,255,255), self.g2scr_pos((ball.x,ball.y)), self.bwidth/10)

            # flip() the display to put your work on screen
            pygame.display.flip()

            clock.tick(60)  # limits FPS to 60

        pygame.quit()


class Launcher: #launches balls
    def __init__(self, balls, x):
        self.balls = balls
        self.x = x

    # angle in radians from vertical; positive = clockwise
    ## launch the first ball in the list
    def launch(self, angle):
        v = 1000
        for ball in self.balls:
            if not ball.launched:
                ball.vx = v*np.sin(angle)
                ball.vy = v*np.cos(angle)
                ball.launched = True
                ball.update()

class Block:
    def __init__(self, game, tot_health, x, y, width, height, space_height, token=False):
        self.game = game
        self.health = tot_health
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        self.sh = space_height # height of space containing block
        self.token = token

    def hit(self):
        self.health -= 1
        if self.health == 0:
            # delete block from game
            self.game.destroy_block(self)

    def shift_down(self):
        self.y = self.y - self.sh

class Ball:
    def __init__(self, x, y, vx, vy, radius, dt, game):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.dt = dt
        self.game = game
        # if it's in the launcher, this is False until it's launched
        self.launched = False 

    def update(self, end_x=None):
        if self.launched:
            if self.y < self.radius: # if ball hits the bottom
                self.launched = False
                self.y = self.radius # set position to bottom
                self.game.set_launcher(self.x)
                # if another ball has hit the bottom first, consolidate to that position
                if end_x != None: 
                    self.x = end_x
                self.vx = 0
                self.vy = 0

            else: # position change by current velocity
                self.detect_collisions()
                self.x = self.x + self.vx * self.dt 
                self.y = self.y + self.vy * self.dt 

    def detect_collisions(self):
        # bounce off left wall
        if self.x <= self.radius and self.vx < 0:
            self.vx *= -1
            self.x = self.radius # reset to not inside of wall
        # bounce off right wall
        elif self.x >= self.game.width - self.radius and self.vx > 0:
            self.vx *= -1 
            self.x = self.game.width - self.radius
        
        # bounce off top
        if self.y >= self.game.height - self.radius and self.vy > 0:
            self.vy *= -1
            self.y = self.game.height - self.radius

        idx = self.game.pos_to_idx((self.x, self.y))
        curr_loc = self.game.blocks.get(idx)
        
        #if self.vy > 0: # ball moving up; would hit a bottom edge
        #    vert_block = self.game.blocks.get((idx[0],idx[1]-1))
        #elif self.vy < 0: # ball moving down; would hit a top edge
        #    vert_block = self.game.blocks.get((idx[0],idx[1]+1))
        # above block is reduced into the below

        if -1 < idx[0] + np.sign(self.vx) < self.game.nblocks:
            horz_block = self.game.blocks.get((idx[0]+np.sign(self.vx),idx[1]), None)
            if horz_block != None:
                if np.sign(self.vx)*(self.x+np.sign(self.vx)*self.radius) > np.sign(self.vx)*(horz_block.x - np.sign(self.vx)*horz_block.w/2):
                    if not horz_block.token:
                        self.x = horz_block.x - np.sign(self.vx)*(horz_block.w/2+self.radius)
                        self.vx*= -1
                    else:
                        self.game.add_ball()
                    horz_block.hit()

        if idx[1] - np.sign(self.vy) > -1:
            vert_block = self.game.blocks.get((idx[0],idx[1]-np.sign(self.vy)), None)
            if vert_block != None:
                if np.sign(self.vy)*(self.y+np.sign(self.vy)*self.radius) > np.sign(self.vy)*(vert_block.y - np.sign(self.vy)*vert_block.h/2):
                    if not vert_block.token:
                        self.y = vert_block.y - np.sign(self.vy)*(vert_block.h/2+self.radius)
                        self.vy*= -1
                    else:
                        self.game.add_ball()
                    vert_block.hit()

        new_pos = self.project_positions()

        i_inc = np.sign(self.x % self.game.bwidth - self.game.bwidth/2)
        j_inc = -np.sign(self.y % self.game.bwidth - self.game.bwidth/2)

       
    def project_to_next_idx(self):
        new_x = self.x + self.vx*self.dt
        new_y = self.y + self.vy*self.dt
        return new_x, new_y

if __name__ == "__main__":
    game = Game(700, 950, 7)
    game.run()
