import pygame
import numpy as np

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_magnitude(self):
        return np.sqrt(self.x**2 + self.y**2)

    def set_magnitude(self, new_mag):
        # Changes magnitude to new_mag keeping the same direction
        self.x = self.x/self.get_magnitude()*new_mag
        self.y = self.y/self.get_magnitude()*new_mag

    def dot(self, other):
        return self.x*other.x + self.y*other.y

    def __add__(self, other):
        return Vector(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return Vector(self.x-other.x, self.y-other.y)

    def __mul__(self, other): # same as self.dot()
        if isinstance(other, Vector):
            return self.x*other.x + self.y*other.y
        else: # if isinstance(other, (int,float))
            return Vector(self.x*other, self.y*other)

    def __truediv__(self, k):
        return Vector(self.x/k, self.y/k)

class Game: # holds everything
    def __init__(self, width, height, nblocks, seed=None):
        np.random.seed(seed)
        # a dictionary mapping block index position to the block
        ## index position starts at (x,y) ~ (0,0)
        self.blocks = {} 
        self.dt = 0.016

        # Going to assume game origin is at bottom left
        self.width = width
        self.height = height
        self.nblocks = 7 # number of block spaces per row
        self.bwidth = width/nblocks
        self.limit = 8

        # Number of rows made so far (determines health of block/tot score)
        self.index = 0 
        self.block_chance = 0.6
        self.double_chance = 0.1
        self.block_space = self.bwidth/20 # space between blocks
        self.ball_rad = self.bwidth/9
        self.balls = [Ball(width/2, self.ball_rad, 0, 0, self.ball_rad, self.dt, self)]
        self.launcher = Launcher(self.balls, self.width/2)

        self.num_new = 0

        self.add_row()
        
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
                                               self.height-self.bwidth*3/2, self.bwidth/3,
                                               self.bwidth/3, self.bwidth, token=True)

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
            if j+1 == self.limit:
                if not self.blocks[(i,j)].token:
                    self.game_over()
                else:
                    self.blocks[(i,j)].hit()
        self.blocks = temp_blocks

    def destroy_block(self, block):
        self.blocks.pop(self.pos_to_idx((block.x,block.y)),0)

    def add_ball(self):
        self.balls.append(Ball(self.launcher.x, self.ball_rad, 0, 0, self.ball_rad, self.dt, self))
        self.launcher.return_ball(self.balls[-1])

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
        return (float(pos[0]), float(self.height - pos[1]))

    def game_over(self):
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        
    def step(self, angle, render=False):
        pygame.init()
        self.launcher.unlock()
        self.launcher.launch(angle)
        frame_time = 0
        clock = pygame.time.Clock()
        terminated = False
        running = True

        if render:
            screen = pygame.display.set_mode((self.width, self.height))
        while running:
            frame_time += clock.get_time()
            if len(self.launcher.queued) == self.launcher.tot_balls:
                for _ in range(self.num_new):
                    self.add_ball()
                    self.num_new = 0
                running = False
                self.add_row()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        terminated = True
                self.launcher.unlock()

            if not self.launcher.balls:
                self.launcher.lock()

            if frame_time >= 100 and len(self.launcher.balls) < self.launcher.tot_balls:
                self.launcher.launch(angle)
                frame_time = 0

            for ball in self.balls:
                ball.update()

            if render:
                screen.fill("black")
                self.render(screen)
                clock.tick(600)

        pygame.quit()
        return terminated
                    
    def render(self, screen):
        surface = pygame.Surface((self.width, self.height))
        
        font_obj = pygame.font.SysFont("Arial", 64, bold=True)
        text_surface_obj = font_obj.render(str(self.index), True, (255,255,255), (0,0,0))
        screen.blit(text_surface_obj, (self.width/2, self.bwidth/4))

        for ball in self.balls:
            ball.update()
            pygame.draw.circle(screen, (255,255,255), self.g2scr_pos((ball.x,ball.y)), self.ball_rad)

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

        # flip() the display to put your work on screen
        pygame.display.flip()



    def run(self):
        # pygame setup
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        running = True
        surface = pygame.Surface((self.width, self.height))

        frame_time = 0 # time between frames
        while running:
            frame_time += clock.get_time()
            # poll for events
            # pygame.QUIT event means the user clicked X to close your window

            if len(self.launcher.queued) == self.launcher.tot_balls:
                for _ in range(self.num_new):
                    self.add_ball()
                    self.num_new = 0

                self.add_row()
                self.launcher.unlock()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        for ball in self.balls:
                            curr_vel = Vector(ball.vx, ball.vy)
                            if curr_vel.get_magnitude():
                                curr_vel.set_magnitude(4*curr_vel.get_magnitude())
                                ball.vx = curr_vel.x
                                ball.vy = curr_vel.y
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = self.g2scr_pos(event.pos)
                    #block = self.blocks.get(self.pos_to_idx(mouse_pos), Block(self,0,0,0,0,0,0))
                    #block.hit()
                    if len(self.launcher.balls) >= self.launcher.tot_balls:
                        self.launcher.unlock()
                        angle = np.atan((mouse_pos[0] - self.launcher.x)/(mouse_pos[1]))
                        self.launcher.launch(angle)
                        frame_time = 0

            # fill the screen with a color to wipe away anything from last frame
            screen.fill("black")

            if not self.launcher.balls:
                self.launcher.lock()

            if frame_time >= 100 and len(self.launcher.balls) < self.launcher.tot_balls:
                self.launcher.launch(angle)
                frame_time = 0

            font_obj = pygame.font.SysFont("Arial", 64, bold=True)
            text_surface_obj = font_obj.render(str(self.index), True, (255,255,255), (0,0,0))
            screen.blit(text_surface_obj, (self.width/2, self.bwidth/4))

            for ball in self.balls:
                ball.update()
                pygame.draw.circle(screen, (255,255,255), self.g2scr_pos((ball.x,ball.y)), self.ball_rad)

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

            # flip() the display to put your work on screen
            pygame.display.flip()

            clock.tick(1/self.dt)  # limits FPS to 60

        pygame.quit()

        print("\033[31m###################################")
        print("             GAME OVER")
        print("           Final Score:", self.index)
        print("###################################\033[0m")



class Launcher: #launches balls
    def __init__(self, balls, x):
        # launcher starts off locked with all balls in queue
        self.balls = [] 
        self.x = x
        self.locked = True
        self.queued = balls
        self.tot_balls = len(self.balls)

    # angle in radians from vertical; positive = clockwise
    ## launch the first ball in the list
    def launch(self, angle):
        v = 1000
        if self.balls and not self.locked:
            self.balls[0].vx = v*np.sin(angle)
            self.balls[0].vy = v*np.cos(angle)
            self.balls[0].update()
            self.balls = self.balls[1:]

    def return_ball(self, ball):
        if not self.queued:
            self.x = ball.x
        else:
            ball.x = self.x
        self.queued.append(ball)

    def unlock(self):
        if self.locked and len(self.queued) >= self.tot_balls:
            self.x = self.queued[0].x
            self.balls = self.queued
            self.queued = []
            self.locked = False
            for ball in self.balls:
                ball.x = self.x
            self.tot_balls = len(self.balls)

    def lock(self):
        self.locked = True

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

    def update(self):
        if self.y < self.radius: # if ball hits the bottom
            self.y = self.radius # set position to bottom
            self.vx = 0
            self.vy = 0
            self.game.launcher.return_ball(self)
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
        
        # find the closest locations adjacent to current one based on velocity
        i_inc = np.sign(self.vx)
        j_inc = -np.sign(self.vy)
        adjacent_locs = [(idx[0],idx[1]), (idx[0]+i_inc,idx[1]), (idx[0],idx[1]+j_inc), (idx[0]+i_inc,idx[1]+j_inc)]

        for loc in adjacent_locs:
            adj_block = self.game.blocks.get(loc, None)
            if adj_block:
                p_bl_ba = Vector(self.x-adj_block.x, self.y-adj_block.y) # position vector block to ball
                dist = p_bl_ba.get_magnitude()
                angle_from_vert = np.asin((self.x-adj_block.x)/dist)
                angle_from_horz = np.asin((self.y-adj_block.y)/dist)

                if np.abs(angle_from_vert) <= np.pi/4: # top or bottom quadrant
                    above_below = np.sign(self.y - adj_block.y)
                    if self.x > adj_block.x+adj_block.w/2: # if outside of corner
                        intersection = Vector(adj_block.x+adj_block.w/2, adj_block.y+above_below*adj_block.h/2)
                    elif self.x < adj_block.x - adj_block.w/2:
                        intersection = Vector(adj_block.x-adj_block.w/2, adj_block.y+above_below*adj_block.h/2)
                    else:
                        intersection = Vector(self.x, adj_block.y+above_below*adj_block.h/2)

                else: # side quadrants
                    left_right = np.sign(self.x - adj_block.x)
                    if self.y > adj_block.y+adj_block.w/2: # if outside of corner
                        intersection = Vector(adj_block.x+left_right*adj_block.w/2, adj_block.y+adj_block.h/2)
                    elif self.y < adj_block.y - adj_block.w/2:
                        intersection = Vector(adj_block.x+left_right*adj_block.w/2, adj_block.y-adj_block.h/2)
                    else:
                        intersection = Vector(adj_block.x+left_right*adj_block.w/2, self.y)

                v_int_ball = Vector(self.x, self.y) - intersection
                overlap = v_int_ball.get_magnitude() - self.radius
                if overlap <= 0: # if ball's edge overlaps box's edge
                    if adj_block.token:
                        self.game.num_new += 1
                    else:
                        new_dist = v_int_ball.get_magnitude() - overlap
                        v_int_ball.set_magnitude(new_dist)
                        ball_p_vec = intersection + v_int_ball
                        self.x = ball_p_vec.x
                        self.y = ball_p_vec.y

                        norm = Vector(self.x, self.y) - intersection
                        vel = Vector(self.vx, self.vy)
                        proj_vel_norm = norm*((vel*norm)/norm.get_magnitude()**2)
                        num_proj = ((vel*norm)/norm.get_magnitude()**2)
                        reflection = vel - proj_vel_norm*2
                        self.vx = reflection.x
                        self.vy = reflection.y

                    adj_block.hit() 

    def project_to_next_idx(self):
        new_x = self.x + self.vx*self.dt
        new_y = self.y + self.vy*self.dt
        return new_x, new_y

if __name__ == "__main__":
    # game = Game(700, 950, 7)
    # game.run()
    pass
