import gymnasium as gym
from gymnasium import spaces
from ballz_game import *
import numpy as np

class BallzEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    def __init__(self, width=700, height=900, nblocks=7, render_mode="human"):
        self.game = Game(width, height, nblocks)

        # grid dimensions with two channels (one for block health and one for tokens)
        self.dims = (self.game.limit+2, self.game.nblocks, 2)
        self.blocks = np.zeros(self.dims, dtype=int)
        # self.tokens = np.zeros(self.dims, dtype=int)

        self.width = width
        self.height = height
        self.nblocks = nblocks
        
        # spaces for observation
        blocks_space = spaces.Box(low=0, high=1, shape=self.dims, dtype=np.float32)
        # token_space = spaces.Box(low=0, high=1, shape=self.dims, dtype=int)
        # spaces.MultiBinary(self.dims) 
        nballs_space = spaces.Box(low=0, high=1, shape=(), dtype=np.float32)
        pos_space = spaces.Box(low=0, high=1, shape=(), dtype=np.float32)
        
        self.pos = self.game.launcher.x/self.width # launcher position
        self.nballs = len(self.game.balls)/self.game.index # number of balls

        self.observation_space = gym.spaces.Dict(
            {"blocks": blocks_space,
             # "tokens": token_space,
             "nballs": nballs_space,
             "position": pos_space
            })
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.block_total = 0

    def _get_obs(self):
        blocks = np.zeros(self.dims, dtype=np.float32)
        tokens = np.zeros(self.dims, dtype=np.float32)
        for block_coord in self.game.blocks:
            block = self.game.blocks[block_coord]
            blocks[block_coord[1], block_coord[0], 0] = np.float32(block.health)
            blocks[block_coord[1], block_coord[0], 1] = np.float32(block.token)
            # tokens[block_coord[::-1]] = int(block.token)
        if np.max(blocks[:,:,0]) > 0:
            blocks[:,:,0] = blocks[:,:,0]/self.game.index
        self.blocks = blocks
        # self.tokens = tokens
        self.nballs = len(self.game.balls)/self.game.index
        self.pos = self.game.launcher.x/self.width
            
        observation = {
            "blocks": self.blocks,
            # "tokens": self.tokens,
            "nballs": self.nballs,
            "position": self.pos
        }
        return observation

    def _get_info(self):
        return {"score": self.game.index}
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.terminated = False
        # self.window = None
        self.clock = None
        game_seed = seed if seed is not None else self.np_random.integers(0, 2**32 - 1)
        self.game = Game(self.width, self.height, self.nblocks, seed=game_seed)
        observation = self._get_obs()
        self.block_total = np.sum(observation["blocks"], dtype=np.float32)
        return observation, self._get_info()
        
    def step(self, action):
        if self.terminated:
            print("already termed")
            return self._get_obs(), 0, self.terminated, False, self._get_info()

        render=False
        if self.render_mode == "human":
            render=True

        if render and self.window == None and self.render_mode == "human":
            # pygame setup
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
        if render and self.clock == None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        game_over, sub_steps = self.game.step(np.pi/2*action, render=render, screen=self.window)
        observation = self._get_obs()
        self.terminated = game_over
        # change in tot health between steps decreasing penalty of increasing total blocks with 0.9 factor
        new_block_tot = np.sum(observation["blocks"][:,:,0])
        block_change = self.block_total - new_block_tot*0.8
        bonus = self.game.index if new_block_tot < 2 and not game_over else 0
        # - 0.1*sub_steps/observation['nballs']
        # reward = (1.1*self.game.index + block_change + bonus)/observation["nballs"] if not game_over else (self.game.index - 2*self.game.limit)/observation["nballs"]
        #reward = 1 + block_change if not game_over else -self.game.limit
        block_locs = np.array(list(self.game.blocks.keys()))
        if len(block_locs) > 0:
            impending_doom = -1.75*np.max(block_locs[:,1])/self.game.limit
        else:
            impending_doom = 0
        # reward = 1.25 + block_change + bonus if not game_over else -1
        reward = 1.25 + impending_doom if not game_over else -self.game.limit
        self.block_total = new_block_tot

        terminated = (self.game.index >= 1000) or game_over
        info = self._get_info()

        return observation, reward, terminated, False, info

    # def render(self):
    #     if self.window is None and self.render_mode == "human":
    #         # pygame setup
    #         pygame.init()
    #         self.window = pygame.display.set_mode((self.width, self.height))
    #     if self.clock is None and self.render_mode == "human":
    #         self.clock = pygame.time.Clock()

    #     surface = pygame.Surface((self.width, self.height))

    #     # fill the screen with a color to wipe away anything from last frame
    #     self.window.fill("black")

    #     font_obj = pygame.font.SysFont("Arial", 64, bold=True)
    #     text_surface_obj = font_obj.render(str(self.game.index), True, (255,255,255), (0,0,0))
    #     self.window.blit(text_surface_obj, (self.width/2, self.game.bwidth/4))

    #     for ball in self.game.balls:
    #         pygame.draw.circle(self.window, (255,255,255), self.game.g2scr_pos((ball.x,ball.y)), self.game.ball_rad)

    #     # render the game
    #     for b_loc in self.game.blocks:
    #         block = self.game.blocks[b_loc]
    #         if not block.token:
    #             rect = pygame.Rect(block.x-block.w/2, self.height-(block.y+block.h/2), block.w, block.h)
    #             pygame.draw.rect(self.window, (255,0,0), rect)
    #             font_obj = pygame.font.SysFont("Arial", 64, bold=True)
    #             text_surface_obj = font_obj.render(str(block.health), True, (0,0,0), (255,0,0))
    #             self.window.blit(text_surface_obj, np.array(rect.center) - np.array(font_obj.size(str(block.health)))/2)
    #         elif block.token:
    #             pygame.draw.circle(self.window, (255,255,0), self.game.g2scr_pos(self.game.idx_to_pos(b_loc)), self.game.bwidth/4)

    #         # flip() the display to put your work on screen
    #         pygame.display.flip()

    #     self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.display.quit()
        pygame.quit()
        self.window = None
        self.clock = None
