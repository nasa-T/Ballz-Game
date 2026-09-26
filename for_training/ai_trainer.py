import gymnasium as gym
from gymnasium import spaces
from ballz_game import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

class BallzEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    def __init__(self, width, height, nblocks, render_mode):
        self.game = Game(width, height, nblocks)

        # grid dimensions with two channels (one for block health and one for tokens)
        self.dims = (self.game.limit+1, self.game.nblocks, 2)
        self.blocks = np.zeros(self.dims, dtype=int)
        # self.tokens = np.zeros(self.dims, dtype=int)

        self.width = width
        self.height = height
        self.nblocks = nblocks
        
        # spaces for observation
        blocks_space = spaces.Box(low=0, high=10000, shape=self.dims, dtype=int)
        # token_space = spaces.Box(low=0, high=1, shape=self.dims, dtype=int)
        # spaces.MultiBinary(self.dims) 
        nballs_space = spaces.Box(low=0, high=10000, shape=(), dtype=int)
        pos_space = spaces.Box(low=0, high=width, shape=(), dtype=int)
        
        self.pos = self.game.launcher.x # launcher position
        self.nballs = len(self.game.balls) # number of balls

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

    def _get_obs(self):
        blocks = np.zeros(self.dims, dtype=int)
        tokens = np.zeros(self.dims, dtype=int)
        for block_coord in self.game.blocks:
            block = self.game.blocks[block_coord]
            blocks[block_coord[1], block_coord[0], 0] = int(block.health)
            blocks[block_coord[1], block_coord[0], 1] = int(block.token)
            # tokens[block_coord[::-1]] = int(block.token)

        self.blocks = blocks
        # self.tokens = tokens
        self.nballs = len(self.game.balls)
        self.pos = self.game.launcher.x
            
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

        self.window = None
        self.clock = None
        game_seed = seed if seed is not None else self.np_random.integers(0, 2**32 - 1)
        self.game = Game(self.width, self.height, self.nblocks, seed=game_seed)
        observation = self._get_obs()
        return observation, self._get_info()
        
    def step(self, action):
        game_over = self.game.step(np.pi/2*action)
        observation = self._get_obs()

        reward = 1/self.game.limit if not game_over else -1

        terminated = (self.game.index >= 1000) or game_over
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        if self.window is None and self.render_mode == "human":
            # pygame setup
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((self.width, self.height))

        # fill the screen with a color to wipe away anything from last frame
        self.window.fill("black")

        font_obj = pygame.font.SysFont("Arial", 64, bold=True)
        text_surface_obj = font_obj.render(str(self.game.index), True, (255,255,255), (0,0,0))
        self.window.blit(text_surface_obj, (self.width/2, self.game.bwidth/4))

        for ball in self.game.balls:
            pygame.draw.circle(self.window, (255,255,255), self.game.g2scr_pos((ball.x,ball.y)), self.game.ball_rad)

        # render the game
        for b_loc in self.game.blocks:
            block = self.game.blocks[b_loc]
            if not block.token:
                rect = pygame.Rect(block.x-block.w/2, self.height-(block.y+block.h/2), block.w, block.h)
                pygame.draw.rect(self.window, (255,0,0), rect)
                font_obj = pygame.font.SysFont("Arial", 64, bold=True)
                text_surface_obj = font_obj.render(str(block.health), True, (0,0,0), (255,0,0))
                self.window.blit(text_surface_obj, np.array(rect.center) - np.array(font_obj.size(str(block.health)))/2)
            elif block.token:
                pygame.draw.circle(self.window, (255,255,0), self.game.g2scr_pos(self.game.idx_to_pos(b_loc)), self.game.bwidth/4)

            # flip() the display to put your work on screen
            pygame.display.flip()

        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class BallzAgent(models.Model):
    def __init__(self, env):
        self.env = env
        obs_space = env.observation_space
        self.grid_dims = obs_space["blocks"].shape
        self.cnn = models.Sequential([
             layers.Input(self.grid_dims, name='Input'),
             layers.Conv2D(16, (3,3), padding='same', activation='relu', name='Conv2d Layer 1'),
             layers.Conv2D(32, (3,3), padding='same', activation='relu', name='Conv2d Layer 2'),
             layers.Flatten()
            ])

        self.obs_flat_shape = self.cnn.output_shape[-1] + len(obs_space)-1
        
        self.actor = models.Sequential([
                layers.Input(shape=(self.obs_flat_shape,), name='Input'),
                layers.Dense(36, activation='relu'),
                layers.Dense(2, activation='tanh')
            ])

        self.critic = models.Sequential([
                layers.Input(shape=(self.obs_flat_shape,), name='Input'),
                layers.Dense(36, activation='relu'),
                layers.Dense(1)
            ])

        self.learning_rate = 1e-3
        self.cnn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.a_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.gamma = 0.99

    def forward(self, obs_space):
        flat_blocks = self.cnn(tf.constant([obs_space["blocks"]]))
        scalar_tensor = tf.stack([obs_space["nballs"], obs_space["position"]],0)
        state_tensor = tf.concat([flat_blocks, [scalar_tensor]], 1)
        return state_tensor
        
    def train(self, n_episodes=1000):
        rewards = []
        for _ in range(n_episodes):
            state, info = self.env.reset()
            state_tensor = self.forward(state)
            episode_reward, new_state = self.run_episode(state_tensor)


        return rewards

    def run_episode(self, state_tensor):
        terminated = False
        tot_reward = 0
        while not terminated:
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape, tf.GradientTape() as cnn_tape:
                # Actor mean and std prediction assuming gaussian distribution
                a_mean, a_std = self.actor(state_tensor)[0]
                print("m,std: ", a_mean, a_std)
                action = np.clip(np.random.normal(a_mean, np.abs(a_std)), a_min=-0.9, a_max=0.9) # sample from gaussian
                print("action: ", action)
                observation, reward, terminated, _, info = self.env.step(action)
                print("reward: ", reward)
                new_state_tensor = self.forward(observation) # observation as a tensor

                curr_val = self.critic(state_tensor)[0][0]
                next_val = self.critic(new_state_tensor)[0][0]
                print(curr_val, next_val+reward)

                if terminated:
                    # no next state exists
                    target = reward
                else:
                    # what the predicted reward actually is given next state
                    target = reward + self.gamma*next_val

                advantage = target - curr_val

                # log of the probability density for a gaussian
                log_pi = lambda x: tf.math.log(1/(a_std*np.sqrt(2*np.pi))) - tf.math.square(x - a_mean)/(2*tf.math.square(a_std)) 
                actor_loss = log_pi(action) * tf.stop_gradient(advantage)
                critic_loss = tf.math.square(advantage)

            # use critic loss to update cnn featureinator
            cnn_gradient = cnn_tape.gradient(critic_loss, self.cnn.trainable_weights)
            self.cnn_optimizer.apply_gradients(zip(cnn_gradient, self.cnn.trainable_weights))
            # update critic and actor
            c_gradient = critic_tape.gradient(critic_loss, self.critic.trainable_weights)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_weights))
            a_gradient = actor_tape.gradient(actor_loss, self.actor.trainable_weights)
            self.a_optimizer.apply_gradients(zip(a_gradient, self.actor.trainable_weights))

            print(reward,terminated)
            state_tensor = new_state_tensor
            tot_reward += reward

        return tot_reward, state_tensor 
