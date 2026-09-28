import gymnasium as gym
from gymnasium import spaces
from ballz_game import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BallzEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    def __init__(self, width, height, nblocks, render_mode):
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
        indx_space = spaces.Box(low=1, high=10000, shape=(), dtype=int)
        
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

        # self.window = None
        self.clock = None
        game_seed = seed if seed is not None else self.np_random.integers(0, 2**32 - 1)
        self.game = Game(self.width, self.height, self.nblocks, seed=game_seed)
        observation = self._get_obs()
        self.block_total = np.sum(observation["blocks"], dtype=np.float32)
        return observation, self._get_info()
        
    def step(self, action):
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

        # change in tot health between steps decreasing penalty of increasing total blocks with 0.9 factor
        new_block_tot = np.sum(observation["blocks"][:,:,0])
        block_change = self.block_total - new_block_tot*0.99
        bonus = self.game.index if new_block_tot < 2 and not game_over else 0
        # - 0.1*sub_steps/observation['nballs']
        # reward = (1.1*self.game.index + block_change + bonus)/observation["nballs"] if not game_over else (self.game.index - 2*self.game.limit)/observation["nballs"]
        #reward = 1 + block_change if not game_over else -self.game.limit
        block_locs = np.array(list(self.game.blocks.keys()))
        if len(block_locs) > 0:
            impending_doom = -1.75*np.max(block_locs[:,1])/self.game.limit
        else:
            impending_doom = 0
        reward = 1.25 + impending_doom if not game_over else -1
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

class BallzAgent(models.Model):
    def __init__(self, env):
        self.env = env
        obs_space = env.observation_space
        self.grid_dims = obs_space["blocks"].shape
        self.cnn = models.Sequential([
             layers.Input(self.grid_dims, name='Input'),
             layers.Conv2D(24, (2,2), padding='same', activation='relu', name='Conv2d Layer 1'),
             layers.Conv2D(32, (2,2), padding='same', activation='relu', name='Conv2d Layer 2'),
             layers.Flatten(),
             layers.Dense(20, activation='relu')
            ])

        self.obs_flat_shape = self.cnn.output_shape[-1] + len(obs_space)-1
        
        self.actor = models.Sequential([
                layers.Input(shape=(self.obs_flat_shape,), name='Input'),
                layers.Dense(32, activation='tanh'),
                layers.Dense(32, activation='relu'),
                layers.Dense(2, activation='tanh')
            ])

        self.critic = models.Sequential([
                layers.Input(shape=(self.obs_flat_shape,), name='Input'),
                layers.Dense(32, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(1)
            ])

        self.learning_rate = 1e-4
        self.cnn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.a_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate*3)

        self.gamma = 0.9

        self.a_losses = []
        self.c_losses = []
        self.rewards = []
        self.scores = []
        self.fig = None
        self.ax = None
        self.recent_rewards = []
        self.recent_rewards = []
        self.recent_mean_action = []
        self.recent_std_action = []

    def forward(self, obs_space):
        flat_blocks = self.cnn(tf.constant([obs_space["blocks"]]))
        scalar_tensor = tf.stack([obs_space["nballs"], obs_space["position"]],0)
        state_tensor = tf.concat([flat_blocks, [scalar_tensor]], 1)
        return state_tensor
        
    def train(self, n_episodes=1000, live_plot=False):
        # rewards = []
        if live_plot and self.fig == None and self.ax == None:
            self.fig, self.ax = plt.subplots(4,1)
            line, = self.ax[0].plot(self.rewards, label="rewards")
            line, = self.ax[0].plot(self.scores, label="scores")
            line, = self.ax[1].plot(self.recent_rewards, label="step rewards")
            line, = self.ax[2].plot(self.recent_mean_action, label="action mean")
            line, = self.ax[2].plot(self.recent_std_action, label="action std")
            line, = self.ax[3].plot(self.c_losses, label="critic loss")
            line, = self.ax[3].plot(self.a_losses, label="actor loss")
            plt.rcParams['text.color'] = '#00FF00'
            plt.rcParams['axes.labelcolor'] = '#00FF00'
            plt.rcParams['xtick.color'] = '#00FF00'
            plt.rcParams['ytick.color'] = '#00FF00'
            plt.rcParams['legend.facecolor'] = '#000000'
            plt.rcParams['legend.edgecolor'] = '#000000'
    
        for _ in range(n_episodes):
            try:
                if live_plot:
                    fig = self.ax[0].get_figure()
                    fig.patch.set_facecolor("black")
                    self.ax[0].set_facecolor("black")
                state, info = self.env.reset()
                episode_reward, new_state, score = self.run_episode(state, live_plot=live_plot)
                #self.cnn.save('models/feature_extractor.keras')
                #self.critic.save('models/critic.keras')
                #self.actor.save('models/actor.keras')
                self.rewards.append(episode_reward)
                self.scores.append(score)
                #ani = FuncAnimation(self.fig, self.animate,cache_frame_data=False)
                # ax = plt.gca()
                if live_plot:
                    self.ax[0].axhline(ls='--')
                    self.ax[0].lines[0].set_xdata(range(len(self.rewards)))
                    self.ax[0].lines[1].set_xdata(range(len(self.scores)))
                    self.ax[0].lines[0].set_ydata(np.array(self.rewards))
                    self.ax[0].lines[1].set_ydata(np.array(self.scores))
                    self.ax[0].lines[0].set_color("green")
                    self.ax[0].lines[1].set_color("red")
                    self.ax[0].spines['bottom'].set_color("green")
                    self.ax[0].spines['top'].set_color("green")
                    self.ax[0].spines['left'].set_color("green")
                    self.ax[0].spines['right'].set_color("green")
                    self.ax[0].tick_params(labelcolor="green", color="green")
                    self.ax[0].relim(); self.ax[0].autoscale_view(); 
                    self.ax[0].legend()
                    plt.pause(0.01);


            except Exception as e:
                self.env.close()
                raise e

        self.env.close()

        return self.rewards

    def animate(self, i):
        self.ax.clear()
        self.ax.plot(range(len(self.rewards)),self.rewards)
        plt.show()

    def run_episode(self, state, live_plot=False):
        terminated = False
        tot_reward = 0
        score = 0
        discount = 1
        self.recent_rewards = []
        self.recent_mean_action = []
        self.recent_std_action = []
        while not terminated:
            if live_plot:
                self.ax[1].set_facecolor("black")
                self.ax[1].axhline(ls='--')
                self.ax[1].lines[0].set_xdata(range(len(self.recent_rewards)))
                self.ax[1].lines[0].set_ydata(self.recent_rewards) 
                self.ax[1].lines[0].set_color("green")
                self.ax[1].spines['bottom'].set_color("green")
                self.ax[1].spines['top'].set_color("green")
                self.ax[1].spines['left'].set_color("green")
                self.ax[1].spines['right'].set_color("green")
                self.ax[1].tick_params(labelcolor="green", color="green")
                self.ax[1].relim(); self.ax[1].autoscale_view(); 
                self.ax[1].legend()
                self.ax[2].set_facecolor("black")
                self.ax[2].axhline(ls='--')
                self.ax[2].lines[0].set_xdata(range(len(self.recent_mean_action)))
                self.ax[2].lines[0].set_ydata(self.recent_mean_action) 
                self.ax[2].lines[1].set_xdata(range(len(self.recent_std_action)))
                self.ax[2].lines[1].set_ydata(self.recent_std_action) 
                self.ax[2].lines[0].set_color("green")
                self.ax[2].lines[1].set_color("red")
                self.ax[2].spines['bottom'].set_color("green")
                self.ax[2].spines['top'].set_color("green")
                self.ax[2].spines['left'].set_color("green")
                self.ax[2].spines['right'].set_color("green")
                self.ax[2].tick_params(labelcolor="green", color="green")
                self.ax[2].relim(); self.ax[2].autoscale_view(); 
                self.ax[2].legend()
                self.ax[3].set_facecolor("black")
                self.ax[3].axhline(ls='--')
                self.ax[3].lines[0].set_xdata(range(len(self.c_losses[-50:])))
                self.ax[3].lines[0].set_ydata(self.c_losses[-50:]) 
                self.ax[3].lines[1].set_xdata(range(len(self.a_losses[-50:])))
                self.ax[3].lines[1].set_ydata(self.a_losses[-50:]) 
                self.ax[3].lines[0].set_color("green")
                self.ax[3].lines[1].set_color("red")
                self.ax[3].spines['bottom'].set_color("green")
                self.ax[3].spines['top'].set_color("green")
                self.ax[3].spines['left'].set_color("green")
                self.ax[3].spines['right'].set_color("green")
                self.ax[3].tick_params(labelcolor="green", color="green")
                self.ax[3].relim(); self.ax[3].autoscale_view(); 
                self.ax[3].legend()
                plt.pause(0.001)
            # self.learning_rate = np.clip(self.learning_rate*1.05, a_min=0, a_max=1e-3)

            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape, tf.GradientTape(persistent=True) as cnn_tape:
                state_tensor = self.forward(state)
                # Actor mean and std prediction assuming gaussian distribution
                a_mean, a_std = self.actor(state_tensor)[0]
                # action = np.clip(np.random.normal(a_mean, np.abs(a_std))/2, a_min=-0.95, a_max=0.95) # sample from gaussian
                # a_std = tf.math.maximum(1e-3, tf.abs(a_std))
                a_std = tf.math.exp(a_std)

                pre_norm_action = tf.random.normal([1],a_mean, a_std)[0] # sample from gaussian
                action = 0.95*tf.math.tanh(pre_norm_action)
                if np.isnan(action):
                    raise Exception("Action is nan, something went awry.")
                observation, reward, terminated, _, info = self.env.step(action)
                self.recent_rewards.append(reward)
                self.recent_mean_action.append(a_mean)
                self.recent_std_action.append(a_std)
                new_state_tensor = self.forward(observation) # observation as a tensor

                curr_val = self.critic(state_tensor)[0][0]
                next_val = self.critic(new_state_tensor)[0][0]

                if terminated:
                    # no next state exists
                    target = reward
                else:
                    # what the predicted reward actually is given next state
                    target = tf.stop_gradient(reward + self.gamma*next_val)

                advantage = target - curr_val
                # log of the probability density for a truncated gaussian
                phi = lambda x: 1/tf.sqrt(2*np.pi) * tf.exp(-1/2 * x**2)
                PHI = lambda x: 1/2*(1 + tf.math.erf(x/np.sqrt(2)))
                inp = lambda x: (tf.cast(x, tf.float32) - a_mean)/a_std
                # log_pi = lambda x: tf.math.log( 1/a_std*phi(inp(x))/(PHI(inp(0.99)) - PHI(inp(-0.99))) )
                log_pi = lambda x: tf.math.log(1/(tf.abs(a_std)*np.sqrt(2*np.pi))) - tf.square(tf.cast(x, tf.float32) - a_mean)/(2*tf.math.square(a_std)) 
                log_prob = log_pi(pre_norm_action)
                actor_loss = -discount*log_prob * tf.stop_gradient(advantage)
                #actor_loss = -log_pi(action) * target
                critic_loss = tf.math.square(advantage)
                joint_loss = tf.add(actor_loss, critic_loss)
                self.a_losses.append(actor_loss)
                self.c_losses.append(critic_loss)

                # print("T,V:", target,curr_val.numpy())
                #print("A,R,L:", advantage.numpy(), reward, critic_loss.numpy())

            # update critic and actor
            c_gradient = critic_tape.gradient(critic_loss, self.critic.trainable_weights)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_weights))
            a_gradient = actor_tape.gradient(actor_loss, self.actor.trainable_weights)
            self.a_optimizer.apply_gradients(zip(a_gradient, self.actor.trainable_weights))
            # use joint loss to update cnn featureinator
            cnn_gradient = cnn_tape.gradient(joint_loss, self.cnn.trainable_weights)
            self.cnn_optimizer.apply_gradients(zip(cnn_gradient, self.cnn.trainable_weights))
            discount *= self.gamma

            score = info['score']

            state_tensor = new_state_tensor
            tot_reward += reward

        return tot_reward, state_tensor, score 
