import gymnasium as gym
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BallzAgent(models.Model):
    def __init__(self, env, alpha=1e-4, gamma=0.9, epsilon=0.2, seed=None):
        np.random.seed(seed)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict to first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')

            # Enable memory growth
            tf.config.experimental.set_memory_growth(gpus[0], True)

        # tf.keras.mixed_precision.set_global_policy("mixed_float16")

        self.env = env
        obs_space = env.observation_space
        if isinstance(env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
            self.grid_dims = obs_space["blocks"].shape[1:]
            self.batch_size = obs_space["blocks"].shape[0]
        else:
            self.grid_dims = obs_space["blocks"].shape
            self.batch_size = 1

        self.cnn = models.Sequential([
             layers.Input(self.grid_dims, name='Input', batch_size=None),
             layers.Conv2D(16, (5,5), padding='same', activation='relu', name='Conv2d_Layer_1'),
             layers.Conv2D(16, (5,5), padding='same', activation='relu', name='Conv2d_Layer_2'),
             layers.Flatten(dtype=tf.float32),
             # layers.Dense(20, activation='relu')
            ])

        self.obs_flat_shape = self.cnn.output_shape[-1] + len(obs_space)-1
        
        self.actor = models.Sequential([
                layers.Input(shape=(self.obs_flat_shape,), name='Input', batch_size=None),
                layers.Dense(16, activation='sigmoid'),
                # layers.Dense(12, activation='relu'),
                layers.Dense(2, activation='tanh', dtype=tf.float32)
            ])

        self.critic = models.Sequential([
                layers.Input(shape=(self.obs_flat_shape,), name='Input', batch_size=None),
                layers.Dense(16, activation='sigmoid'),
                # layers.Dense(12, activation='relu'),
                layers.Dense(1, dtype=tf.float32)
            ])

        self.learning_rate = alpha
        self.cnn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # self.cnn_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.cnn_optimizer)
        self.a_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # self.a_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.a_optimizer)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate*3)
        # self.c_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.c_optimizer)

        self.gamma = gamma
        self.epsilon = epsilon

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
        self.logger = None
        self.log_logger = None
        self.logpath = None
        self.update_step = 1
        self.episode_num = 1

    @tf.function
    def forward(self, obs_space):
        if isinstance(self.env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
            flat_blocks = self.cnn(obs_space["blocks"])
            scalar_tensor = tf.stack([obs_space["nballs"], obs_space["position"]],1)
            state_tensor = tf.concat([flat_blocks, scalar_tensor], 1)
        else:
            flat_blocks = self.cnn([obs_space["blocks"]])
            scalar_tensor = tf.stack([obs_space["nballs"], obs_space["position"]],0)
            state_tensor = tf.concat([flat_blocks, [scalar_tensor]], 1)
        return state_tensor
        
    def train(self, n_episodes=1000, live_plot=False, ppo=False, mc=False):
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
            plt.rcParams['legend.loc'] = 'lower right'
    
        self.logpath = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logger = tf.summary.create_file_writer(self.logpath)
        self.update_step = 0
        self.episode_num = 0
        for _ in range(n_episodes):
            try:
                if live_plot:
                    fig = self.ax[0].get_figure()
                    fig.patch.set_facecolor("black")
                    self.ax[0].set_facecolor("black")

                state, info = self.env.reset()

                if mc:
                    episode_reward, new_state, score = self.run_episode_mc(state, live_plot=live_plot, ppo=ppo)
                else:
                    episode_reward, new_state, score = self.run_episode(state, live_plot=live_plot, ppo=ppo)

                #self.cnn.save('models/feature_extractor.keras')
                #self.critic.save('models/critic.keras')
                #self.actor.save('models/actor.keras')
                self.rewards.append(episode_reward)
                self.scores.append(score)
                # with self.logger.as_default():
                #     tf.summary.scalar("episode reward", episode_reward, step=self.episode_num)
                #     tf.summary.scalar("score", score, step=self.episode_num)
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

                self.episode_num += 1

            except Exception as e:
                self.env.close()
                raise e

        self.env.close()

        return self.rewards

    def run_episode(self, state, live_plot=False, ppo=False):
        terminated = np.array([False]*self.batch_size)
        tot_reward = 0
        score = 0
        discount = 1
        self.recent_rewards = []
        self.recent_mean_action = []
        self.recent_std_action = []
        prev_state_action_prob = None
        prev_mean = 1
        prev_std = 1

        while not terminated.all():
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
            with tf.profiler.experimental.Trace('train', step_num=self.update_step, _r=1), tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape, tf.GradientTape() as cnn_tape:
                state_tensor = self.forward(state)
                # Actor mean and std prediction assuming gaussian distribution
                a_mean, a_std = tf.transpose(self.actor(state_tensor))
                # action = np.clip(np.random.normal(a_mean, np.abs(a_std))/2, a_min=-0.95, a_max=0.95) # sample from gaussian
                # a_std = tf.math.maximum(1e-3, tf.abs(a_std))
                a_std = tf.math.exp(a_std)

                pre_norm_action = tf.random.normal([self.batch_size],a_mean, a_std) # sample from gaussian
                action = 0.95*tf.math.tanh(pre_norm_action)

                if np.isnan(action).any():
                    raise Exception("Action is nan, something went awry.")
                if isinstance(self.env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
                    observation, reward, step_terminated, _, info = self.env.step(action)
                else:
                    observation, reward, step_terminated, _, info = self.env.step(action[0])
                self.recent_rewards.append(tf.reduce_mean(reward))
                self.recent_mean_action.append(tf.reduce_mean(a_mean))
                self.recent_std_action.append(tf.reduce_mean(a_std))
                new_state_tensor = self.forward(observation) # observation as a tensor

                curr_val = self.critic(state_tensor)
                next_val = self.critic(new_state_tensor)

                if not isinstance(self.env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
                    if step_terminated:
                        # no next state exists
                        target = reward
                    else:
                        # what the predicted reward actually is given next state
                        target = tf.stop_gradient(reward + discount*next_val)
                else:
                    target = []
                    for i, t in enumerate(step_terminated):
                        if t:
                            target.append(tf.Variable([reward[i]], shape=(1,), dtype=tf.float32))
                            with self.logger.as_default():
                                tf.summary.scalar("score", info["score"][i], step=self.episode_num)
                                tf.summary.scalar("episode reward", tot_reward[i], step=self.episode_num)
                            tot_reward[i] = 0
                        else:
                            target.append(tf.stop_gradient(reward[i] + discount*next_val[i]))

                    if step_terminated.any():
                        self.episode_num += 1

                target = tf.convert_to_tensor(target, dtype=tf.float32)

                advantage = target - curr_val
                # log of the probability density for a truncated gaussian
                phi = lambda x: 1/tf.sqrt(2*np.pi) * tf.exp(-1/2 * x**2)
                PHI = lambda x: 1/2*(1 + tf.math.erf(x/np.sqrt(2)))
                inp = lambda x: (tf.cast(x, tf.float32) - a_mean)/a_std
                # log_pi = lambda x: tf.math.log( 1/a_std*phi(inp(x))/(PHI(inp(0.99)) - PHI(inp(-0.99))) )
                log_pi = lambda x, m, s: tf.math.log(1/(tf.abs(s)*np.sqrt(2*np.pi))) - tf.square(tf.cast(x, tf.float32) - m)/(2*tf.math.square(s)) 
                pi = lambda x, m, s: 1/(tf.sqrt(2*np.pi*tf.square(s)))*tf.exp(-tf.square(x - m)/(2*tf.square(s)))
                log_prob = log_pi(pre_norm_action, a_mean, a_std)

                if ppo:
                    if prev_state_action_prob:
                        old_state, old_action, old_log_prob, old_adv = prev_state_action_prob
                        # Mean and std based on previous state with updated actor
                        ps_mean, ps_std = tf.transpose(self.actor(old_state))
                        old_adv = tf.reduce_mean(old_adv)
                        old_action = tf.reduce_mean(old_action)
                        ps_mean = tf.reduce_mean(ps_mean)
                        ps_std = tf.reduce_mean(ps_std)
                        old_log_prob = tf.reduce_mean(old_log_prob)

                        ps_log_prob = log_pi(old_action, ps_mean, ps_std)
                        r = tf.math.exp(ps_log_prob - old_log_prob)
                        actor_loss = -tf.math.minimum(r*old_adv, tf.clip_by_value(r, 1-self.epsilon, 1+self.epsilon)*old_adv)
                        critic_loss = tf.reduce_mean(tf.math.square(advantage))
                        joint_loss = tf.add(actor_loss, critic_loss)
                        with self.logger.as_default():
                            tf.summary.scalar("critic loss", critic_loss, step=self.update_step)
                            tf.summary.scalar("value", tf.reduce_mean(curr_val), step=self.update_step)
                            tf.summary.scalar("reward", tf.reduce_mean(reward), step=self.update_step)
                            tf.summary.scalar("actor loss", actor_loss, step=self.update_step)

                else:
                    actor_loss = -tf.reduce_mean(discount*log_prob * tf.stop_gradient(advantage))
                    #scaled_actor_loss = self.a_optimizer.scale_loss(actor_loss)
                    #actor_loss = -log_pi(action) * target
                    critic_loss = tf.reduce_mean(tf.math.square(advantage))
                    #scaled_critic_loss = self.c_optimizer.scale_loss(critic_loss)
                    joint_loss = tf.add(actor_loss, critic_loss)
                    
                    if np.isnan(actor_loss) or np.isnan(critic_loss) or np.isnan(joint_loss):
                        raise Exception(f"At least one loss is Nan; something went awry: actor--{actor_loss}, critic--{critic_loss}, joint--{joint_loss}.")

                    with self.logger.as_default():
                        tf.summary.scalar("critic loss", critic_loss, step=self.update_step)
                        tf.summary.scalar("value", tf.reduce_mean(curr_val), step=self.update_step)
                        tf.summary.scalar("reward", tf.reduce_mean(reward), step=self.update_step)
                        tf.summary.scalar("actor loss", actor_loss, step=self.update_step)
                self.update_step += 1


                
            if not ppo or prev_state_action_prob:
                # update critic and actor
                c_gradient = critic_tape.gradient(critic_loss, self.critic.trainable_weights)

                self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_weights))
                a_gradient = actor_tape.gradient(actor_loss, self.actor.trainable_weights)

                self.a_optimizer.apply_gradients(zip(a_gradient, self.actor.trainable_weights))
                # use joint loss to update cnn featureinator
                cnn_gradient = cnn_tape.gradient(joint_loss, self.cnn.trainable_weights)

                self.cnn_optimizer.apply_gradients(zip(cnn_gradient, self.cnn.trainable_weights))
                discount *= self.gamma
                state = observation

                self.a_losses.append(actor_loss)
                self.c_losses.append(critic_loss)

            prev_state_action_prob = (state_tensor, pre_norm_action, log_prob, advantage)
            score = info['score']

            terminated = terminated | step_terminated
            # state_tensor = new_state_tensor
            state = observation # update state
            tot_reward += reward

        self.logger.flush()

        return tot_reward, state_tensor, tf.reduce_mean(score)

    def run_episode_mc(self, state, live_plot=False, ppo=False):
        terminated = np.array([False]*self.batch_size)
        tot_reward = 0
        score = 0
        discount = 1
        self.recent_rewards = []
        self.recent_mean_action = []
        self.recent_std_action = []
        log_probs = []
        pred_vals = [] # values predicted by critic
        states, actions = [], []
        prev_state_action_prob = None
        prev_mean = 1
        prev_std = 1
        while not terminated.all():
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

            state_tensor = self.forward(state)
            # Actor mean and std prediction assuming gaussian distribution
            a_mean, a_std = tf.transpose(self.actor(state_tensor))
            # action = np.clip(np.random.normal(a_mean, np.abs(a_std))/2, a_min=-0.95, a_max=0.95) # sample from gaussian
            # a_std = tf.math.maximum(1e-3, tf.abs(a_std))
            a_std = tf.math.exp(a_std)

            pre_norm_action = tf.random.normal([self.batch_size],a_mean, a_std) # sample from gaussian
            action = 0.95*tf.math.tanh(pre_norm_action)

            states.append(state)
            actions.append(pre_norm_action)
            # if np.isnan(action):
            #     raise Exception("Action is nan, something went awry.")
            if isinstance(self.env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
                observation, reward, step_terminated, _, info = self.env.step(action)
            else:
                observation, reward, step_terminated, _, info = self.env.step(action[0])

            self.recent_rewards.append(reward)
            self.recent_mean_action.append(a_mean)
            self.recent_std_action.append(a_std)
            new_state_tensor = self.forward(observation) # observation as a tensor

            curr_val = self.critic(state_tensor)
            next_val = self.critic(new_state_tensor)
            pred_vals.append(curr_val)

            target = []
            for i, t in enumerate(step_terminated):
                if t:
                    target.append(tf.Variable([reward[i]], shape=(1,), dtype=tf.float32))
                    with self.logger.as_default():
                        tf.summary.scalar("score", info["score"][i], step=self.episode_num)
                        tf.summary.scalar("episode reward", tot_reward[i], step=self.episode_num)
                    returns = self.compute_returns(i)
                    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape, tf.GradientTape(persistent=True) as cnn_tape:
                        states_batch = {
                            'blocks': tf.stack([s['blocks'][i] for s in states]),
                            'nballs': tf.stack([s['nballs'][i] for s in states]),
                            'position': tf.stack([s['position'][i] for s in states])
                        }
                        states_tensors = self.forward(states_batch)
                        means, stds = tf.transpose(self.actor(states_tensors))
                        log_pi = lambda x, m, s: tf.math.log(1/(tf.abs(s)*np.sqrt(2*np.pi))) - tf.square(tf.cast(x, tf.float32) - m)/(2*tf.math.square(s)) 
                        actions_tensor = tf.concat(actions, 0)
                        stds = tf.math.exp(stds)
                        log_pis = log_pi(actions_tensor, means, stds)
                        pred_vals = self.critic(states_tensors)
                        advantages = returns - pred_vals
                        actor_loss = -tf.reduce_mean(log_pis * tf.stop_gradient(advantages))
                        #actor_loss = -log_pi(action) * target
                        critic_loss = tf.reduce_mean(tf.math.square(advantages))
                        joint_loss = tf.add(actor_loss, critic_loss)
                        with self.logger.as_default():
                            tf.summary.scalar("critic loss", critic_loss, step=self.update_step)
                            tf.summary.scalar("value", tf.reduce_mean(curr_val), step=self.update_step)
                            tf.summary.scalar("reward", tf.reduce_mean(reward), step=self.update_step)
                            tf.summary.scalar("actor loss", actor_loss, step=self.update_step)
                        tot_reward[i] = 0
                else:
                    target.append(tf.stop_gradient(reward[i] + discount*next_val[i]))

            if step_terminated.any():
                self.episode_num += 1            

            terminated = terminated | step_terminated
            
            log_pi = lambda x, m, s: tf.math.log(1/(tf.abs(s)*np.sqrt(2*np.pi))) - tf.square(tf.cast(x, tf.float32) - m)/(2*tf.math.square(s)) 
            pi = lambda x, m, s: 1/(tf.sqrt(2*np.pi*tf.square(s)))*tf.exp(-tf.square(x - m)/(2*tf.square(s)))
            log_prob = log_pi(pre_norm_action, a_mean, a_std)
            log_probs.append(log_prob)


                # print("T,V:", target,curr_val.numpy())
                #print("A,R,L:", advantage.numpy(), reward, critic_loss.numpy())

            # prev_state_action_prob = (state_tensor, pre_norm_action, log_prob, advantage)
            score = info['score']

            # state_tensor = new_state_tensor
            state = observation # update state
            tot_reward += reward

        if ppo:
            if prev_state_action_prob:
                old_state, old_action, old_log_prob, old_adv = prev_state_action_prob
                # Mean and std based on previous state with updated actor
                ps_mean, ps_std = self.actor(old_state)
                ps_log_prob = log_pi(old_action, ps_mean, ps_std)
                r = tf.math.exp(ps_log_prob - old_log_prob)
                actor_loss = -tf.math.minimum(r*old_adv, tf.clip_by_value(r, 1-self.epsilon, 1+self.epsilon)*old_adv)
                critic_loss = tf.math.square(advantage)
                joint_loss = tf.add(actor_loss, critic_loss)
                with self.logger.as_default():
                    tf.summary.scalar("critic loss", critic_loss, step=self.update_step)
                    tf.summary.scalar("value", tf.reduce_mean(curr_val), step=self.update_step)
                    tf.summary.scalar("reward", tf.reduce_mean(reward), step=self.update_step)
                    tf.summary.scalar("actor loss", actor_loss, step=self.update_step)
                    
        else:
            returns = self.compute_returns()
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape, tf.GradientTape(persistent=True) as cnn_tape:
                states_batch = {
                    'blocks': tf.stack([s['blocks'][0] for s in states]),
                    'nballs': tf.stack([s['nballs'][0] for s in states]),
                    'position': tf.stack([s['position'][0] for s in states])
                }
                states_tensors = self.forward(states_batch)
                means, stds = tf.transpose(self.actor(states_tensors))
                log_pi = lambda x, m, s: tf.math.log(1/(tf.abs(s)*np.sqrt(2*np.pi))) - tf.square(tf.cast(x, tf.float32) - m)/(2*tf.math.square(s)) 
                actions_tensor = tf.concat(actions, 0)
                stds = tf.math.exp(stds)
                log_pis = log_pi(actions_tensor, means, stds)
                pred_vals = self.critic(states_tensors)
                advantages = returns - pred_vals
                actor_loss = -tf.reduce_mean(log_pis * tf.stop_gradient(advantages))
                #actor_loss = -log_pi(action) * target
                critic_loss = tf.reduce_mean(tf.math.square(advantages))
                joint_loss = tf.add(actor_loss, critic_loss)
                with self.logger.as_default():
                    tf.summary.scalar("critic loss", critic_loss, step=self.update_step)
                    tf.summary.scalar("value", tf.reduce_mean(curr_val), step=self.update_step)
                    tf.summary.scalar("reward", tf.reduce_mean(reward), step=self.update_step)
                    tf.summary.scalar("actor loss", actor_loss, step=self.update_step)
                    
            # update critic and actor
            c_gradient = critic_tape.gradient(critic_loss, self.critic.trainable_weights)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_weights))
            a_gradient = actor_tape.gradient(actor_loss, self.actor.trainable_weights)
            self.a_optimizer.apply_gradients(zip(a_gradient, self.actor.trainable_weights))
            # use joint loss to update cnn featureinator
            cnn_gradient = cnn_tape.gradient(joint_loss, self.cnn.trainable_weights)
            self.cnn_optimizer.apply_gradients(zip(cnn_gradient, self.cnn.trainable_weights))
            discount *= self.gamma
            state = observation
            
            self.a_losses.append(actor_loss)
            self.c_losses.append(critic_loss)

        states, actions, log_probs, pred_vals = [], [], [], []
        return tot_reward, state_tensor, score 

    def compute_returns(self, i):
        returns = np.zeros_like(self.recent_rewards, np.float32)
        G = 0
        for t in range(len(self.recent_rewards)-1, -1, -1):
            G = self.recent_rewards[t] + G*self.gamma
            returns[t] = G
        return returns

    def run_episode_mc_clean(self, state, live_plot=False, ppo=False):
        terminated = np.array([False]*self.batch_size)
        tot_reward = 0
        score = 0
        discount = 1
        self.recent_rewards = []
        self.recent_mean_action = []
        self.recent_std_action = []
        log_probs = []
        pred_vals = [] # values predicted by critic
        states, actions = [], []
        prev_state_action_prob = None
        prev_mean = 1
        prev_std = 1
        while not terminated.all():
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

            state_tensor = self.forward(state)
            # Actor mean and std prediction assuming gaussian distribution
            a_mean, a_std = self.actor(state_tensor)
            # action = np.clip(np.random.normal(a_mean, np.abs(a_std))/2, a_min=-0.95, a_max=0.95) # sample from gaussian
            # a_std = tf.math.maximum(1e-3, tf.abs(a_std))
            a_std = tf.math.exp(a_std)

            pre_norm_action = tf.random.normal([self.batch_size],a_mean, a_std) # sample from gaussian
            action = 0.95*tf.math.tanh(pre_norm_action)

            states.append(state)
            actions.append(pre_norm_action)
            # if np.isnan(action):
            #     raise Exception("Action is nan, something went awry.")
            if isinstance(self.env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
                observation, reward, step_terminated, _, info = self.env.step(action)
            else:
                observation, reward, step_terminated, _, info = self.env.step(action[0])

            target = []
            for i, t in enumerate(step_terminated):
                if t:
                    target.append(tf.Variable([reward[i]], shape=(1,), dtype=tf.float32))
                    with self.logger.as_default():
                        tf.summary.scalar("score", info["score"][i], step=self.episode_num)
                        tf.summary.scalar("episode reward", tot_reward[i], step=self.episode_num)
                    tot_reward[i] = 0
                else:
                    target.append(tf.stop_gradient(reward[i] + discount*next_val[i]))

            if step_terminated.any():
                self.episode_num += 1            

            terminated = terminated | step_terminated

            self.recent_rewards.append(reward)
            self.recent_mean_action.append(a_mean)
            self.recent_std_action.append(a_std)
            new_state_tensor = self.forward(observation) # observation as a tensor

            curr_val = self.critic(state_tensor)
            next_val = self.critic(new_state_tensor)
            pred_vals.append(curr_val)

            log_pi = lambda x, m, s: tf.math.log(1/(tf.abs(s)*np.sqrt(2*np.pi))) - tf.square(tf.cast(x, tf.float32) - m)/(2*tf.math.square(s)) 
            pi = lambda x, m, s: 1/(tf.sqrt(2*np.pi*tf.square(s)))*tf.exp(-tf.square(x - m)/(2*tf.square(s)))
            log_prob = log_pi(pre_norm_action, a_mean, a_std)
            log_probs.append(log_prob)


                # print("T,V:", target,curr_val.numpy())
                #print("A,R,L:", advantage.numpy(), reward, critic_loss.numpy())

            # prev_state_action_prob = (state_tensor, pre_norm_action, log_prob, advantage)
            score = info['score']

            # state_tensor = new_state_tensor
            state = observation # update state
            tot_reward += reward

        if ppo:
            if prev_state_action_prob:
                old_state, old_action, old_log_prob, old_adv = prev_state_action_prob
                # Mean and std based on previous state with updated actor
                ps_mean, ps_std = self.actor(old_state)
                ps_log_prob = log_pi(old_action, ps_mean, ps_std)
                r = tf.math.exp(ps_log_prob - old_log_prob)
                actor_loss = -tf.math.minimum(r*old_adv, tf.clip_by_value(r, 1-self.epsilon, 1+self.epsilon)*old_adv)
                critic_loss = tf.math.square(advantage)
                joint_loss = tf.add(actor_loss, critic_loss)
                with self.logger.as_default():
                    tf.summary.scalar("critic loss", critic_loss, step=self.update_step)
                    tf.summary.scalar("value", tf.reduce_mean(curr_val), step=self.update_step)
                    tf.summary.scalar("reward", tf.reduce_mean(reward), step=self.update_step)
                    tf.summary.scalar("actor loss", actor_loss, step=self.update_step)
                    
        else:
            returns = self.compute_returns()
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape, tf.GradientTape(persistent=True) as cnn_tape:
                states_batch = {
                    'blocks': tf.stack([s['blocks'][0] for s in states]),
                    'nballs': tf.stack([s['nballs'][0] for s in states]),
                    'position': tf.stack([s['position'][0] for s in states])
                }
                states_tensors = self.forward(states_batch)
                means, stds = tf.transpose(self.actor(states_tensors))
                log_pi = lambda x, m, s: tf.math.log(1/(tf.abs(s)*np.sqrt(2*np.pi))) - tf.square(tf.cast(x, tf.float32) - m)/(2*tf.math.square(s)) 
                actions_tensor = tf.concat(actions, 0)
                stds = tf.math.exp(stds)
                log_pis = log_pi(actions_tensor, means, stds)
                pred_vals = self.critic(states_tensors)
                advantages = returns - pred_vals
                actor_loss = -tf.reduce_mean(log_pis * tf.stop_gradient(advantages))
                #actor_loss = -log_pi(action) * target
                critic_loss = tf.reduce_mean(tf.math.square(advantages))
                joint_loss = tf.add(actor_loss, critic_loss)
                with self.logger.as_default():
                    tf.summary.scalar("critic loss", critic_loss, step=self.update_step)
                    tf.summary.scalar("value", tf.reduce_mean(curr_val), step=self.update_step)
                    tf.summary.scalar("reward", tf.reduce_mean(reward), step=self.update_step)
                    tf.summary.scalar("actor loss", actor_loss, step=self.update_step)
                    
            # update critic and actor
            c_gradient = critic_tape.gradient(critic_loss, self.critic.trainable_weights)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_weights))
            a_gradient = actor_tape.gradient(actor_loss, self.actor.trainable_weights)
            self.a_optimizer.apply_gradients(zip(a_gradient, self.actor.trainable_weights))
            # use joint loss to update cnn featureinator
            cnn_gradient = cnn_tape.gradient(joint_loss, self.cnn.trainable_weights)
            self.cnn_optimizer.apply_gradients(zip(cnn_gradient, self.cnn.trainable_weights))
            discount *= self.gamma
            state = observation
            
            self.a_losses.append(actor_loss)
            self.c_losses.append(critic_loss)

        states, actions, log_probs, pred_vals = [], [], [], []
        return tot_reward, state_tensor, score 
