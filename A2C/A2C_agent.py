import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

class ActorCritic:
    def __init__(self, state_size, is_eval=False, actor_name="", critic_name=""):
        self.state_size = state_size
        self.states = []
        self.actions = []
        self.rewards = []

        self.action_size = 7  # sit, long, short
        self.in_trade = False

        if not is_eval:
            self.actor = self.build_actor()
            self.critic = self.build_critic()
        else:
            self.actor = tf.keras.models.load_model(actor_name)
            self.critic = tf.keras.models.load_model(critic_name)

    def build_actor(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(1024, input_shape=(self.state_size, 4)))
        model.add(tf.keras.layers.Dense(self.action_size, activation='softmax'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
        model.compile(loss='categorical_crossentropy', optimizer=opt)

        return model

    def build_critic(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(1024, input_shape=(self.state_size, 4)))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
        model.compile(loss='mse', optimizer=opt)

        return model

    def remember(self, state, action, reward):
        state = state.reshape(1, self.state_size, 4)
        self.states.append(state)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)

    def act(self, state):
        state = state.reshape(1, self.state_size, 4)
        dist = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=dist)
        return action

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99  # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add
            discounted_r = discounted_r.astype(np.float64)

        discounted_r -= np.mean(discounted_r)  # standardizing the result
        discounted_r /= np.std(discounted_r)  # divide by standard deviation
        return discounted_r.reshape(len(discounted_r),1)

    def train(self):
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)

        # Compute discounted rewards
        discounted_r = self.discount_rewards(self.rewards)

        # Get Critic network predictions
        values = self.critic.predict(states)[:, 0]
        values = values.reshape(len(values),1)
        # Compute advantages
        advantages = discounted_r - values
        # training Actor and Critic networks
        print("--- TRAINING MODELS ---")
        self.actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0, shuffle=True)
        self.critic.fit(states, discounted_r, epochs=1, verbose=0, shuffle=True)
        # reset training memory
        self.states, self.actions, self.rewards = [], [], []
