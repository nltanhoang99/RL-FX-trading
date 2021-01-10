import tensorflow as tf
import numpy as np
import random
from collections import deque

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3  # sit, long, short
        self.memory = deque(maxlen=1000)
        self.in_trade = False
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model_name = model_name
        if is_eval:
            tf.keras.models.load_model(model_name)
        else:
            self.model = self.build_actor()

    def build_actor(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(512, input_shape=(self.state_size, 4)))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
        model.compile(loss='mse', optimizer=opt)

        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state = state.reshape(1, self.state_size, 4)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batchsize):
        mini_batch = random.sample(self.memory, batchsize)

        for state, action, reward, next_state, done in mini_batch:
            next_state = next_state.reshape(1, self.state_size, 4)
            state = state.reshape(1, self.state_size, 4)

            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                target = reward

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
