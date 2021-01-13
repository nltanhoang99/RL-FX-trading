import numpy as np
from tqdm import tqdm
import pandas as pd
from DQN_agent import Agent


def getState(data, start, window):
    result = data[start:start + window]
    return np.array(result)


def evalTrade(a, tr, next_step):
    if tr[0] == 1:  # LONG
        if next_step[-1][1] >= tr[3]:  # TP
            r = 1
            a.in_trade = False
            return [1, r]

        elif next_step[-1][3] <= tr[2]:  # SL
            r = -1
            a.in_trade = False
            return [1, r]

    elif tr[0] == 2:  # SHORT
        if next_step[-1][3] <= tr[3]:  # TP
            r = 1
            a.in_trade = False
            return [2, r]

        elif next_step[-1][1] >= tr[2]:  # SL
            r = -1
            a.in_trade = False
            return [2, r]
    return [0, 0]

def evalPass(current, next):
    # Reward pass if action prevented SL
    long = takeTrade(1, current[-1])
    short = takeTrade(2, current[-1])
    if long[2] >= next[-1][3] or short[2] <= next[-1][1]:
        return 0.1
    return 0


def invScale(data, minimum, maximum):
    return data * (maximum - minimum) + minimum


def takeTrade(action, ohlc):
    unscaled_close = invScale(ohlc, data_min, data_max)[3]
    close = ohlc[3]

    if action == 1:  # (long/short, entry, SL, TP)
        sl = (unscaled_close - 0.0005 - data_min) / (data_max - data_min)
        tp = (unscaled_close + 0.0015 - data_min) / (data_max - data_min)
        return action, close, sl, tp

    elif action == 2:
        sl = (unscaled_close + 0.0005 - data_min) / (data_max - data_min)
        tp = (unscaled_close - 0.0015 - data_min) / (data_max - data_min)
        return action, close, sl, tp


df = pd.read_csv('EU_M5_2017_2020.csv',
                 index_col='Local time',
                 parse_dates=True,
                 skiprows=1,
                 nrows=5000,
                 names=['Local time', 'Open', 'High', 'Low', 'Close', 'Volume'])
df = df.drop('Volume', 1)

data_max = df['High'].max()
data_min = df['Low'].min()
df = np.array(df)
df = (df - data_min) / (data_max - data_min)

MIN_MEM_SIZE = 128
BATCH_SIZE = 32
WINDOW_SIZE = 90
EPISODE_LENGTH = len(df) - WINDOW_SIZE
ACTIONS = ['PASS', 'LONG', 'SHORT']

agent = Agent(WINDOW_SIZE)
win, loss = 0, 0

loop = tqdm(total=EPISODE_LENGTH, position=0, leave=False)
for ep in range(10):

    print('starting ep ', ep)
    ret = 0

    for t in range(EPISODE_LENGTH):
        loop.set_description('Training ...'.format(t))
        loop.update(1)

        done = False
        if t == 0:
            state = getState(df, t, WINDOW_SIZE)  # (90, 4) size

        close = state[-1][3]
        reward = 0
        if not agent.in_trade:
            action = agent.act(state)

            if action == 1:
                trade = takeTrade(action, state[-1])
                entry_state = state
                entry_t = t
                agent.in_trade = True
                print(f'--t:{t} OPEN {ACTIONS[action]} --')

            elif action == 2:
                trade = takeTrade(action, state[-1])
                entry_state = state
                entry_t = t
                agent.in_trade = True
                print(f'--t:{t} OPEN {ACTIONS[action]} --')

        next_state = getState(df, t + 1, WINDOW_SIZE)

        if agent.in_trade:
            result = evalTrade(agent, trade, next_state)
            if result[1] != 0:
                if result[1] >= 0:
                    win += 1
                    ret += 3
                else:
                    loss += 1
                    ret -= 1
                reward += result[1]
                agent.memory.append((entry_state, action, reward, next_state, done))

                print(f'--t:{t + 1} CLOSING {ACTIONS[result[0]]} ({reward})')

                if len(agent.memory) >= MIN_MEM_SIZE:
                    print('--- Fitting new Weights ---')
                    agent.expReplay(BATCH_SIZE)

        # Passing
        else:
            reward = evalPass(state, next_state)
            print(f'--t:{t} NO TRADE ({reward})')
            agent.memory.append((state, action, reward, next_state, done))
            if len(agent.memory) >= MIN_MEM_SIZE:
                print('--- Fitting new Weights ---')
                agent.expReplay(BATCH_SIZE)

        if t == EPISODE_LENGTH - 1 or ret <= -100:
            done = True
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            print(f"--- Cumulative return Episode {ep}: {ret} ---")
            if win + loss > 0:
                print(f"--- Winrate: {win / (win + loss)} ---")
            break

        state = next_state

    agent.model.save(f'episode-{ep}-trained-agent.h5')
