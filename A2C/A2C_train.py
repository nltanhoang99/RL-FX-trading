import numpy as np
from tqdm import tqdm
import pandas as pd
import random
from A2C_agent import ActorCritic


def getState(data, start, window):
    result = data[start:start + window]
    return np.array(result)


def evalTrade(a, tr, next_step):
    if tr[0] == 1:  # LONG
        if next_step[-1][1] >= tr[3]:  # TP
            r = 3
            a.in_trade = False
            return [1, r]

        elif next_step[-1][3] <= tr[2]:  # SL
            r = -1
            a.in_trade = False
            return [1, r]

    elif tr[0] == 2:  # SHORT
        if next_step[-1][3] <= tr[3]:  # TP
            r = 3
            a.in_trade = False
            return [2, r]

        elif next_step[-1][1] >= tr[2]:  # SL
            r = -1
            a.in_trade = False
            return [2, r]
    return [0, 0]


def invScale(data, minimum, maximum):
    return data * (maximum - minimum) + minimum


def takeTrade(action, ohlc):
    unscaled_close = invScale(ohlc, data_min, data_max)[3]
    close = ohlc[3]

    # (long/short, entry, SL, TP)
    if action == 1:
        sl = (unscaled_close - 0.0005 - data_min) / (data_max - data_min)
        tp = (unscaled_close + 0.0015 - data_min) / (data_max - data_min)
        return action, close, sl, tp

    elif action == 2:
        sl = (unscaled_close + 0.0005 - data_min) / (data_max - data_min)
        tp = (unscaled_close - 0.0015 - data_min) / (data_max - data_min)
        return action, close, sl, tp


def evalPass(current, next):
    # Reward for preventing SL and punish unnecessary pass
    long = takeTrade(1, current[-1])
    short = takeTrade(2, current[-1])
    if long[2] >= next[-1][3] and short[2] <= next[-1][1]:
        return 0.5
    elif long[2] >= next[-1][3] or short[2] <= next[-1][1]:
        return 0.25
    return -0.25


WINDOW_SIZE = 90
EPISODE_LENGTH = 2500 - WINDOW_SIZE
EPISODE_NUM = 6500

agent = ActorCritic(WINDOW_SIZE)
win, loss = 0, 0
ps, lg, st = 0, 0, 0
chkpnt = 0

loop = tqdm(total=EPISODE_NUM, position=0, leave=False)
for episode in range(EPISODE_NUM):
    loop.set_description('Training ...'.format(episode))
    loop.update(1)

    # Randomize trading environment every episode
    chunk = random.randint(0, 79)
    print(f"--- Starting chunk {chunk} ---")
    srow = int(chunk * 2500 + 1)
    df = pd.read_csv('EU_M5_2017_2020.csv',
                     index_col='Local time',
                     parse_dates=True,
                     skiprows=srow,
                     nrows=2500,
                     names=['Local time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df = df.drop('Volume', 1)

    df = np.array(df)
    data_max = max([max(l) for l in df])
    data_min = min([min(l) for l in df])

    df = (df - data_min) / (data_max - data_min)

    ret = 0
    win, loss = 0, 0
    ps, lg, st = 0, 0, 0

    for t in range(EPISODE_LENGTH):
        """Agent can only take 1 trade at a time and takes next action after trade/pass completes"""

        if t == 0:
            state = getState(df, t, WINDOW_SIZE)

        close = state[-1][3]
        reward = 0
        if not agent.in_trade:
            action = agent.act(state)

            if action == 1:
                trade = takeTrade(action, state[-1])
                entry_state = state
                entry_t = t
                agent.in_trade = True

            elif action == 2:
                trade = takeTrade(action, state[-1])
                entry_state = state
                entry_t = t
                agent.in_trade = True

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
                if result[0] == 1:
                    lg += 1
                else:
                    st += 1
                reward = result[1]
                agent.remember(entry_state, action, reward)

        # Passing
        else:
            reward = evalPass(state, next_state)
            ps += 1
            agent.remember(state, action, reward)

        if t == EPISODE_LENGTH - 1 or ret <= -100:

            print(f"--- Cumulative return Episode {episode} (chunk {chunk}): {ret} ---")
            print(f"--- Wins: {win}     Losses: {loss} ---")
            if win + loss > 0:
                print(f"--- Winrate: {win / (win + loss)} ---")
            print(f"--- Passes: {ps}    Longs: {lg}     Shorts: {st} ---")

            agent.train()
            break

        state = next_state

    chkpnt += 1
    if chkpnt == 20:
        print("--- Saving models ---")
        agent.actor.save(f"ep-{episode}-actor.h5")
        agent.critic.save(f"ep-{episode}-critic.h5")
        chkpnt = 0
