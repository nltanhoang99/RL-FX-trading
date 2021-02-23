import numpy as np
from tqdm import tqdm
import pandas as pd
import random
from A2C_agent import ActorCritic


def getState(data, start, window):
    result = data[start:start + window]
    return np.array(result)


def evalTrade(a, tr, next_step):
    if tr[0] == 1 or tr[0] == 2 or tr[0] == 3:  # LONG
        if next_step[-1][1] >= tr[3]:  # TP
            r = 3
            a.in_trade = False
            return [tr[0], r]

        elif next_step[-1][3] <= tr[2]:  # SL
            r = -1
            a.in_trade = False
            return [tr[0], r]

    elif tr[0] == 4 or tr[0] == 5 or tr[0] == 6:  # SHORT
        if next_step[-1][3] <= tr[3]:  # TP
            r = 3
            a.in_trade = False
            return [tr[0], r]

        elif next_step[-1][1] >= tr[2]:  # SL
            r = -1
            a.in_trade = False
            return [tr[0], r]
    return [0, 0]


def invScale(data, minimum, maximum):
    return data * (maximum - minimum) + minimum


def takeTrade(action, ohlc):
    unscaled_close = invScale(ohlc, data_min, data_max)[3]
    close = ohlc[3]

    # (long/short, entry, SL, TP)
    if action == 1:
        sl = (unscaled_close - 0.0002 - data_min) / (data_max - data_min)
        tp = (unscaled_close + 0.0006 - data_min) / (data_max - data_min)
        return action, close, sl, tp

    elif action == 2:
        sl = (unscaled_close - 0.0005 - data_min) / (data_max - data_min)
        tp = (unscaled_close + 0.0015 - data_min) / (data_max - data_min)
        return action, close, sl, tp

    elif action == 3:
        sl = (unscaled_close - 0.0010 - data_min) / (data_max - data_min)
        tp = (unscaled_close + 0.0030 - data_min) / (data_max - data_min)
        return action, close, sl, tp

    elif action == 4:
        sl = (unscaled_close + 0.0002 - data_min) / (data_max - data_min)
        tp = (unscaled_close - 0.0006 - data_min) / (data_max - data_min)
        return action, close, sl, tp

    elif action == 5:
        sl = (unscaled_close + 0.0005 - data_min) / (data_max - data_min)
        tp = (unscaled_close - 0.0015 - data_min) / (data_max - data_min)
        return action, close, sl, tp

    elif action == 6:
        sl = (unscaled_close + 0.0010 - data_min) / (data_max - data_min)
        tp = (unscaled_close - 0.0030 - data_min) / (data_max - data_min)
        return action, close, sl, tp


def evalPass(current, next):
    # Reward pass if action prevented SL
    long = takeTrade(1, current[-1])
    short = takeTrade(2, current[-1])
    if long[2] >= next[-1][3] or short[2] <= next[-1][1]:
        return 0.1
    return 0


WINDOW_SIZE = 90
EPISODE_LENGTH = 2500 - WINDOW_SIZE
EPISODE_NUM = 10000

agent = ActorCritic(WINDOW_SIZE)
agent = ActorCritic(WINDOW_SIZE, is_eval=True, actor_name="ep-2800-actor.h5", critic_name="ep-2800-critic.h5")
win, loss = 0, 0
ps, lg, st = 0, 0, 0


loop = tqdm(total=EPISODE_NUM, position=0, leave=False)
for episode in range(EPISODE_NUM):
    ACTIONS = [0, 0, 0, 0, 0, 0, 0]
    loop.set_description('Training ...'.format(episode))
    loop.update(1)

    # Randomize trading environment every episode
    chunk = random.randint(0, 79)
    print(f"Starting chunk {chunk}")
    srow = int(chunk * 2500 + 1)
    df = pd.read_csv('EU_M5_2017_2020.csv',
                     index_col='Local time',
                     parse_dates=True,
                     skiprows=srow,
                     nrows=2500,
                     names=['Local time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df = df.drop('Volume', 1)

    data_max = df['High'].max()
    data_min = df['Low'].min()
    df = np.array(df)
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

            if action != 0:
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
                ACTIONS[result[0]] += 1
                reward = result[1]
                agent.remember(entry_state, action, reward)

        # Passing
        else:
            reward = evalPass(state, next_state)
            ACTIONS[0] += 1
            agent.remember(state, action, reward)

        if t == EPISODE_LENGTH - 1 or ret <= -100:

            print("--------------------------------------------------------------------")
            print(f"Cumulative return Episode {episode} (chunk {chunk}): {ret}")
            print(f"Wins: {win}     Losses: {loss}")
            if win + loss > 0:
                print(f"Winrate: {win / (win + loss)}")
            print(f"Passes: {ACTIONS[0]}")
            print(f"L2: {ACTIONS[1]}     L5: {ACTIONS[2]}      L10:  {ACTIONS[3]}")
            print(f"S2: {ACTIONS[4]}     S5: {ACTIONS[5]}      S10:  {ACTIONS[6]}")
            print("--------------------------------------------------------------------")
            agent.train()

            break

        state = next_state

    if episode % 25 == 0:
        print("SAVING MODELS")
        agent.actor.save(f"ep-{episode + 2800}-actor.h5")
        agent.critic.save(f"ep-{episode + 2800}-critic.h5")
