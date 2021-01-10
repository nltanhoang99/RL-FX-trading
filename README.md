# RL-FX-trading
Personal AI deep reinforcement learning project for FX trading using EURUSD data retrieved from dukascopy.com.
This project consists of two RL algorithms: DQN and A2C.
The implementation of these algorithms use my own custom environment (without using the Gym library).

---
# Custom environment & agent
The trading environment consists of a sliding 90 bar (5 min OHLC data per observation) window. 
The agent has three possible actions 0: pass, 1: long, 2: short.
Taking a trade has a fixed Stop-Loss (SL) and Take-profit (TP) and therefore a fixed Risk-to-reward. 
In addition, the agent can take only 1 trade at a time and the environment will notify the agent if the trade ended in a win or loss.
The game then continues from the state where the trade or pass ended.

---
## License & copyright
© Tan Hoang
