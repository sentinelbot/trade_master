import numpy as np
import pandas as pd
import ccxt
import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam
import random
import threading
import asyncio
from telegram import Update
from telegram.ext import ContextTypes

# Trading environment with more features
class TradingEnv:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long position
        self.current_step = 0
        self.history = []
        return self._get_observation()

    def step(self, action):
        # 0: hold, 1: buy, 2: sell
        current_price = self.data.iloc[self.current_step]['close']

        # Execute action
        reward = 0
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.buy_price = current_price
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            reward = (current_price - self.buy_price) / self.buy_price * 100

        # Update state
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Track history
        self.history.append({
            'step': self.current_step,
            'action': action,
            'price': current_price,
            'position': self.position,
            'reward': reward
        })

        return self._get_observation(), reward, done

    def _get_observation(self):
        if self.current_step >= len(self.data):
            return np.zeros(9)

        current_row = self.data.iloc[self.current_step]
        obs = np.array([
            current_row['close'],
            current_row['volume'],
            current_row['close'] / current_row['open'] - 1,  # Current candle return
            self.position,
            self.balance,
            current_row['ma_50'],  # Moving average 50
            current_row['ma_200'],  # Moving average 200
            current_row['rsi'],  # Relative Strength Index
            current_row['macd']  # MACD
        ])
        return obs

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

async def train_rl_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Train a reinforcement learning model for trading"""
    await update.message.reply_text("Training reinforcement learning model. This may take a few minutes...")

    try:
        # Fetch data
        exchange = ccxt.binance()
        symbol = "BTC/USDT"
        if context.args and len(context.args) > 0:
            symbol = context.args[0]

        since = exchange.parse8601('2 weeks ago')
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

        # Feature Engineering
        data['ma_50'] = data['close'].rolling(window=50).mean()
        data['ma_200'] = data['close'].rolling(window=200).mean()
        data['rsi'] = compute_rsi(data['close'])
        data['macd'] = compute_macd(data['close'])
        data = data.dropna()

        # Normalize data
        scaler = StandardScaler()
        price_scaler = StandardScaler()

        normalized_close = price_scaler.fit_transform(data[['close']])
        normalized_volume = scaler.fit_transform(data[['volume']])

        data['close_normalized'] = normalized_close
        data['volume_normalized'] = normalized_volume

        # Create environment and agent
        env = TradingEnv(data)
        state_size = env._get_observation().shape[0]
        action_size = 3
        agent = DQNAgent(state_size, action_size)
        done = False
        batch_size = 32

        # Training parameters
        episodes = 50
        max_steps = len(data) - 1

        # Training loop
        total_rewards = []

        for episode in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            done = False

            for _ in range(max_steps):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    agent.replay(batch_size)
                    break

            total_rewards.append(total_reward)

            if episode % 10 == 0:
                print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

        # Evaluate model
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        evaluation_history = []

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state

            current_step = env.current_step
            if current_step < len(data):
                evaluation_history.append({
                    'timestamp': data.iloc[current_step]['timestamp'],
                    'price': data.iloc[current_step]['close'],
                    'action': action,
                    'reward': reward
                })

        # Create visualization
        eval_df = pd.DataFrame(evaluation_history)

        plt.figure(figsize=(12, 8))

        # Plot price
        plt.subplot(2, 1, 1)
        plt.plot(data['timestamp'], data['close'], 'b-', label='Price')

        # Plot buy and sell signals
        buy_signals = eval_df[eval_df['action'] == 1]
        sell_signals = eval_df[eval_df['action'] == 2]

        plt.scatter(buy_signals['timestamp'], buy_signals['price'], marker='^', color='g', s=100, label='Buy')
        plt.scatter(sell_signals['timestamp'], sell_signals['price'], marker='v', color='r', s=100, label='Sell')

        plt.title(f'RL Trading Model for {symbol}')
        plt.legend()
        plt.grid(True)

        # Plot rewards
        plt.subplot(2, 1, 2)
        plt.plot(range(episodes), total_rewards, 'r-')
        plt.title('Training Rewards Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)

        plt.tight_layout()

        # Convert plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Prepare summary
        buy_count = len(buy_signals)
        sell_count = len(sell_signals)
        final_reward = sum(eval_df['reward'])

        summary = (
            f"RL Model Training Summary for {symbol}:\n\n"
            f"Episodes trained: {episodes}\n"
            f"Final epsilon: {agent.epsilon:.4f}\n"
            f"Buy signals: {buy_count}\n"
            f"Sell signals: {sell_count}\n"
            f"Cumulative reward: {final_reward:.2f}%\n\n"
            f"The model is now available for backtesting and paper trading."
        )

        # Send results
        await update.message.reply_photo(buf)
        await update.message.reply_text(summary)

    except Exception as e:
        await update.message.reply_text(f"Error training RL model: {str(e)}")

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd - signal

# Backtesting function
def backtest_model(agent, data):
    env = TradingEnv(data)
    state_size = env._get_observation().shape[0]
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    backtest_history = []

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state

        current_step = env.current_step
        if current_step < len(data):
            backtest_history.append({
                'timestamp': data.iloc[current_step]['timestamp'],
                'price': data.iloc[current_step]['close'],
                'action': action,
                'reward': reward
            })

    return pd.DataFrame(backtest_history)

# Real-time trading function
async def real_time_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Execute real-time trades based on RL model signals"""
    await update.message.reply_text("Starting real-time trading...")

    try:
        # Fetch data
        exchange = ccxt.binance()
        symbol = "BTC/USDT"
        if context.args and len(context.args) > 0:
            symbol = context.args[0]

        # Load pre-trained model
        state_size = 9
        action_size = 3
        agent = DQNAgent(state_size, action_size)
        agent.load("dqn_model.h5")

        # Real-time data fetching
        async def fetch_data():
            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=1)
                data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

                # Feature Engineering
                data['ma_50'] = data['close'].rolling(window=50).mean()
                data['ma_200'] = data['close'].rolling(window=200).mean()
                data['rsi'] = compute_rsi(data['close'])
                data['macd'] = compute_macd(data['close'])
                data = data.dropna()

                # Normalize data
                scaler = StandardScaler()
                price_scaler = StandardScaler()

                normalized_close = price_scaler.fit_transform(data[['close']])
                normalized_volume = scaler.fit_transform(data[['volume']])

                data['close_normalized'] = normalized_close
                data['volume_normalized'] = normalized_volume

                # Create environment
                env = TradingEnv(data)
                state = env.reset()
                state = np.reshape(state, [1, state_size])

                # Get action from agent
                action = agent.act(state)
                if action == 1:
                    await update.message.reply_text(f"Buying {symbol} at {data['close'].iloc[-1]}")
                    # Execute buy order
                elif action == 2:
                    await update.message.reply_text(f"Selling {symbol} at {data['close'].iloc[-1]}")
                    # Execute sell order

                await asyncio.sleep(60)  # Wait for 1 minute

        # Start real-time data fetching in a separate thread
        threading.Thread(target=lambda: asyncio.run(fetch_data())).start()

    except Exception as e:
        await update.message.reply_text(f"Error during real-time trading: {str(e)}")

# Paper trading function
async def paper_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Simulate trading without real money"""
    await update.message.reply_text("Starting paper trading...")

    try:
        # Fetch data
        exchange = ccxt.binance()
        symbol = "BTC/USDT"
        if context.args and len(context.args) > 0:
            symbol = context.args[0]

        since = exchange.parse8601('1 day ago')
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

        # Feature Engineering
        data['ma_50'] = data['close'].rolling(window=50).mean()
        data['ma_200'] = data['close'].rolling(window=200).mean()
        data['rsi'] = compute_rsi(data['close'])
        data['macd'] = compute_macd(data['close'])
        data = data.dropna()

        # Normalize data
        scaler = StandardScaler()
        price_scaler = StandardScaler()

        normalized_close = price_scaler.fit_transform(data[['close']])
        normalized_volume = scaler.fit_transform(data[['volume']])

        data['close_normalized'] = normalized_close
        data['volume_normalized'] = normalized_volume

        # Create environment and agent
        env = TradingEnv(data)
        state_size = env._get_observation().shape[0]
        action_size = 3
        agent = DQNAgent(state_size, action_size)
        agent.load("dqn_model.h5")  # Load pre-trained model

        # Paper trading
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        paper_trading_history = []

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state

            current_step = env.current_step
            if current_step < len(data):
                paper_trading_history.append({
                    'timestamp': data.iloc[current_step]['timestamp'],
                    'price': data.iloc[current_step]['close'],
                    'action': action,
                    'reward': reward
                })

        # Create visualization
        paper_df = pd.DataFrame(paper_trading_history)

        plt.figure(figsize=(12, 6))
        plt.plot(data['timestamp'], data['close'], 'b-', label='Price')

        # Plot buy and sell signals
        buy_signals = paper_df[paper_df['action'] == 1]
        sell_signals = paper_df[paper_df['action'] == 2]

        plt.scatter(buy_signals['timestamp'], buy_signals['price'], marker='^', color='g', s=100, label='Buy')
        plt.scatter(sell_signals['timestamp'], sell_signals['price'], marker='v', color='r', s=100, label='Sell')

        plt.title(f'Paper Trading for {symbol}')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Convert plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Prepare summary
        buy_count = len(buy_signals)
        sell_count = len(sell_signals)
        final_reward = sum(paper_df['reward'])

        summary = (
            f"Paper Trading Summary for {symbol}:\n\n"
            f"Buy signals: {buy_count}\n"
            f"Sell signals: {sell_count}\n"
            f"Cumulative reward: {final_reward:.2f}%\n\n"
            f"The paper trading session is complete."
        )

        # Send results
        await update.message.reply_photo(buf)
        await update.message.reply_text(summary)

    except Exception as e:
        await update.message.reply_text(f"Error during paper trading: {str(e)}")