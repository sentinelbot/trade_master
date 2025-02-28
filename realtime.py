import json
import asyncio
import websockets
import aiohttp
import pandas as pd
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
# ParseMode is now in telegram.constants in newer versions
from telegram.constants import ParseMode

# Global variable to store market data
market_data_cache = {}

# Start WebSocket connection for real-time updates
async def websocket_client():
    """
    Establishes a WebSocket connection to receive real-time market data.
    """
    uri = "wss://stream.binance.com:9443/ws/btcusdt@trade/ethusdt@trade"
    
    async with websockets.connect(uri) as websocket:
        print("WebSocket connection established")
        
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                # Update the cache with the latest data
                symbol = data.get('s', '').replace('USDT', '/USDT')
                if symbol:
                    price = float(data.get('p', 0))
                    market_data_cache[symbol] = {
                        'price': price,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'volume': float(data.get('q', 0)),
                        'change_24h': 0  # This would be calculated elsewhere
                    }
            except Exception as e:
                print(f"WebSocket error: {e}")
                await asyncio.sleep(5)
                break  # Break to reconnect

def start_websocket():
    """
    Starts the WebSocket client in an event loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while True:
        try:
            loop.run_until_complete(websocket_client())
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
        
        print("Reconnecting to WebSocket in 5 seconds...")
        loop.run_until_complete(asyncio.sleep(5))

async def get_market_data(update: Update, context):
    """
    Gets market data for a specific exchange and symbol.
    Can be called from either a direct command or a callback query.
    """
    # Determine if function was called from a command or callback query
    is_callback = hasattr(update, 'callback_query') and update.callback_query
    
    if is_callback:
        query = update.callback_query
        chat_id = query.message.chat_id
        message_id = query.message.message_id
    else:
        chat_id = update.message.chat_id
        message_id = None
    
    # Check if arguments are provided
    if not context.args or len(context.args) < 2:
        message = "Please provide an exchange and a symbol. Example: /market binance BTC/USDT"
        
        if is_callback:
            await query.edit_message_text(message)
        else:
            await context.bot.send_message(chat_id=chat_id, text=message)
        return
    
    exchange = context.args[0].lower()
    symbol = context.args[1].upper()
    
    # First check cache for real-time data
    if symbol in market_data_cache:
        data = market_data_cache[symbol]
        message = f"📊 *{symbol} Market Data (Real-time)*\n\n"
        message += f"🏦 Exchange: {exchange.capitalize()}\n"
        message += f"💰 Price: ${data['price']:,.2f}\n"
        message += f"📈 24h Change: {data.get('change_24h', 0):.2f}%\n"
        message += f"🔄 Volume: ${data.get('volume', 0):,.2f}\n"
        message += f"⏰ Last Update: {data['timestamp']}"
        
        # Add interactive buttons
        keyboard = [
            [
                InlineKeyboardButton("Refresh", callback_data=f"market_{exchange}_{symbol}"),
                InlineKeyboardButton("Chart", callback_data=f"chart_{symbol}_1h")
            ],
            [
                InlineKeyboardButton("Set Alert", callback_data=f"alert_setup_{symbol}"),
                InlineKeyboardButton("Add to Watchlist", callback_data=f"addwatch_{symbol}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if is_callback:
            await query.edit_message_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
        return
    
    # If not in cache, fetch from API
    await fetch_market_data(update, context, exchange, symbol, is_callback, chat_id, message_id)

async def fetch_market_data(update, context, exchange, symbol, is_callback, chat_id, message_id):
    """
    Fetches market data from API if not available in cache.
    """
    # Send a "loading" message
    if is_callback:
        query = update.callback_query
        await query.edit_message_text(f"Fetching {symbol} data from {exchange}...")
    else:
        loading_message = await context.bot.send_message(
            chat_id=chat_id,
            text=f"Fetching {symbol} data from {exchange}..."
        )
    
    try:
        # Simulate API call (replace with actual API call)
        await asyncio.sleep(1)  # Simulate network delay
        
        # Format symbol for API (different exchanges have different formats)
        api_symbol = symbol.replace("/", "")
        
        async with aiohttp.ClientSession() as session:
            url = f"https://api.{exchange}.com/api/v3/ticker/24hr?symbol={api_symbol}"
            
            # For demo, simulate response instead of actual API call
            # In production, use: async with session.get(url) as response:
            #                     data = await response.json()
            
            # Simulated data
            price = 51234.56 if "BTC" in symbol else 3456.78
            change = 2.45 if "BTC" in symbol else -1.23
            volume = 1230456.78
            
            # Update cache
            market_data_cache[symbol] = {
                'price': price,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'volume': volume,
                'change_24h': change
            }
            
            # Prepare response message
            message = f"📊 *{symbol} Market Data*\n\n"
            message += f"🏦 Exchange: {exchange.capitalize()}\n"
            message += f"💰 Price: ${price:,.2f}\n"
            message += f"📈 24h Change: {change:.2f}%\n"
            message += f"🔄 Volume: ${volume:,.2f}\n"
            message += f"⏰ Last Update: {market_data_cache[symbol]['timestamp']}"
            
            # Add interactive buttons
            keyboard = [
                [
                    InlineKeyboardButton("Refresh", callback_data=f"market_{exchange}_{symbol}"),
                    InlineKeyboardButton("Chart", callback_data=f"chart_{symbol}_1h")
                ],
                [
                    InlineKeyboardButton("Set Alert", callback_data=f"alert_setup_{symbol}"),
                    InlineKeyboardButton("Add to Watchlist", callback_data=f"addwatch_{symbol}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if is_callback:
                await query.edit_message_text(
                    message,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=reply_markup
                )
            else:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading_message.message_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=reply_markup
                )
    
    except Exception as e:
        error_message = f"Error fetching market data: {str(e)}"
        if is_callback:
            await query.edit_message_text(error_message)
        else:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_message.message_id,
                text=error_message
            )

async def get_ticker_data(exchange, symbol, timeframe='1h', limit=100):
    """
    Gets historical ticker data for a symbol.
    
    Args:
        exchange (str): Exchange name (e.g., 'binance')
        symbol (str): Trading pair (e.g., 'BTC/USDT')
        timeframe (str): Candle timeframe (e.g., '1h', '1d')
        limit (int): Number of candles to retrieve
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    try:
        # Format symbol for API
        api_symbol = symbol.replace("/", "")
        
        # In a real implementation, you would call the exchange API
        # For now, generate sample data
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
        
        # Generate sample data based on symbol
        base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        
        # Create sample OHLCV data
        import numpy as np
        np.random.seed(42)  # For reproducible results
        
        data = {
            'timestamp': dates,
            'open': [base_price + np.random.normal(0, base_price * 0.02) for _ in range(limit)],
            'high': [0] * limit,
            'low': [0] * limit,
            'close': [0] * limit,
            'volume': [np.random.randint(100, 1000) * base_price for _ in range(limit)]
        }
        
        # Adjust high, low, close based on open
        for i in range(limit):
            data['close'][i] = data['open'][i] * (1 + np.random.normal(0, 0.02))
            data['high'][i] = max(data['open'][i], data['close'][i]) * (1 + abs(np.random.normal(0, 0.01)))
            data['low'][i] = min(data['open'][i], data['close'][i]) * (1 - abs(np.random.normal(0, 0.01)))
        
        df = pd.DataFrame(data)
        return df
    
    except Exception as e:
        print(f"Error getting ticker data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error