import os
import json
import aiohttp
import asyncio
from web3 import Web3
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler, ApplicationBuilder
from etherscan import Etherscan
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

INFURA_PROJECT_ID = os.getenv('INFURA_PROJECT_ID')
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')
DEFAULT_ADDRESS = os.getenv('DEFAULT_ETH_ADDRESS')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Initialize Web3 connection
def get_web3():
    """Initialize and return a Web3 instance connected to Infura."""
    if not INFURA_PROJECT_ID:
        raise ValueError("INFURA_PROJECT_ID environment variable is not set.")
    infura_url = f"https://mainnet.infura.io/v3/{INFURA_PROJECT_ID}"
    return Web3(Web3.HTTPProvider(infura_url))

# Initialize Etherscan client
def get_etherscan():
    """Initialize and return an Etherscan client."""
    if not ETHERSCAN_API_KEY:
        raise ValueError("ETHERSCAN_API_KEY environment variable is not set.")
    return Etherscan(ETHERSCAN_API_KEY)

# ERC20 ABI for token balance checking
ERC20_ABI = json.loads('''[
    {"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"},
    {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
    {"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"}
]''')

# Check ETH balance
async def check_eth_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Check ETH balance for a specific address.
    Usage: /ethbalance [address]
    """
    # Get address from args or use default
    address = DEFAULT_ADDRESS
    if context.args and Web3.isAddress(context.args[0]):
        address = context.args[0]
    
    if not Web3.is_address(address):
        await update.message.reply_text("Invalid Ethereum address. Please provide a valid address.")
        return
    
    await update.message.reply_text(f"Checking Ethereum balance for {address}...")
    
    try:
        # Get ETH balance
        web3 = get_web3()
        eth_balance = web3.eth.get_balance(address)
        eth_balance_eth = web3.from_wei(eth_balance, 'ether')
        
        # Get token balances using Etherscan
        token_balances = await get_token_balances(address)
        
        # Format the response
        response = f"*Ethereum Balance*\n\n"
        response += f"*Address:* `{address}`\n"
        response += f"*ETH Balance:* `{eth_balance_eth:.4f} ETH`\n\n"
        
        if token_balances:
            response += "*Token Balances:*\n"
            for token in token_balances[:5]:  # Show top 5 tokens by value
                response += f"• {token['name']} ({token['symbol']}): {token['balance']} (${token['value_usd']:.2f})\n"
            
            if len(token_balances) > 5:
                response += f"\n_...and {len(token_balances) - 5} more tokens_"
        
        # Add gas price information
        gas_price = web3.eth.gas_price
        gas_price_gwei = web3.from_wei(gas_price, 'gwei')
        response += f"\n\n*Current Gas Price:* `{gas_price_gwei:.2f} Gwei`"
        
        await update.message.reply_text(response, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"Error checking ETH balance: {str(e)}")

async def get_token_balances(address):
    """Get ERC20 token balances for an address using Etherscan."""
    etherscan = get_etherscan()
    
    try:
        # Get token balances
        tokens = etherscan.get_token_balance(address)
        
        # Get token prices
        token_prices = await get_token_prices([token['contractAddress'] for token in tokens])
        
        # Format token balances
        formatted_tokens = []
        for token in tokens:
            token_address = token['contractAddress']
            decimals = int(token['tokenDecimal'])
            raw_balance = int(token['value'])
            balance = raw_balance / (10 ** decimals)
            
            # Calculate USD value if price available
            price_usd = token_prices.get(token_address.lower(), 0)
            value_usd = balance * price_usd
            
            formatted_tokens.append({
                'name': token['name'],
                'symbol': token['symbol'],
                'balance': balance,
                'value_usd': value_usd
            })
        
        # Sort by USD value (highest first)
        return sorted(formatted_tokens, key=lambda x: x['value_usd'], reverse=True)
    
    except Exception as e:
        print(f"Error getting token balances: {str(e)}")
        return []

async def get_token_prices(token_addresses):
    """Get token prices from CoinGecko API."""
    if not token_addresses:
        return {}
    
    async with aiohttp.ClientSession() as session:
        try:
            # Format addresses for CoinGecko
            addresses_param = ','.join([addr.lower() for addr in token_addresses])
            url = f"https://api.coingecko.com/api/v3/simple/token_price/ethereum?contract_addresses={addresses_param}&vs_currencies=usd"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {addr: data[addr]['usd'] for addr in data if 'usd' in data[addr]}
                else:
                    print(f"Error fetching token prices: Status {response.status}")
                    return {}
        
        except Exception as e:
            print(f"Error getting token prices: {str(e)}")
            return {}

# Get DeFi protocol positions (Compound, Aave, etc.)
async def get_defi_positions(address):
    """Get DeFi positions for an address."""
    # This would integrate with DeFi protocols' APIs or use an aggregator like Zapper.fi
    # Example implementation would be added here
    return {
        "lending": [
            {"protocol": "Compound", "asset": "ETH", "supplied": 1.5, "value_usd": 3000},
            {"protocol": "Aave", "asset": "USDC", "supplied": 1000, "value_usd": 1000}
        ],
        "borrowing": [
            {"protocol": "Compound", "asset": "USDC", "borrowed": 500, "value_usd": 500}
        ]
    }

# Check gas prices
async def check_gas_prices(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get current Ethereum gas prices."""
    try:
        web3 = get_web3()
        gas_price = web3.eth.gas_price
        gas_price_gwei = web3.from_wei(gas_price, 'gwei')
        
        # For more detailed gas information, use a gas tracker API
        async with aiohttp.ClientSession() as session:
            async with session.get("https://ethgasstation.info/api/ethgasAPI.json") as response:
                if response.status == 200:
                    gas_data = await response.json()
                    response_text = (
                        f"*Current Gas Prices*\n\n"
                        f"🐢 Slow: {gas_data['safeLow'] / 10:.2f} Gwei (wait ~{gas_data['safeLowWait']} mins)\n"
                        f"🚶 Average: {gas_data['average'] / 10:.2f} Gwei (wait ~{gas_data['avgWait']} mins)\n"
                        f"🏎️ Fast: {gas_data['fast'] / 10:.2f} Gwei (wait ~{gas_data['fastWait']} mins)\n"
                    )
                    await update.message.reply_text(response_text, parse_mode='Markdown')
                    return
        
        # Fallback if API call fails
        response_text = f"*Current Gas Price:* `{gas_price_gwei:.2f} Gwei`"
        await update.message.reply_text(response_text, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"Error checking gas prices: {str(e)}")

# Get DeFi protocols information
async def get_defi_protocols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get information about DeFi protocols."""
    protocols = [
        {"name": "Aave", "tvl": "$5.2B", "description": "Lending and borrowing protocol"},
        {"name": "Uniswap", "tvl": "$3.8B", "description": "Decentralized exchange"},
        {"name": "Compound", "tvl": "$2.1B", "description": "Lending and borrowing protocol"}
    ]
    
    response = "*Top DeFi Protocols*\n\n"
    for protocol in protocols:
        response += f"*{protocol['name']}* - {protocol['tvl']}\n{protocol['description']}\n\n"
    
    await update.message.reply_text(response, parse_mode='Markdown')

# Main function to test the code
if __name__ == "__main__":
    # Example usage
    async def test():
        class MockUpdate:
            def __init__(self):
                self.message = MockMessage()
        
        class MockMessage:
            async def reply_text(self, text, parse_mode=None):
                print(text)
        
        mock_update = MockUpdate()
        class MockContext:
            def __init__(self):
                self.args = []

        mock_context = MockContext()
        
        await check_eth_balance(mock_update, mock_context)
        await check_gas_prices(mock_update, mock_context)
        await get_defi_protocols(mock_update, mock_context)
    
    asyncio.run(test())