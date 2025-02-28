import os
import json
import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from telegram import Bot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get telegram token with fallback for development
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
WEBHOOK_PORT = int(os.getenv('TRADINGVIEW_WEBHOOK_PORT', '8080'))
WEBHOOK_SECRET = os.getenv('TRADINGVIEW_WEBHOOK_SECRET', 'your_webhook_secret')

# Get chat IDs from environment
TELEGRAM_CHAT_IDS = os.getenv('TELEGRAM_CHAT_IDS', '')

# Check if we're in development mode
DEVELOPMENT_MODE = os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'

# Global variables
webhook_running = False
httpd = None

# Modified initialization with better error handling and development mode
if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN.strip() == "":
    if DEVELOPMENT_MODE:
        print("WARNING: Running in development mode with placeholder bot token")
        TELEGRAM_BOT_TOKEN = "placeholder_token_for_development"
        # Create a mock bot that logs instead of sending real messages
        class MockBot:
            async def send_message(self, chat_id, text, parse_mode=None):
                print(f"[MOCK] Would send to {chat_id}:\n{text}")
        bot = MockBot()
    else:
        print("ERROR: TELEGRAM_BOT_TOKEN is not set")
        print("Please set TELEGRAM_BOT_TOKEN in your .env file or enable DEVELOPMENT_MODE=true for testing")
        raise SystemExit(1)
else:
    # Initialize real bot if token exists
    bot = Bot(token=TELEGRAM_BOT_TOKEN)

class TradingViewWebhookHandler(BaseHTTPRequestHandler):
    def _set_response(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            # Parse JSON data
            payload = json.loads(post_data.decode('utf-8'))

            # Verify webhook secret if provided
            if 'secret' not in payload or payload['secret'] != WEBHOOK_SECRET:
                print("Invalid webhook secret")
                self._set_response(403)
                self.wfile.write(json.dumps({"status": "error", "message": "Invalid secret"}).encode('utf-8'))
                return

            # Process the alert
            self._process_tradingview_alert(payload)

            # Send successful response
            self._set_response()
            self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))

        except json.JSONDecodeError:
            print("Error decoding JSON")
            self._set_response(400)
            self.wfile.write(json.dumps({"status": "error", "message": "Invalid JSON"}).encode('utf-8'))

        except Exception as e:
            print(f"Error processing webhook: {str(e)}")
            self._set_response(500)
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))

    def _process_tradingview_alert(self, payload):
        """Process the TradingView alert data and take appropriate action"""
        print(f"Received TradingView alert: {payload}")

        # Extract important information
        strategy = payload.get('strategy', 'Unknown')
        symbol = payload.get('ticker', 'Unknown')
        action = payload.get('action', 'Unknown')
        price = payload.get('price', 'Unknown')
        message = payload.get('message', '')

        # Determine which users should receive this alert
        target_chat_ids = self._get_subscribed_users(symbol, strategy)

        # Format the message
        formatted_message = (
            f"🔔 *TradingView Alert*\n\n"
            f"*Strategy:* {strategy}\n"
            f"*Symbol:* {symbol}\n"
            f"*Action:* {action}\n"
            f"*Price:* {price}\n\n"
            f"*Message:* {message}"
        )

        # Send messages asynchronously
        for chat_id in target_chat_ids:
            asyncio.run(self._send_telegram_message(chat_id, formatted_message))

    def _get_subscribed_users(self, symbol, strategy):
        """
        Get list of chat IDs for users subscribed to this symbol/strategy
        Read from environment variable or database
        """
        # Get chat IDs from environment variable
        if not TELEGRAM_CHAT_IDS:
            # Development fallback
            if DEVELOPMENT_MODE:
                print("[MOCK] No chat IDs configured, using development fallback")
                return [123456789]  # Development fallback ID
            return []  # No chat IDs configured

        # Parse comma-separated list of chat IDs
        try:
            chat_ids = [int(cid.strip()) for cid in TELEGRAM_CHAT_IDS.split(',') if cid.strip()]
            # Filter based on symbol/strategy if needed (future enhancement)
            return chat_ids
        except ValueError:
            print("Error parsing TELEGRAM_CHAT_IDS. Make sure it's a comma-separated list of integers.")
            return []

    async def _send_telegram_message(self, chat_id, message):
        """Send a Telegram message asynchronously"""
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='Markdown'
            )
            print(f"Alert sent to {chat_id}")
        except Exception as e:
            print(f"Error sending message to {chat_id}: {str(e)}")

async def start_tradingview_webhook():
    """Start the webhook server in a separate thread"""
    global webhook_running, httpd
    if not webhook_running:
        server_address = ('', WEBHOOK_PORT)
        httpd = HTTPServer(server_address, TradingViewWebhookHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        webhook_running = True
        print(f"✅ TradingView webhook server started on port {WEBHOOK_PORT}")
        print(f"Webhook URL: http://your-server-ip:{WEBHOOK_PORT}/")
        print(f"Development mode: {DEVELOPMENT_MODE}")
        print(f"Chat IDs configured: {TELEGRAM_CHAT_IDS or 'None'}")
        if TELEGRAM_CHAT_IDS:
            await bot.send_message(chat_id=TELEGRAM_CHAT_IDS.split(',')[0], text="✅ TradingView webhook server started.")

def stop_tradingview_webhook():
    """Stop the webhook server"""
    global webhook_running, httpd
    if webhook_running and httpd:
        httpd.shutdown()
        webhook_running = False
        print("❌ TradingView webhook server stopped")
        if TELEGRAM_CHAT_IDS:
            asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_IDS.split(',')[0], text="❌ TradingView webhook server stopped."))

# For testing the webhook directly
if __name__ == "__main__":
    try:
        # Test if you can receive webhooks
        print("TradingView webhook server is starting...")
        print("Make sure you've set the following environment variables:")
        print(f"- TELEGRAM_BOT_TOKEN: {'✓ Set' if TELEGRAM_BOT_TOKEN and not DEVELOPMENT_MODE else '✗ Not set'}")
        print(f"- TRADINGVIEW_WEBHOOK_PORT: {WEBHOOK_PORT}")
        print(f"- TRADINGVIEW_WEBHOOK_SECRET: {'✓ Set (custom)' if WEBHOOK_SECRET != 'your_webhook_secret' else '✗ Using default'}")
        print(f"- TELEGRAM_CHAT_IDS: {'✓ Set' if TELEGRAM_CHAT_IDS else '✗ Not set'}")

        print("\nExample TradingView alert message format:")
        print("""{
  "secret": "your_webhook_secret",
  "strategy": "RSI Strategy",
  "ticker": "BTCUSDT",
  "action": "BUY",
  "price": 45000,
  "message": "RSI crossed below 30, potential buy signal"
}""")

        # Start the webhook server
        asyncio.run(start_tradingview_webhook())

        # Keep the main thread alive to keep the server running
        while webhook_running:
            asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Server shutdown requested...")
        stop_tradingview_webhook()
    except Exception as e:
        print(f"Error starting server: {e}")
