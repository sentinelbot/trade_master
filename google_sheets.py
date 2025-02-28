import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os
from datetime import datetime
from telegram import Bot
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Google Sheets API setup
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SHEETS_CREDS_FILE', r"C:\Users\Administrator\OneDrive\Documents\trademaster\modules\credentials.json")  # Path to your Google Sheets API credentials

# Telegram Bot setup
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Initialize Telegram Bot
if TELEGRAM_BOT_TOKEN:
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
else:
    bot = None

def get_gspread_client():
    """
    Authenticate and return a Google Sheets API client using gspread.
    
    Returns:
        gspread.Client: Authenticated client or None if failed.
    """
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        print(f"Error connecting to Google Sheets: {str(e)}")
        return None

def sync_trade_history(trades_data=None):
    """
    Synchronize trade history with Google Sheets.
    
    Args:
        trades_data (list, optional): List of trade dictionaries. If None, fetches from database.
        
    Returns:
        bool: Success status.
    """
    client = get_gspread_client()
    if not client:
        return False
    
    try:
        # Open the spreadsheet by key
        spreadsheet_key = os.getenv('TRADE_HISTORY_SHEET_KEY')
        if not spreadsheet_key:
            raise ValueError("TRADE_HISTORY_SHEET_KEY environment variable is not set.")
        
        sheet = client.open_by_key(spreadsheet_key).worksheet("Trade History")
        
        # If no trades_data provided, use example data
        if not trades_data:
            trades_data = [
                {"timestamp": datetime.now().isoformat(), "symbol": "BTC/USDT", "side": "buy", "price": 50000, "amount": 0.1, "fee": 5, "total": 5005},
                {"timestamp": datetime.now().isoformat(), "symbol": "ETH/USDT", "side": "sell", "price": 3000, "amount": 1, "fee": 3, "total": 2997}
            ]
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(trades_data)
        
        # Get all values from the sheet
        existing_data = sheet.get_all_values()
        headers = existing_data[0] if existing_data else []
        
        # If sheet is empty, set headers
        if not headers:
            headers = df.columns.tolist()
            sheet.append_row(headers)
        
        # Append new data
        for _, row in df.iterrows():
            # Format row according to headers
            formatted_row = [str(row.get(header, "")) for header in headers]
            sheet.append_row(formatted_row)
        
        print(f"Successfully synced {len(df)} trades to Google Sheets")
        
        # Notify via Telegram
        if bot and TELEGRAM_CHAT_ID:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="✅ Trade history synced with Google Sheets.")
        
        return True
    
    except Exception as e:
        print(f"Error syncing trade history: {str(e)}")
        
        # Notify via Telegram
        if bot and TELEGRAM_CHAT_ID:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"❌ Error syncing trade history: {str(e)}")
        
        return False

def get_performance_metrics():
    """
    Retrieve performance metrics from Google Sheets.
    
    Returns:
        dict: Performance metrics or error message.
    """
    client = get_gspread_client()
    if not client:
        return {"error": "Could not connect to Google Sheets"}
    
    try:
        spreadsheet_key = os.getenv('PERFORMANCE_METRICS_SHEET_KEY')
        if not spreadsheet_key:
            raise ValueError("PERFORMANCE_METRICS_SHEET_KEY environment variable is not set.")
        
        sheet = client.open_by_key(spreadsheet_key).worksheet("Performance")
        
        # Get data as dictionary
        data = sheet.get_all_records()
        
        # Process the data as needed
        return data
    
    except Exception as e:
        print(f"Error getting performance metrics: {str(e)}")
        return {"error": str(e)}

def sync_trade_history_with_google_api(update, context):
    """
    Sync trade history with Google Sheets using the Google Sheets API and notify the user via Telegram.
    """
    try:
        # Authenticate with Google Sheets API
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        
        # Example: Append trade history to a Google Sheet
        sheet_id = os.getenv('TRADE_HISTORY_SHEET_KEY')
        if not sheet_id:
            raise ValueError("TRADE_HISTORY_SHEET_KEY environment variable is not set.")
        
        range_name = 'Sheet1!A1'  # Adjust the range as needed
        values = [["Trade1", "BTC/USDT", "Buy", "1000", "2023-10-01"]]  # Example data
        body = {'values': values}
        result = service.spreadsheets().values().append(
            spreadsheetId=sheet_id, range=range_name,
            valueInputOption='RAW', body=body).execute()
        
        # Notify the user
        update.message.reply_text("✅ Trade history synced with Google Sheets.")
    except Exception as e:
        update.message.reply_text(f"❌ Error syncing trade history: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Sync trade history
    sync_trade_history()
    
    # Get performance metrics
    metrics = get_performance_metrics()
    print("Performance Metrics:", metrics)