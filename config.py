"""
Configuration file for Customer Support Chatbot
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Dialogflow Configuration
DIALOGFLOW_PROJECT_ID = os.getenv('DIALOGFLOW_PROJECT_ID', 'customersupportbot-bsqb')
DIALOGFLOW_SESSION_ID = os.getenv('DIALOGFLOW_SESSION_ID', 'default-session')
DIALOGFLOW_LANGUAGE_CODE = os.getenv('DIALOGFLOW_LANGUAGE_CODE', 'en')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = '8385290070:AAEeDHqDYckjXk0K6BUpz3GBv62RHlSsFE8'

# OpenAI Configuration (Optional)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')

# Bot Configuration
BOT_NAME = "Customer Support Bot"
BOT_DESCRIPTION = "AI-powered customer support assistant"
BOT_VERSION = "1.0.0"

# Dialogflow Bot URL
DIALOGFLOW_BOT_URL = "https://bot.dialogflow.com/826bc631-37f7-4d24-ba69-6f73fae1fe1c"
