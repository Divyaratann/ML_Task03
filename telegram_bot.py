"""
Telegram Bot Integration for Customer Support Chatbot
This creates a Telegram bot that uses the core chatbot functionality
"""

import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from chatbot_core import CustomerSupportChatbot
from config import TELEGRAM_BOT_TOKEN, BOT_NAME, BOT_DESCRIPTION

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramCustomerSupportBot:
    def __init__(self, token: str):
        self.token = token
        self.chatbot = CustomerSupportChatbot()
        self.application = Application.builder().token(token).build()
        self.setup_handlers()
    
    def setup_handlers(self):
        """Set up command and message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("export", self.export_command))
        
        # Message handler for all text messages
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command"""
        welcome_message = f"""
🤖 Welcome to {BOT_NAME}!

{BOT_DESCRIPTION}

I can help you with:
• Order status and tracking
• Shipping and delivery information
• Returns and refunds
• Payment methods and billing
• Product information
• Account issues
• Contact information

Type /help to see all available commands or just start chatting with me!

How can I assist you today? 😊
        """
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command"""
        help_message = """
📋 Available Commands:

/start - Start the bot and see welcome message
/help - Show this help message
/stats - View bot analytics and statistics
/clear - Clear conversation history
/export - Export conversation data

💬 You can also just type your questions directly!

Examples:
• "Where is my order?"
• "What's your return policy?"
• "How long does shipping take?"
• "I need help with my account"
• "What payment methods do you accept?"
        """
        await update.message.reply_text(help_message)
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /stats command"""
        analytics = self.chatbot.get_analytics()
        
        stats_message = f"""
📊 Bot Statistics:

📈 Performance Metrics:
• Total Requests: {analytics['total_requests']}
• Success Rate: {analytics['success_rate']}
• Average Response Time: {analytics['average_response_time']}

🎯 Intent Distribution:
"""
        
        for intent, count in analytics['intent_distribution'].items():
            stats_message += f"• {intent.title()}: {count}\n"
        
        await update.message.reply_text(stats_message)
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /clear command"""
        self.chatbot.clear_history()
        await update.message.reply_text("🗑️ Conversation history cleared! Start fresh with your questions.")
    
    async def export_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /export command"""
        filename = self.chatbot.export_conversations()
        await update.message.reply_text(f"📥 Conversation data exported to: {filename}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages"""
        user_message = update.message.text
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        logger.info(f"Message from {username} ({user_id}): {user_message}")
        
        # Process the message using our chatbot
        response = self.chatbot.process_message(user_message, str(user_id))
        
        # Format response with intent information
        response_text = f"{response['text']}\n\n"
        response_text += f"🎯 Intent: {response['intent'].title()}\n"
        response_text += f"📊 Confidence: {response['confidence']:.2f}\n"
        response_text += f"⏱️ Response Time: {response['response_time']:.2f}s"
        
        # Send the response back to the user
        await update.message.reply_text(response_text)
        
        # Log the interaction
        logger.info(f"Bot response to {username}: {response['text']}")
    
    def run(self):
        """Start the bot"""
        logger.info(f"Starting {BOT_NAME}...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """Main function to run the Telegram bot"""
    # Get bot token from environment variable
    bot_token = TELEGRAM_BOT_TOKEN
    
    if not bot_token:
        print("❌ Error: TELEGRAM_BOT_TOKEN not found in environment variables!")
        print("Please set your Telegram bot token in the config.py file or environment variables.")
        print("Get your token from @BotFather on Telegram.")
        return
    
    # Create and run the bot
    bot = TelegramCustomerSupportBot(bot_token)
    bot.run()

if __name__ == '__main__':
    main()
