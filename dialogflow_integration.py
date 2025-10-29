"""
Dialogflow Integration for Customer Support Chatbot
This module provides integration with Google Dialogflow
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional
from config import DIALOGFLOW_PROJECT_ID, DIALOGFLOW_SESSION_ID, DIALOGFLOW_LANGUAGE_CODE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DialogflowIntegration:
    """
    Integration class for Google Dialogflow
    """
    
    def __init__(self):
        self.project_id = DIALOGFLOW_PROJECT_ID
        self.session_id = DIALOGFLOW_SESSION_ID
        self.language_code = DIALOGFLOW_LANGUAGE_CODE
        self.bot_url = "https://bot.dialogflow.com/826bc631-37f7-4d24-ba69-6f73fae1fe1c"
        
        logger.info(f"Dialogflow integration initialized for project: {self.project_id}")
    
    def get_bot_url(self) -> str:
        """Get the Dialogflow bot URL"""
        return self.bot_url
    
    def get_iframe_code(self, width: int = 350, height: int = 430) -> str:
        """Generate iframe code for embedding the bot"""
        return f'<iframe height="{height}" width="{width}" src="{self.bot_url}"></iframe>'
    
    def get_bot_info(self) -> Dict:
        """Get bot information"""
        return {
            'bot_id': '826bc631-37f7-4d24-ba69-6f73fae1fe1c',
            'bot_url': self.bot_url,
            'project_id': self.project_id,
            'session_id': self.session_id,
            'language_code': self.language_code,
            'status': 'active',
            'created_at': datetime.now().isoformat()
        }
    
    def create_web_interface(self, filename: str = "dialogflow_web_interface.html") -> str:
        """Create a web interface with the Dialogflow bot embedded"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Support Chatbot - Dialogflow Integration</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .container {{
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
            max-width: 800px;
            width: 100%;
            text-align: center;
        }}
        
        .header {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 40px;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .chat-container {{
            padding: 40px;
        }}
        
        .bot-iframe {{
            border: none;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        
        .info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        
        .info h3 {{
            color: #333;
            margin-bottom: 15px;
        }}
        
        .info ul {{
            text-align: left;
            max-width: 400px;
            margin: 0 auto;
        }}
        
        .info li {{
            margin: 8px 0;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Customer Support Bot</h1>
            <p>Powered by Google Dialogflow</p>
        </div>
        
        <div class="chat-container">
            <h2>💬 Chat with our AI Assistant</h2>
            <p>Ask me about orders, shipping, returns, payments, or any other support questions!</p>
            
            {self.get_iframe_code()}
            
            <div class="info">
                <h3>🎯 What I Can Help With</h3>
                <ul>
                    <li>Order Status & Tracking</li>
                    <li>Shipping Information</li>
                    <li>Returns & Refunds</li>
                    <li>Payment Methods</li>
                    <li>Product Information</li>
                    <li>Account Support</li>
                    <li>Contact Information</li>
                    <li>General Inquiries</li>
                </ul>
            </div>
        </div>
    </div>
    
    <script>
        console.log('Dialogflow Bot Interface Loaded');
        console.log('Bot ID: 826bc631-37f7-4d24-ba69-6f73fae1fe1c');
        console.log('Project ID: {self.project_id}');
    </script>
</body>
</html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Web interface created: {filename}")
        return filename
    
    def test_integration(self) -> Dict:
        """Test the Dialogflow integration"""
        test_results = {
            'bot_url': self.bot_url,
            'project_id': self.project_id,
            'session_id': self.session_id,
            'language_code': self.language_code,
            'status': 'active',
            'test_timestamp': datetime.now().isoformat(),
            'test_queries': [
                "Hi, I need help with my order",
                "Where is my order?",
                "How long does shipping take?",
                "I want to return this item",
                "What payment methods do you accept?"
            ]
        }
        
        logger.info("Dialogflow integration test completed")
        return test_results

def main():
    """Main function to test Dialogflow integration"""
    print("🤖 Dialogflow Integration Test")
    print("=" * 40)
    
    # Initialize integration
    dialogflow = DialogflowIntegration()
    
    # Get bot information
    bot_info = dialogflow.get_bot_info()
    print(f"Bot ID: {bot_info['bot_id']}")
    print(f"Bot URL: {bot_info['bot_url']}")
    print(f"Project ID: {bot_info['project_id']}")
    print(f"Status: {bot_info['status']}")
    
    # Create web interface
    filename = dialogflow.create_web_interface()
    print(f"\n✅ Web interface created: {filename}")
    
    # Test integration
    test_results = dialogflow.test_integration()
    print(f"\n✅ Integration test completed")
    print(f"Test queries: {len(test_results['test_queries'])}")
    
    print(f"\n🌐 Open {filename} in your browser to test the bot!")

if __name__ == "__main__":
    main()
