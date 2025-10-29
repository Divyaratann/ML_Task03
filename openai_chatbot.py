"""
Enhanced Chatbot with OpenAI Integration
This provides advanced AI capabilities using OpenAI's GPT models
"""

import os
import openai
from typing import Dict, List, Optional
from datetime import datetime
import logging
from config import OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIChatbot:
    """Enhanced chatbot with OpenAI integration"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.client = None
        self.conversation_history = []
        
        if self.api_key and self.api_key != 'your-openai-api-key-here':
            try:
                openai.api_key = self.api_key
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.warning("OpenAI API key not provided. Using fallback responses.")
    
    def get_openai_response(self, user_message: str, context: str = "") -> Dict:
        """Get response from OpenAI GPT model"""
        if not self.client:
            return self._get_fallback_response(user_message)
        
        try:
            # Prepare the conversation context
            system_prompt = self._get_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\nUser: {user_message}"}
            ]
            
            # Add conversation history
            for entry in self.conversation_history[-6:]:  # Keep last 6 exchanges
                messages.append({"role": "user", "content": entry["user"]})
                messages.append({"role": "assistant", "content": entry["assistant"]})
            
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            bot_response = response.choices[0].message.content.strip()
            
            # Store in conversation history
            self.conversation_history.append({
                "user": user_message,
                "assistant": bot_response,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "text": bot_response,
                "intent": "openai_response",
                "confidence": 0.95,
                "response_time": 0.0,
                "model": "gpt-3.5-turbo"
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._get_fallback_response(user_message)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for OpenAI"""
        return """You are a helpful customer support assistant for an e-commerce company. 
        You can help with:
        - Order status and tracking
        - Shipping and delivery information
        - Returns and refunds
        - Payment methods and billing
        - Product information
        - Account issues
        - General customer support
        
        Be friendly, helpful, and professional. If you don't know something specific, 
        offer to connect the customer with a human agent. Keep responses concise but informative."""
    
    def _get_fallback_response(self, user_message: str) -> Dict:
        """Fallback response when OpenAI is not available"""
        fallback_responses = [
            "I'm here to help! I can assist with order status, shipping, returns, payment, and product information.",
            "I'd be happy to help you with your inquiry. What specific information do you need?",
            "I can help with various customer support topics. How can I assist you today?",
            "I'm ready to help! Please let me know what you need assistance with."
        ]
        
        import random
        return {
            "text": random.choice(fallback_responses),
            "intent": "fallback",
            "confidence": 0.5,
            "response_time": 0.0,
            "model": "fallback"
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of user message"""
        if not self.client:
            return {"sentiment": "neutral", "confidence": 0.5}
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Analyze the sentiment of the following text. Respond with only: positive, negative, or neutral"},
                    {"role": "user", "content": text}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            sentiment = response.choices[0].message.content.strip().lower()
            return {"sentiment": sentiment, "confidence": 0.8}
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}
    
    def generate_summary(self, conversation: List[Dict]) -> str:
        """Generate a summary of the conversation"""
        if not self.client or not conversation:
            return "No conversation to summarize."
        
        try:
            # Prepare conversation text
            conv_text = ""
            for entry in conversation:
                conv_text += f"User: {entry.get('user', '')}\n"
                conv_text += f"Assistant: {entry.get('assistant', '')}\n\n"
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Summarize this customer support conversation in 2-3 sentences. Focus on the main issues and resolutions."},
                    {"role": "user", "content": conv_text}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return "Unable to generate summary."
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

def main():
    """Test the OpenAI chatbot"""
    print("OpenAI Chatbot Test")
    print("=" * 30)
    
    # Initialize chatbot
    chatbot = OpenAIChatbot()
    
    if chatbot.client:
        print("✅ OpenAI integration active")
        print("Model: GPT-3.5-turbo")
    else:
        print("⚠️ OpenAI not configured - using fallback responses")
        print("To enable OpenAI:")
        print("1. Get API key from https://platform.openai.com/")
        print("2. Set OPENAI_API_KEY in config.py")
    
    print("\nTesting chatbot...")
    
    # Test messages
    test_messages = [
        "Hi, I need help with my order",
        "Where is my order #12345?",
        "I'm not happy with my purchase",
        "What payment methods do you accept?",
        "How long does shipping take?"
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        response = chatbot.get_openai_response(message)
        print(f"Bot: {response['text']}")
        print(f"Intent: {response['intent']} (Confidence: {response['confidence']:.2f})")
        print(f"Model: {response.get('model', 'unknown')}")
    
    # Test sentiment analysis
    if chatbot.client:
        print(f"\nSentiment Analysis Test:")
        sentiment = chatbot.analyze_sentiment("I'm very happy with my order!")
        print(f"Sentiment: {sentiment['sentiment']} (Confidence: {sentiment['confidence']:.2f})")
    
    print(f"\nConversation History: {len(chatbot.get_conversation_history())} exchanges")

if __name__ == "__main__":
    main()
