"""
Core Chatbot Implementation for Customer Support
This module contains the main chatbot logic and intent handling
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerSupportChatbot:
    """
    Main chatbot class that handles customer support conversations
    """
    
    def __init__(self):
        self.conversation_history = []
        self.analytics = {
            'total_requests': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'intent_counts': {},
            'response_times': []
        }
        
        # Initialize intents and responses
        self.intents = self._initialize_intents()
        self.greeting_messages = [
            "Hi! Welcome to our customer support. How can I help you today?",
            "Hello! I'm here to assist you with any questions or concerns.",
            "Hi there! Welcome to our support center. What can I help you with?",
            "Hello! How can I make your day better today?"
        ]
        
        logger.info("Customer Support Chatbot initialized successfully")
    
    def _initialize_intents(self) -> Dict:
        """Initialize all customer support intents and responses"""
        return {
            "greeting": {
                "keywords": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"],
                "responses": [
                    "Hi! Welcome to our customer support. How can I help you today?",
                    "Hello! I'm here to assist you with any questions or concerns.",
                    "Hi there! Welcome to our support center. What can I help you with?",
                    "Hello! How can I make your day better today?"
                ]
            },
            "order_status": {
                "keywords": ["order", "status", "tracking", "shipped", "delivery", "where is my order", "track my order"],
                "responses": [
                    "I can help you track your order! Please provide your order number and I'll check the status for you.",
                    "To check your order status, I'll need your order number. What's your order number?",
                    "I can look up your order status. Please share your order number with me.",
                    "Let me help you track your order. Could you please provide your order number?"
                ]
            },
            "shipping": {
                "keywords": ["shipping", "delivery", "shipping time", "how long", "when will it arrive", "delivery time"],
                "responses": [
                    "We offer standard shipping (3-5 business days) and express shipping (1-2 business days). Which option would you like to know more about?",
                    "Shipping times vary by location. Standard delivery is 3-5 days, express is 1-2 days.",
                    "We have two shipping options: standard (3-5 days) and express (1-2 days). Which would you prefer?",
                    "Standard shipping takes 3-5 business days, while express shipping takes 1-2 business days."
                ]
            },
            "returns": {
                "keywords": ["return", "refund", "exchange", "cancel", "return policy", "how to return"],
                "responses": [
                    "We offer a 30-day return policy. Items must be in original condition with packaging. Would you like me to help you start a return?",
                    "Returns are accepted within 30 days of purchase. Please keep your receipt and original packaging.",
                    "You can return items within 30 days. I can help you start the return process if you'd like.",
                    "Our return policy allows returns within 30 days. Items must be in original condition."
                ]
            },
            "payment": {
                "keywords": ["payment", "billing", "charge", "credit card", "payment method", "invoice", "billing issue"],
                "responses": [
                    "We accept all major credit cards, PayPal, and bank transfers. How can I help with your payment?",
                    "You can pay with Visa, MasterCard, American Express, PayPal, or bank transfer.",
                    "We accept multiple payment methods including credit cards and PayPal. What payment issue can I help with?",
                    "For payment assistance, we accept credit cards, PayPal, and bank transfers."
                ]
            },
            "product_info": {
                "keywords": ["product", "specifications", "features", "size", "color", "availability", "product details"],
                "responses": [
                    "I'd be happy to help with product information. Which product are you interested in?",
                    "Let me get you the details on that product. What specific information do you need?",
                    "I can provide product specifications and availability. What product are you asking about?",
                    "I'd love to help with product details. Which product would you like to know more about?"
                ]
            },
            "contact": {
                "keywords": ["contact", "phone", "email", "support", "help", "speak to someone", "human agent"],
                "responses": [
                    "You can reach our support team at support@company.com or call 1-800-SUPPORT. Our team is available 24/7!",
                    "For immediate assistance, call us at 1-800-SUPPORT or email support@company.com.",
                    "Our support team is available 24/7. Call 1-800-SUPPORT or email support@company.com.",
                    "You can contact us at support@company.com or call 1-800-SUPPORT for immediate help."
                ]
            },
            "account": {
                "keywords": ["account", "login", "password", "profile", "sign up", "register", "account issue"],
                "responses": [
                    "I can help with account issues. Are you having trouble logging in or creating an account?",
                    "For account support, please provide your email address and I'll assist you.",
                    "I can help with account-related questions. What specific issue are you experiencing?",
                    "Let me help you with your account. What's the problem you're facing?"
                ]
            },
            "complaint": {
                "keywords": ["complaint", "problem", "issue", "dissatisfied", "unhappy", "bad experience"],
                "responses": [
                    "I'm sorry to hear about your experience. Let me help resolve this issue for you. Can you tell me more details?",
                    "I apologize for any inconvenience. I'm here to help make things right. What happened?",
                    "I'm sorry you're having a problem. Let me assist you in resolving this issue.",
                    "I understand your concern. Let me help you with this issue right away."
                ]
            },
            "goodbye": {
                "keywords": ["bye", "goodbye", "thanks", "thank you", "see you", "farewell"],
                "responses": [
                    "You're welcome! Have a great day!",
                    "Thank you for contacting us! Feel free to reach out anytime.",
                    "Happy to help! Take care!",
                    "You're welcome! Don't hesitate to contact us if you need anything else."
                ]
            }
        }
    
    def process_message(self, user_input: str, session_id: str = None) -> Dict:
        """
        Process user input and return bot response
        """
        if not user_input.strip():
            return self._create_response("Please enter a message so I can help you!", "empty_input", 0.0)
        
        start_time = datetime.now()
        
        try:
            # Find the best matching intent
            intent, confidence = self._find_best_intent(user_input)
            
            # Generate response
            response_text = self._generate_response(intent, user_input)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Record analytics
            self._record_interaction(user_input, intent, response_text, confidence, response_time, session_id)
            
            return self._create_response(response_text, intent, confidence, response_time)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            self.analytics['failed_responses'] += 1
            return self._create_response("I'm sorry, I encountered an error. Please try again.", "error", 0.0)
    
    def _find_best_intent(self, user_input: str) -> Tuple[str, float]:
        """Find the best matching intent for user input"""
        user_input_lower = user_input.lower()
        best_match_score = 0
        best_intent = "fallback"
        
        for intent_name, intent_data in self.intents.items():
            score = 0
            for keyword in intent_data["keywords"]:
                if keyword in user_input_lower:
                    score += 1
            
            if score > best_match_score:
                best_match_score = score
                best_intent = intent_name
        
        # Calculate confidence (0.0 to 1.0)
        confidence = min(best_match_score / 3.0, 1.0) if best_match_score > 0 else 0.0
        
        return best_intent, confidence
    
    def _generate_response(self, intent: str, user_input: str) -> str:
        """Generate response based on intent"""
        if intent in self.intents:
            responses = self.intents[intent]["responses"]
            return random.choice(responses)
        else:
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Get fallback response for unknown queries"""
        fallback_responses = [
            "I'm sorry, I didn't quite understand that. Could you please rephrase your question?",
            "I'm not sure I can help with that specific question. Could you try asking in a different way?",
            "I don't have information about that. Let me connect you with a human agent who can help better.",
            "That's an interesting question! I might not have the exact answer, but I can connect you with our support team.",
            "I'm still learning! Could you try asking about order status, shipping, returns, or product information?"
        ]
        return random.choice(fallback_responses)
    
    def _create_response(self, text: str, intent: str, confidence: float, response_time: float = 0.0) -> Dict:
        """Create standardized response dictionary"""
        return {
            'text': text,
            'intent': intent,
            'confidence': confidence,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def _record_interaction(self, user_input: str, intent: str, response_text: str, 
                          confidence: float, response_time: float, session_id: str = None):
        """Record interaction for analytics"""
        # Update analytics
        self.analytics['total_requests'] += 1
        self.analytics['successful_responses'] += 1
        self.analytics['response_times'].append(response_time)
        
        # Count intents
        self.analytics['intent_counts'][intent] = self.analytics['intent_counts'].get(intent, 0) + 1
        
        # Record conversation
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id or 'default',
            'user_input': user_input,
            'intent': intent,
            'response': response_text,
            'confidence': confidence,
            'response_time': response_time
        })
        
        logger.info(f"Intent: {intent}, Confidence: {confidence:.2f}, Response Time: {response_time:.2f}s")
    
    def get_analytics(self) -> Dict:
        """Get conversation analytics"""
        if self.analytics['total_requests'] > 0:
            avg_response_time = sum(self.analytics['response_times']) / len(self.analytics['response_times'])
            success_rate = (self.analytics['successful_responses'] / self.analytics['total_requests']) * 100
        else:
            avg_response_time = 0
            success_rate = 0
        
        return {
            'total_requests': self.analytics['total_requests'],
            'successful_responses': self.analytics['successful_responses'],
            'failed_responses': self.analytics['failed_responses'],
            'success_rate': f"{success_rate:.1f}%",
            'average_response_time': f"{avg_response_time:.2f}s",
            'intent_distribution': self.analytics['intent_counts'],
            'recent_conversations': self.conversation_history[-10:]
        }
    
    def clear_history(self):
        """Clear conversation history and reset analytics"""
        self.conversation_history = []
        self.analytics = {
            'total_requests': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'intent_counts': {},
            'response_times': []
        }
        logger.info("Conversation history and analytics cleared")
    
    def export_conversations(self, filename: str = None) -> str:
        """Export conversation history to JSON file"""
        if not filename:
            filename = f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'analytics': self.analytics,
            'conversations': self.conversation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversations exported to {filename}")
        return filename
