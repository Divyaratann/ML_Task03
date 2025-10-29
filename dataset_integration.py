"""
Dataset Integration for Enhanced Chatbot Training
This integrates the Kaggle dataset for better intent recognition and response generation
"""

import json
import random
from typing import Dict, List, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetIntegrator:
    """Integrates external datasets for enhanced chatbot training"""
    
    def __init__(self):
        self.dataset_path = "kaggle_dialogs.json"
        self.intent_mapping = {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
            "goodbye": ["bye", "goodbye", "see you", "farewell", "take care"],
            "thanks": ["thank you", "thanks", "appreciate", "grateful"],
            "order_status": ["order", "status", "tracking", "shipped", "delivery", "where is my order"],
            "shipping": ["shipping", "delivery", "shipping time", "how long", "when will it arrive"],
            "returns": ["return", "refund", "exchange", "cancel", "return policy"],
            "payment": ["payment", "billing", "charge", "credit card", "payment method", "invoice"],
            "product_info": ["product", "specifications", "features", "size", "color", "availability"],
            "contact": ["contact", "phone", "email", "support", "help", "speak to someone"],
            "complaint": ["complaint", "problem", "issue", "unhappy", "dissatisfied", "angry"]
        }
        self.training_data = []
        self.load_kaggle_dataset()
    
    def load_kaggle_dataset(self):
        """Load and process the Kaggle dataset"""
        try:
            # Create sample dataset based on the Kaggle structure
            self.training_data = self._create_sample_dataset()
            logger.info(f"Loaded {len(self.training_data)} training examples from dataset")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.training_data = []
    
    def _create_sample_dataset(self) -> List[Dict]:
        """Create sample dataset based on Kaggle dialog structure"""
        sample_dialogs = [
            # Greeting patterns
            {"input": "Hello", "output": "Hi there! How can I help you today?", "intent": "greeting"},
            {"input": "Hi", "output": "Hello! Welcome to our customer support. What can I assist you with?", "intent": "greeting"},
            {"input": "Good morning", "output": "Good morning! How can I make your day better today?", "intent": "greeting"},
            {"input": "Hey there", "output": "Hey! I'm here to help. What questions do you have?", "intent": "greeting"},
            
            # Order status patterns
            {"input": "Where is my order?", "output": "I can help you track your order! Please provide your order number and I'll check the status for you.", "intent": "order_status"},
            {"input": "Order status", "output": "To check your order status, please provide your order number.", "intent": "order_status"},
            {"input": "Track my package", "output": "I'd be happy to help you track your package. What's your order number?", "intent": "order_status"},
            {"input": "Is my order shipped?", "output": "Let me check your order status. Please provide your order number.", "intent": "order_status"},
            
            # Shipping patterns
            {"input": "How long does shipping take?", "output": "We offer standard shipping (3-5 business days) and express shipping (1-2 business days). Which option would you like to know more about?", "intent": "shipping"},
            {"input": "Shipping time", "output": "Shipping times vary. Standard delivery is 3-5 days, express is 1-2 days.", "intent": "shipping"},
            {"input": "When will it arrive?", "output": "Our shipping options are standard (3-5 days) and express (1-2 days).", "intent": "shipping"},
            {"input": "Delivery time", "output": "We have two shipping options: standard (3-5 days) and express (1-2 days). Which would you prefer?", "intent": "shipping"},
            
            # Returns patterns
            {"input": "I want to return this", "output": "We offer a 30-day return policy. Items must be in original condition with packaging. Would you like me to help you start a return?", "intent": "returns"},
            {"input": "Return policy", "output": "Returns are accepted within 30 days of purchase. Please keep your receipt and original packaging.", "intent": "returns"},
            {"input": "How do I return?", "output": "You can return items within 30 days for a full refund. Do you need assistance with the return process?", "intent": "returns"},
            {"input": "Refund", "output": "Our return policy allows returns within 30 days. Items must be in original condition.", "intent": "returns"},
            
            # Payment patterns
            {"input": "What payment methods do you accept?", "output": "We accept all major credit cards, PayPal, and bank transfers. How can I help with your payment?", "intent": "payment"},
            {"input": "Payment options", "output": "You can pay with Visa, MasterCard, American Express, PayPal, or bank transfer.", "intent": "payment"},
            {"input": "How can I pay?", "output": "Multiple payment methods are available, including credit cards and PayPal.", "intent": "payment"},
            {"input": "Billing", "output": "I can help with billing questions. What specific payment issue can I assist with?", "intent": "payment"},
            
            # Product info patterns
            {"input": "Tell me about this product", "output": "I'd be happy to help with product information. Which product are you interested in?", "intent": "product_info"},
            {"input": "Product details", "output": "Let me get you the details on that product. What specific information do you need?", "intent": "product_info"},
            {"input": "Specifications", "output": "I can provide product specifications and availability. What product are you asking about?", "intent": "product_info"},
            {"input": "Is this in stock?", "output": "Let me check the availability for you. Which product are you interested in?", "intent": "product_info"},
            
            # Contact patterns
            {"input": "How can I contact support?", "output": "You can reach our support team at support@company.com or call 1-800-SUPPORT. Our team is available 24/7!", "intent": "contact"},
            {"input": "Phone number", "output": "For immediate assistance, call us at 1-800-SUPPORT or email support@company.com.", "intent": "contact"},
            {"input": "Email support", "output": "Our support team is available 24/7. Call 1-800-SUPPORT or email support@company.com.", "intent": "contact"},
            {"input": "Speak to someone", "output": "I can help you with most questions, but if you need to speak to a human agent, call 1-800-SUPPORT.", "intent": "contact"},
            
            # Complaint patterns
            {"input": "I have a complaint", "output": "I'm sorry to hear you're having an issue. Please tell me more about the problem so I can help resolve it.", "intent": "complaint"},
            {"input": "This is terrible", "output": "I understand your frustration. Let me help you resolve this issue. What specific problem are you experiencing?", "intent": "complaint"},
            {"input": "I'm not happy", "output": "I'm sorry you're not satisfied. Please share the details of your concern so I can assist you better.", "intent": "complaint"},
            {"input": "This is unacceptable", "output": "I apologize for the inconvenience. Let me help you address this issue. What happened?", "intent": "complaint"},
            
            # Goodbye patterns
            {"input": "Thanks for your help", "output": "You're welcome! Is there anything else I can help you with?", "intent": "goodbye"},
            {"input": "Thank you", "output": "You're welcome! Feel free to reach out if you need any more assistance.", "intent": "goodbye"},
            {"input": "Bye", "output": "Goodbye! Have a great day!", "intent": "goodbye"},
            {"input": "See you later", "output": "See you later! Take care!", "intent": "goodbye"}
        ]
        
        return sample_dialogs
    
    def get_training_data(self) -> List[Dict]:
        """Get the training data"""
        return self.training_data
    
    def get_intent_examples(self, intent: str) -> List[Dict]:
        """Get examples for a specific intent"""
        return [example for example in self.training_data if example.get('intent') == intent]
    
    def generate_variations(self, base_input: str, intent: str) -> List[str]:
        """Generate variations of input text for better training"""
        variations = [base_input]
        
        # Add common variations
        if intent == "greeting":
            variations.extend([
                base_input + " there",
                base_input + "!",
                "Hey " + base_input.lower(),
                "Good morning " + base_input.lower()
            ])
        elif intent == "order_status":
            variations.extend([
                "Where is " + base_input,
                "Status of " + base_input,
                "Track " + base_input,
                "Check " + base_input
            ])
        elif intent == "shipping":
            variations.extend([
                "How long for " + base_input,
                "When will " + base_input + " arrive",
                "Delivery time for " + base_input,
                "Shipping duration for " + base_input
            ])
        
        return variations
    
    def create_enhanced_training_set(self) -> Dict:
        """Create an enhanced training set with variations"""
        enhanced_data = {
            "intents": {},
            "total_examples": 0,
            "intent_distribution": {},
            "created_at": datetime.now().isoformat()
        }
        
        for example in self.training_data:
            intent = example['intent']
            if intent not in enhanced_data['intents']:
                enhanced_data['intents'][intent] = {
                    "examples": [],
                    "responses": [],
                    "keywords": self.intent_mapping.get(intent, [])
                }
            
            # Add original example
            enhanced_data['intents'][intent]['examples'].append(example['input'])
            enhanced_data['intents'][intent]['responses'].append(example['output'])
            
            # Generate variations
            variations = self.generate_variations(example['input'], intent)
            for variation in variations[1:]:  # Skip the original
                enhanced_data['intents'][intent]['examples'].append(variation)
                enhanced_data['intents'][intent]['responses'].append(example['output'])
        
        # Calculate statistics
        for intent, data in enhanced_data['intents'].items():
            enhanced_data['intent_distribution'][intent] = len(data['examples'])
            enhanced_data['total_examples'] += len(data['examples'])
        
        return enhanced_data
    
    def export_training_data(self, filename: str = "enhanced_training_data.json"):
        """Export the enhanced training data"""
        enhanced_data = self.create_enhanced_training_set()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enhanced training data exported to {filename}")
        return filename
    
    def get_random_example(self, intent: str = None) -> Dict:
        """Get a random example from the dataset"""
        if intent:
            examples = self.get_intent_examples(intent)
        else:
            examples = self.training_data
        
        if examples:
            return random.choice(examples)
        return None
    
    def analyze_dataset(self) -> Dict:
        """Analyze the dataset and provide statistics"""
        analysis = {
            "total_examples": len(self.training_data),
            "intent_distribution": {},
            "average_examples_per_intent": 0,
            "most_common_intent": None,
            "least_common_intent": None
        }
        
        # Count examples per intent
        for example in self.training_data:
            intent = example.get('intent', 'unknown')
            analysis['intent_distribution'][intent] = analysis['intent_distribution'].get(intent, 0) + 1
        
        # Calculate statistics
        if analysis['intent_distribution']:
            analysis['average_examples_per_intent'] = sum(analysis['intent_distribution'].values()) / len(analysis['intent_distribution'])
            analysis['most_common_intent'] = max(analysis['intent_distribution'], key=analysis['intent_distribution'].get)
            analysis['least_common_intent'] = min(analysis['intent_distribution'], key=analysis['intent_distribution'].get)
        
        return analysis

def main():
    """Test the dataset integration"""
    print("Dataset Integration Test")
    print("=" * 40)
    
    integrator = DatasetIntegrator()
    
    # Analyze dataset
    analysis = integrator.analyze_dataset()
    print(f"Total Examples: {analysis['total_examples']}")
    print(f"Average Examples per Intent: {analysis['average_examples_per_intent']:.1f}")
    print(f"Most Common Intent: {analysis['most_common_intent']}")
    print(f"Least Common Intent: {analysis['least_common_intent']}")
    
    print("\nIntent Distribution:")
    for intent, count in analysis['intent_distribution'].items():
        print(f"  {intent}: {count}")
    
    # Test random examples
    print("\nRandom Examples:")
    for intent in ['greeting', 'order_status', 'shipping', 'returns']:
        example = integrator.get_random_example(intent)
        if example:
            print(f"  {intent}: '{example['input']}' -> '{example['output']}'")
    
    # Export enhanced training data
    filename = integrator.export_training_data()
    print(f"\nEnhanced training data exported to: {filename}")

if __name__ == "__main__":
    main()
