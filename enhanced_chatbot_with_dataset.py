"""
Enhanced Chatbot with Dataset Integration and OpenAI
This combines the Kaggle dataset training with OpenAI capabilities
"""

import json
import random
from typing import Dict, List, Tuple
from datetime import datetime
import logging
from chatbot_core import CustomerSupportChatbot
from openai_chatbot import OpenAIChatbot
from dataset_integration import DatasetIntegrator
from config import OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedChatbotWithDataset:
    """Enhanced chatbot that uses dataset training and OpenAI integration"""
    
    def __init__(self):
        self.core_chatbot = CustomerSupportChatbot()
        self.openai_chatbot = OpenAIChatbot()
        self.dataset_integrator = DatasetIntegrator()
        self.training_data = self.dataset_integrator.get_training_data()
        self.enhanced_responses = self._load_enhanced_responses()
        
        # Analytics
        self.analytics = {
            'total_requests': 0,
            'openai_requests': 0,
            'core_requests': 0,
            'intent_distribution': {},
            'response_times': [],
            'user_satisfaction': [],
            'conversation_history': []
        }
        
        logger.info("Enhanced Chatbot with Dataset initialized successfully")
    
    def _load_enhanced_responses(self) -> Dict:
        """Load enhanced responses from the dataset"""
        enhanced_responses = {}
        
        for example in self.training_data:
            intent = example['intent']
            if intent not in enhanced_responses:
                enhanced_responses[intent] = []
            
            enhanced_responses[intent].append({
                'input': example['input'],
                'output': example['output'],
                'variations': self.dataset_integrator.generate_variations(example['input'], intent)
            })
        
        return enhanced_responses
    
    def process_message(self, user_input: str, user_id: str = None, use_openai: bool = True) -> Dict:
        """Process user message with enhanced dataset training and OpenAI"""
        start_time = datetime.now()
        
        # Update analytics
        self.analytics['total_requests'] += 1
        
        # Try OpenAI first if enabled and available
        if use_openai and self.openai_chatbot.client:
            try:
                response = self.openai_chatbot.get_openai_response(user_input)
                self.analytics['openai_requests'] += 1
                
                # Add to conversation history
                self._add_to_history(user_input, response['text'], 'openai', response['confidence'], user_id)
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                self.analytics['response_times'].append(response_time)
                
                return {
                    'text': response['text'],
                    'intent': response['intent'],
                    'confidence': response['confidence'],
                    'response_time': response_time,
                    'source': 'openai',
                    'model': response.get('model', 'gpt-3.5-turbo')
                }
                
            except Exception as e:
                logger.warning(f"OpenAI request failed, falling back to core chatbot: {e}")
        
        # Fallback to core chatbot with enhanced dataset
        response = self._process_with_enhanced_dataset(user_input)
        self.analytics['core_requests'] += 1
        
        # Add to conversation history
        self._add_to_history(user_input, response['text'], 'core', response['confidence'], user_id)
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        self.analytics['response_times'].append(response_time)
        
        return {
            'text': response['text'],
            'intent': response['intent'],
            'confidence': response['confidence'],
            'response_time': response_time,
            'source': 'core_enhanced',
            'model': 'dataset_trained'
        }
    
    def _process_with_enhanced_dataset(self, user_input: str) -> Dict:
        """Process message using enhanced dataset training"""
        user_input_lower = user_input.lower()
        
        # Find best matching intent using enhanced dataset
        best_match = self._find_best_intent_match(user_input_lower)
        
        if best_match:
            intent, confidence, response_text = best_match
            return {
                'text': response_text,
                'intent': intent,
                'confidence': confidence
            }
        
        # Fallback to core chatbot
        return self.core_chatbot.process_message(user_input)
    
    def _find_best_intent_match(self, user_input: str) -> Tuple[str, float, str]:
        """Find the best intent match using enhanced dataset"""
        best_intent = None
        best_confidence = 0.0
        best_response = ""
        
        for intent, examples in self.enhanced_responses.items():
            for example in examples:
                # Check direct match
                if user_input in example['input'].lower():
                    confidence = 0.9
                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence
                        best_response = example['output']
                
                # Check variations
                for variation in example['variations']:
                    if user_input in variation.lower():
                        confidence = 0.8
                        if confidence > best_confidence:
                            best_intent = intent
                            best_confidence = confidence
                            best_response = example['output']
                
                # Check keyword matching
                keywords = self.dataset_integrator.intent_mapping.get(intent, [])
                keyword_matches = sum(1 for keyword in keywords if keyword in user_input)
                
                if keyword_matches > 0:
                    confidence = min(0.7, keyword_matches * 0.2)
                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence
                        best_response = example['output']
        
        return best_intent, best_confidence, best_response
    
    def _add_to_history(self, user_input: str, bot_response: str, source: str, confidence: float, user_id: str = None):
        """Add conversation to history"""
        self.analytics['conversation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_input': user_input,
            'bot_response': bot_response,
            'source': source,
            'confidence': confidence
        })
        
        # Keep only last 100 conversations
        if len(self.analytics['conversation_history']) > 100:
            self.analytics['conversation_history'] = self.analytics['conversation_history'][-100:]
    
    def get_analytics(self) -> Dict:
        """Get comprehensive analytics"""
        total_requests = self.analytics['total_requests']
        response_times = self.analytics['response_times']
        
        analytics = {
            'total_requests': total_requests,
            'openai_requests': self.analytics['openai_requests'],
            'core_requests': self.analytics['core_requests'],
            'openai_usage_percentage': (self.analytics['openai_requests'] / total_requests * 100) if total_requests > 0 else 0,
            'average_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'intent_distribution': self.analytics['intent_distribution'],
            'conversation_count': len(self.analytics['conversation_history']),
            'dataset_examples': len(self.training_data),
            'enhanced_responses': len(self.enhanced_responses)
        }
        
        return analytics
    
    def get_sentiment_analysis(self, text: str) -> Dict:
        """Get sentiment analysis using OpenAI"""
        if self.openai_chatbot.client:
            return self.openai_chatbot.analyze_sentiment(text)
        else:
            return {'sentiment': 'neutral', 'confidence': 0.5}
    
    def generate_conversation_summary(self) -> str:
        """Generate conversation summary using OpenAI"""
        if self.openai_chatbot.client and self.analytics['conversation_history']:
            conversations = self.analytics['conversation_history'][-10:]  # Last 10 conversations
            return self.openai_chatbot.generate_summary(conversations)
        else:
            return "No conversation history available for summary."
    
    def export_conversations(self, filename: str = None) -> str:
        """Export conversation data"""
        if not filename:
            filename = f"enhanced_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'analytics': self.get_analytics(),
            'conversation_history': self.analytics['conversation_history'],
            'training_data': self.training_data,
            'enhanced_responses': self.enhanced_responses,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enhanced conversations exported to {filename}")
        return filename
    
    def get_dataset_statistics(self) -> Dict:
        """Get dataset statistics"""
        return self.dataset_integrator.analyze_dataset()
    
    def test_enhanced_responses(self) -> Dict:
        """Test enhanced responses with sample inputs"""
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        test_cases = [
            "Hello there!",
            "Where is my order #12345?",
            "How long does shipping take?",
            "I want to return this item",
            "What payment methods do you accept?",
            "Tell me about this product",
            "How can I contact support?",
            "I have a complaint about my order",
            "Thanks for your help!",
            "Goodbye!"
        ]
        
        for test_input in test_cases:
            test_results['total_tests'] += 1
            
            try:
                response = self.process_message(test_input, use_openai=False)  # Test core enhanced
                
                if response['confidence'] > 0.3:
                    test_results['passed_tests'] += 1
                    status = 'PASS'
                else:
                    test_results['failed_tests'] += 1
                    status = 'FAIL'
                
                test_results['test_details'].append({
                    'input': test_input,
                    'output': response['text'],
                    'intent': response['intent'],
                    'confidence': response['confidence'],
                    'status': status
                })
                
            except Exception as e:
                test_results['failed_tests'] += 1
                test_results['test_details'].append({
                    'input': test_input,
                    'output': f"Error: {str(e)}",
                    'intent': 'error',
                    'confidence': 0.0,
                    'status': 'ERROR'
                })
        
        return test_results

def main():
    """Test the enhanced chatbot with dataset"""
    print("Enhanced Chatbot with Dataset Test")
    print("=" * 50)
    
    # Initialize enhanced chatbot
    chatbot = EnhancedChatbotWithDataset()
    
    # Show dataset statistics
    dataset_stats = chatbot.get_dataset_statistics()
    print(f"Dataset Statistics:")
    print(f"  Total Examples: {dataset_stats['total_examples']}")
    print(f"  Average Examples per Intent: {dataset_stats['average_examples_per_intent']:.1f}")
    print(f"  Most Common Intent: {dataset_stats['most_common_intent']}")
    
    # Test enhanced responses
    print(f"\nTesting Enhanced Responses:")
    print("-" * 30)
    
    test_messages = [
        "Hi there!",
        "Where is my order?",
        "How long does shipping take?",
        "I want to return this item",
        "What payment methods do you accept?",
        "Tell me about this product",
        "How can I contact support?",
        "I have a complaint",
        "Thanks for your help!",
        "Goodbye!"
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        
        # Test with OpenAI (if available)
        response_openai = chatbot.process_message(message, use_openai=True)
        print(f"OpenAI: {response_openai['text']}")
        print(f"  Intent: {response_openai['intent']} (Confidence: {response_openai['confidence']:.2f})")
        print(f"  Source: {response_openai['source']}")
        
        # Test with enhanced core
        response_core = chatbot.process_message(message, use_openai=False)
        print(f"Core+: {response_core['text']}")
        print(f"  Intent: {response_core['intent']} (Confidence: {response_core['confidence']:.2f})")
        print(f"  Source: {response_core['source']}")
    
    # Run comprehensive test
    print(f"\nComprehensive Test Results:")
    print("-" * 30)
    test_results = chatbot.test_enhanced_responses()
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed_tests']}")
    print(f"Failed: {test_results['failed_tests']}")
    print(f"Success Rate: {(test_results['passed_tests'] / test_results['total_tests'] * 100):.1f}%")
    
    # Show analytics
    analytics = chatbot.get_analytics()
    print(f"\nAnalytics:")
    print(f"  Total Requests: {analytics['total_requests']}")
    print(f"  OpenAI Requests: {analytics['openai_requests']}")
    print(f"  Core Requests: {analytics['core_requests']}")
    print(f"  OpenAI Usage: {analytics['openai_usage_percentage']:.1f}%")
    print(f"  Average Response Time: {analytics['average_response_time']:.2f}s")
    
    # Export data
    filename = chatbot.export_conversations()
    print(f"\nData exported to: {filename}")
    
    print(f"\nEnhanced Chatbot with Dataset is ready for production!")

if __name__ == "__main__":
    main()
