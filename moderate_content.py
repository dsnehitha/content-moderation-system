#!/usr/bin/env python3
"""
Production Content Moderation Script
=====================================

This script demonstrates how to use the deployed content moderation system
for real-world content moderation tasks.

Usage:
    python moderate_content.py "Your text to check here"
    python moderate_content.py --batch file.txt
    python moderate_content.py --interactive
"""

import boto3
import json
import sys
import argparse
from typing import Dict, List, Tuple

class ContentModerator:
    """Production-ready content moderation client"""
    
    def __init__(self, config_path: str = 'aws_config.json', endpoint_info_path: str = 'endpoint_info.json'):
        """Initialize the content moderator"""
        # Load AWS configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load endpoint information
        with open(endpoint_info_path, 'r') as f:
            endpoint_info = json.load(f)
        
        self.endpoint_name = endpoint_info['endpoint_name']
        self.client = boto3.client('sagemaker-runtime', region_name=self.config['region'])
        
        print(f"✅ Connected to endpoint: {self.endpoint_name}")
    
    def moderate_text(self, text: str) -> Dict:
        """
        Moderate a single piece of text
        
        Args:
            text: The text to moderate
            
        Returns:
            Dictionary with moderation results
        """
        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='text/csv',
                Body=text.strip()
            )
            
            toxicity_score = float(response['Body'].read().decode('utf-8'))
            
            return {
                'text': text,
                'toxicity_score': toxicity_score,
                'is_toxic': toxicity_score > 0.5,
                'confidence': 'HIGH' if abs(toxicity_score - 0.5) > 0.3 else 'MEDIUM',
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'text': text,
                'toxicity_score': 0.0,
                'is_toxic': False,
                'confidence': 'LOW',
                'status': 'error',
                'error': str(e)
            }
    
    def moderate_batch(self, texts: List[str]) -> List[Dict]:
        """Moderate multiple texts"""
        results = []
        for i, text in enumerate(texts, 1):
            print(f"Processing {i}/{len(texts)}...", end='\\r')
            result = self.moderate_text(text)
            results.append(result)
        print()  # New line after progress
        return results
    
    def print_result(self, result: Dict):
        """Print a formatted moderation result"""
        if result['status'] == 'error':
            print(f"❌ Error: {result['error']}")
            return
        
        status_emoji = "🔴" if result['is_toxic'] else "✅"
        status_text = "TOXIC" if result['is_toxic'] else "SAFE"
        
        print(f"{status_emoji} {status_text} ({result['toxicity_score']:.1%}) - {result['confidence']} confidence")
        print(f"   Text: \"{result['text'][:100]}{'...' if len(result['text']) > 100 else ''}\"")
        print()

def main():
    parser = argparse.ArgumentParser(description='Content Moderation System')
    parser.add_argument('text', nargs='?', help='Text to moderate')
    parser.add_argument('--batch', help='File containing texts to moderate (one per line)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    # Initialize moderator
    try:
        moderator = ContentModerator()
    except Exception as e:
        print(f"❌ Failed to initialize moderator: {e}")
        sys.exit(1)
    
    # Handle different modes
    if args.interactive:
        print("🎯 Interactive Content Moderation")
        print("Type 'quit' to exit\\n")
        
        while True:
            try:
                text = input("Enter text to moderate: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if text:
                    result = moderator.moderate_text(text)
                    if args.json:
                        print(json.dumps(result, indent=2))
                    else:
                        moderator.print_result(result)
                        
            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
    
    elif args.batch:
        try:
            with open(args.batch, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"📁 Processing {len(texts)} texts from {args.batch}")
            results = moderator.moderate_batch(texts)
            
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                for result in results:
                    moderator.print_result(result)
                
                # Summary
                toxic_count = sum(1 for r in results if r['is_toxic'])
                print(f"📊 Summary: {toxic_count}/{len(results)} texts flagged as toxic")
                
        except FileNotFoundError:
            print(f"❌ File not found: {args.batch}")
            sys.exit(1)
    
    elif args.text:
        result = moderator.moderate_text(args.text)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            moderator.print_result(result)
    
    else:
        # Demo mode
        print("🎯 Content Moderation System Demo")
        print("=" * 40)
        
        demo_texts = [
            "Hello, how are you today?",
            "I hate you so much!",
            "Thank you for your help!",
            "You are an idiot!",
            "Have a wonderful day!",
            "Go kill yourself!",
            "This is a great product!",
            "I will destroy you!"
        ]
        
        results = moderator.moderate_batch(demo_texts)
        
        for result in results:
            moderator.print_result(result)
        
        # Summary
        toxic_count = sum(1 for r in results if r['is_toxic'])
        print(f"📊 Demo completed: {toxic_count}/{len(results)} texts flagged as toxic")

if __name__ == "__main__":
    main()
