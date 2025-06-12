#!/usr/bin/env python3
"""
Amazon Bedrock Integration for Content Moderation
=================================================

This module integrates Amazon Bedrock (Claude/GPT) for:
- Contextual analysis of borderline cases
- Enhanced moderation decisions
- Explanation generation for moderation actions
- Fallback logic for complex content
"""

import boto3
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class BedrockContentAnalyzer:
    """Amazon Bedrock integration for advanced content analysis"""
    
    def __init__(self):
        # Initialize Bedrock availability flag
        self.bedrock_available = True
        self.bedrock_runtime = None
        self.bedrock_client = None
        
        # Load configuration
        with open('aws_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Check if Bedrock should be enabled (default: True, but can be disabled)
        self.bedrock_enabled = self.config.get('bedrock_enabled', True)
        
        # Primary models using inference profile IDs (recommended for production)
        self.available_models = {
            'claude-4-sonnet': 'us.anthropic.claude-sonnet-4-20250514-v1:0',
            'claude-3-haiku': 'us.anthropic.claude-3-haiku-20240307-v1:0',
        }
        
        # Fallback to direct model IDs if inference profiles don't work
        self.direct_model_ids = {
            'claude-4-sonnet': 'anthropic.claude-sonnet-4-20250514-v1:0',
            'claude-3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
        }
        
        # Initialize Bedrock clients if enabled
        if self.bedrock_enabled:
            try:
                self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
                self.bedrock_client = boto3.client('bedrock', region_name='us-east-1')
                
                # Test basic connectivity and model access
                print(f"🔍 Testing Bedrock access...")
                self._test_bedrock_access()
                
                self.bedrock_available = True
                self.default_model = 'claude-4-sonnet'  # Use latest Claude 4
                
                print(f"🤖 Bedrock Content Analyzer initialized")
                print(f"🌍 Region: us-east-1")
                print(f"🎯 Bedrock: ✅ Available")
                
            except Exception as e:
                print(f"⚠️  Bedrock initialization failed: {e}")
                print(f"🔄 Continuing with ML-only mode")
                self._diagnose_bedrock_issue(e)
                self.bedrock_available = False
                self.default_model = None
        else:
            print(f"🚫 Bedrock disabled in configuration")
            print(f"🔄 Running in ML-only mode")
        
        if not self.bedrock_available:
            print(f"📝 Note: System will use ML model + rule-based fallbacks")
            print(f"💡 To enable Bedrock: Set 'bedrock_enabled': true in aws_config.json")
    
    def check_model_availability(self):
        """Check which Bedrock models are available"""
        if not self.bedrock_available:
            print("🚫 Bedrock not available - running in ML-only mode")
            return []
            
        try:
            response = self.bedrock_client.list_foundation_models()
            available_model_ids = [model['modelId'] for model in response['modelSummaries']]
            
            print("🔍 Available Bedrock models (using inference profiles):")
            for name, model_id in self.available_models.items():
                # Check if inference profile exists
                if model_id.startswith('us.'):
                    status = "✅ (inference profile)" if model_id in available_model_ids else "❓ (profile not found)"
                else:
                    status = "✅ (direct model)" if model_id in available_model_ids else "❌ (not accessible)"
                print(f"   {status} {name}: {model_id}")
            
            print("\n🔍 Fallback direct model IDs:")
            for name, model_id in self.direct_model_ids.items():
                status = "✅" if model_id in available_model_ids else "❌"
                print(f"   {status} {name}: {model_id}")
            
            return available_model_ids
            
        except Exception as e:
            print(f"⚠️  Could not check model availability: {e}")
            return []
    
    def analyze_with_bedrock(self, text: str, toxicity_score: float, 
                           risk_indicators: Dict, model_name: str = None) -> Dict:
        """
        Use Bedrock to analyze borderline content and provide context
        
        Args:
            text: The content to analyze
            toxicity_score: ML model toxicity score (0-1)
            risk_indicators: Risk indicators from preprocessing
            model_name: Bedrock model to use
            
        Returns:
            Dictionary with Bedrock analysis results
        """
        
        # If Bedrock is not available, return fallback immediately
        if not self.bedrock_available:
            return self._create_fallback_response(text, toxicity_score, "Bedrock not available")
        
        if model_name is None:
            model_name = self.default_model
        
        model_id = self.available_models.get(model_name)
        if not model_id:
            print(f"⚠️  Model {model_name} not available, falling back to ML-only")
            return self._create_fallback_response(text, toxicity_score, f"Model {model_name} not available")
        
        # Create contextual prompt
        prompt = self._create_analysis_prompt(text, toxicity_score, risk_indicators)
        
        try:
            start_time = time.time()
            
            # Try the requested model first
            response = self._invoke_model_with_fallback(model_id, model_name, prompt)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Parse Bedrock response
            analysis = self._parse_bedrock_response(response, text, toxicity_score)
            analysis['processing_time_ms'] = processing_time
            analysis['model_used'] = model_name
            
            return analysis
            
        except Exception as e:
            print(f"❌ Bedrock analysis failed: {e}")
            return self._create_fallback_response(text, toxicity_score, str(e))
    
    def _invoke_model_with_fallback(self, model_id: str, model_name: str, prompt: str) -> str:
        """Invoke model with fallback to other available models and inference profiles"""
        
        # Try the inference profile first (model_id is already an inference profile)
        try:
            print(f"🔄 Trying inference profile: {model_id}")
            if 'claude' in model_name:
                return self._call_claude(model_id, prompt)
            else:
                raise ValueError(f"Unsupported model type: {model_name}")
                
        except Exception as profile_error:
            error_str = str(profile_error).lower()
            print(f"⚠️  Inference profile {model_name} failed: {profile_error}")
            
            # If inference profile fails, try direct model ID
            if model_name in self.direct_model_ids:
                try:
                    direct_model_id = self.direct_model_ids[model_name]
                    print(f"🔄 Trying direct model ID: {direct_model_id}")
                    
                    if 'claude' in model_name:
                        return self._call_claude(direct_model_id, prompt)
                    else:
                        raise ValueError(f"Unsupported model type: {model_name}")
                        
                except Exception as direct_error:
                    print(f"⚠️  Direct model ID also failed: {direct_error}")
            
            # Try fallback models in order of preference
            fallback_models = ['claude-3-haiku']
            
            for fallback_name in fallback_models:
                if fallback_name != model_name and fallback_name in self.available_models:
                    try:
                        print(f"🔄 Trying fallback model: {fallback_name}")
                        fallback_id = self.available_models[fallback_name]
                        
                        if 'claude' in fallback_name:
                            return self._call_claude(fallback_id, prompt)
                        else:
                            raise ValueError(f"Unsupported model type: {fallback_name}")
                            
                    except Exception as fallback_error:
                        print(f"⚠️  Fallback model {fallback_name} also failed: {fallback_error}")
                        continue
            
            # If all models fail, raise the original error
            raise profile_error
    
    def _create_analysis_prompt(self, text: str, toxicity_score: float, 
                              risk_indicators: Dict) -> str:
        """Create a comprehensive prompt for Bedrock analysis"""
        
        prompt = f"""You are an expert content moderation analyst. Analyze the following text for toxicity, context, and potential harm.

        TEXT TO ANALYZE: "{text}"

        CONTEXT:
        - ML Model Toxicity Score: {toxicity_score:.3f} (0.0 = safe, 1.0 = toxic)
        - Risk Indicators: {json.dumps(risk_indicators, indent=2)}

        Please provide a detailed analysis in the following JSON format:

        {{
            "contextual_toxicity_score": <float between 0.0 and 1.0>,
            "recommended_action": "<allow|review|block>",
            "confidence": "<high|medium|low>",
            "reasoning": "<detailed explanation of your analysis>",
            "harm_categories": [<list of relevant categories: "hate_speech", "harassment", "threats", "profanity", "spam", "none">],
            "context_factors": [<list of mitigating or aggravating factors>],
            "is_sarcasm": <true|false>,
            "is_false_positive": <true|false>,
            "severity_level": "<low|medium|high|critical>",
            "target_audience_risk": "<children|general|adults>",
            "cultural_sensitivity": "<high|medium|low>",
            "explanation_for_user": "<user-friendly explanation if content is moderated>"
        }}

        Consider these factors:
        1. Context and intent of the message
        2. Potential for real-world harm
        3. Sarcasm, humor, or figurative language
        4. Cultural and linguistic nuances
        5. Target audience appropriateness
        6. False positive likelihood

        Be thorough but concise. Focus on actionable insights for content moderation decisions."""
        
        return prompt
    
    def _call_claude(self, model_id: str, prompt: str) -> str:
        """Call Claude models via Bedrock"""
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = self.bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']
    
    def _parse_bedrock_response(self, response_text: str, original_text: str, 
                               ml_score: float) -> Dict:
        """Parse and validate Bedrock response"""
        
        try:
            # Extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            analysis = json.loads(response_text)
            
            # Validate required fields
            required_fields = [
                'contextual_toxicity_score', 'recommended_action', 
                'confidence', 'reasoning', 'harm_categories'
            ]
            
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = self._get_fallback_value(field)
            
            # Ensure score is valid
            if not isinstance(analysis['contextual_toxicity_score'], (int, float)):
                analysis['contextual_toxicity_score'] = ml_score
            else:
                analysis['contextual_toxicity_score'] = max(0.0, min(1.0, 
                    float(analysis['contextual_toxicity_score'])))
            
            # Validate action
            valid_actions = ['allow', 'review', 'block']
            if analysis['recommended_action'] not in valid_actions:
                analysis['recommended_action'] = 'review'
            
            # Add metadata
            analysis['bedrock_enhanced'] = True
            analysis['ml_toxicity_score'] = ml_score
            analysis['score_difference'] = abs(analysis['contextual_toxicity_score'] - ml_score)
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse Bedrock JSON response: {e}")
            return self._create_fallback_response(original_text, ml_score, 
                                                f"JSON parse error: {e}")
        except Exception as e:
            print(f"⚠️  Error processing Bedrock response: {e}")
            return self._create_fallback_response(original_text, ml_score, str(e))
    
    def _get_fallback_value(self, field: str):
        """Get fallback values for missing fields"""
        fallbacks = {
            'contextual_toxicity_score': 0.5,
            'recommended_action': 'review',
            'confidence': 'medium',
            'reasoning': 'Analysis incomplete - using fallback values',
            'harm_categories': ['unknown'],
            'context_factors': [],
            'is_sarcasm': False,
            'is_false_positive': False,
            'severity_level': 'medium'
        }
        return fallbacks.get(field, 'unknown')
    
    def _create_fallback_response(self, text: str, ml_score: float, error: str) -> Dict:
        """Create fallback response when Bedrock analysis fails"""
        
        # Simple fallback logic based on ML score
        if ml_score > 0.8:
            action = 'block'
            confidence = 'medium'
        elif ml_score > 0.6:
            action = 'review'
            confidence = 'low'
        else:
            action = 'allow'
            confidence = 'medium'
        
        return {
            'contextual_toxicity_score': ml_score,
            'recommended_action': action,
            'confidence': confidence,
            'reasoning': f'Bedrock analysis failed: {error}. Using ML model score.',
            'harm_categories': ['unknown'],
            'context_factors': ['bedrock_unavailable'],
            'is_sarcasm': False,
            'is_false_positive': False,
            'severity_level': 'medium',
            'bedrock_enhanced': False,
            'ml_toxicity_score': ml_score,
            'score_difference': 0.0,
            'error': error
        }
    
    def make_enhanced_moderation_decision(self, text: str, ml_score: float, 
                                        risk_indicators: Dict, 
                                        use_bedrock: bool = True) -> Dict:
        """
        Make enhanced moderation decision combining ML and Bedrock analysis
        
        Args:
            text: Content to moderate
            ml_score: ML model toxicity score
            risk_indicators: Risk indicators from preprocessing
            use_bedrock: Whether to use Bedrock for analysis
            
        Returns:
            Enhanced moderation decision
        """
        
        start_time = time.time()
        
        # Determine if we should use Bedrock
        should_use_bedrock = (
            use_bedrock and
            0.3 < ml_score < 0.8  # Borderline cases
        )
        
        if should_use_bedrock:
            print(f"🤖 Using Bedrock for borderline case (score: {ml_score:.3f})")
            bedrock_analysis = self.analyze_with_bedrock(text, ml_score, risk_indicators)
        else:
            print(f"⚡ Using ML-only decision (score: {ml_score:.3f})")
            bedrock_analysis = self._create_ml_only_response(text, ml_score)
        
        # Combine ML and Bedrock insights for final decision
        final_decision = self._combine_analyses(ml_score, bedrock_analysis, risk_indicators)
        
        # Add timing and metadata
        total_time = int((time.time() - start_time) * 1000)
        final_decision.update({
            'text': text,
            'timestamp': datetime.utcnow().isoformat(),
            'total_processing_time_ms': total_time,
            'used_bedrock': should_use_bedrock,
            'version': '2.0'
        })
        
        return final_decision
    
    def _create_ml_only_response(self, text: str, ml_score: float) -> Dict:
        """Create response using only ML model"""
        
        if ml_score > 0.7:
            action = 'block'
            confidence = 'high'
            reasoning = 'High toxicity score from ML model'
        elif ml_score > 0.5:
            action = 'review'
            confidence = 'medium'
            reasoning = 'Moderate toxicity score - needs human review'
        else:
            action = 'allow'
            confidence = 'high'
            reasoning = 'Low toxicity score from ML model'
        
        return {
            'contextual_toxicity_score': ml_score,
            'recommended_action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'harm_categories': ['unknown'],
            'bedrock_enhanced': False,
            'ml_toxicity_score': ml_score,
            'score_difference': 0.0
        }
    
    def _combine_analyses(self, ml_score: float, bedrock_analysis: Dict, 
                         risk_indicators: Dict) -> Dict:
        """Combine ML and Bedrock analyses for final decision"""
        
        # Get Bedrock recommendation
        bedrock_score = bedrock_analysis.get('contextual_toxicity_score', ml_score)
        bedrock_action = bedrock_analysis.get('recommended_action', 'review')
        
        # Weight the scores (60% Bedrock, 40% ML for borderline cases)
        if bedrock_analysis.get('bedrock_enhanced', False):
            final_score = 0.6 * bedrock_score + 0.4 * ml_score
        else:
            final_score = ml_score
        
        # Apply risk indicator adjustments
        if risk_indicators.get('excessive_caps', False):
            final_score += 0.05
        if risk_indicators.get('excessive_punctuation', False):
            final_score += 0.03
        
        # Determine final action
        if final_score > 0.8 or bedrock_action == 'block':
            final_action = 'block'
            confidence = 'high'
        elif final_score > 0.5 or bedrock_action == 'review':
            final_action = 'review'
            confidence = bedrock_analysis.get('confidence', 'medium')
        else:
            final_action = 'allow'
            confidence = 'high'
        
        # Handle false positive detection
        if bedrock_analysis.get('is_false_positive', False):
            final_action = 'allow'
            confidence = 'high'
        
        return {
            'final_toxicity_score': round(final_score, 4),
            'final_action': final_action,
            'final_confidence': confidence,
            'ml_score': ml_score,
            'bedrock_score': bedrock_score,
            'bedrock_analysis': bedrock_analysis,
            'risk_indicators': risk_indicators,
            'reasoning': bedrock_analysis.get('reasoning', 'ML-based decision'),
            'harm_categories': bedrock_analysis.get('harm_categories', []),
            'user_explanation': bedrock_analysis.get('explanation_for_user', 
                                                   'Content flagged by automated system')
        }
    
    def _test_bedrock_access(self):
        """Test Bedrock access and find available models"""
        try:
            # First test basic connectivity
            response = self.bedrock_client.list_foundation_models()
            available_model_ids = [model['modelId'] for model in response['modelSummaries']]
            
            print(f"🔍 Found {len(available_model_ids)} total models")
            
            # Test a simple model invocation
            test_models = ['claude-4-sonnet', 'claude-3-haiku']
            working_model = None
            
            for model_name in test_models:
                if model_name in self.available_models:
                    model_id = self.available_models[model_name]
                    try:
                        print(f"🧪 Testing {model_name}...")
                        self._test_model_invocation(model_id, model_name)
                        working_model = model_name
                        print(f"✅ {model_name} is working!")
                        break
                    except Exception as e:
                        print(f"⚠️  {model_name} test failed: {e}")
                        continue
            
            if working_model:
                self.default_model = working_model
                print(f"🎯 Default model set to: {working_model}")
            else:
                raise Exception("No working models found")
                
        except Exception as e:
            raise Exception(f"Bedrock access test failed: {e}")
    
    def _test_model_invocation(self, model_id: str, model_name: str):
        """Test a simple model invocation"""
        test_prompt = "Hello, respond with just 'OK'"
        
        if 'claude' in model_name:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": test_prompt}]
            }
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        response = self.bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        
        # If we get here without exception, the model works
        return True
    
    def _diagnose_bedrock_issue(self, error):
        """Diagnose and provide guidance for Bedrock issues"""
        error_str = str(error).lower()
        
        print(f"\n🔍 Diagnosing Bedrock Issue:")
        print(f"Error: {error}")
        
        if 'accessdenied' in error_str or 'forbidden' in error_str:
            print(f"\n💡 Bedrock Access Issue Detected:")
            print(f"1. Go to AWS Console → Bedrock → Model access")
            print(f"2. Request access to Claude 4 Sonnet and Claude 3 Haiku models")
            print(f"3. Wait for approval (can take a few minutes)")
            print(f"4. Re-run this script")
            
        elif 'validation' in error_str and 'inference profile' in error_str:
            print(f"\n💡 Inference Profile Issue Detected:")
            print(f"1. Use inference profiles instead of direct model IDs")
            print(f"2. Check available inference profiles in AWS Console")
            print(f"3. Update model configurations")
            
        elif 'region' in error_str:
            print(f"\n💡 Region Issue Detected:")
            print(f"1. Try different regions (us-west-2, us-east-1)")
            print(f"2. Check model availability in your region")
            
        else:
            print(f"\n💡 General Bedrock Setup:")
            print(f"1. Ensure Bedrock service is available in your region")
            print(f"2. Check IAM permissions for Bedrock")
            print(f"3. Verify account has Bedrock access enabled")
        
        print(f"\n🔄 System will continue with ML-only mode for now...")

class BedrockModerationIntegration:
    """Integration class for Bedrock-enhanced content moderation"""
    
    def __init__(self):
        self.analyzer = BedrockContentAnalyzer()
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        
        # Load endpoint info
        try:
            with open('endpoint_info.json', 'r') as f:
                endpoint_info = json.load(f)
            self.endpoint_name = endpoint_info['endpoint_name']
        except FileNotFoundError:
            print("⚠️  endpoint_info.json not found. Deploy SageMaker endpoint first.")
            self.endpoint_name = None
    
    def moderate_content_enhanced(self, text: str, use_bedrock: bool = True) -> Dict:
        """
        Complete content moderation with Bedrock enhancement
        
        Args:
            text: Content to moderate
            use_bedrock: Whether to use Bedrock for analysis
            
        Returns:
            Complete moderation result
        """
        
        if not self.endpoint_name:
            raise Exception("SageMaker endpoint not available")
        
        # Step 1: Get ML model prediction
        try:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='text/csv',
                Body=text.strip()
            )
            
            result = json.loads(response['Body'].read().decode())
            ml_score = float(result[0]) if isinstance(result, list) else float(result)
            
        except Exception as e:
            print(f"❌ SageMaker prediction failed: {e}")
            ml_score = 0.5  # Default to uncertain
        
        # Step 2: Extract risk indicators (simplified)
        risk_indicators = {
            'excessive_caps': len([c for c in text if c.isupper()]) > len(text) * 0.5,
            'excessive_punctuation': text.count('!') + text.count('?') > 5,
            'repeated_chars': any(c * 4 in text for c in 'abcdefghijklmnopqrstuvwxyz'),
            'very_short': len(text.strip()) < 10,
            'very_long': len(text.strip()) > 1000
        }
        
        # Step 3: Get enhanced analysis
        enhanced_result = self.analyzer.make_enhanced_moderation_decision(
            text, ml_score, risk_indicators, use_bedrock
        )
        
        return enhanced_result

def test_bedrock_integration():
    """Test the Bedrock integration with sample content"""
    
    print("🧪 Testing Bedrock Integration")
    print("=" * 40)
    
    # Initialize integration
    integration = BedrockModerationIntegration()
    
    # Check model availability
    integration.analyzer.check_model_availability()
    
    # Test cases
    test_cases = [
        {
            "text": "You're such an idiot, I can't believe how stupid you are!",
            "description": "Clear toxic content"
        },
        {
            "text": "That movie was absolutely terrible, what a waste of time.",
            "description": "Negative opinion (potential false positive)"
        },
        {
            "text": "Oh great, another meeting. Just what I needed today... NOT!",
            "description": "Sarcastic but not toxic"
        },
        {
            "text": "I'm going to destroy you in this video game!",
            "description": "Gaming context (ambiguous)"
        },
        {
            "text": "Thank you so much for your help! You're amazing!",
            "description": "Clearly positive content"
        }
    ]
    
    print(f"\n🔍 Testing {len(test_cases)} cases with Bedrock analysis:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Text: \"{test_case['text']}\"")
        
        try:
            # Test with Bedrock
            result = integration.moderate_content_enhanced(test_case['text'], use_bedrock=True)
            
            print(f"✅ ML Score: {result['ml_score']:.3f}")
            print(f"🤖 Bedrock Score: {result['bedrock_score']:.3f}")
            print(f"🎯 Final Score: {result['final_toxicity_score']:.3f}")
            print(f"🚦 Action: {result['final_action']} ({result['final_confidence']} confidence)")
            print(f"💭 Reasoning: {result['reasoning'][:100]}...")
            
            if result['bedrock_analysis'].get('bedrock_enhanced'):
                categories = result['harm_categories']
                print(f"🏷️  Categories: {', '.join(categories) if categories else 'None'}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        print("-" * 60)
    
    print("\n✅ Bedrock integration testing completed!")

def main():
    """Main function for Bedrock setup and testing"""
    
    print("🤖 Amazon Bedrock Integration Setup")
    print("=" * 50)
    
    try:
        # Test Bedrock integration
        test_bedrock_integration()
        
        print("\n" + "=" * 50)
        print("✅ Bedrock Integration Complete!")
        print("=" * 50)
        print("📋 Next steps:")
        print("1. Test integration: python bedrock_integration.py")
        print("2. Set up monitoring: python cloudwatch_monitoring.py")
        print("3. Deploy Step Functions: python step_functions_orchestration.py")
        
    except Exception as e:
        print(f"❌ Bedrock setup failed: {e}")
        print("💡 Make sure you have Bedrock access enabled in your AWS account")
        return False
    
    return True

if __name__ == "__main__":
    main()
