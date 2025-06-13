import json
import boto3
from datetime import datetime
import time
import logging
import re
import html
from bedrock_integration import BedrockModerationIntegration

# Initialize CloudWatch client for custom metrics
cloudwatch = boto3.client('cloudwatch')

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Get toxicity prediction from SageMaker endpoint and make moderation decision
    Enhanced with Bedrock integration and CloudWatch monitoring
    """
    
    runtime = boto3.client('sagemaker-runtime')
    lambda_client = boto3.client('lambda')
    
    # Initialize Bedrock integration (reuse existing comprehensive implementation)
    try:
        bedrock_integration = BedrockModerationIntegration()
    except Exception as e:
        logger.warning(f"Failed to initialize Bedrock integration: {e}")
        bedrock_integration = None
    
    # Use the specific endpoint name from endpoint_info.json
    endpoint_name = 'content-moderation-endpoint-1749581246'
    
    def send_cloudwatch_metric(metric_name, value, unit='Count', dimensions=None):
        """Send custom metric to CloudWatch"""
        try:
            cloudwatch.put_metric_data(
                Namespace='ContentModeration',
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Timestamp': datetime.utcnow(),
                        'Dimensions': dimensions or []
                    }
                ]
            )
        except Exception as e:
            logger.warning(f"Failed to send metric {metric_name}: {str(e)}")
    
    def get_enhanced_moderation_decision(text, use_bedrock=True):
        """Use existing comprehensive Bedrock integration for enhanced moderation"""
        try:
            if not bedrock_integration:
                return None
                
            # Use the complete moderate_content_enhanced method from bedrock_integration.py
            enhanced_result = bedrock_integration.moderate_content_enhanced(
                text=text,
                use_bedrock=use_bedrock
            )
            
            send_cloudwatch_metric('BedrockAnalysisSuccess', 1)
            
            # Return the complete enhanced result
            return enhanced_result
                
        except Exception as e:
            logger.warning(f"Bedrock enhanced analysis failed: {str(e)}")
            send_cloudwatch_metric('BedrockAnalysisError', 1)
            return None
    
    def preprocess_text(text, metadata=None):
        """Preprocess text for content moderation (inline preprocessing)"""
        
        def clean_text(text):
            """Comprehensive text cleaning for content moderation"""
            if not text or not isinstance(text, str):
                return ""
            
            # Decode HTML entities
            text = html.unescape(text)
            
            # Remove URLs (various formats)
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove excessive punctuation (but keep some for context)
            text = re.sub(r'[.]{3,}', '...', text)  # Normalize ellipsis
            text = re.sub(r'[!]{2,}', '!!', text)   # Normalize exclamations
            text = re.sub(r'[?]{2,}', '??', text)   # Normalize questions
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Trim and ensure we have content
            text = text.strip()
            
            return text
        
        def extract_features(text):
            """Extract features that might be useful for moderation"""
            features = {
                'length': len(text),
                'word_count': len(text.split()),
                'has_caps': bool(re.search(r'[A-Z]{3,}', text)),  # Excessive caps
                'has_repeated_chars': bool(re.search(r'(.)\1{3,}', text)),  # aaaa
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'has_numbers': bool(re.search(r'\d', text)),
                'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1)
            }
            return features
        
        def get_risk_indicators(text, features):
            """Get risk indicators based on text patterns"""
            risk_score = 0
            indicators = []
            
            # Length-based risks
            if features['length'] > 1000:
                risk_score += 1
                indicators.append('very_long_text')
            elif features['length'] < 5:
                risk_score += 1
                indicators.append('very_short_text')
            
            # Pattern-based risks
            if features['has_caps']:
                risk_score += 2
                indicators.append('excessive_caps')
            
            if features['has_repeated_chars']:
                risk_score += 1
                indicators.append('repeated_characters')
            
            if features['exclamation_count'] > 3:
                risk_score += 1
                indicators.append('excessive_exclamations')
            
            # Content patterns
            if re.search(r'\b(kill|die|hate|stupid|idiot)\b', text, re.IGNORECASE):
                risk_score += 3
                indicators.append('potential_offensive_language')
            
            return {
                'risk_score': risk_score,
                'indicators': indicators,
                'needs_human_review': risk_score > 5
            }
        
        try:
            original_text = text
            cleaned_text = clean_text(text)
            
            if not cleaned_text:
                return {
                    'success': False,
                    'error': 'Text is empty after cleaning'
                }
            
            features = extract_features(cleaned_text)
            risk_indicators = get_risk_indicators(cleaned_text, features)
            
            return {
                'success': True,
                'cleaned_text': cleaned_text,
                'original_text': original_text,
                'features': features,
                'risk_indicators': risk_indicators
            }
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return {
                'success': False,
                'error': f'Preprocessing failed: {str(e)}'
            }
    
    def make_moderation_decision(toxicity_score, features=None):
        """
        Make moderation decision based on toxicity score and additional features
        """
        # Base threshold
        base_threshold = 0.5
        
        # Adjust threshold based on features
        if features:
            # Lower threshold for content with risk indicators
            if features.get('excessive_caps', False):
                base_threshold -= 0.1
            if features.get('excessive_punctuation', False):
                base_threshold -= 0.05
            if features.get('repeated_chars', False):
                base_threshold -= 0.05
        
        # Ensure threshold stays within reasonable bounds
        adjusted_threshold = max(0.3, min(0.8, base_threshold))
        
        is_toxic = toxicity_score > adjusted_threshold
        
        # Determine action
        if toxicity_score > 0.8:
            action = 'block'
            confidence = 'high'
        elif toxicity_score > adjusted_threshold:
            action = 'review'  # Human review needed
            confidence = 'medium'
        else:
            action = 'allow'
            confidence = 'high' if toxicity_score < 0.2 else 'medium'
        
        return {
            'is_toxic': is_toxic,
            'action': action,
            'confidence': confidence,
            'threshold_used': adjusted_threshold
        }
    
    try:
        # Extract text from event
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        else:
            body = event
        
        # Get raw text
        raw_text = body.get('text', '')
        
        if not raw_text:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'No text provided for prediction',
                    'success': False
                })
            }
        
        # Preprocess text inline (no external Lambda needed)
        preprocess_result = preprocess_text(raw_text, body.get('metadata'))
        
        if not preprocess_result['success']:
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': f'Preprocessing failed: {preprocess_result.get("error", "Unknown error")}',
                    'success': False
                })
            }
        
        text = preprocess_result['cleaned_text']
        original_text = preprocess_result['original_text']
        features = preprocess_result['features']
        risk_indicators = preprocess_result['risk_indicators']
        
        if not text:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'No text provided for prediction',
                    'success': False
                })
            }
        
        # Handle short or empty text
        if len(text.strip()) < 3:
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'text': original_text,
                    'cleaned_text': text,
                    'toxicity_score': 0.0,
                    'is_toxic': False,
                    'action': 'allow',
                    'confidence': 'high',
                    'reason': 'Text too short for meaningful analysis',
                    'processing_time_ms': 0,
                    'success': True
                })
            }
        
        start_time = time.time()
        
        # Send request start metric
        send_cloudwatch_metric('PredictionRequestStart', 1)
        
        # Call SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=text
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        # Handle different response formats
        if isinstance(result, list):
            toxicity_score = float(result[0])
        elif isinstance(result, dict) and 'predictions' in result:
            toxicity_score = float(result['predictions'][0])
        else:
            toxicity_score = float(result)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Send performance metrics
        send_cloudwatch_metric('PredictionLatency', processing_time, 'Milliseconds')
        send_cloudwatch_metric('ToxicityScore', toxicity_score, 'None')
        
        # Use comprehensive Bedrock integration for enhanced moderation decision
        bedrock_start_time = time.time()
        enhanced_result = get_enhanced_moderation_decision(text, use_bedrock=True)
        bedrock_time = int((time.time() - bedrock_start_time) * 1000)
        
        if enhanced_result:
            send_cloudwatch_metric('BedrockLatency', bedrock_time, 'Milliseconds')
            
            # Use enhanced result directly (it already combines ML and Bedrock)
            final_decision = {
                'is_toxic': enhanced_result.get('final_toxicity_score', toxicity_score) > 0.5,
                'action': enhanced_result.get('final_action', 'review'),
                'confidence': enhanced_result.get('final_confidence', 'medium'),
                'threshold_used': 0.5,
                'combined_score': enhanced_result.get('final_toxicity_score', toxicity_score)
            }
        else:
            # Fallback to simple ML-based decision
            final_decision = make_moderation_decision(toxicity_score, risk_indicators)
        
        # Send decision metrics
        send_cloudwatch_metric('ModerationDecision', 1, 'Count', [
            {'Name': 'Action', 'Value': final_decision['action']},
            {'Name': 'Confidence', 'Value': final_decision['confidence']}
        ])
        
        # Prepare enhanced response
        response_data = {
            'text': original_text,
            'cleaned_text': text,
            'toxicity_score': round(toxicity_score, 4),
            'is_toxic': final_decision['is_toxic'],
            'action': final_decision['action'],
            'confidence': final_decision['confidence'],
            'threshold_used': final_decision['threshold_used'],
            'features': features,
            'risk_indicators': risk_indicators,
            'processing_time_ms': processing_time,
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': '2.0',  # Updated for Day 2
            'success': True
        }
        
        # Add Bedrock analysis if available
        if enhanced_result:
            bedrock_analysis_summary = enhanced_result.get('bedrock_analysis', {})
            response_data['bedrock_analysis'] = {
                'enhanced_score': enhanced_result.get('final_toxicity_score'),
                'ml_score': enhanced_result.get('ml_score'),
                'bedrock_score': enhanced_result.get('bedrock_score'),
                'concerns': enhanced_result.get('harm_categories', []),
                'explanation': enhanced_result.get('reasoning', ''),
                'confidence': enhanced_result.get('final_confidence', 'medium'),
                'processing_time_ms': bedrock_time,
                'used_bedrock': enhanced_result.get('used_bedrock', False)
            }
            response_data['combined_score'] = final_decision.get('combined_score')
        
        # Add detailed reasoning
        if final_decision['action'] == 'review':
            response_data['review_reason'] = 'Content requires human review based on enhanced analysis'
            if enhanced_result and enhanced_result.get('harm_categories'):
                concerns = enhanced_result.get('harm_categories', [])
                if concerns and concerns != ['unknown']:
                    response_data['review_reason'] += f" (Potential concerns: {', '.join(concerns)})"
        elif final_decision['action'] == 'block':
            response_data['block_reason'] = 'Content blocked based on enhanced toxicity analysis'
            if enhanced_result and enhanced_result.get('reasoning'):
                response_data['block_reason'] += f" (Analysis: {enhanced_result['reasoning'][:100]}...)"
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_data)
        }
        
    except boto3.client('sagemaker-runtime').exceptions.ModelError as e:
        send_cloudwatch_metric('ModelError', 1)
        logger.error(f"Model prediction error: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Model prediction error: {str(e)}',
                'success': False
            })
        }
        
    except Exception as e:
        send_cloudwatch_metric('PredictionError', 1)
        logger.error(f"Prediction error: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Prediction failed: {str(e)}',
                'success': False
            })
        }
    