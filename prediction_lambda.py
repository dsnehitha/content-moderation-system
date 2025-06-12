import json
import boto3
from datetime import datetime
import time
import logging
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
    
    def call_preprocessing_lambda(text, metadata=None):
        """Call the preprocessing Lambda function"""
        try:
            payload = {
                'text': text,
                'metadata': metadata or {}
            }
            
            response = lambda_client.invoke(
                FunctionName='content-moderation-preprocessing',
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            result = json.loads(response['Payload'].read().decode())
            
            if result.get('statusCode') == 200:
                body = json.loads(result['body'])
                if body.get('success'):
                    return {
                        'success': True,
                        'cleaned_text': body.get('cleaned_text', ''),
                        'original_text': body.get('original_text', text),
                        'features': body.get('features', {}),
                        'risk_indicators': body.get('risk_indicators', {})
                    }
            
            return {'success': False, 'error': 'Preprocessing failed'}
            
        except Exception as e:
            return {'success': False, 'error': f'Preprocessing error: {str(e)}'}
    
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
        
        # Always call preprocessing Lambda to ensure consistent processing
        preprocess_result = call_preprocessing_lambda(raw_text, body.get('metadata'))
        
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
    