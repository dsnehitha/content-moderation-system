import json
import boto3
from datetime import datetime
import time

def lambda_handler(event, context):
    """
    Get toxicity prediction from SageMaker endpoint and make moderation decision
    """
    
    runtime = boto3.client('sagemaker-runtime')
    lambda_client = boto3.client('lambda')
    
    # Use the specific endpoint name from endpoint_info.json
    endpoint_name = 'content-moderation-endpoint-1749581246'
    
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
        
        # Make moderation decision
        decision = make_moderation_decision(toxicity_score, risk_indicators)
        
        # Prepare response
        response_data = {
            'text': original_text,
            'cleaned_text': text,
            'toxicity_score': round(toxicity_score, 4),
            'is_toxic': decision['is_toxic'],
            'action': decision['action'],
            'confidence': decision['confidence'],
            'threshold_used': decision['threshold_used'],
            'features': features,
            'risk_indicators': risk_indicators,
            'processing_time_ms': processing_time,
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': '1.0',
            'success': True
        }
        
        # Add detailed reasoning for review cases
        if decision['action'] == 'review':
            response_data['review_reason'] = 'Toxicity score in uncertain range - human review recommended'
        elif decision['action'] == 'block':
            response_data['block_reason'] = 'High toxicity score detected'
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_data)
        }
        
    except boto3.client('sagemaker-runtime').exceptions.ModelError as e:
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
        # Log the error (in production, use proper logging)
        print(f"Prediction error: {str(e)}")
        
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
    