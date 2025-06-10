import json
import boto3
import re
import html

def lambda_handler(event, context):
    """
    Preprocess text before sending to content moderation model
    Handles text cleaning, normalization, and basic validation
    """
    
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
        
        # Basic profanity masking for logging (optional)
        # text = re.sub(r'\b(fuck|shit|damn)\b', '***', text, flags=re.IGNORECASE)
        
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
            'contains_numbers': bool(re.search(r'\d', text)),
        }
        return features
    
    try:
        # Parse the event to extract text
        if 'body' in event:
            # API Gateway format
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
            text = body.get('text', '')
            metadata = body.get('metadata', {})
        else:
            # Direct invocation format
            text = event.get('text', '')
            metadata = event.get('metadata', {})
        
        # Validate input
        if not text:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'No text provided for moderation',
                    'success': False
                })
            }
        
        # Check text length limits
        if len(text) > 10000:  # 10KB limit
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Text too long (max 10,000 characters)',
                    'success': False
                })
            }
        
        # Clean the text
        original_text = text
        cleaned_text = clean_text(text)
        
        # Extract features
        features = extract_features(cleaned_text)
        
        # Quick checks for obviously problematic content
        risk_indicators = {
            'too_short': len(cleaned_text.strip()) < 3,
            'excessive_caps': features['has_caps'] and len(text) > 20,
            'excessive_punctuation': features['exclamation_count'] > 5,
            'repeated_chars': features['has_repeated_chars']
        }
        
        response_data = {
            'original_text': original_text,
            'cleaned_text': cleaned_text,
            'features': features,
            'risk_indicators': risk_indicators,
            'metadata': metadata,
            'success': True,
            'preprocessing_version': '1.0'
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_data)
        }
        
    except json.JSONDecodeError as e:
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Invalid JSON in request body: {str(e)}',
                'success': False
            })
        }
        
    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"Preprocessing error: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Preprocessing failed: {str(e)}',
                'success': False
            })
        }
    