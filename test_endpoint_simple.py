#!/usr/bin/env python3
"""
Simple endpoint test to debug the inference issue
"""

import boto3
import json

# Load config
with open('aws_config.json', 'r') as f:
    config = json.load(f)

# Load endpoint info
with open('endpoint_info.json', 'r') as f:
    endpoint_info = json.load(f)

endpoint_name = endpoint_info['endpoint_name']
print(f"Testing endpoint: {endpoint_name}")

# Create SageMaker runtime client
runtime_client = boto3.client('sagemaker-runtime', region_name=config['region'])

# Test different input formats
test_cases = [
    {
        'name': 'Simple string',
        'data': 'hello world',
        'content_type': 'text/csv'
    },
    {
        'name': 'JSON format',
        'data': json.dumps({'text': 'hello world'}),
        'content_type': 'application/json'
    },
    {
        'name': 'JSON list',
        'data': json.dumps(['hello world']),
        'content_type': 'application/json'
    }
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test_case['name']}")
    print(f"Data: {test_case['data']}")
    print(f"Content-Type: {test_case['content_type']}")
    
    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=test_case['content_type'],
            Body=test_case['data']
        )
        
        result = response['Body'].read()
        print(f"✅ Success: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
