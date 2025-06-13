#!/usr/bin/env python3
"""
AWS Step Functions Orchestration for Content Moderation
======================================================

This module creates Step Functions to orchestrate the complete
content moderation workflow including:
- Preprocessing
- ML prediction
- Bedrock analysis for borderline cases
- Decision making
- Alerting and logging
"""

import boto3
import json
import time
import zipfile
import os
from typing import Dict, List, Optional

# Import existing modules instead of recreating functionality
from bedrock_integration import BedrockContentAnalyzer, BedrockModerationIntegration
from cloudwatch_monitoring import ContentModerationMonitoring, MetricLogger

class ContentModerationStepFunctions:
    """Step Functions orchestration for content moderation workflow using existing modules"""
    
    def __init__(self):
        self.stepfunctions = boto3.client('stepfunctions')
        self.iam = boto3.client('iam')
        self.lambda_client = boto3.client('lambda')
        
        # Load configuration
        with open('aws_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.region = self.config['region']
        self.account_id = self.config['account_id']
        
        # Load component information
        self.endpoint_info = self._load_json_file('endpoint_info.json')
        self.monitoring_config = self._load_json_file('monitoring_config.json', required=False)
        
        # Initialize existing modules instead of creating new ones
        self.bedrock_analyzer = BedrockContentAnalyzer()
        self.monitoring = ContentModerationMonitoring()
        self.metric_logger = MetricLogger()
        
        print(f"🔄 Step Functions Orchestration Setup (Refactored)")
        print(f"🌍 Region: {self.region}")
        print(f"🏗️  Account: {self.account_id}")
        print(f"🤖 Using existing Bedrock integration")
        print(f"📊 Using existing CloudWatch monitoring")
        
        # Verify existing modules
        self._verify_module_integration()
    
    def _load_json_file(self, filename: str, required: bool = True) -> Optional[Dict]:
        """Load JSON configuration file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            if required:
                raise Exception(f"{filename} not found. Deploy components first.")
            return None
    
    def _verify_module_integration(self):
        """Verify that existing modules are properly integrated"""
        print("🔍 Verifying module integration...")
        
        # Test Bedrock integration
        try:
            test_result = self.bedrock_analyzer.make_enhanced_moderation_decision(
                text="test", 
                ml_score=0.5, 
                risk_indicators={}, 
                use_bedrock=False  # Don't actually call Bedrock for verification
            )
            print("✅ Bedrock integration verified")
        except Exception as e:
            print(f"⚠️  Bedrock integration issue: {e}")
        
        # Test CloudWatch monitoring
        try:
            # Test metric logger
            self.metric_logger.log_prediction_metrics(0.5, 'medium', 100, 'review')
            print("✅ CloudWatch monitoring verified")
        except Exception as e:
            print(f"⚠️  CloudWatch monitoring issue: {e}")
    
    def create_step_functions_role(self):
        """Create IAM role for Step Functions execution"""
        
        role_name = 'ContentModerationStepFunctionsRole'
        
        # Trust policy for Step Functions
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "states.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # Permission policy
        permission_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "lambda:InvokeFunction"
                    ],
                    "Resource": [
                        f"arn:aws:lambda:{self.region}:{self.account_id}:function:content-moderation-*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "sagemaker:InvokeEndpoint"
                    ],
                    "Resource": [
                        f"arn:aws:sagemaker:{self.region}:{self.account_id}:endpoint/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel"
                    ],
                    "Resource": [
                        "arn:aws:bedrock:*::foundation-model/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:DescribeLogGroups",
                        "logs:DescribeLogStreams"
                    ],
                    "Resource": [
                        f"arn:aws:logs:{self.region}:{self.account_id}:*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "cloudwatch:PutMetricData"
                    ],
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "sns:Publish"
                    ],
                    "Resource": [
                        f"arn:aws:sns:{self.region}:{self.account_id}:content-moderation-*"
                    ]
                }
            ]
        }
        
        try:
            # Create role
            response = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Role for Content Moderation Step Functions'
            )
            role_arn = response['Role']['Arn']
            
            # Attach permission policy
            self.iam.put_role_policy(
                RoleName=role_name,
                PolicyName='ContentModerationStepFunctionsPolicy',
                PolicyDocument=json.dumps(permission_policy)
            )
            
            print(f"✅ Created Step Functions role: {role_name}")
            print(f"🔐 Role ARN: {role_arn}")
            
            return role_arn
            
        except self.iam.exceptions.EntityAlreadyExistsException:
            # Role already exists, get its ARN
            response = self.iam.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            print(f"📋 Using existing role: {role_name}")
            return role_arn
        except Exception as e:
            print(f"❌ Error creating Step Functions role: {e}")
            return None
    
    def create_content_moderation_state_machine(self, role_arn: str):
        """Create the main content moderation state machine"""
        
        endpoint_name = self.endpoint_info['endpoint_name'] if self.endpoint_info else 'content-moderation-endpoint'
        
        # Define state machine definition
        state_machine_definition = {
            "Comment": "Content Moderation Workflow with ML and Bedrock integration",
            "StartAt": "ValidateInput",
            "States": {
                "ValidateInput": {
                    "Type": "Pass",
                    "Parameters": {
                        "text.$": "$.text",
                        "metadata.$": "$.metadata",
                        "timestamp.$": "$$.State.EnteredTime",
                        "execution_id.$": "$$.Execution.Name"
                    },
                    "Next": "GetPrediction"
                },
                "GetPrediction": {
                    "Type": "Task",
                    "Resource": f"arn:aws:lambda:{self.region}:{self.account_id}:function:content-moderation-prediction",
                    "Parameters": {
                        "text.$": "$.text",
                        "metadata.$": "$.metadata"
                    },
                    "ResultPath": "$.preprocessing_result",
                    "Retry": [
                        {
                            "ErrorEquals": ["States.TaskFailed"],
                            "IntervalSeconds": 2,
                            "MaxAttempts": 3
                        }
                    ],
                    "Catch": [
                        {
                            "ErrorEquals": ["States.ALL"],
                            "Next": "PreprocessingError",
                            "ResultPath": "$.error"
                        }
                    ],
                    "Next": "GetMLPrediction"
                },
                "GetMLPrediction": {
                    "Type": "Task",
                    "Resource": f"arn:aws:lambda:{self.region}:{self.account_id}:function:content-moderation-prediction",
                    "Parameters": {
                        "text.$": "$.preprocessing_result.cleaned_text",
                        "endpoint_name": endpoint_name
                    },
                    "ResultPath": "$.ml_prediction",
                    "Retry": [
                        {
                            "ErrorEquals": ["States.TaskFailed"],
                            "IntervalSeconds": 2,
                            "MaxAttempts": 3
                        }
                    ],
                    "Catch": [
                        {
                            "ErrorEquals": ["States.ALL"],
                            "Next": "MLPredictionError",
                            "ResultPath": "$.error"
                        }
                    ],
                    "Next": "ParseMLResult"
                },
                "ParseMLResult": {
                    "Type": "Pass",
                    "Parameters": {
                        "text.$": "$.text",
                        "preprocessing_result.$": "$.preprocessing_result",
                        "ml_score.$": "$.ml_prediction.toxicity_score",
                        "timestamp.$": "$.timestamp",
                        "execution_id.$": "$.execution_id"
                    },
                    "Next": "EvaluateToxicityScore"
                },
                "EvaluateToxicityScore": {
                    "Type": "Choice",
                    "Choices": [
                        {
                            "Variable": "$.ml_score",
                            "NumericGreaterThan": 0.8,
                            "Next": "HighToxicityDetected"
                        },
                        {
                            "And": [
                                {
                                    "Variable": "$.ml_score",
                                    "NumericGreaterThan": 0.3
                                },
                                {
                                    "Variable": "$.ml_score",
                                    "NumericLessThanEquals": 0.8
                                }
                            ],
                            "Next": "BorderlineCase"
                        }
                    ],
                    "Default": "LowToxicityDetected"
                },
                "BorderlineCase": {
                    "Type": "Task",
                    "Resource": f"arn:aws:lambda:{self.region}:{self.account_id}:function:content-moderation-bedrock-analysis",
                    "Parameters": {
                        "text.$": "$.text",
                        "ml_score.$": "$.ml_score",
                        "preprocessing_result.$": "$.preprocessing_result"
                    },
                    "ResultPath": "$.bedrock_analysis",
                    "Retry": [
                        {
                            "ErrorEquals": ["States.TaskFailed"],
                            "IntervalSeconds": 3,
                            "MaxAttempts": 2
                        }
                    ],
                    "Catch": [
                        {
                            "ErrorEquals": ["States.ALL"],
                            "Next": "BedrockFallback",
                            "ResultPath": "$.bedrock_error"
                        }
                    ],
                    "Next": "MakeEnhancedDecision"
                },
                "BedrockFallback": {
                    "Type": "Pass",
                    "Parameters": {
                        "text.$": "$.text",
                        "ml_score.$": "$.ml_score",
                        "action": "review",
                        "confidence": "medium",
                        "reasoning": "Bedrock analysis failed, defaulting to human review",
                        "bedrock_enhanced": False,
                        "fallback_used": True
                    },
                    "Next": "LogDecision"
                },
                "MakeEnhancedDecision": {
                    "Type": "Choice",
                    "Choices": [
                        {
                            "Variable": "$.bedrock_analysis.recommended_action",
                            "StringEquals": "block",
                            "Next": "HighToxicityDetected"
                        },
                        {
                            "Variable": "$.bedrock_analysis.recommended_action",
                            "StringEquals": "review",
                            "Next": "RequiresReview"
                        }
                    ],
                    "Default": "LowToxicityDetected"
                },
                "HighToxicityDetected": {
                    "Type": "Pass",
                    "Parameters": {
                        "text.$": "$.text",
                        "ml_score.$": "$.ml_score",
                        "action": "block",
                        "confidence": "high",
                        "reasoning": "High toxicity score detected",
                        "bedrock_analysis.$": "$.bedrock_analysis",
                        "final_decision": "BLOCKED"
                    },
                    "Next": "LogDecision"
                },
                "RequiresReview": {
                    "Type": "Pass",
                    "Parameters": {
                        "text.$": "$.text",
                        "ml_score.$": "$.ml_score",
                        "action": "review",
                        "confidence": "medium",
                        "reasoning": "Content requires human review",
                        "bedrock_analysis.$": "$.bedrock_analysis",
                        "final_decision": "REVIEW"
                    },
                    "Next": "LogDecision"
                },
                "LowToxicityDetected": {
                    "Type": "Pass",
                    "Parameters": {
                        "text.$": "$.text",
                        "ml_score.$": "$.ml_score",
                        "action": "allow",
                        "confidence": "high",
                        "reasoning": "Low toxicity score detected",
                        "bedrock_analysis.$": "$.bedrock_analysis",
                        "final_decision": "ALLOWED"
                    },
                    "Next": "LogDecision"
                },
                "LogDecision": {
                    "Type": "Task",
                    "Resource": f"arn:aws:lambda:{self.region}:{self.account_id}:function:content-moderation-logger",
                    "Parameters": {
                        "decision.$": "$",
                        "timestamp.$": "$.timestamp",
                        "execution_id.$": "$.execution_id"
                    },
                    "ResultPath": "$.logging_result",
                    "Retry": [
                        {
                            "ErrorEquals": ["States.TaskFailed"],
                            "IntervalSeconds": 1,
                            "MaxAttempts": 2
                        }
                    ],
                    "Catch": [
                        {
                            "ErrorEquals": ["States.ALL"],
                            "Next": "Success",
                            "ResultPath": "$.logging_error"
                        }
                    ],
                    "Next": "CheckAlerts"
                },
                "CheckAlerts": {
                    "Type": "Choice",
                    "Choices": [
                        {
                            "Variable": "$.final_decision",
                            "StringEquals": "BLOCKED",
                            "Next": "SendAlert"
                        }
                    ],
                    "Default": "Success"
                },
                "SendAlert": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::sns:publish",
                    "Parameters": {
                        "TopicArn": f"arn:aws:sns:{self.region}:{self.account_id}:content-moderation-alerts",
                        "Message.$": "$.reasoning",
                        "Subject": "Content Blocked - High Toxicity Detected"
                    },
                    "Catch": [
                        {
                            "ErrorEquals": ["States.ALL"],
                            "Next": "Success",
                            "ResultPath": "$.alert_error"
                        }
                    ],
                    "Next": "Success"
                },
                "Success": {
                    "Type": "Pass",
                    "Parameters": {
                        "statusCode": 200,
                        "body": {
                            "text.$": "$.text",
                            "action.$": "$.action",
                            "confidence.$": "$.confidence",
                            "ml_score.$": "$.ml_score",
                            "reasoning.$": "$.reasoning",
                            "execution_id.$": "$.execution_id",
                            "bedrock_enhanced.$": "$.bedrock_analysis.bedrock_enhanced"
                        }
                    },
                    "End": True
                },
                "PreprocessingError": {
                    "Type": "Pass",
                    "Parameters": {
                        "statusCode": 500,
                        "error": "Preprocessing failed",
                        "details.$": "$.error"
                    },
                    "End": True
                },
                "MLPredictionError": {
                    "Type": "Pass",
                    "Parameters": {
                        "statusCode": 500,
                        "error": "ML prediction failed",
                        "details.$": "$.error"
                    },
                    "End": True
                }
            }
        }
        
        state_machine_name = f"ContentModerationWorkflow-{int(time.time())}"
        
        try:
            response = self.stepfunctions.create_state_machine(
                name=state_machine_name,
                definition=json.dumps(state_machine_definition),
                roleArn=role_arn,
                type='STANDARD',
                tags=[
                    {
                        'key': 'Environment',
                        'value': 'ContentModeration'
                    },
                    {
                        'key': 'Purpose',
                        'value': 'WorkflowOrchestration'
                    }
                ]
            )
            
            state_machine_arn = response['stateMachineArn']
            
            print(f"✅ Created state machine: {state_machine_name}")
            print(f"🔗 State machine ARN: {state_machine_arn}")
            
            return state_machine_arn, state_machine_name
            
        except Exception as e:
            print(f"❌ Error creating state machine: {e}")
            return None, None
    
    def create_logging_lambda(self):
        """Create Lambda function for logging decisions using existing cloudwatch_monitoring module"""
        
        lambda_client = boto3.client('lambda')
        
        # Create Lambda package with existing cloudwatch_monitoring module
        # Create a zip file with the required modules
        zip_path = '/tmp/logging_lambda.zip'
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            # Add the cloudwatch_monitoring module
            zip_file.write('cloudwatch_monitoring.py', 'cloudwatch_monitoring.py')
            
            # Add Lambda handler code that uses existing MetricLogger
            logging_code = '''
            import json
            import boto3
            import time
            from datetime import datetime
            from cloudwatch_monitoring import MetricLogger

            def lambda_handler(event, context):
                """Log content moderation decisions using existing MetricLogger"""
                
                # Initialize the existing MetricLogger
                metric_logger = MetricLogger()
                
                decision = event.get('decision', {})
                execution_id = event.get('execution_id', 'unknown')
                
                # Log to CloudWatch Logs
                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'execution_id': execution_id,
                    'text_preview': decision.get('text', '')[:100],
                    'ml_score': decision.get('ml_score', 0),
                    'action': decision.get('action', 'unknown'),
                    'confidence': decision.get('confidence', 'unknown'),
                    'reasoning': decision.get('reasoning', ''),
                    'bedrock_enhanced': decision.get('bedrock_analysis', {}).get('bedrock_enhanced', False)
                }
                
                print(f"MODERATION_DECISION: {json.dumps(log_entry)}")
                
                # Use existing MetricLogger for consistent metric logging
                try:
                    ml_score = decision.get('ml_score', 0)
                    confidence = decision.get('confidence', 'medium')
                    action = decision.get('action', 'unknown')
                    latency_ms = decision.get('processing_time_ms', 0)
                    
                    # Use the existing log_prediction_metrics method
                    metric_logger.log_prediction_metrics(
                        toxicity_score=ml_score,
                        confidence=confidence,
                        latency_ms=latency_ms,
                        action=action
                    )
                    
                    # Log Bedrock metrics if enhanced
                    if decision.get('bedrock_analysis', {}).get('bedrock_enhanced', False):
                        bedrock_latency = decision.get('bedrock_analysis', {}).get('processing_time_ms', 0)
                        metric_logger.log_bedrock_metrics(
                            latency_ms=bedrock_latency,
                            success=True
                        )
                
                except Exception as e:
                    print(f"Error logging metrics: {e}")
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'logged': True,
                        'execution_id': execution_id
                    })
                }
            '''
            zip_file.writestr('lambda_function.py', logging_code)
        
        # Read the zip file
        with open(zip_path, 'rb') as zip_file:
            zip_data = zip_file.read()
        
        function_name = 'content-moderation-logger'
        
        try:
            # Create Lambda function with packaged module
            response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=self.config['lambda_role'],
                Handler='lambda_function.lambda_handler',
                Code={
                    'ZipFile': zip_data
                },
                Description='Log content moderation decisions using existing cloudwatch_monitoring module',
                Timeout=30,
                MemorySize=128,
                Environment={
                    'Variables': {
                        'REGION': self.region
                    }
                },
                Tags={
                    'Environment': 'ContentModeration',
                    'Purpose': 'Logging'
                }
            )
            
            function_arn = response['FunctionArn']
            
            print(f"✅ Created logging Lambda with existing module: {function_name}")
            print(f"🔗 Function ARN: {function_arn}")
            
            # Clean up temp file
            os.remove(zip_path)
            
            return function_arn
            
        except lambda_client.exceptions.ResourceConflictException:
            # Function already exists
            response = lambda_client.get_function(FunctionName=function_name)
            function_arn = response['Configuration']['FunctionArn']
            print(f"📋 Using existing logging Lambda: {function_name}")
            return function_arn
        except Exception as e:
            print(f"❌ Error creating logging Lambda: {e}")
            return None
    
    def create_bedrock_analysis_lambda(self):
        """Create Lambda function for Bedrock analysis using existing bedrock_integration module"""
        
        lambda_client = boto3.client('lambda')
        
        # Create Lambda package with existing bedrock_integration module
        
        # Create a zip file with the required modules
        zip_path = '/tmp/bedrock_lambda.zip'
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            # Add the bedrock_integration module
            zip_file.write('bedrock_integration.py', 'bedrock_integration.py')
            zip_file.write('aws_config.json', 'aws_config.json')
            
            # Add Lambda handler code that uses existing BedrockContentAnalyzer
            bedrock_code = '''
            import json
            import boto3
            import time
            from datetime import datetime
            from bedrock_integration import BedrockContentAnalyzer

            def lambda_handler(event, context):
                """Perform Bedrock analysis using existing BedrockContentAnalyzer"""
                
                # Initialize the existing BedrockContentAnalyzer
                try:
                    bedrock_analyzer = BedrockContentAnalyzer()
                except Exception as e:
                    return {
                        'statusCode': 500,
                        'body': json.dumps({
                            'error': f'Failed to initialize Bedrock analyzer: {str(e)}',
                            'bedrock_enhanced': False
                        })
                    }
                
                text = event.get('text', '')
                ml_score = event.get('ml_score', 0.5)
                preprocessing_result = event.get('preprocessing_result', {})
                
                try:
                    start_time = time.time()
                    
                    # Extract risk indicators from preprocessing result
                    risk_indicators = preprocessing_result.get('risk_indicators', {})
                    
                    # Use the existing make_enhanced_moderation_decision method
                    enhanced_result = bedrock_analyzer.make_enhanced_moderation_decision(
                        text=text,
                        ml_score=ml_score,
                        risk_indicators=risk_indicators,
                        use_bedrock=True
                    )
                    
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    # Extract relevant fields for Step Functions
                    result = {
                        'contextual_toxicity_score': enhanced_result.get('final_toxicity_score', ml_score),
                        'recommended_action': enhanced_result.get('final_action', 'review'),
                        'confidence': enhanced_result.get('final_confidence', 'medium'),
                        'reasoning': enhanced_result.get('reasoning', 'Enhanced Bedrock analysis'),
                        'bedrock_enhanced': enhanced_result.get('used_bedrock', True),
                        'processing_time_ms': processing_time,
                        'harm_categories': enhanced_result.get('harm_categories', []),
                        'context_factors': enhanced_result.get('context_factors', [])
                    }
                    
                    return {
                        'statusCode': 200,
                        'body': json.dumps(result)
                    }
                    
                except Exception as e:
                    print(f"Bedrock analysis error: {e}")
                    return {
                        'statusCode': 500,
                        'body': json.dumps({
                            'error': f'Bedrock analysis failed: {str(e)}',
                            'bedrock_enhanced': False,
                            'contextual_toxicity_score': ml_score,
                            'recommended_action': 'review',
                            'confidence': 'low',
                            'reasoning': f'Bedrock analysis failed: {str(e)}'
                        })
                    }
            '''
            zip_file.writestr('lambda_function.py', bedrock_code)
        
        # Read the zip file
        with open(zip_path, 'rb') as zip_file:
            zip_data = zip_file.read()
        
        function_name = 'content-moderation-bedrock-analysis'
        
        try:
            # Create Lambda function with packaged module
            response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=self.config['lambda_role'],
                Handler='lambda_function.lambda_handler',
                Code={
                    'ZipFile': zip_data
                },
                Description='Bedrock analysis using existing bedrock_integration module',
                Timeout=60,
                MemorySize=256,
                Environment={
                    'Variables': {
                        'REGION': self.region
                    }
                },
                Tags={
                    'Environment': 'ContentModeration',
                    'Purpose': 'BedrockAnalysis'
                }
            )
            
            function_arn = response['FunctionArn']
            
            print(f"✅ Created Bedrock analysis Lambda with existing module: {function_name}")
            print(f"🔗 Function ARN: {function_arn}")
            
            # Clean up temp file
            os.remove(zip_path)
            
            return function_arn
            
        except lambda_client.exceptions.ResourceConflictException:
            # Function already exists
            response = lambda_client.get_function(FunctionName=function_name)
            function_arn = response['Configuration']['FunctionArn']
            print(f"📋 Using existing Bedrock analysis Lambda: {function_name}")
            return function_arn
        except Exception as e:
            print(f"❌ Error creating Bedrock analysis Lambda: {e}")
            return None
    
    def create_prediction_lambda(self):
        """Create Lambda function for prediction using existing prediction_lambda module"""
        
        lambda_client = boto3.client('lambda')
        
        # Create Lambda package with existing prediction_lambda module
        zip_path = '/tmp/prediction_lambda.zip'
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            # Add the prediction_lambda module
            zip_file.write('prediction_lambda.py', 'prediction_lambda.py')
            zip_file.write('bedrock_integration.py', 'bedrock_integration.py')
            zip_file.write('aws_config.json', 'aws_config.json')
            zip_file.write('endpoint_info.json', 'endpoint_info.json')
            
            # Add Lambda handler code that uses existing prediction logic
            prediction_code = '''
            import json
            import boto3
            import time
            from datetime import datetime
            from prediction_lambda import lambda_handler as prediction_handler

            def lambda_handler(event, context):
                """Wrapper for existing prediction lambda logic"""
                return prediction_handler(event, context)
            '''
            zip_file.writestr('lambda_function.py', prediction_code)
        
        # Read the zip file
        with open(zip_path, 'rb') as zip_file:
            zip_data = zip_file.read()
        
        function_name = 'content-moderation-prediction'
        
        try:
            # Create Lambda function with packaged module
            response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=self.config['lambda_role'],
                Handler='lambda_function.lambda_handler',
                Code={
                    'ZipFile': zip_data
                },
                Description='Content moderation prediction using existing prediction_lambda module',
                Timeout=30,
                MemorySize=256,
                Environment={
                    'Variables': {
                        'REGION': self.region
                    }
                },
                Tags={
                    'Environment': 'ContentModeration',
                    'Purpose': 'Prediction'
                }
            )
            
            function_arn = response['FunctionArn']
            
            print(f"✅ Created prediction Lambda with existing module: {function_name}")
            print(f"🔗 Function ARN: {function_arn}")
            
            # Clean up temp file
            os.remove(zip_path)
            
            return function_arn
            
        except lambda_client.exceptions.ResourceConflictException:
            # Function already exists
            response = lambda_client.get_function(FunctionName=function_name)
            function_arn = response['Configuration']['FunctionArn']
            print(f"📋 Using existing prediction Lambda: {function_name}")
            return function_arn
        except Exception as e:
            print(f"❌ Error creating prediction Lambda: {e}")
            return None
    
    def test_state_machine(self, state_machine_arn: str):
        """Test the state machine with sample input"""
        
        test_cases = [
            {
                "text": "You are such an idiot, I hate you!",
                "metadata": {"source": "test", "user_id": "test_user_1"}
            },
            {
                "text": "That movie was terrible, what a waste of time.",
                "metadata": {"source": "test", "user_id": "test_user_2"}
            },
            {
                "text": "Thank you for your help!",
                "metadata": {"source": "test", "user_id": "test_user_3"}
            }
        ]
        
        print(f"\n🧪 Testing state machine with {len(test_cases)} cases...")
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                execution_name = f"test-execution-{int(time.time())}-{i}"
                
                response = self.stepfunctions.start_execution(
                    stateMachineArn=state_machine_arn,
                    name=execution_name,
                    input=json.dumps(test_case)
                )
                
                execution_arn = response['executionArn']
                
                print(f"✅ Started test execution {i}: {execution_name}")
                print(f"   Text: \"{test_case['text']}\"")
                print(f"   Execution ARN: {execution_arn}")
                
                # Wait a moment and check status
                time.sleep(2)
                
                status_response = self.stepfunctions.describe_execution(
                    executionArn=execution_arn
                )
                
                status = status_response['status']
                print(f"   Status: {status}")
                
                if status == 'SUCCEEDED':
                    output = json.loads(status_response.get('output', '{}'))
                    if 'body' in output:
                        action = output['body'].get('action', 'unknown')
                        confidence = output['body'].get('confidence', 'unknown')
                        print(f"   Result: {action} ({confidence} confidence)")
                
                print()
                
            except Exception as e:
                print(f"❌ Error testing case {i}: {e}")
    
    def setup_complete_orchestration(self):
        """Set up complete Step Functions orchestration"""
        
        print("🚀 Setting up Step Functions orchestration...")
        
        # Step 1: Create IAM role
        print("\n1. Creating Step Functions IAM role...")
        role_arn = self.create_step_functions_role()
        if not role_arn:
            return None
        
        # Step 2: Create supporting Lambda functions
        print("\n2. Creating prediction Lambda function...")
        prediction_lambda_arn = self.create_prediction_lambda()
        
        print("\n3. Creating logging Lambda function...")
        logging_lambda_arn = self.create_logging_lambda()
        
        print("\n4. Creating Bedrock analysis Lambda function...")
        bedrock_lambda_arn = self.create_bedrock_analysis_lambda()
        
        # Wait for role propagation
        print("\n5. Waiting for IAM role propagation...")
        time.sleep(10)
        
        # Step 6: Create state machine
        print("\n6. Creating content moderation state machine...")
        state_machine_arn, state_machine_name = self.create_content_moderation_state_machine(role_arn)
        
        if not state_machine_arn:
            return None
        
        # Step 7: Test state machine
        print("\n7. Testing state machine...")
        self.test_state_machine(state_machine_arn)
        
        # Save orchestration configuration
        orchestration_config = {
            'state_machine_arn': state_machine_arn,
            'state_machine_name': state_machine_name,
            'role_arn': role_arn,
            'logging_lambda_arn': logging_lambda_arn,
            'bedrock_lambda_arn': bedrock_lambda_arn,
            'prediction_lambda_arn': prediction_lambda_arn,
            'region': self.region,
            'account_id': self.account_id,
            'setup_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('orchestration_config.json', 'w') as f:
            json.dump(orchestration_config, f, indent=2)
        
        return orchestration_config

def main():
    """Main function for Step Functions setup"""
    
    print("🔄 AWS Step Functions Orchestration Setup")
    print("=" * 50)
    
    try:
        # Initialize Step Functions
        sf_setup = ContentModerationStepFunctions()
        
        # Set up complete orchestration
        config = sf_setup.setup_complete_orchestration()
        
        if config:
            print("\n" + "=" * 50)
            print("✅ Step Functions Orchestration Complete!")
            print("=" * 50)
            print(f"🔄 State Machine: {config['state_machine_name']}")
            print(f"🔗 ARN: {config['state_machine_arn']}")
            print(f"🌐 Console: https://{config['region']}.console.aws.amazon.com/states/home?region={config['region']}#/statemachines")
            
            print(f"\n📋 Next steps:")
            print("1. View executions in Step Functions console")
            print("2. Create frontend: python create_demo_frontend.py")
            print("3. Run full system test: python test_complete_system.py")
        else:
            print("❌ Step Functions setup failed")
            return False
        
    except Exception as e:
        print(f"❌ Step Functions setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
