#!/usr/bin/env python3
"""
CloudWatch Monitoring Setup for Content Moderation System
=========================================================

🎯 PURPOSE: Set up continuous production monitoring infrastructure

📊 CREATES:
- CloudWatch dashboards for system metrics
- Custom metrics for model performance
- Automated alerts for performance degradation
- SNS notifications for critical issues
- Log groups for centralized logging

⏱️  DURATION: Continuous (24/7 production monitoring)

🚀 USAGE:
    # Set up monitoring infrastructure (one-time)
    python cloudwatch_monitoring.py
    
    # View dashboards at:
    # https://console.aws.amazon.com/cloudwatch/home#dashboards:

💡 NOTE: Use monitor_pipeline.py for training pipeline monitoring
"""

import boto3
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class ContentModerationMonitoring:
    """CloudWatch monitoring setup for content moderation system"""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.logs = boto3.client('logs')
        self.sns = boto3.client('sns')
        
        # Load configuration
        with open('aws_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.region = self.config['region']
        self.account_id = self.config['account_id']
        
        # Load endpoint and API info
        self.endpoint_info = self._load_json_file('endpoint_info.json')
        self.api_info = self._load_json_file('api_info.json', required=False)
        
        self.namespace = 'ContentModeration'
        
        print(f"📊 CloudWatch Monitoring Setup")
        print(f"🌍 Region: {self.region}")
        print(f"📈 Namespace: {self.namespace}")
    
    def _load_json_file(self, filename: str, required: bool = True) -> Optional[Dict]:
        """Load JSON configuration file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            if required:
                raise Exception(f"{filename} not found. Deploy components first.")
            return None
    
    def create_log_groups(self):
        """Create CloudWatch log groups for different components"""
        
        log_groups = [
            {
                'name': '/aws/lambda/content-moderation-prediction',
                'retention': 30
            },
            {
                'name': '/content-moderation/model-performance',
                'retention': 90
            },
            {
                'name': '/content-moderation/api-access',
                'retention': 30
            },
            {
                'name': '/content-moderation/alerts',
                'retention': 7
            }
        ]
        
        created_groups = []
        
        for log_group in log_groups:
            try:
                self.logs.create_log_group(
                    logGroupName=log_group['name'],
                    tags={
                        'Environment': 'ContentModeration',
                        'Component': 'Monitoring',
                        'Purpose': 'SystemLogs'
                    }
                )
                
                # Set retention policy
                self.logs.put_retention_policy(
                    logGroupName=log_group['name'],
                    retentionInDays=log_group['retention']
                )
                
                print(f"✅ Created log group: {log_group['name']}")
                created_groups.append(log_group['name'])
                
            except self.logs.exceptions.ResourceAlreadyExistsException:
                print(f"📋 Log group already exists: {log_group['name']}")
                created_groups.append(log_group['name'])
            except Exception as e:
                print(f"❌ Error creating log group {log_group['name']}: {e}")
        
        return created_groups
    
    def create_custom_metrics(self):
        """Create custom CloudWatch metrics for model performance"""
        
        # Define custom metrics
        metrics = [
            {
                'MetricName': 'ToxicityPredictionLatency',
                'Unit': 'Milliseconds',
                'Value': 0
            },
            {
                'MetricName': 'ToxicityPredictionCount',
                'Unit': 'Count',
                'Value': 0
            },
            {
                'MetricName': 'HighConfidencePredictions',
                'Unit': 'Percent',
                'Value': 0
            },
            {
                'MetricName': 'FalsePositiveRate',
                'Unit': 'Percent',
                'Value': 0
            },
            {
                'MetricName': 'ContentBlockedCount',
                'Unit': 'Count',
                'Value': 0
            },
            {
                'MetricName': 'ContentReviewCount',
                'Unit': 'Count',
                'Value': 0
            },
            {
                'MetricName': 'BedrockAnalysisCount',
                'Unit': 'Count',
                'Value': 0
            },
            {
                'MetricName': 'BedrockLatency',
                'Unit': 'Milliseconds',
                'Value': 0
            }
        ]
        
        # Put sample metrics to create them
        for metric in metrics:
            try:
                self.cloudwatch.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=[
                        {
                            'MetricName': metric['MetricName'],
                            'Value': metric['Value'],
                            'Unit': metric['Unit'],
                            'Timestamp': datetime.utcnow(),
                            'Dimensions': [
                                {
                                    'Name': 'Environment',
                                    'Value': 'Production'
                                }
                            ]
                        }
                    ]
                )
                print(f"✅ Created metric: {metric['MetricName']}")
                
            except Exception as e:
                print(f"❌ Error creating metric {metric['MetricName']}: {e}")
        
        return [m['MetricName'] for m in metrics]
    
    def create_dashboard(self):
        """Create CloudWatch dashboard for monitoring"""
        
        # Get endpoint name for SageMaker metrics
        endpoint_name = self.endpoint_info['endpoint_name'] if self.endpoint_info else 'content-moderation-endpoint'
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [self.namespace, "ToxicityPredictionCount", "Environment", "Production"],
                            [self.namespace, "ContentBlockedCount", "Environment", "Production"],
                            [self.namespace, "ContentReviewCount", "Environment", "Production"]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": self.region,
                        "title": "Content Moderation Activity",
                        "yAxis": {
                            "left": {
                                "min": 0
                            }
                        }
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [self.namespace, "ToxicityPredictionLatency", "Environment", "Production"],
                            [self.namespace, "BedrockLatency", "Environment", "Production"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region,
                        "title": "Response Latency (ms)",
                        "yAxis": {
                            "left": {
                                "min": 0
                            }
                        }
                    }
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "Invocations", "EndpointName", endpoint_name],
                            ["AWS/SageMaker", "InvocationErrors", "EndpointName", endpoint_name]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": self.region,
                        "title": "SageMaker Endpoint Metrics"
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [self.namespace, "HighConfidencePredictions", "Environment", "Production"],
                            [self.namespace, "FalsePositiveRate", "Environment", "Production"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region,
                        "title": "Model Performance Metrics (%)"
                    }
                }
            ]
        }
        
        # Add API Gateway metrics if available
        if self.api_info:
            api_id = self.api_info.get('api_id')
            if api_id:
                dashboard_body["widgets"].append({
                    "type": "metric",
                    "x": 0,
                    "y": 12,
                    "width": 24,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/ApiGateway", "Count", "ApiName", "content-moderation-api"],
                            ["AWS/ApiGateway", "Latency", "ApiName", "content-moderation-api"],
                            ["AWS/ApiGateway", "4XXError", "ApiName", "content-moderation-api"],
                            ["AWS/ApiGateway", "5XXError", "ApiName", "content-moderation-api"]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": self.region,
                        "title": "API Gateway Metrics"
                    }
                })
        
        # Create dashboard
        dashboard_name = "ContentModerationDashboard"
        
        try:
            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            
            dashboard_url = f"https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={dashboard_name}"
            
            print(f"✅ Created dashboard: {dashboard_name}")
            print(f"🌐 Dashboard URL: {dashboard_url}")
            
            return dashboard_name, dashboard_url
            
        except Exception as e:
            print(f"❌ Error creating dashboard: {e}")
            return None, None
    
    def create_sns_topic(self):
        """Create SNS topic for alerts"""
        
        topic_name = "content-moderation-alerts"
        
        try:
            response = self.sns.create_topic(
                Name=topic_name,
                Tags=[
                    {
                        'Key': 'Environment',
                        'Value': 'ContentModeration'
                    },
                    {
                        'Key': 'Purpose',
                        'Value': 'Alerts'
                    }
                ]
            )
            
            topic_arn = response['TopicArn']
            
            print(f"✅ Created SNS topic: {topic_name}")
            print(f"📧 Topic ARN: {topic_arn}")
            
            return topic_arn
            
        except Exception as e:
            print(f"❌ Error creating SNS topic: {e}")
            return None
    
    def create_alarms(self, topic_arn: str):
        """Create CloudWatch alarms for key metrics"""
        
        endpoint_name = self.endpoint_info['endpoint_name'] if self.endpoint_info else 'content-moderation-endpoint'
        
        alarms = [
            {
                'AlarmName': 'ContentModeration-HighErrorRate',
                'MetricName': 'InvocationErrors',
                'Namespace': 'AWS/SageMaker',
                'Statistic': 'Sum',
                'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}],
                'Threshold': 10,
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 2,
                'Period': 300,
                'Description': 'High error rate on SageMaker endpoint'
            },
            {
                'AlarmName': 'ContentModeration-HighLatency',
                'MetricName': 'ToxicityPredictionLatency',
                'Namespace': self.namespace,
                'Statistic': 'Average',
                'Dimensions': [{'Name': 'Environment', 'Value': 'Production'}],
                'Threshold': 2000,  # 2 seconds
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 3,
                'Period': 300,
                'Description': 'High prediction latency detected'
            },
            {
                'AlarmName': 'ContentModeration-LowConfidence',
                'MetricName': 'HighConfidencePredictions',
                'Namespace': self.namespace,
                'Statistic': 'Average',
                'Dimensions': [{'Name': 'Environment', 'Value': 'Production'}],
                'Threshold': 70,  # Less than 70% high confidence
                'ComparisonOperator': 'LessThanThreshold',
                'EvaluationPeriods': 5,
                'Period': 300,
                'Description': 'Model confidence degradation detected'
            },
            {
                'AlarmName': 'ContentModeration-HighFalsePositiveRate',
                'MetricName': 'FalsePositiveRate',
                'Namespace': self.namespace,
                'Statistic': 'Average',
                'Dimensions': [{'Name': 'Environment', 'Value': 'Production'}],
                'Threshold': 20,  # More than 20% false positives
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 5,
                'Period': 900,  # 15 minutes
                'Description': 'High false positive rate detected'
            }
        ]
        
        created_alarms = []
        
        for alarm in alarms:
            try:
                self.cloudwatch.put_metric_alarm(
                    AlarmName=alarm['AlarmName'],
                    ComparisonOperator=alarm['ComparisonOperator'],
                    EvaluationPeriods=alarm['EvaluationPeriods'],
                    MetricName=alarm['MetricName'],
                    Namespace=alarm['Namespace'],
                    Period=alarm['Period'],
                    Statistic=alarm['Statistic'],
                    Threshold=alarm['Threshold'],
                    ActionsEnabled=True,
                    AlarmActions=[topic_arn],
                    AlarmDescription=alarm['Description'],
                    Dimensions=alarm['Dimensions'],
                    Unit='None'
                )
                
                print(f"✅ Created alarm: {alarm['AlarmName']}")
                created_alarms.append(alarm['AlarmName'])
                
            except Exception as e:
                print(f"❌ Error creating alarm {alarm['AlarmName']}: {e}")
        
        return created_alarms
    
    def create_metric_filters(self):
        """Create CloudWatch metric filters for log analysis"""
        
        filters = [
            {
                'log_group': '/aws/lambda/content-moderation-prediction',
                'filter_name': 'ToxicContentDetected',
                'filter_pattern': '[timestamp, request_id, level="INFO", message="TOXIC*"]',
                'metric_name': 'ToxicContentDetected',
                'metric_namespace': self.namespace,
                'metric_value': '1'
            },
            {
                'log_group': '/aws/lambda/content-moderation-prediction',
                'filter_name': 'PredictionErrors',
                'filter_pattern': '[timestamp, request_id, level="ERROR", ...]',
                'metric_name': 'PredictionErrors',
                'metric_namespace': self.namespace,
                'metric_value': '1'
            }
        ]
        
        created_filters = []
        
        for filter_config in filters:
            try:
                self.logs.put_metric_filter(
                    logGroupName=filter_config['log_group'],
                    filterName=filter_config['filter_name'],
                    filterPattern=filter_config['filter_pattern'],
                    metricTransformations=[
                        {
                            'metricName': filter_config['metric_name'],
                            'metricNamespace': filter_config['metric_namespace'],
                            'metricValue': filter_config['metric_value'],
                            'defaultValue': 0
                        }
                    ]
                )
                
                print(f"✅ Created metric filter: {filter_config['filter_name']}")
                created_filters.append(filter_config['filter_name'])
                
            except Exception as e:
                # Log group might not exist yet
                print(f"⚠️  Could not create filter {filter_config['filter_name']}: {e}")
        
        return created_filters
    
    def setup_complete_monitoring(self):
        """Set up complete monitoring infrastructure"""
        
        print("🚀 Setting up CloudWatch monitoring...")
        
        # Step 1: Create log groups
        print("\n1. Creating log groups...")
        log_groups = self.create_log_groups()
        
        # Step 2: Create custom metrics
        print("\n2. Creating custom metrics...")
        metrics = self.create_custom_metrics()
        
        # Step 3: Create SNS topic for alerts
        print("\n3. Creating SNS topic for alerts...")
        topic_arn = self.create_sns_topic()
        
        # Step 4: Create alarms
        if topic_arn:
            print("\n4. Creating CloudWatch alarms...")
            alarms = self.create_alarms(topic_arn)
        else:
            alarms = []
        
        # Step 5: Create dashboard
        print("\n5. Creating CloudWatch dashboard...")
        dashboard_name, dashboard_url = self.create_dashboard()
        
        # Step 6: Create metric filters
        print("\n6. Creating metric filters...")
        filters = self.create_metric_filters()
        
        # Save monitoring configuration
        monitoring_config = {
            'log_groups': log_groups,
            'custom_metrics': metrics,
            'sns_topic_arn': topic_arn,
            'alarms': alarms,
            'dashboard_name': dashboard_name,
            'dashboard_url': dashboard_url,
            'metric_filters': filters,
            'namespace': self.namespace,
            'setup_time': datetime.utcnow().isoformat()
        }
        
        with open('monitoring_config.json', 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        return monitoring_config

class MetricLogger:
    """Helper class for logging custom metrics from application code"""
    
    def __init__(self, namespace: str = 'ContentModeration'):
        self.cloudwatch = boto3.client('cloudwatch')
        self.namespace = namespace
    
    def log_prediction_metrics(self, toxicity_score: float, confidence: str, 
                             latency_ms: int, action: str):
        """Log metrics for a content moderation prediction"""
        
        metrics = [
            {
                'MetricName': 'ToxicityPredictionCount',
                'Value': 1,
                'Unit': 'Count'
            },
            {
                'MetricName': 'ToxicityPredictionLatency',
                'Value': latency_ms,
                'Unit': 'Milliseconds'
            }
        ]
        
        # Add confidence metric
        if confidence == 'high':
            metrics.append({
                'MetricName': 'HighConfidencePredictions',
                'Value': 1,
                'Unit': 'Count'
            })
        
        # Add action metrics
        if action == 'block':
            metrics.append({
                'MetricName': 'ContentBlockedCount',
                'Value': 1,
                'Unit': 'Count'
            })
        elif action == 'review':
            metrics.append({
                'MetricName': 'ContentReviewCount',
                'Value': 1,
                'Unit': 'Count'
            })
        
        # Send metrics to CloudWatch
        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        **metric,
                        'Timestamp': datetime.utcnow(),
                        'Dimensions': [
                            {
                                'Name': 'Environment',
                                'Value': 'Production'
                            }
                        ]
                    }
                    for metric in metrics
                ]
            )
        except Exception as e:
            print(f"⚠️  Failed to log metrics: {e}")
    
    def log_bedrock_metrics(self, latency_ms: int, success: bool):
        """Log Bedrock analysis metrics"""
        
        metrics = [
            {
                'MetricName': 'BedrockAnalysisCount',
                'Value': 1,
                'Unit': 'Count'
            }
        ]
        
        if success:
            metrics.append({
                'MetricName': 'BedrockLatency',
                'Value': latency_ms,
                'Unit': 'Milliseconds'
            })
        
        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        **metric,
                        'Timestamp': datetime.utcnow(),
                        'Dimensions': [
                            {
                                'Name': 'Environment',
                                'Value': 'Production'
                            }
                        ]
                    }
                    for metric in metrics
                ]
            )
        except Exception as e:
            print(f"⚠️  Failed to log Bedrock metrics: {e}")

def main():
    """Main function for monitoring setup"""
    
    print("📊 CloudWatch Monitoring Setup for Content Moderation")
    print("=" * 60)
    
    try:
        # Initialize monitoring
        monitoring = ContentModerationMonitoring()
        
        # Set up complete monitoring
        config = monitoring.setup_complete_monitoring()
        
        print("\n" + "=" * 60)
        print("✅ CloudWatch Monitoring Setup Complete!")
        print("=" * 60)
        
        if config['dashboard_url']:
            print(f"🌐 Dashboard: {config['dashboard_url']}")
        
        if config['sns_topic_arn']:
            print(f"📧 SNS Topic: {config['sns_topic_arn']}")
            print("💡 Subscribe to SNS topic for email alerts:")
            print(f"   aws sns subscribe --topic-arn {config['sns_topic_arn']} --protocol email --notification-endpoint your-email@example.com")
        
        print(f"\n📋 Next steps:")
        print("1. Subscribe to SNS topic for alerts")
        print("3. Deploy Step Functions: python step_functions_orchestration.py")
        
    except Exception as e:
        print(f"❌ Monitoring setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
