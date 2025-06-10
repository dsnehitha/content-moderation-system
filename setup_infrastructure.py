#!/usr/bin/env python3
"""
AWS Infrastructure Setup for Content Moderation System
Creates S3 buckets, IAM roles, and necessary permissions
"""

import boto3
import json
import time
from botocore.exceptions import ClientError

class InfrastructureSetup:
    def __init__(self):
        self.sts = boto3.client('sts')
        self.iam = boto3.client('iam')
        self.s3 = boto3.client('s3')
        
        # Get account ID
        self.account_id = self.sts.get_caller_identity()['Account']
        self.region = boto3.Session().region_name or 'us-east-1'
        
        print(f"Setting up infrastructure for account: {self.account_id}")
        print(f"Region: {self.region}")
    
    def create_s3_buckets(self):
        """Create S3 buckets for the content moderation system"""
        bucket_configs = [
            {
                'name': 'content-moderation-system-datastore',
                'folders': ['data/raw/', 'data/processed/', 'models/', 'logs/']
            },
            {
                'name': 'content-moderation-lambda-code',
                'folders': ['preprocessing/', 'prediction/', 'deployment-packages/']
            }
        ]
        
        created_buckets = []
        
        for bucket_config in bucket_configs:
            bucket_name = f"{bucket_config['name']}-{self.account_id}"
            
            try:
                # Create bucket
                if self.region == 'us-east-1':
                    self.s3.create_bucket(Bucket=bucket_name)
                else:
                    self.s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                
                print(f"✅ Created S3 bucket: {bucket_name}")
                
                # Create folders by uploading empty objects
                for folder in bucket_config['folders']:
                    self.s3.put_object(Bucket=bucket_name, Key=folder, Body='')
                
                created_buckets.append(bucket_name)
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                    print(f"✅ S3 bucket already exists: {bucket_name}")
                    created_buckets.append(bucket_name)
                else:
                    print(f"❌ Error creating bucket {bucket_name}: {e}")
        
        return created_buckets
    
    def create_sagemaker_execution_role(self):
        """Create IAM role for SageMaker execution"""
        role_name = 'SageMakerExecutionRole'
        
        # Trust policy for SageMaker
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "sagemaker.amazonaws.com"
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
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::content-moderation-*",
                        f"arn:aws:s3:::content-moderation-*/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:DescribeLogStreams"
                    ],
                    "Resource": "arn:aws:logs:*:*:*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "ecr:GetAuthorizationToken",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        try:
            # Create role
            role_response = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Execution role for SageMaker content moderation'
            )
            
            print(f"✅ Created IAM role: {role_name}")
            
            # Attach AWS managed policy
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
            
            # Create and attach custom policy
            policy_name = f'{role_name}CustomPolicy'
            self.iam.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(permission_policy),
                Description='Custom permissions for content moderation SageMaker role'
            )
            
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=f'arn:aws:iam::{self.account_id}:policy/{policy_name}'
            )
            
            role_arn = role_response['Role']['Arn']
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                print(f"✅ IAM role already exists: {role_name}")
                role_arn = f'arn:aws:iam::{self.account_id}:role/{role_name}'
            else:
                print(f"❌ Error creating IAM role: {e}")
                return None
        
        return role_arn
    
    def create_lambda_execution_role(self):
        """Create IAM role for Lambda execution"""
        role_name = 'ContentModerationLambdaRole'
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        permission_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "sagemaker:InvokeEndpoint"
                    ],
                    "Resource": f"arn:aws:sagemaker:{self.region}:{self.account_id}:endpoint/*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::content-moderation-*/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "arn:aws:logs:*:*:*"
                }
            ]
        }
        

        try:
            role_response = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Execution role for content moderation Lambda functions'
            )
            
            print(f"✅ Created Lambda IAM role: {role_name}")
            
            # Attach AWS managed policy for basic Lambda execution
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AWSLambdaBasicExecutionRole'
            )
            
            # Create and attach custom policy
            policy_name = f'{role_name}CustomPolicy'
            self.iam.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(permission_policy),
                Description='Custom permissions for content moderation Lambda'
            )
            
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=f'arn:aws:iam::{self.account_id}:policy/{policy_name}'
            )
            
            role_arn = role_response['Role']['Arn']
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                print(f"✅ Lambda IAM role already exists: {role_name}")
                role_arn = f'arn:aws:iam::{self.account_id}:role/{role_name}'
            else:
                print(f"❌ Error creating Lambda IAM role: {e}")
                return None
        
        return role_arn
    
    def save_configuration(self, buckets, sagemaker_role, lambda_role):
        """Save configuration to file for other scripts to use"""
        config = {
            'account_id': self.account_id,
            'region': self.region,
            'buckets': buckets,
            'sagemaker_role': sagemaker_role,
            'lambda_role': lambda_role,
            'datastore_bucket': f"content-moderation-system-datastore-{self.account_id}",
            'lambda_bucket': f"content-moderation-lambda-code-{self.account_id}"
        }
        
        with open('aws_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to aws_config.json")
    
    def setup_all(self):
        """Run complete infrastructure setup"""
        print("🚀 Starting AWS infrastructure setup...")
        print("=" * 60)
        
        # Create S3 buckets
        print("\n1. Setting up S3 buckets...")
        buckets = self.create_s3_buckets()
        
        # Create SageMaker role
        print("\n2. Setting up SageMaker IAM role...")
        sagemaker_role = self.create_sagemaker_execution_role()
        
        # Wait for role to propagate
        print("⏳ Waiting for IAM role propagation...")
        time.sleep(10)
        
        # Create Lambda role
        print("\n3. Setting up Lambda IAM role...")
        lambda_role = self.create_lambda_execution_role()
        
        # Save configuration
        print("\n4. Saving configuration...")
        self.save_configuration(buckets, sagemaker_role, lambda_role)
        
        print("\n" + "=" * 60)
        print("✅ Infrastructure setup complete!")
        print(f"📊 Created {len(buckets)} S3 buckets")
        print(f"🔐 SageMaker role: {sagemaker_role}")
        print(f"🔐 Lambda role: {lambda_role}")
        print("\nNext steps:")
        print("1. Run: python data_preparation.py")
        print("2. Run: python launch_training.py")
        print("3. Run: python deploy_endpoint.py")

if __name__ == "__main__":
    setup = InfrastructureSetup()
    setup.setup_all()
