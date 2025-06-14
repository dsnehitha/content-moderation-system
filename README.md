# Content Moderation System

A production-ready AI system that automatically moderates user-generated content (text + images) using AWS services.

## Architecture Overview
User Content → S3 → Lambda (preprocessing) → SageMaker (classification) → Bedrock (context analysis) → API Gateway → Real-time decision

## Part 1 Implementation Plan

Create an inline policy for the user with the policies attached in `./trust-policy.json`

### Core Components:
1. **S3 Setup**: Buckets for raw data, processed data, and model artifacts
2. **Data Ingestion**: Upload datasets to S3
3. **SageMaker Training**: Train text classification models
4. **Lambda Functions**: Data preprocessing and prediction orchestration
5. **SageMaker Endpoints**: Deploy models for real-time inference
6. **API Gateway**: REST API endpoints

### Quick Start
1. Run data preparation: `python data_preparation.py`
2. Setup AWS infrastructure: `python setup_infrastructure.py`
3. Train the model: `python launch_training.py`
4. Deploy endpoint: `python deploy_endpoint.py`
5. Test the system: `python test_system.py`

## Data Sources
- Toxic Comment Classification Dataset (Kaggle)
- Custom synthetic data generation
- Future: Image moderation datasets

## Files Overview

### Core Infrastructure
- `data_preparation.py`: Generate and prepare training data
- `setup_infrastructure.py`: Create S3 buckets and IAM roles
- `launch_training.py`: Launch SageMaker training job
- `deploy_endpoint.py`: Deploy model endpoint

### Processing Components
- `prediction_lambda.py`: Text preprocessing and prediction Lambda (combined)
- `bedrock_integration.py`: Amazon Bedrock content analysis
- `cloudwatch_monitoring.py`: CloudWatch monitoring and metrics

### Orchestration & APIs
- `step_functions_orchestration.py`: Complete Step Functions workflow orchestration
- `api_gateway_setup.py`: API Gateway configuration

### Testing & Utilities
- `test_system.py`: End-to-end testing
- `test_complete_system.py`: Comprehensive system testing
- `moderate_content.py`: Direct content moderation utility