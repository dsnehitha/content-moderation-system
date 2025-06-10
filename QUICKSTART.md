# Content Moderation System - Day 1 Quick Start Guide

## 🚀 Getting Started

### Prerequisites
1. **AWS Account** with appropriate permissions
2. **AWS CLI configured** (`aws configure`) 
3. **Python 3.8+** installed
4. **Virtual environment** (recommended)

### Option 1: Automated Setup (Recommended)
```bash
# Run the complete Day 1 setup
python run_day1_setup.py
```
This will automatically:
- Install dependencies
- Setup AWS infrastructure  
- Prepare training data
- Train the ML model
- Deploy SageMaker endpoint
- Setup API Gateway
- Run system tests

### Option 2: Manual Step-by-Step
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup AWS infrastructure
python setup_infrastructure.py

# 3. Prepare training data
python data_preparation.py

# 4. Train the model (10-15 minutes)
python launch_training.py

# 5. Deploy endpoint (5-10 minutes)  
python deploy_endpoint.py

# 6. Setup API Gateway
python api_gateway_setup.py

# 7. Test the system
python test_system.py
```

## 🧪 Testing Your API

After setup, test your content moderation API:

```bash
# Get your API endpoint from api_info.json
curl -X POST https://YOUR-API-ID.execute-api.REGION.amazonaws.com/prod/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "I hate you so much!"}'
```

Expected response:
```json
{
  "text": "I hate you so much!",
  "toxicity_score": 0.8542,
  "is_toxic": true,
  "action": "block",
  "confidence": "high"
}
```

## 📊 System Components

### Core Infrastructure
- **S3 Buckets**: Data storage and model artifacts
- **IAM Roles**: Security for SageMaker and Lambda
- **SageMaker**: ML model training and hosting
- **Lambda**: Preprocessing and prediction logic
- **API Gateway**: REST API endpoints

### Data Pipeline
1. **Input** → Text content for moderation
2. **Preprocessing** → Clean and normalize text
3. **Prediction** → SageMaker toxicity classification
4. **Decision** → Block/Allow/Review based on thresholds
5. **Response** → JSON with moderation decision

## 🔧 Configuration Files

After setup, these files contain your system configuration:
- `aws_config.json` - AWS resources and identifiers
- `api_info.json` - API Gateway endpoints and functions
- `endpoint_info.json` - SageMaker endpoint details
- `model_info.json` - Trained model metadata

## 📈 Performance

Expected performance metrics:
- **Response Time**: < 500ms per request
- **Accuracy**: ~85-90% on synthetic test data
- **Throughput**: 100+ requests/minute per endpoint

## 🛠️ Troubleshooting

### Common Issues

1. **AWS Credentials Error**
   ```bash
   aws configure
   # Enter your Access Key ID, Secret Key, Region
   ```

2. **Permission Denied**
   - Ensure your AWS user has SageMaker, Lambda, API Gateway, S3, and IAM permissions

3. **Training Job Failed**
   - Check CloudWatch logs in AWS Console
   - Verify S3 bucket permissions

4. **Endpoint Not Found**
   - Run `python deploy_endpoint.py` again
   - Check SageMaker console for endpoint status

### Cleanup Resources
To avoid AWS charges, delete resources when done:
```bash
# Delete SageMaker endpoint
aws sagemaker delete-endpoint --endpoint-name content-moderation-endpoint

# Delete S3 buckets (after emptying them)
aws s3 rb s3://your-bucket-name --force
```

## 🎯 Next Steps (Day 2)

After Day 1 completion, you'll be ready for:
- Image content moderation
- Amazon Bedrock integration
- Real-time streaming analysis
- Advanced ML model optimization
- Production monitoring and logging

## 📞 Support

If you encounter issues:
1. Check the test output: `python test_system.py`
2. Review AWS CloudWatch logs
3. Verify all configuration files exist
4. Ensure AWS permissions are correct

---

🎉 **Congratulations!** You now have a production-ready content moderation system!
