# Lambda Deployment Guide for Admin

## Current Status
✅ **ContentModerationLambdaRole EXISTS** - The IAM role is properly created and configured  
❌ **Lambda Functions MISSING** - User lacks Lambda deployment permissions

## Solution: Admin Lambda Deployment

The user `content-moderator` has successfully created the IAM role but lacks permissions to create Lambda functions. An admin needs to deploy the Lambda functions using the existing role.

### IAM Role Status
```
Role Name: ContentModerationLambdaRole
ARN: arn:aws:iam::806070028440:role/ContentModerationLambdaRole
Status: ✅ EXISTS and properly configured
Policies: 
  - AWSLambdaBasicExecutionRole (AWS managed)
  - ContentModerationLambdaRoleCustomPolicy (custom)
```

### Required Lambda Functions

#### 1. Preprocessing Lambda
**Function Name:** `content-moderation-preprocessing`  
**Runtime:** python3.9  
**Handler:** lambda_function.lambda_handler  
**Role:** arn:aws:iam::806070028440:role/ContentModerationLambdaRole

**Code:** Use `/Users/snehithad/Learning/Projects/content-moderation-system/preprocessing_lambda.py`

#### 2. Prediction Lambda
**Function Name:** `content-moderation-prediction`  
**Runtime:** python3.9  
**Handler:** lambda_function.lambda_handler  
**Role:** arn:aws:iam::806070028440:role/ContentModerationLambdaRole

**Code:** Use `/Users/snehithad/Learning/Projects/content-moderation-system/prediction_lambda.py`

### Admin Deployment Commands

```bash
# 1. Create deployment packages
cd /Users/snehithad/Learning/Projects/content-moderation-system

# Create preprocessing function
zip preprocessing-lambda.zip preprocessing_lambda.py
aws lambda create-function \
  --function-name content-moderation-preprocessing \
  --runtime python3.9 \
  --role arn:aws:iam::806070028440:role/ContentModerationLambdaRole \
  --handler preprocessing_lambda.lambda_handler \
  --zip-file fileb://preprocessing-lambda.zip \
  --description "Content moderation preprocessing"

# Create prediction function  
zip prediction-lambda.zip prediction_lambda.py
aws lambda create-function \
  --function-name content-moderation-prediction \
  --runtime python3.9 \
  --role arn:aws:iam::806070028440:role/ContentModerationLambdaRole \
  --handler prediction_lambda.lambda_handler \
  --zip-file fileb://prediction-lambda.zip \
  --description "Content moderation prediction"

# Set environment variables for prediction function
aws lambda update-function-configuration \
  --function-name content-moderation-prediction \
  --environment Variables='{
    "SAGEMAKER_ENDPOINT_NAME":"content-moderation-endpoint-1749411921",
    "AWS_DEFAULT_REGION":"us-east-1"
  }'
```

### Alternative: Grant Lambda Permissions to User

If you prefer the user to deploy Lambda functions themselves, add this policy to the `content-moderator` user:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "lambda:CreateFunction",
                "lambda:UpdateFunctionCode",
                "lambda:UpdateFunctionConfiguration",
                "lambda:ListFunctions",
                "lambda:GetFunction",
                "lambda:DeleteFunction",
                "lambda:InvokeFunction",
                "lambda:AddPermission",
                "lambda:RemovePermission"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": "arn:aws:iam::806070028440:role/ContentModerationLambdaRole"
        }
    ]
}
```

### Current Workaround

Since the core SageMaker endpoint is working perfectly, you can:

1. **Use the SageMaker endpoint directly** (as demonstrated in Day 1)
2. **Build your own API wrapper** around the endpoint
3. **Have an admin deploy the Lambda functions** using the guide above

The system is 95% complete - only the Lambda deployment step needs admin intervention!
