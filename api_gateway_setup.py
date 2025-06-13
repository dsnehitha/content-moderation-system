"""
API Gateway Setup for Content Moderation System
Creates REST API endpoints that orchestrate preprocessing and prediction
"""

import boto3
import json
import zipfile
import os
import time
from botocore.exceptions import ClientError

class APIGatewaySetup:
    def __init__(self):
        self.apigateway = boto3.client('apigateway')
        self.lambda_client = boto3.client('lambda')
        self.iam = boto3.client('iam')
        
        # Load configuration
        with open('aws_config.json', 'r') as f:
            self.config = json.load(f)
    
    def create_lambda_deployment_package(self, lambda_name, source_file):
        """Create deployment package for Lambda function"""
        zip_path = f"{lambda_name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            zip_file.write(source_file, 'lambda_function.py')
        
        return zip_path
    
    def deploy_lambda_function(self, function_name, source_file, description):
        """Deploy Lambda function"""
        print(f"📦 Deploying Lambda function: {function_name}")
        
        # Create deployment package
        zip_path = self.create_lambda_deployment_package(function_name, source_file)
        
        try:
            with open(zip_path, 'rb') as zip_file:
                zip_content = zip_file.read()
            
            # Try to update existing function first
            try:
                response = self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zip_content
                )
                print(f"✅ Updated existing Lambda function: {function_name}")
                function_arn = response['FunctionArn']
                
            except ClientError as e:
                if 'ResourceNotFoundException' in str(e):
                    # Create new function
                    response = self.lambda_client.create_function(
                        FunctionName=function_name,
                        Runtime='python3.9',
                        Role=self.config['lambda_role'],
                        Handler='lambda_function.lambda_handler',
                        Code={'ZipFile': zip_content},
                        Description=description,
                        Timeout=30,
                        MemorySize=256
                    )
                    print(f"✅ Created new Lambda function: {function_name}")
                    function_arn = response['FunctionArn']
                else:
                    raise e
            
            # Clean up zip file
            os.remove(zip_path)
            
            return function_arn
            
        except Exception as e:
            print(f"❌ Error deploying Lambda function {function_name}: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return None
    
    def create_api_gateway(self):
        """Create API Gateway REST API"""
        api_name = 'content-moderation-api'
        
        try:
            # Check if API already exists
            apis = self.apigateway.get_rest_apis()
            existing_api = None
            
            for api in apis['items']:
                if api['name'] == api_name:
                    existing_api = api
                    break
            
            if existing_api:
                print(f"✅ Using existing API Gateway: {api_name}")
                api_id = existing_api['id']
            else:
                # Create new API
                response = self.apigateway.create_rest_api(
                    name=api_name,
                    description='Content Moderation System API',
                    endpointConfiguration={'types': ['REGIONAL']}
                )
                api_id = response['id']
                print(f"✅ Created API Gateway: {api_name}")
            
            return api_id
            
        except Exception as e:
            print(f"❌ Error creating API Gateway: {e}")
            return None
    
    def setup_api_resources_and_methods(self, api_id, prediction_function_arn):
        """Setup API Gateway resources and methods"""
        try:
            # Get root resource
            resources = self.apigateway.get_resources(restApiId=api_id)
            root_resource_id = None
            
            for resource in resources['items']:
                if resource['path'] == '/':
                    root_resource_id = resource['id']
                    break
            
            # Create /moderate resource (or find existing)
            moderate_resource_id = None
            
            # Check if /moderate resource already exists
            for resource in resources['items']:
                if resource['path'] == '/moderate':
                    moderate_resource_id = resource['id']
                    print("✅ Found existing /moderate resource")
                    break
            
            if not moderate_resource_id:
                moderate_resource = self.apigateway.create_resource(
                    restApiId=api_id,
                    parentId=root_resource_id,
                    pathPart='moderate'
                )
                moderate_resource_id = moderate_resource['id']
                print("✅ Created new /moderate resource")
            
            # Create POST method for /moderate (handle existing)
            try:
                self.apigateway.put_method(
                    restApiId=api_id,
                    resourceId=moderate_resource_id,
                    httpMethod='POST',
                    authorizationType='NONE'
                )
                print("✅ Created POST method")
            except ClientError as e:
                if 'ConflictException' in str(e):
                    print("✅ POST method already exists")
                else:
                    raise e
            
            # Enable CORS (handle existing)
            try:
                self.apigateway.put_method(
                    restApiId=api_id,
                    resourceId=moderate_resource_id,
                    httpMethod='OPTIONS',
                    authorizationType='NONE'
                )
                print("✅ Created OPTIONS method")
            except ClientError as e:
                if 'ConflictException' in str(e):
                    print("✅ OPTIONS method already exists")
                else:
                    raise e
            
            # Set up integration for POST method
            # Direct integration: Request -> Prediction Lambda -> SageMaker -> Response
            integration_uri = f"arn:aws:apigateway:{self.config['region']}:lambda:path/2015-03-31/functions/{prediction_function_arn}/invocations"
            
            try:
                self.apigateway.put_integration(
                    restApiId=api_id,
                    resourceId=moderate_resource_id,
                    httpMethod='POST',
                    type='AWS_PROXY',
                    integrationHttpMethod='POST',
                    uri=integration_uri
                )
                print("✅ Created POST integration")
            except ClientError as e:
                if 'ConflictException' in str(e):
                    print("✅ POST integration already exists")
                else:
                    raise e
            
            # CORS integration
            try:
                self.apigateway.put_integration(
                    restApiId=api_id,
                    resourceId=moderate_resource_id,
                    httpMethod='OPTIONS',
                    type='MOCK',
                    requestTemplates={
                        'application/json': '{"statusCode": 200}'
                    }
                )
                print("✅ Created OPTIONS integration")
            except ClientError as e:
                if 'ConflictException' in str(e):
                    print("✅ OPTIONS integration already exists")
                else:
                    raise e
            
            # CORS method response
            try:
                self.apigateway.put_method_response(
                    restApiId=api_id,
                    resourceId=moderate_resource_id,
                    httpMethod='OPTIONS',
                    statusCode='200',
                    responseParameters={
                        'method.response.header.Access-Control-Allow-Headers': False,
                        'method.response.header.Access-Control-Allow-Methods': False,
                        'method.response.header.Access-Control-Allow-Origin': False
                    }
                )
                print("✅ Created OPTIONS method response")
            except ClientError as e:
                if 'ConflictException' in str(e):
                    print("✅ OPTIONS method response already exists")
                else:
                    raise e
            
            # CORS integration response
            try:
                self.apigateway.put_integration_response(
                    restApiId=api_id,
                    resourceId=moderate_resource_id,
                    httpMethod='OPTIONS',
                    statusCode='200',
                    responseParameters={
                        'method.response.header.Access-Control-Allow-Headers': "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
                        'method.response.header.Access-Control-Allow-Methods': "'POST,OPTIONS'",
                        'method.response.header.Access-Control-Allow-Origin': "'*'"
                    }
                )
                print("✅ Created OPTIONS integration response")
            except ClientError as e:
                if 'ConflictException' in str(e):
                    print("✅ OPTIONS integration response already exists")
                else:
                    raise e
            
            print("✅ API Gateway resources and methods configured")
            
            return moderate_resource_id
            
        except Exception as e:
            print(f"❌ Error setting up API resources: {e}")
            return None
    
    def grant_api_gateway_permission(self, function_arn, api_id):
        """Grant API Gateway permission to invoke Lambda"""
        function_name = function_arn.split(':')[-1]
        
        try:
            self.lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'api-gateway-invoke-{int(time.time())}',
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com',
                SourceArn=f"arn:aws:execute-api:{self.config['region']}:{self.config['account_id']}:{api_id}/*/*"
            )
            print(f"✅ Granted API Gateway permission for {function_name}")
            
        except ClientError as e:
            if 'ResourceConflictException' in str(e):
                print(f"✅ Permission already exists for {function_name}")
            else:
                print(f"❌ Error granting permission: {e}")
    
    def deploy_api(self, api_id):
        """Deploy API to a stage"""
        try:
            # Check if stage already exists
            try:
                stage_info = self.apigateway.get_stage(
                    restApiId=api_id,
                    stageName='prod'
                )
                print("✅ Production stage already exists - updating deployment")
                
                # Create new deployment and update stage
                deployment = self.apigateway.create_deployment(
                    restApiId=api_id,
                    description=f'Updated deployment - {time.strftime("%Y-%m-%d %H:%M:%S")}'
                )
                
                # Update stage to point to new deployment
                self.apigateway.update_stage(
                    restApiId=api_id,
                    stageName='prod',
                    patchOperations=[
                        {
                            'op': 'replace',
                            'path': '/deploymentId',
                            'value': deployment['id']
                        }
                    ]
                )
                
            except ClientError as stage_error:
                if 'NotFoundException' in str(stage_error):
                    print("✅ Creating new production stage")
                    # Create new deployment with stage
                    deployment = self.apigateway.create_deployment(
                        restApiId=api_id,
                        stageName='prod',
                        description='Production deployment'
                    )
                else:
                    raise stage_error
            
            api_url = f"https://{api_id}.execute-api.{self.config['region']}.amazonaws.com/prod"
            
            print(f"✅ API deployed to production stage")
            print(f"🌐 API URL: {api_url}")
            
            return api_url
            
        except Exception as e:
            print(f"❌ Error deploying API: {e}")
            return None
    
    def setup_complete_api(self):
        """Setup complete API Gateway with Lambda integration"""
        print("🚀 Setting up API Gateway for Content Moderation System...")
        print("=" * 60)
        
        # Deploy Lambda functions
        print("\n1. Deploying Lambda function...")
        
        # Check if functions already exist
        existing_functions = []
        try:
            response = self.lambda_client.list_functions()
            existing_functions = [f['FunctionName'] for f in response['Functions']]
        except Exception as e:
            print(f"⚠️  Could not check existing functions: {e}")
        
        if 'content-moderation-prediction' in existing_functions:
            print("ℹ️  Prediction function already exists - will update")
        
        prediction_arn = self.deploy_lambda_function(
            'content-moderation-prediction',
            'prediction_lambda.py',
            'Get toxicity predictions from SageMaker'
        )
        
        if not prediction_arn:
            print("❌ Failed to deploy Lambda function")
            return False
        
        # Create API Gateway
        print("\n2. Creating API Gateway...")
        api_id = self.create_api_gateway()
        if not api_id:
            return False
        
        # Setup resources and methods
        print("\n3. Setting up API resources...")
        resource_id = self.setup_api_resources_and_methods(api_id, prediction_arn)
        if not resource_id:
            return False
        
        # Grant permissions
        print("\n4. Setting up permissions...")
        self.grant_api_gateway_permission(prediction_arn, api_id)
        
        # Deploy API
        print("\n5. Deploying API...")
        api_url = self.deploy_api(api_id)
        if not api_url:
            return False
        
        # Save API information
        api_info = {
            'api_id': api_id,
            'api_url': api_url,
            'endpoints': {
                'moderate': f"{api_url}/moderate"
            },
            'prediction_function_arn': prediction_arn,
            'deployment_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('api_info.json', 'w') as f:
            json.dump(api_info, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✅ API Gateway setup complete!")
        print(f"🌐 API URL: {api_url}")
        print(f"📡 Moderation endpoint: {api_url}/moderate")
        print("\n📋 Usage example:")
        print(f"""
curl -X POST {api_url}/moderate \\
  -H "Content-Type: application/json" \\
  -d '{{"text": "Your content to moderate"}}'
        """)
        
        return True

if __name__ == "__main__":
    try:
        setup = APIGatewaySetup()
        success = setup.setup_complete_api()
        
        if success:
            print("\n🎉 All systems ready! The content moderation API is live.")
        else:
            print("\n❌ Setup failed. Please check the logs above.")
            exit(1)
            
    except FileNotFoundError:
        print("❌ aws_config.json not found. Please run setup_infrastructure.py first.")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)
