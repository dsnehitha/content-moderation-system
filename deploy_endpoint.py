import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import json
import time

def load_aws_config():
    """Load AWS configuration"""
    try:
        with open('aws_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ aws_config.json not found. Please run setup_infrastructure.py first.")
        return None

def load_model_info():
    """Load model information from training - prioritize pipeline model"""
    
    # First, try to load pipeline model (preferred)
    try:
        with open('pipeline_model_info.json', 'r') as f:
            pipeline_model_info = json.load(f)
        
        print("✅ Found pipeline model - using pipeline-trained model")
        return pipeline_model_info['model_data'], pipeline_model_info
        
    except FileNotFoundError:
        print("🔍 No pipeline model found, checking for standalone training model...")
    
    # Fallback to standalone training model
    try:
        with open('model_path.txt', 'r') as f:
            model_data = f.read().strip()
        
        # Try to load additional model info
        try:
            with open('model_info.json', 'r') as f:
                model_info = json.load(f)
        except FileNotFoundError:
            model_info = {}
        
        print("✅ Found standalone training model")
        return model_data, model_info
        
    except FileNotFoundError:
        print("❌ No trained model found. Please run sagemaker_pipeline.py first (preferred) or launch_training.py as fallback.")
        return None, None

def deploy_endpoint():
    """Deploy SageMaker endpoint for content moderation"""
    
    # Load configuration
    config = load_aws_config()
    if not config:
        return None
    
    # Check if we should use a pipeline model instead
    print("🔍 Checking for pipeline models...")
    
    model_data, model_info = load_model_info()
    if not model_data:
        print("❌ No trained model found. Please run sagemaker_pipeline.py first (preferred) or launch_training.py as fallback.")
        return None
    
    print("🚀 Deploying SageMaker endpoint...")
    print(f"📁 Model location: {model_data}")
    print(f"🔐 Using role: {config['sagemaker_role']}")
    
    # Indicate which model type is being used
    if 'pipeline_execution_time' in model_info:
        print(f"🔧 Using PIPELINE model (training job: {model_info.get('training_job_name', 'unknown')})")
    else:
        print(f"🔧 Using STANDALONE model (training job: {model_info.get('training_job_name', 'unknown')})")
    
    # Create model
    sklearn_model = SKLearnModel(
        model_data=model_data,
        role=config['sagemaker_role'],
        entry_point='ml_scripts/train.py',
        framework_version='0.23-1',
        py_version='py3',
        name=f"content-moderation-model-{int(time.time())}"
    )
    
    endpoint_name = f'content-moderation-endpoint-{int(time.time())}'
    print(endpoint_name)
    try:
        print(f"📡 Deploying endpoint: {endpoint_name}")
        print("⏳ This may take 5-10 minutes...")
        
        # Deploy endpoint
        predictor = sklearn_model.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',
            endpoint_name=endpoint_name
        )
        
        print("✅ Endpoint deployed successfully!")
        
        # Test the endpoint with sample data
        print("\n🧪 Testing endpoint with sample data...")
        
        test_cases = [
            "I hate you so much!",  # Should be toxic
            "Thank you for your help!",  # Should be non-toxic
            "You are amazing!",  # Should be non-toxic
            "Go kill yourself!"  # Should be toxic
        ]
        
        # Use SageMaker runtime client for direct testing
        runtime_client = boto3.client('sagemaker-runtime', region_name=config.get('region', 'us-east-1'))
        
        for i, text in enumerate(test_cases, 1):
            try:
                response = runtime_client.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='text/csv',
                    Body=text
                )
                
                result = response['Body'].read().decode('utf-8')
                toxicity_score = float(result)
                
                print(f"Test {i}: '{text}'")
                print(f"   Toxicity Score: {toxicity_score:.3f}")
                print(f"   Classification: {'🔴 TOXIC' if toxicity_score > 0.5 else '✅ SAFE'}")
                print()
                
            except Exception as e:
                print(f"❌ Error testing case {i}: {e}")
        
        # Save endpoint information
        endpoint_info = {
            'endpoint_name': endpoint_name,
            'model_data': model_data,
            'instance_type': 'ml.t2.medium',
            'framework_version': '0.23-1',
            'deployment_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'region': config.get('region', 'us-east-1'),
            'model_info': model_info if model_info else {},
            'model_type': 'pipeline' if 'pipeline_execution_time' in model_info else 'standalone',
            'training_job_name': model_info.get('training_job_name', 'unknown'),
            'status': 'deployed'
        }
        
        # Save locally
        with open('endpoint_info.json', 'w') as f:
            json.dump(endpoint_info, f, indent=2)
        
        # Save to S3 bucket as well
        try:
            s3_client = boto3.client('s3', region_name=config.get('region', 'us-east-1'))
            s3_key = f"endpoints/{endpoint_name}_info.json"
            s3_client.put_object(
                Bucket=config['datastore_bucket'],
                Key=s3_key,
                Body=json.dumps(endpoint_info, indent=2),
                ContentType='application/json'
            )
            print(f"✅ Endpoint info also saved to S3: s3://{config['datastore_bucket']}/{s3_key}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to save endpoint info to S3: {e}")
        
        print(f"📊 Endpoint info saved to endpoint_info.json")
        print(f"🎯 Endpoint name: {endpoint_name}")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return None

def check_existing_endpoint():
    """Check if any content moderation endpoints exist"""
    config = load_aws_config()
    if not config:
        return False
        
    sagemaker_client = boto3.client('sagemaker', region_name=config.get('region', 'us-east-1'))
    
    try:
        # List all endpoints that contain 'content-moderation'
        response = sagemaker_client.list_endpoints(
            StatusEquals='InService',
            NameContains='content-moderation'
        )
        
        if response['Endpoints']:
            print(f"✅ Found {len(response['Endpoints'])} existing content moderation endpoint(s):")
            for ep in response['Endpoints']:
                print(f"   - {ep['EndpointName']} (Status: {ep['EndpointStatus']})")
            return True
        else:
            print("📡 No existing content moderation endpoints found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking endpoints: {e}")
        return False

if __name__ == "__main__":
    print("Content Moderation System - Endpoint Deployment")
    print("=" * 50)
    
    print("📋 Model Priority Order:")
    print("   1. Pipeline model (pipeline_model_info.json) - PREFERRED")
    print("   2. Standalone model (model_path.txt) - Fallback")
    print()
    
    # Check for existing endpoints first
    print("🔍 Checking for existing endpoints...")
    check_existing_endpoint()
    
    # Deploy new endpoint
    predictor = deploy_endpoint()
    
    if predictor:
        print(f"\n🎉 Deployment completed successfully!")
        print("Next steps:")
        print("1. Run: python test_system.py")
        print("2. Setup API Gateway with: python api_gateway_setup.py")
    else:
        print("\n❌ Deployment failed. Please check the logs above.")
        exit(1)
