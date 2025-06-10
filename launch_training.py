import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import json
import os

def load_aws_config():
    """Load AWS configuration from setup"""
    try:
        with open('aws_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ aws_config.json not found. Please run setup_infrastructure.py first.")
        return None

def launch_training_job():
    """Launch SageMaker training job for content moderation"""
    
    # Load configuration
    config = load_aws_config()
    if not config:
        return None
    
    print("🚀 Launching SageMaker training job...")
    print(f"📊 Using bucket: {config['datastore_bucket']}")
    print(f"🔐 Using role: {config['sagemaker_role']}")
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Create sklearn estimator
    sklearn_estimator = SKLearn(
        entry_point='train.py',
        role=config['sagemaker_role'],
        instance_type='ml.m5.large',
        framework_version='0.23-1',
        py_version='py3',
        script_mode=True,
        sagemaker_session=sagemaker_session
    )
    
    # Define input channels
    train_input = f"s3://{config['datastore_bucket']}/data/raw/"
    test_input = f"s3://{config['datastore_bucket']}/data/raw/"
    
    print(f"📥 Training data location: {train_input}")
    print(f"📥 Test data location: {test_input}")
    
    try:
        # Start training with both train and test data
        sklearn_estimator.fit({
            'train': train_input,
            'test': test_input
        })
        
        # Save model information
        model_data = sklearn_estimator.model_data
        
        model_info = {
            'model_data': model_data,
            'training_job_name': sklearn_estimator.latest_training_job.name,
            'framework_version': '0.23-1',
            'instance_type': 'ml.m5.large'
        }
        
        with open('model_path.txt', 'w') as f:
            f.write(model_data)
        
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("✅ Training job completed successfully!")
        print(f"📁 Model artifacts saved at: {model_data}")
        print(f"🏷️  Training job name: {sklearn_estimator.latest_training_job.name}")
        
        return model_data
        
    except Exception as e:
        print(f"❌ Training job failed: {e}")
        return None

def check_data_availability():
    """Check if training data is available in S3"""
    config = load_aws_config()
    if not config:
        return False
    
    s3_client = boto3.client('s3')
    bucket_name = config['datastore_bucket']
    
    try:
        # Check if training data exists
        s3_client.head_object(Bucket=bucket_name, Key='data/raw/train_data.csv')
        print("✅ Training data found in S3")
        return True
    except:
        print("❌ Training data not found in S3")
        print("Please run data_preparation.py to upload training data")
        return False

if __name__ == "__main__":
    print("Content Moderation System - Training Job Launcher")
    print("=" * 50)
    
    # Check if data is available
    if not check_data_availability():
        print("\n🔄 Uploading training data to S3...")
        # Try to upload data if it exists locally
        if os.path.exists('train_data.csv'):
            exec(open('data_preparation.py').read())
        else:
            print("❌ No local training data found. Please run data_preparation.py first.")
            exit(1)
    
    # Launch training
    model_path = launch_training_job()
    
    if model_path:
        print(f"\n🎉 Training completed! Next steps:")
        print("1. Run: python deploy_endpoint.py")
        print("2. Run: python test_system.py")
    else:
        print("\n❌ Training failed. Please check the logs above.")
        exit(1)
