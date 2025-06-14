#!/usr/bin/env python3
"""
Storage Manager for Content Moderation System
Handles both local structured storage and S3 bucket operations
"""
import os
import json
import boto3
from datetime import datetime
from botocore.exceptions import ClientError

class StorageManager:
    """Manages local and S3 storage for content moderation system"""
    
    def __init__(self):
        # Load AWS configuration
        with open('aws_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.s3_client = boto3.client('s3', region_name=self.config['region'])
        self.datastore_bucket = self.config['datastore_bucket']
        
        # Define local storage structure
        self.local_storage = {
            'models': 'local_storage/models/',
            'pipelines': 'local_storage/pipelines/',
            'endpoints': 'local_storage/endpoints/',
            'training': 'local_storage/training/',
            'logs': 'local_storage/logs/',
            'artifacts': 'local_storage/artifacts/'
        }
        
        # Create local directories
        self._create_local_directories()
        
        print(f"📁 Storage Manager initialized")
        print(f"🪣 S3 Bucket: {self.datastore_bucket}")
        print(f"📂 Local storage: local_storage/")
    
    def _create_local_directories(self):
        """Create local storage directory structure"""
        for category, path in self.local_storage.items():
            os.makedirs(path, exist_ok=True)
        print("✅ Local storage directories created")
    
    def save_model_info(self, model_data, model_type='standalone', training_job_name=None, 
                       pipeline_execution_time=None, additional_info=None):
        """Save model information both locally and to S3"""
        
        timestamp = datetime.now().isoformat()
        
        model_info = {
            'model_data': model_data,
            'training_job_name': training_job_name,
            'model_type': model_type,
            'framework_version': '0.23-1',
            'instance_type': 'ml.m5.large',
            'timestamp': timestamp,
            'region': self.config['region']
        }
        
        if pipeline_execution_time:
            model_info['pipeline_execution_time'] = pipeline_execution_time
            model_info['pipeline_type'] = 'automated'
        
        if additional_info:
            model_info.update(additional_info)
        
        # Save locally
        if model_type == 'pipeline':
            local_file = os.path.join(self.local_storage['models'], 'pipeline_model_info.json')
            # Also maintain backward compatibility
            legacy_file = 'pipeline_model_info.json'
        else:
            local_file = os.path.join(self.local_storage['models'], 'standalone_model_info.json')
            legacy_file = 'model_info.json'
        
        with open(local_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Maintain backward compatibility
        with open(legacy_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Save to S3
        s3_key = f"models/{model_type}_model_info_{int(datetime.now().timestamp())}.json"
        try:
            self.s3_client.put_object(
                Bucket=self.datastore_bucket,
                Key=s3_key,
                Body=json.dumps(model_info, indent=2),
                ContentType='application/json'
            )
            print(f"✅ Model info saved to S3: s3://{self.datastore_bucket}/{s3_key}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to save to S3: {e}")
        
        if model_type == 'standalone' and model_data:
            # Also save model path for backward compatibility
            model_path_file = os.path.join(self.local_storage['models'], 'model_path.txt')
            with open(model_path_file, 'w') as f:
                f.write(model_data)
            with open('model_path.txt', 'w') as f:  # Legacy location
                f.write(model_data)
        
        print(f"✅ {model_type.title()} model info saved locally and to S3")
        return local_file
    
    def save_pipeline_info(self, pipeline_name, pipeline_arn, deployment_time, status='deployed'):
        """Save pipeline information both locally and to S3"""
        
        pipeline_info = {
            'pipeline_name': pipeline_name,
            'pipeline_arn': pipeline_arn,
            'region': self.config['region'],
            'bucket': self.datastore_bucket,
            'role': self.config['sagemaker_role'],
            'deployment_time': deployment_time,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save locally
        local_file = os.path.join(self.local_storage['pipelines'], f'{pipeline_name}_info.json')
        with open(local_file, 'w') as f:
            json.dump(pipeline_info, f, indent=2)
        
        # Maintain backward compatibility
        with open('pipeline_info.json', 'w') as f:
            json.dump(pipeline_info, f, indent=2)
        
        # Save to S3
        s3_key = f"pipeline/info/{pipeline_name}_info.json"
        try:
            self.s3_client.put_object(
                Bucket=self.datastore_bucket,
                Key=s3_key,
                Body=json.dumps(pipeline_info, indent=2),
                ContentType='application/json'
            )
            print(f"✅ Pipeline info saved to S3: s3://{self.datastore_bucket}/{s3_key}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to save pipeline info to S3: {e}")
        
        print(f"✅ Pipeline info saved locally and to S3")
        return local_file
    
    def save_endpoint_info(self, endpoint_name, model_data, model_type, deployment_time, 
                          training_job_name, additional_info=None):
        """Save endpoint information both locally and to S3"""
        
        endpoint_info = {
            'endpoint_name': endpoint_name,
            'model_data': model_data,
            'model_type': model_type,
            'instance_type': 'ml.t2.medium',
            'framework_version': '0.23-1',
            'deployment_time': deployment_time,
            'region': self.config['region'],
            'training_job_name': training_job_name,
            'status': 'deployed',
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_info:
            endpoint_info.update(additional_info)
        
        # Save locally
        local_file = os.path.join(self.local_storage['endpoints'], f'{endpoint_name}_info.json')
        with open(local_file, 'w') as f:
            json.dump(endpoint_info, f, indent=2)
        
        # Maintain backward compatibility
        with open('endpoint_info.json', 'w') as f:
            json.dump(endpoint_info, f, indent=2)
        
        # Save to S3
        s3_key = f"endpoints/{endpoint_name}_info.json"
        try:
            self.s3_client.put_object(
                Bucket=self.datastore_bucket,
                Key=s3_key,
                Body=json.dumps(endpoint_info, indent=2),
                ContentType='application/json'
            )
            print(f"✅ Endpoint info saved to S3: s3://{self.datastore_bucket}/{s3_key}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to save endpoint info to S3: {e}")
        
        print(f"✅ Endpoint info saved locally and to S3")
        return local_file
    
    def get_latest_model_info(self, model_type='pipeline'):
        """Get the latest model info, prioritizing pipeline models"""
        
        # Try pipeline model first
        pipeline_file = os.path.join(self.local_storage['models'], 'pipeline_model_info.json')
        if os.path.exists(pipeline_file):
            with open(pipeline_file, 'r') as f:
                return json.load(f), 'pipeline'
        
        # Fallback to legacy location
        if os.path.exists('pipeline_model_info.json'):
            with open('pipeline_model_info.json', 'r') as f:
                return json.load(f), 'pipeline'
        
        # Try standalone model
        standalone_file = os.path.join(self.local_storage['models'], 'standalone_model_info.json')
        if os.path.exists(standalone_file):
            with open(standalone_file, 'r') as f:
                return json.load(f), 'standalone'
        
        # Fallback to legacy locations
        if os.path.exists('model_info.json'):
            with open('model_info.json', 'r') as f:
                return json.load(f), 'standalone'
        
        if os.path.exists('model_path.txt'):
            with open('model_path.txt', 'r') as f:
                model_data = f.read().strip()
            return {'model_data': model_data}, 'standalone'
        
        return None, None
    
    def list_s3_pipeline_artifacts(self):
        """List all pipeline artifacts in S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.datastore_bucket,
                Prefix='pipeline/'
            )
            
            if 'Contents' in response:
                artifacts = []
                for obj in response['Contents']:
                    artifacts.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    })
                return artifacts
            return []
        except Exception as e:
            print(f"❌ Error listing S3 artifacts: {e}")
            return []
    
    def sync_local_storage(self):
        """Sync local storage structure with current files"""
        print("🔄 Syncing local storage structure...")
        
        # Move existing files to structured storage
        file_mappings = {
            'pipeline_info.json': ('pipelines', 'latest_pipeline_info.json'),
            'pipeline_model_info.json': ('models', 'pipeline_model_info.json'),
            'model_info.json': ('models', 'standalone_model_info.json'),
            'model_path.txt': ('models', 'model_path.txt'),
            'endpoint_info.json': ('endpoints', 'latest_endpoint_info.json'),
        }
        
        for source_file, (category, target_name) in file_mappings.items():
            if os.path.exists(source_file):
                target_path = os.path.join(self.local_storage[category], target_name)
                if not os.path.exists(target_path):
                    import shutil
                    shutil.copy2(source_file, target_path)
                    print(f"📁 Copied {source_file} to {target_path}")
        
        print("✅ Local storage sync completed")

def main():
    """Test the storage manager"""
    storage = StorageManager()
    
    print("\n📊 Storage Manager Test")
    print("=" * 40)
    
    # Test getting latest model info
    model_info, model_type = storage.get_latest_model_info()
    if model_info:
        print(f"✅ Found {model_type} model: {model_info.get('training_job_name', 'unknown')}")
        print(f"📁 Model data: {model_info.get('model_data', 'unknown')}")
    else:
        print("❌ No model info found")
    
    # List S3 artifacts
    artifacts = storage.list_s3_pipeline_artifacts()
    print(f"\n📦 Found {len(artifacts)} pipeline artifacts in S3")
    
    # Sync local storage
    storage.sync_local_storage()

if __name__ == "__main__":
    main()
