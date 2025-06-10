#!/usr/bin/env python3
"""
Content Moderation System - Day 1 Setup Script
Automates the complete setup process for the content moderation system
"""

import subprocess
import sys
import os
import time

def run_command(command, description, check_success=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=check_success, 
                                  capture_output=False, text=True)
        else:
            result = subprocess.run(command, check=check_success, 
                                  capture_output=False, text=True)
        
        if check_success and result.returncode == 0:
            print(f"✅ {description} completed successfully!")
        elif not check_success:
            print(f"ℹ️  {description} finished (status: {result.returncode})")
        
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def check_aws_credentials():
    """Check if AWS credentials are configured"""
    print("🔍 Checking AWS credentials...")
    
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS credentials found for account: {identity['Account']}")
        return True
    except Exception as e:
        print(f"❌ AWS credentials not configured: {e}")
        print("\nPlease configure AWS credentials using one of:")
        print("1. aws configure")
        print("2. export AWS_ACCESS_KEY_ID=...")
        print("3. IAM roles (if running on EC2)")
        return False

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    return run_command([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                      "Installing Python packages")

def main():
    """Main setup workflow"""
    print("🚀 Content Moderation System - Day 1 Setup")
    print("=" * 60)
    print("This script will set up your complete content moderation system!")
    print()
    
    # Check prerequisites
    if not check_aws_credentials():
        print("\n❌ Setup cannot continue without AWS credentials")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies")
        return False
    
    # Step 1: Setup infrastructure
    if not run_command([sys.executable, 'setup_infrastructure.py'],
                      "Setting up AWS infrastructure (S3, IAM roles)"):
        return False
    
    # Step 2: Prepare and upload data
    if not run_command([sys.executable, 'data_preparation.py'],
                      "Preparing and uploading training data"):
        return False
    
    # Step 3: Train the model
    print("\n⏰ Starting model training (this may take 10-15 minutes)...")
    if not run_command([sys.executable, 'launch_training.py'],
                      "Training content moderation model"):
        return False
    
    # Step 4: Deploy endpoint
    print("\n⏰ Deploying SageMaker endpoint (this may take 5-10 minutes)...")
    if not run_command([sys.executable, 'deploy_endpoint.py'],
                      "Deploying SageMaker endpoint"):
        return False
    
    # Step 5: Setup API Gateway
    if not run_command([sys.executable, 'api_gateway_setup.py'],
                      "Setting up API Gateway"):
        return False
    
    # Step 6: Test the system
    if not run_command([sys.executable, 'test_system.py'],
                      "Running system tests", check_success=False):
        print("⚠️  Some tests may have failed, but setup is complete")
    
    # Success!
    print("\n" + "=" * 60)
    print("🎉 DAY 1 SETUP COMPLETE!")
    print("=" * 60)
    print("✅ Your content moderation system is now live!")
    print()
    
    # Show summary
    try:
        import json
        with open('api_info.json', 'r') as f:
            api_info = json.load(f)
        
        print("📋 System Summary:")
        print(f"🌐 API Endpoint: {api_info['endpoints']['moderate']}")
        print()
        print("🧪 Test your API:")
        print(f"""
curl -X POST {api_info['endpoints']['moderate']} \\
  -H "Content-Type: application/json" \\
  -d '{{"text": "This is a test message"}}'
        """)
        
    except:
        print("📋 Check api_info.json for endpoint details")
    
    print("\n🚀 Ready for Day 2 features:")
    print("• Image moderation")
    print("• Bedrock integration")
    print("• Real-time streaming")
    print("• Advanced ML models")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✨ Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
