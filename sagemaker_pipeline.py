#!/usr/bin/env python3
"""
SageMaker Pipeline for Content Moderation Model Training
========================================================

This creates an automated ML pipeline with:
- Data validation and quality checks
- Model training using existing train.py
- Model evaluation leveraging train.py metrics
- Simplified workflow that works with existing components
"""

import os
import json
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.model_metrics import MetricsSource, ModelMetrics
import time

class ContentModerationPipeline:
    """SageMaker Pipeline for automated content moderation model training"""
    
    def __init__(self):
        self.sagemaker_session = sagemaker.Session()
        self.region = self.sagemaker_session.boto_region_name
        self.role = self.load_sagemaker_role()
        self.bucket = self.load_bucket_name()
        self.pipeline_name = f"content-moderation-pipeline-{int(time.time())}"
        
        print(f"🚀 Initializing SageMaker Pipeline: {self.pipeline_name}")
        print(f"📍 Region: {self.region}")
        print(f"🔐 Role: {self.role}")
        print(f"📦 Bucket: {self.bucket}")
        print(f"🗄️ Storage: Direct S3 integration")
    
    def load_sagemaker_role(self):
        """Load SageMaker execution role from config"""
        try:
            with open('aws_config.json', 'r') as f:
                config = json.load(f)
            return config['sagemaker_role']
        except FileNotFoundError:
            raise Exception("aws_config.json not found. Run setup_infrastructure.py first.")
    
    def load_bucket_name(self):
        """Load S3 bucket name from config"""
        try:
            with open('aws_config.json', 'r') as f:
                config = json.load(f)
            return config['datastore_bucket']
        except FileNotFoundError:
            raise Exception("aws_config.json not found. Run setup_infrastructure.py first.")
        
    def create_data_validation_step(self):
        """Create simplified data validation step"""
        
        # Use the existing validated data validation script instead of inline
        validation_processor = ScriptProcessor(
            image_uri=sagemaker.image_uris.retrieve(
                framework="sklearn",
                region=self.region,
                version="0.23-1",
                py_version="py3",
                instance_type="ml.t3.medium"
            ),
            role=self.role,
            instance_type="ml.t3.medium",
            instance_count=1,
            base_job_name="content-moderation-data-validation",
            sagemaker_session=self.sagemaker_session,
            command=["python3"]
        )
        
        # Define processing step using existing validation script
        validation_step = ProcessingStep(
            name="DataValidation",
            processor=validation_processor,
            code="ml_scripts/data_validation.py",
            inputs=[
                ProcessingInput(
                    source=f"s3://{self.bucket}/data/raw/",
                    destination="/opt/ml/processing/input"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="validated-data",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{self.bucket}/data/validated/"
                )
            ],
            job_arguments=["--input-path", "/opt/ml/processing/input", "--output-path", "/opt/ml/processing/output"]
        )
        
        return validation_step
    
    def create_training_step(self, validation_step):
        """Create model training step using existing train.py script"""
        
        # Use existing train.py script - no need to duplicate training logic
        sklearn_estimator = SKLearn(
            entry_point="ml_scripts/train.py",  # Use consolidated training script
            role=self.role,
            instance_type="ml.m5.large",  # Use fixed instance type to avoid parameter issues
            framework_version="0.23-1",
            py_version="py3",
            script_mode=True,
            base_job_name="content-moderation-training",
            sagemaker_session=self.sagemaker_session,
        )
        
        # Define training step with both train and test data channels
        training_step = TrainingStep(
            name="TrainContentModerationModel",
            estimator=sklearn_estimator,
            inputs={
                "train": sagemaker.inputs.TrainingInput(
                    s3_data=f"s3://{self.bucket}/data/validated/",
                    content_type="text/csv"
                ),
                "test": sagemaker.inputs.TrainingInput(
                    s3_data=f"s3://{self.bucket}/data/validated/",
                    content_type="text/csv"
                )
            },
            depends_on=[validation_step]  # Explicit dependency
        )
        
        return training_step
    
    def create_evaluation_step(self, training_step):
        """Create model evaluation step that extracts metrics from train.py output"""
        
        # Create evaluation processor
        evaluation_processor = ScriptProcessor(
            image_uri=sagemaker.image_uris.retrieve(
                framework="sklearn",
                region=self.region,
                version="0.23-1",
                py_version="py3",
                instance_type="ml.t3.medium"
            ),
            role=self.role,
            instance_type="ml.t3.medium",
            instance_count=1,
            base_job_name="content-moderation-evaluation",
            sagemaker_session=self.sagemaker_session,
            command=["python3"]
        )
        
        # Create evaluation step - use string path for model artifacts
        evaluation_step = ProcessingStep(
            name="EvaluateModel",
            processor=evaluation_processor,
            code="ml_scripts/model_evaluation.py",
            inputs=[
                ProcessingInput(
                    source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{self.bucket}/pipeline/evaluation/"
                )
            ],
            property_files=[
                PropertyFile(
                    name="EvaluationReport", 
                    output_name="evaluation",
                    path="evaluation.json"
                )
            ],
            depends_on=[training_step]
        )
        
        return evaluation_step
    
    def create_model_registration_step(self, training_step, evaluation_step):
        """Create simplified model registration step without complex conditions"""
        
        # Create model registration step (always register for manual approval)
        register_model = RegisterModel(
            name="RegisterContentModerationModel",
            estimator=training_step.estimator,
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["text/csv", "application/json"],
            response_types=["application/json"],
            inference_instances=["ml.t2.medium", "ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name="content-moderation-model-group",
            approval_status="PendingManualApproval",
            depends_on=[evaluation_step]
        )
        
        return register_model
    
    def create_pipeline(self):
        """Create the complete SageMaker pipeline"""
        
        print("🔨 Building SageMaker Pipeline...")
        
        # Create pipeline steps
        validation_step = self.create_data_validation_step()
        training_step = self.create_training_step(validation_step)
        evaluation_step = self.create_evaluation_step(training_step)
        registration_step = self.create_model_registration_step(training_step, evaluation_step)
        
        # Define pipeline
        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=[
                ParameterString(name="TrainingInstanceType", default_value="ml.m5.large"),
                ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
            ],
            steps=[
                validation_step,
                training_step,
                evaluation_step,
                registration_step
            ],
            sagemaker_session=self.sagemaker_session
        )
        
        return pipeline
    
    def deploy_pipeline(self):
        """Deploy the SageMaker pipeline"""
        
        print("🚀 Deploying SageMaker Pipeline...")
        
        try:
            # Create pipeline
            pipeline = self.create_pipeline()
            
            # Upsert pipeline
            pipeline.upsert(role_arn=self.role)
            
            print(f"✅ Pipeline '{self.pipeline_name}' deployed successfully!")
            
        except Exception as e:
            print(f"❌ Pipeline deployment failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        # Save pipeline information
        deployment_time = time.strftime('%Y-%m-%d %H:%M:%S')
        pipeline_arn = pipeline.describe()["PipelineArn"]
        
        pipeline_info = {
            "pipeline_name": self.pipeline_name,
            "pipeline_arn": pipeline_arn,
            "region": self.region,
            "bucket": self.bucket,
            "role": self.role,
            "deployment_time": deployment_time,
            "status": "deployed"
        }
        
        # Save locally
        with open("pipeline_info.json", 'w') as f:
            json.dump(pipeline_info, f, indent=2)
        
        # Save to S3 bucket as well
        try:
            s3_client = boto3.client('s3', region_name=self.region)
            s3_key = f"pipeline/info/{self.pipeline_name}_info.json"
            s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=json.dumps(pipeline_info, indent=2),
                ContentType='application/json'
            )
            print(f"✅ Pipeline info also saved to S3: s3://{self.bucket}/{s3_key}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to save pipeline info to S3: {e}")
        
        print(f"📄 Pipeline info saved to pipeline_info.json")
        
        return pipeline
    
    def execute_pipeline(self, pipeline_execution_name=None):
        """Execute the pipeline"""
        
        if pipeline_execution_name is None:
            pipeline_execution_name = f"content-moderation-execution-{int(time.time())}"
        
        print(f"▶️  Starting pipeline execution: {pipeline_execution_name}")
        
        # Load pipeline info
        with open("pipeline_info.json", 'r') as f:
            pipeline_info = json.load(f)
        
        pipeline_name = pipeline_info["pipeline_name"]
        
        # Start execution
        sm_client = boto3.client('sagemaker', region_name=self.region)
        
        response = sm_client.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineExecutionDisplayName=pipeline_execution_name,
            PipelineParameters=[
                {
                    'Name': 'TrainingInstanceType',
                    'Value': 'ml.m5.large'
                }
            ]
        )
        
        execution_arn = response['PipelineExecutionArn']
        
        print(f"📊 Pipeline execution started!")
        print(f"🔗 Execution ARN: {execution_arn}")
        print(f"🌐 View in console: https://{self.region}.console.aws.amazon.com/sagemaker/home?region={self.region}#/pipelines")
        
        # Save execution info
        execution_info = {
            "execution_arn": execution_arn,
            "execution_name": pipeline_execution_name,
            "pipeline_name": pipeline_name,
            "start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "status": "executing",
            "region": self.region,
            "bucket": self.bucket
        }
        
        # Save locally
        with open("pipeline_execution_info.json", 'w') as f:
            json.dump(execution_info, f, indent=2)
        
        # Save to S3 bucket as well
        try:
            s3_client = boto3.client('sagemaker', region_name=self.region)
            s3_key = f"pipeline/executions/{pipeline_execution_name}_execution.json"
            s3_client = boto3.client('s3', region_name=self.region)
            s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=json.dumps(execution_info, indent=2),
                ContentType='application/json'
            )
            print(f"✅ Execution info also saved to S3: s3://{self.bucket}/{s3_key}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to save execution info to S3: {e}")
        
        return execution_arn
    
    def save_pipeline_model_info(self, execution_arn):
        """Save model information after pipeline execution completes"""
        
        print("🔍 Checking pipeline execution status and saving model info...")
        
        sm_client = boto3.client('sagemaker', region_name=self.region)
        
        try:
            # Get execution details
            execution_response = sm_client.describe_pipeline_execution(
                PipelineExecutionArn=execution_arn
            )
            
            execution_status = execution_response['PipelineExecutionStatus']
            print(f"Pipeline execution status: {execution_status}")
            
            if execution_status == 'Succeeded':
                # List pipeline execution steps to find training job
                steps_response = sm_client.list_pipeline_execution_steps(
                    PipelineExecutionArn=execution_arn
                )
                
                training_job_name = None
                model_data = None
                
                # Find the training step
                for step in steps_response['PipelineExecutionSteps']:
                    if step['StepName'] == 'TrainContentModerationModel':
                        if 'Metadata' in step and 'TrainingJob' in step['Metadata']:
                            training_job_name = step['Metadata']['TrainingJob']['Arn'].split('/')[-1]
                            break
                
                if training_job_name:
                    # Get training job details
                    training_response = sm_client.describe_training_job(
                        TrainingJobName=training_job_name
                    )
                    
                    model_data = training_response['ModelArtifacts']['S3ModelArtifacts']
                    
                    # Create model info similar to launch_training.py
                    model_info = {
                        'model_data': model_data,
                        'training_job_name': training_job_name,
                        'framework_version': '0.23-1',
                        'instance_type': training_response.get('ResourceConfig', {}).get('InstanceType', 'ml.m5.large'),
                        'model_type': 'pipeline',
                        'pipeline_execution_time': time.time(),
                        'pipeline_execution_arn': execution_arn,
                        'training_start_time': training_response.get('TrainingStartTime', '').isoformat() if training_response.get('TrainingStartTime') else '',
                        'training_end_time': training_response.get('TrainingEndTime', '').isoformat() if training_response.get('TrainingEndTime') else '',
                        'training_job_status': training_response.get('TrainingJobStatus', 'Unknown'),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'region': self.region
                    }
                    
                    # Save locally (this will be picked up by deploy_endpoint.py)
                    with open('pipeline_model_info.json', 'w') as f:
                        json.dump(model_info, f, indent=2)
                    
                    # Save model path to model_path.txt for immediate compatibility
                    with open('model_path.txt', 'w') as f:
                        f.write(model_data)
                    
                    # Save to S3 bucket as well
                    try:
                        s3_client = boto3.client('s3', region_name=self.region)
                        s3_key = f"models/pipeline_model_info_{int(time.time())}.json"
                        s3_client.put_object(
                            Bucket=self.bucket,
                            Key=s3_key,
                            Body=json.dumps(model_info, indent=2),
                            ContentType='application/json'
                        )
                        print(f"✅ Pipeline model info also saved to S3: s3://{self.bucket}/{s3_key}")
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to save model info to S3: {e}")
                    
                    print(f"✅ Pipeline model information saved!")
                    print(f"📁 Model location: {model_data}")
                    print(f"🏷️  Training job: {training_job_name}")
                    print(f"📝 Model path also saved to model_path.txt for compatibility")
                    
                    return model_data
                else:
                    print("❌ Could not find training job in pipeline execution")
                    return None
            else:
                print(f"⏳ Pipeline execution not yet completed (status: {execution_status})")
                print("💡 You can run this method later when the pipeline completes")
                return None
                
        except Exception as e:
            print(f"❌ Failed to save pipeline model info: {e}")
            return None

    def approve_latest_model(self):
        """Approve the latest model package in the model registry"""
        
        print("🔍 Finding and approving latest model package...")
        
        sm_client = boto3.client('sagemaker', region_name=self.region)
        
        try:
            # List model packages for our model group
            response = sm_client.list_model_packages(
                ModelPackageGroupName="content-moderation-model-group",
                ModelApprovalStatus="PendingManualApproval",
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=1
            )
            
            if not response['ModelPackageSummaryList']:
                print("❌ No pending model packages found for approval")
                return False
            
            # Get the latest pending model package
            latest_model = response['ModelPackageSummaryList'][0]
            model_package_arn = latest_model['ModelPackageArn']
            
            print(f"📦 Found pending model: {model_package_arn}")
            
            # Approve the model
            sm_client.update_model_package(
                ModelPackageArn=model_package_arn,
                ModelApprovalStatus='Approved'
            )
            
            print(f"✅ Model approved successfully!")
            print(f"🔗 Model Package ARN: {model_package_arn}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to approve model: {e}")
            return False
    
    def list_model_packages(self):
        """List all model packages in the model registry"""
        
        print("📋 Listing model packages...")
        
        sm_client = boto3.client('sagemaker', region_name=self.region)
        
        try:
            # List all model packages
            response = sm_client.list_model_packages(
                ModelPackageGroupName="content-moderation-model-group"
            )
            
            if not response['ModelPackageSummaryList']:
                print("❌ No model packages found")
                return []
            
            models = response['ModelPackageSummaryList']
            
            print(f"📊 Found {len(models)} model packages:")
            print("-" * 80)
            
            for i, model in enumerate(models, 1):
                status = model['ModelApprovalStatus']
                status_emoji = "✅" if status == "Approved" else "⏳" if status == "PendingManualApproval" else "❌"
                
                print(f"{i}. {status_emoji} Status: {status}")
                print(f"   ARN: {model['ModelPackageArn']}")
                print(f"   Created: {model['CreationTime']}")
                if 'ModelPackageDescription' in model:
                    print(f"   Description: {model['ModelPackageDescription']}")
                print("-" * 80)
            
            return models
            
        except Exception as e:
            print(f"❌ Failed to list model packages: {e}")
            return []

def main():
    """Main function to demonstrate pipeline deployment and execution"""
    
    print("🎯 Content Moderation SageMaker Pipeline Setup")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        cm_pipeline = ContentModerationPipeline()
        
        # Deploy pipeline
        pipeline = cm_pipeline.deploy_pipeline()
        
        # Execute pipeline
        execution_arn = cm_pipeline.execute_pipeline()
        
        print("\n" + "=" * 50)
        print("✅ Pipeline Setup Complete!")
        print("=" * 50)
        print(f"📋 Next steps:")
        print(f"1. Monitor execution in SageMaker Console")
        print(f"2. Check pipeline_execution_info.json for status")
        print(f"3. When pipeline completes, model info will be automatically available")
        print(f"4. Run 'python deploy_endpoint.py' to deploy the pipeline model")
        print(f"5. Run 'python bedrock_integration.py' for Bedrock setup")
        print(f"6. Approve model: python -c \"from sagemaker_pipeline import ContentModerationPipeline; p=ContentModerationPipeline(); p.approve_latest_model()\"")
        
        # Optional: Wait for pipeline to complete and save model info
        user_input = input("\n❓ Do you want to wait for pipeline completion and save model info? (y/n): ")
        if user_input.lower() == 'y':
            print("⏳ Waiting for pipeline to complete...")
            import time
            while True:
                model_data = cm_pipeline.save_pipeline_model_info(execution_arn)
                if model_data:
                    print(f"\n🎉 Pipeline model ready for deployment!")
                    
                    # Ask if user wants to approve the model
                    approve_input = input("\n❓ Do you want to approve the model now? (y/n): ")
                    if approve_input.lower() == 'y':
                        cm_pipeline.approve_latest_model()
                    
                    break
                else:
                    print("⏳ Still waiting... (checking again in 60 seconds)")
                    time.sleep(60)
        else:
            print(f"\n💡 You can save model info later by running:")
            print(f"   python -c \"from sagemaker_pipeline import ContentModerationPipeline; p=ContentModerationPipeline(); p.save_pipeline_model_info('{execution_arn}')\"")
            print(f"\n💡 You can approve models by running:")
            print(f"   python -c \"from sagemaker_pipeline import ContentModerationPipeline; p=ContentModerationPipeline(); p.approve_latest_model()\"")
        
    except Exception as e:
        print(f"❌ Pipeline setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
