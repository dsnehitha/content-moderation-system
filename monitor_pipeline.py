#!/usr/bin/env python3
"""
SageMaker Pipeline Monitoring Script
====================================

🎯 PURPOSE: Monitor SageMaker Pipeline executions during model training

📊 MONITORS:
- Pipeline execution status (Running/Succeeded/Failed)
- Individual step progress (Data validation, Training, Evaluation) 
- Step failure reasons and training logs
- Training job completion times

⏱️  DURATION: Temporary (runs while pipeline executes - minutes to hours)

🚀 USAGE:
    # Start monitoring active pipeline
    python monitor_pipeline.py
    
    # Get execution summary
    python monitor_pipeline.py --summary

💡 NOTE: Use cloudwatch_monitoring.py for 24/7 production monitoring
"""

import json
import boto3
import time
from datetime import datetime

def monitor_pipeline_execution():
    """Monitor the pipeline execution status"""
    
    # Load execution info
    try:
        with open("pipeline_execution_info.json", 'r') as f:
            execution_info = json.load(f)
    except FileNotFoundError:
        print("❌ No pipeline execution found. Run sagemaker_pipeline.py first.")
        return
    
    execution_arn = execution_info["execution_arn"]
    execution_name = execution_info["execution_name"]
    
    print(f"🔍 Monitoring Pipeline Execution: {execution_name}")
    print(f"🔗 Execution ARN: {execution_arn}")
    print("=" * 60)
    
    # Initialize SageMaker client
    sm_client = boto3.client('sagemaker')
    
    last_status = None
    step_statuses = {}
    
    while True:
        try:
            # Get execution status
            response = sm_client.describe_pipeline_execution(
                PipelineExecutionArn=execution_arn
            )
            
            current_status = response['PipelineExecutionStatus']
            
            # Print status update if changed
            if current_status != last_status:
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] 📊 Pipeline Status: {current_status}")
                last_status = current_status
            
            # Get step details
            steps_response = sm_client.list_pipeline_execution_steps(
                PipelineExecutionArn=execution_arn
            )
            
            # Check for step status changes
            for step in steps_response['PipelineExecutionSteps']:
                step_name = step['StepName']
                step_status = step['StepStatus']
                
                if step_name not in step_statuses or step_statuses[step_name] != step_status:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    if step_status == 'Executing':
                        print(f"[{timestamp}] ▶️  Step '{step_name}': {step_status}")
                    elif step_status == 'Succeeded':
                        print(f"[{timestamp}] ✅ Step '{step_name}': {step_status}")
                    elif step_status == 'Failed':
                        print(f"[{timestamp}] ❌ Step '{step_name}': {step_status}")
                        if 'FailureReason' in step:
                            print(f"    💡 Reason: {step['FailureReason']}")
                    else:
                        print(f"[{timestamp}] 📋 Step '{step_name}': {step_status}")
                    
                    step_statuses[step_name] = step_status
            
            # Check if pipeline completed
            if current_status in ['Succeeded', 'Failed', 'Stopped']:
                print("\n" + "=" * 60)
                if current_status == 'Succeeded':
                    print("🎉 Pipeline execution completed successfully!")
                    
                    # Get final step statuses
                    print("\n📊 Final Step Summary:")
                    for step_name, status in step_statuses.items():
                        status_icon = "✅" if status == "Succeeded" else "❌" if status == "Failed" else "⏸️"
                        print(f"   {status_icon} {step_name}: {status}")
                    
                elif current_status == 'Failed':
                    print("❌ Pipeline execution failed!")
                    
                    # Show failed steps
                    print("\n📊 Failed Steps:")
                    for step_name, status in step_statuses.items():
                        if status == "Failed":
                            print(f"   ❌ {step_name}")
                
                else:
                    print(f"⏸️ Pipeline execution {current_status.lower()}")
                
                break
            
            # Wait before next check
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n⚠️ Monitoring interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error monitoring pipeline: {e}")
            break
    
    print(f"\n🌐 View details in console:")
    print(f"   https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/pipelines")

def get_pipeline_summary():
    """Get a summary of pipeline execution"""
    
    try:
        with open("pipeline_execution_info.json", 'r') as f:
            execution_info = json.load(f)
    except FileNotFoundError:
        print("❌ No pipeline execution found.")
        return
    
    execution_arn = execution_info["execution_arn"]
    sm_client = boto3.client('sagemaker')
    
    try:
        # Get execution details
        response = sm_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )
        
        print("📊 Pipeline Execution Summary")
        print("=" * 40)
        print(f"Name: {execution_info['execution_name']}")
        print(f"Status: {response['PipelineExecutionStatus']}")
        print(f"Started: {execution_info['start_time']}")
        
        if 'CreationTime' in response:
            print(f"AWS Start Time: {response['CreationTime']}")
        if 'LastModifiedTime' in response:
            print(f"Last Modified: {response['LastModifiedTime']}")
        
        # Get step details
        steps_response = sm_client.list_pipeline_execution_steps(
            PipelineExecutionArn=execution_arn
        )
        
        print(f"\n📋 Steps ({len(steps_response['PipelineExecutionSteps'])}):")
        for step in steps_response['PipelineExecutionSteps']:
            step_name = step['StepName']
            step_status = step['StepStatus']
            status_icon = "✅" if step_status == "Succeeded" else "❌" if step_status == "Failed" else "▶️" if step_status == "Executing" else "⏸️"
            print(f"   {status_icon} {step_name}: {step_status}")
            
            if step_status == 'Failed' and 'FailureReason' in step:
                print(f"      💡 {step['FailureReason']}")
        
    except Exception as e:
        print(f"❌ Error getting pipeline summary: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        get_pipeline_summary()
    else:
        monitor_pipeline_execution()
