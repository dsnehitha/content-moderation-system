#!/usr/bin/env python3
"""
Comprehensive System Test for Content Moderation Day 2
====================================================

This script tests all Day 2 components:
- SageMaker Pipeline execution
- Bedrock integration
- CloudWatch metrics
- Step Functions workflow
- End-to-end system performance
"""

import boto3
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional

class ComprehensiveSystemTest:
    """Complete system testing for Day 2 features"""
    
    def __init__(self):
        # Initialize AWS clients
        self.sagemaker = boto3.client('sagemaker')
        self.stepfunctions = boto3.client('stepfunctions')
        self.cloudwatch = boto3.client('cloudwatch')
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Load configurations
        self.configs = self._load_configurations()
        
        # Test results
        self.test_results = {
            "day1_components": {},
            "sagemaker_pipeline": {},
            "bedrock_integration": {},
            "cloudwatch_monitoring": {},
            "step_functions": {},
            "end_to_end": {},
            "performance": {}
        }
        
        print("🧪 Comprehensive Content Moderation System Test")
        print("=" * 60)
    
    def _load_configurations(self):
        """Load all configuration files"""
        config_files = [
            'aws_config.json',
            'endpoint_info.json',
            'pipeline_info.json',
            'monitoring_config.json',
            'orchestration_config.json'
        ]
        
        configs = {}
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    configs[config_file.replace('.json', '')] = json.load(f)
                print(f"✅ Loaded {config_file}")
            except FileNotFoundError:
                print(f"⚠️  {config_file} not found - some tests may be skipped")
                configs[config_file.replace('.json', '')] = None
        
        return configs
    
    def test_day1_components(self):
        """Test Day 1 components are still working"""
        
        print("\n🔍 Testing Day 1 Components...")
        
        results = {}
        
        # Test SageMaker endpoint
        if self.configs.get('endpoint_info'):
            endpoint_name = self.configs['endpoint_info']['endpoint_name']
            results['sagemaker_endpoint'] = self._test_sagemaker_endpoint(endpoint_name)
        else:
            results['sagemaker_endpoint'] = {"status": "skipped", "reason": "No endpoint info"}
        
        # Test basic functionality
        test_text = "This is a test message."
        if results['sagemaker_endpoint']['status'] == 'passed':
            try:
                sagemaker_runtime = boto3.client('sagemaker-runtime')
                response = sagemaker_runtime.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='text/csv',
                    Body=test_text
                )
                
                result = json.loads(response['Body'].read().decode())
                toxicity_score = float(result[0]) if isinstance(result, list) else float(result)
                
                results['basic_prediction'] = {
                    "status": "passed",
                    "toxicity_score": toxicity_score,
                    "test_text": test_text
                }
                
            except Exception as e:
                results['basic_prediction'] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        self.test_results["day1_components"] = results
        return results
    
    def _test_sagemaker_endpoint(self, endpoint_name):
        """Test if SageMaker endpoint is available"""
        try:
            response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            if status == 'InService':
                return {
                    "status": "passed",
                    "endpoint_status": status,
                    "instance_type": response.get('ProductionVariants', [{}])[0].get('InstanceType', 'unknown')
                }
            else:
                return {
                    "status": "failed",
                    "endpoint_status": status,
                    "reason": "Endpoint not in service"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_sagemaker_pipeline(self):
        """Test SageMaker Pipeline functionality"""
        
        print("\n📊 Testing SageMaker Pipeline...")
        
        results = {}
        
        if not self.configs.get('pipeline_info'):
            results = {"status": "skipped", "reason": "Pipeline not deployed"}
            self.test_results["sagemaker_pipeline"] = results
            return results
        
        pipeline_name = self.configs['pipeline_info']['pipeline_name']
        
        try:
            # Check pipeline exists
            response = self.sagemaker.describe_pipeline(PipelineName=pipeline_name)
            
            results['pipeline_status'] = {
                "status": "passed",
                "pipeline_name": pipeline_name,
                "pipeline_status": response['PipelineStatus']
            }
            
            # Check recent executions
            executions = self.sagemaker.list_pipeline_executions(
                PipelineName=pipeline_name,
                MaxResults=5
            )
            
            if executions['PipelineExecutionSummaries']:
                latest_execution = executions['PipelineExecutionSummaries'][0]
                results['recent_execution'] = {
                    "status": "passed",
                    "execution_status": latest_execution['PipelineExecutionStatus'],
                    "start_time": latest_execution['StartTime'].isoformat()
                }
            else:
                results['recent_execution'] = {
                    "status": "no_executions",
                    "message": "No pipeline executions found"
                }
            
        except Exception as e:
            results['pipeline_status'] = {
                "status": "failed",
                "error": str(e)
            }
        
        self.test_results["sagemaker_pipeline"] = results
        return results
    
    def test_bedrock_integration(self):
        """Test Bedrock integration"""
        
        print("\n🤖 Testing Bedrock Integration...")
        
        results = {}
        
        # Test Bedrock model availability
        try:
            bedrock_client = boto3.client('bedrock', region_name='us-east-1')
            models = bedrock_client.list_foundation_models()
            
            claude_models = [m for m in models['modelSummaries'] 
                           if 'claude' in m['modelId'].lower()]
            
            if claude_models:
                results['model_availability'] = {
                    "status": "passed",
                    "available_models": len(claude_models),
                    "sample_model": claude_models[0]['modelId']
                }
            else:
                results['model_availability'] = {
                    "status": "no_models",
                    "message": "No Claude models available"
                }
                
        except Exception as e:
            results['model_availability'] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test enhanced moderation function
        try:
            from bedrock_integration import BedrockModerationIntegration
            
            integration = BedrockModerationIntegration()
            
            test_cases = [
                "You are such an idiot!",
                "That movie was terrible.",
                "Thank you for your help!"
            ]
            
            test_results = []
            for text in test_cases:
                try:
                    result = integration.moderate_content_enhanced(text, use_bedrock=True)
                    test_results.append({
                        "text": text,
                        "status": "passed",
                        "ml_score": result.get('ml_score', 0),
                        "final_score": result.get('final_toxicity_score', 0),
                        "action": result.get('final_action', 'unknown'),
                        "bedrock_enhanced": result.get('bedrock_analysis', {}).get('bedrock_enhanced', False)
                    })
                except Exception as e:
                    test_results.append({
                        "text": text,
                        "status": "failed",
                        "error": str(e)
                    })
            
            results['enhanced_moderation'] = {
                "status": "passed",
                "test_cases": test_results
            }
            
        except Exception as e:
            results['enhanced_moderation'] = {
                "status": "failed",
                "error": str(e)
            }
        
        self.test_results["bedrock_integration"] = results
        return results
    
    def test_cloudwatch_monitoring(self):
        """Test CloudWatch monitoring setup"""
        
        print("\n📈 Testing CloudWatch Monitoring...")
        
        results = {}
        
        if not self.configs.get('monitoring_config'):
            results = {"status": "skipped", "reason": "Monitoring not configured"}
            self.test_results["cloudwatch_monitoring"] = results
            return results
        
        monitoring_config = self.configs['monitoring_config']
        
        # Test dashboard exists
        try:
            dashboard_name = monitoring_config.get('dashboard_name')
            if dashboard_name:
                response = self.cloudwatch.get_dashboard(DashboardName=dashboard_name)
                results['dashboard'] = {
                    "status": "passed",
                    "dashboard_name": dashboard_name
                }
            else:
                results['dashboard'] = {
                    "status": "not_configured",
                    "message": "No dashboard name in config"
                }
                
        except Exception as e:
            results['dashboard'] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test custom metrics
        try:
            metrics = self.cloudwatch.list_metrics(
                Namespace='ContentModeration'
            )
            
            metric_names = [m['MetricName'] for m in metrics['Metrics']]
            
            results['custom_metrics'] = {
                "status": "passed",
                "metric_count": len(metric_names),
                "sample_metrics": metric_names[:5]
            }
            
        except Exception as e:
            results['custom_metrics'] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test alarms
        try:
            alarms = self.cloudwatch.describe_alarms(
                AlarmNamePrefix='ContentModeration'
            )
            
            alarm_names = [a['AlarmName'] for a in alarms['MetricAlarms']]
            
            results['alarms'] = {
                "status": "passed",
                "alarm_count": len(alarm_names),
                "alarm_names": alarm_names
            }
            
        except Exception as e:
            results['alarms'] = {
                "status": "failed",
                "error": str(e)
            }
        
        self.test_results["cloudwatch_monitoring"] = results
        return results
    
    def test_step_functions(self):
        """Test Step Functions workflow"""
        
        print("\n🔄 Testing Step Functions Workflow...")
        
        results = {}
        
        if not self.configs.get('orchestration_config'):
            results = {"status": "skipped", "reason": "Step Functions not deployed"}
            self.test_results["step_functions"] = results
            return results
        
        state_machine_arn = self.configs['orchestration_config']['state_machine_arn']
        
        # Test state machine exists
        try:
            response = self.stepfunctions.describe_state_machine(
                stateMachineArn=state_machine_arn
            )
            
            results['state_machine'] = {
                "status": "passed",
                "name": response['name'],
                "status": response['status']
            }
            
        except Exception as e:
            results['state_machine'] = {
                "status": "failed",
                "error": str(e)
            }
            self.test_results["step_functions"] = results
            return results
        
        # Test workflow execution
        test_cases = [
            {
                "text": "You are an idiot!",
                "metadata": {"source": "test", "test_case": "toxic"}
            },
            {
                "text": "Thank you for your help!",
                "metadata": {"source": "test", "test_case": "benign"}
            }
        ]
        
        execution_results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                execution_name = f"comprehensive-test-{int(time.time())}-{i}"
                
                # Start execution
                start_response = self.stepfunctions.start_execution(
                    stateMachineArn=state_machine_arn,
                    name=execution_name,
                    input=json.dumps(test_case)
                )
                
                execution_arn = start_response['executionArn']
                
                # Wait for completion (max 30 seconds)
                timeout = 30
                elapsed = 0
                
                while elapsed < timeout:
                    time.sleep(2)
                    elapsed += 2
                    
                    status_response = self.stepfunctions.describe_execution(
                        executionArn=execution_arn
                    )
                    
                    status = status_response['status']
                    
                    if status in ['SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED']:
                        break
                
                execution_results.append({
                    "test_case": test_case['text'],
                    "execution_status": status,
                    "execution_arn": execution_arn,
                    "elapsed_time": elapsed
                })
                
            except Exception as e:
                execution_results.append({
                    "test_case": test_case['text'],
                    "execution_status": "failed",
                    "error": str(e)
                })
        
        results['workflow_executions'] = {
            "status": "passed" if execution_results else "failed",
            "executions": execution_results
        }
        
        self.test_results["step_functions"] = results
        return results
    
    def test_end_to_end_performance(self):
        """Test end-to-end system performance"""
        
        print("\n⚡ Testing End-to-End Performance...")
        
        results = {}
        
        # Performance test cases
        test_cases = [
            "This is a simple test message.",
            "You are such an idiot, I hate you!",
            "That movie was absolutely terrible, what a waste of time.",
            "Thank you so much for your help, you're amazing!",
            "I'm going to destroy you in this video game!"
        ]
        
        # Test direct SageMaker endpoint performance
        if self.configs.get('endpoint_info'):
            endpoint_name = self.configs['endpoint_info']['endpoint_name']
            sagemaker_runtime = boto3.client('sagemaker-runtime')
            
            sagemaker_times = []
            sagemaker_results = []
            
            for text in test_cases:
                try:
                    start_time = time.time()
                    
                    response = sagemaker_runtime.invoke_endpoint(
                        EndpointName=endpoint_name,
                        ContentType='text/csv',
                        Body=text
                    )
                    
                    result = json.loads(response['Body'].read().decode())
                    toxicity_score = float(result[0]) if isinstance(result, list) else float(result)
                    
                    latency = (time.time() - start_time) * 1000
                    sagemaker_times.append(latency)
                    
                    sagemaker_results.append({
                        "text": text,
                        "toxicity_score": toxicity_score,
                        "latency_ms": latency,
                        "status": "success"
                    })
                    
                except Exception as e:
                    sagemaker_results.append({
                        "text": text,
                        "status": "failed",
                        "error": str(e)
                    })
            
            results['sagemaker_performance'] = {
                "status": "passed",
                "avg_latency_ms": sum(sagemaker_times) / len(sagemaker_times) if sagemaker_times else 0,
                "min_latency_ms": min(sagemaker_times) if sagemaker_times else 0,
                "max_latency_ms": max(sagemaker_times) if sagemaker_times else 0,
                "success_rate": len([r for r in sagemaker_results if r.get('status') == 'success']) / len(test_cases),
                "results": sagemaker_results
            }
        
        # Test Bedrock integration performance
        try:
            from bedrock_integration import BedrockModerationIntegration
            integration = BedrockModerationIntegration()
            
            bedrock_times = []
            bedrock_results = []
            
            for text in test_cases:
                try:
                    start_time = time.time()
                    
                    result = integration.moderate_content_enhanced(text, use_bedrock=True)
                    
                    latency = (time.time() - start_time) * 1000
                    bedrock_times.append(latency)
                    
                    bedrock_results.append({
                        "text": text,
                        "final_score": result.get('final_toxicity_score', 0),
                        "action": result.get('final_action', 'unknown'),
                        "bedrock_enhanced": result.get('bedrock_analysis', {}).get('bedrock_enhanced', False),
                        "latency_ms": latency,
                        "status": "success"
                    })
                    
                except Exception as e:
                    bedrock_results.append({
                        "text": text,
                        "status": "failed",
                        "error": str(e)
                    })
            
            results['bedrock_performance'] = {
                "status": "passed",
                "avg_latency_ms": sum(bedrock_times) / len(bedrock_times) if bedrock_times else 0,
                "min_latency_ms": min(bedrock_times) if bedrock_times else 0,
                "max_latency_ms": max(bedrock_times) if bedrock_times else 0,
                "success_rate": len([r for r in bedrock_results if r.get('status') == 'success']) / len(test_cases),
                "bedrock_usage_rate": len([r for r in bedrock_results if r.get('bedrock_enhanced')]) / len(test_cases),
                "results": bedrock_results
            }
            
        except Exception as e:
            results['bedrock_performance'] = {
                "status": "failed",
                "error": str(e)
            }
        
        self.test_results["performance"] = results
        return results
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        # Overall summary
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            if isinstance(tests, dict):
                for test_name, test_result in tests.items():
                    total_tests += 1
                    if isinstance(test_result, dict) and test_result.get('status') == 'passed':
                        passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"🎯 Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)")
        print(f"📅 Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Category summaries
        print(f"\n📊 Test Category Results:")
        for category, tests in self.test_results.items():
            if isinstance(tests, dict):
                category_passed = sum(1 for test in tests.values() 
                                    if isinstance(test, dict) and test.get('status') == 'passed')
                category_total = len(tests)
                category_rate = (category_passed / category_total * 100) if category_total > 0 else 0
                
                status_icon = "✅" if category_rate >= 80 else "⚠️" if category_rate >= 50 else "❌"
                category_name = category.replace('_', ' ').title()
                
                print(f"   {status_icon} {category_name}: {category_rate:.0f}% ({category_passed}/{category_total})")
        
        # Performance summary
        if 'performance' in self.test_results:
            perf = self.test_results['performance']
            
            print(f"\n⚡ Performance Summary:")
            
            if 'sagemaker_performance' in perf:
                sm_perf = perf['sagemaker_performance']
                print(f"   🔬 SageMaker Endpoint:")
                print(f"      Average Latency: {sm_perf.get('avg_latency_ms', 0):.1f}ms")
                print(f"      Success Rate: {sm_perf.get('success_rate', 0)*100:.1f}%")
            
            if 'bedrock_performance' in perf:
                br_perf = perf['bedrock_performance']
                print(f"   🤖 Bedrock Integration:")
                print(f"      Average Latency: {br_perf.get('avg_latency_ms', 0):.1f}ms")
                print(f"      Success Rate: {br_perf.get('success_rate', 0)*100:.1f}%")
                print(f"      Bedrock Usage: {br_perf.get('bedrock_usage_rate', 0)*100:.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if success_rate >= 90:
            print("   🎉 Excellent! Your system is performing optimally.")
            print("   🚀 Ready for production deployment.")
        elif success_rate >= 70:
            print("   👍 Good performance with minor issues.")
            print("   🔧 Review failed tests and optimize as needed.")
        else:
            print("   ⚠️  Several components need attention.")
            print("   🛠️  Focus on fixing critical components first.")
        
        # Failed test details
        failed_tests = []
        for category, tests in self.test_results.items():
            if isinstance(tests, dict):
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict) and test_result.get('status') == 'failed':
                        failed_tests.append({
                            'category': category,
                            'test': test_name,
                            'error': test_result.get('error', 'Unknown error')
                        })
        
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for fail in failed_tests[:5]:  # Show first 5 failures
                print(f"   • {fail['category']}.{fail['test']}: {fail['error'][:100]}...")
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": success_rate
            },
            "results": self.test_results
        }
        
        with open('comprehensive_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: comprehensive_test_report.json")
        
        return success_rate >= 70
    
    def run_all_tests(self):
        """Run all comprehensive tests"""
        
        print("🚀 Starting comprehensive system test...")
        
        # Test Day 1 components
        self.test_day1_components()
        
        # Test Day 2 components
        self.test_sagemaker_pipeline()
        self.test_bedrock_integration()
        self.test_cloudwatch_monitoring()
        self.test_step_functions()
        
        # Performance tests
        self.test_end_to_end_performance()
        
        # Generate report
        success = self.generate_test_report()
        
        return success

def main():
    """Main function"""
    
    tester = ComprehensiveSystemTest()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎊 All tests completed successfully!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the report for details.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
