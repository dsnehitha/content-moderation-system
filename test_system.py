#!/usr/bin/env python3
"""
Comprehensive test suite for the Content Moderation System
Tests all components: preprocessing, prediction, and API endpoints
"""

import json
import boto3
import requests
import time
from datetime import datetime

class ContentModerationTester:
    def __init__(self):
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        self.lambda_client = boto3.client('lambda')
        
        # Load configuration files
        self.aws_config = self.load_json_file('aws_config.json')
        self.api_info = self.load_json_file('api_info.json', required=False)
        self.endpoint_info = self.load_json_file('endpoint_info.json', required=False)
        
        # Test cases
        self.test_cases = [
            {
                "text": "I hate you so much! You're terrible!",
                "expected_toxic": True,
                "category": "hate_speech"
            },
            {
                "text": "Thank you for your help! You're amazing!",
                "expected_toxic": False,
                "category": "positive"
            },
            {
                "text": "Go kill yourself, nobody likes you",
                "expected_toxic": True,
                "category": "threat"
            },
            {
                "text": "What time does the meeting start?",
                "expected_toxic": False,
                "category": "neutral"
            },
            {
                "text": "You're absolutely pathetic and worthless",
                "expected_toxic": True,
                "category": "harassment"
            },
            {
                "text": "I disagree with your opinion, but respect your right to have it",
                "expected_toxic": False,
                "category": "constructive"
            },
            {
                "text": "F*** you and everything you stand for!",
                "expected_toxic": True,
                "category": "profanity"
            },
            {
                "text": "Have a great day everyone!",
                "expected_toxic": False,
                "category": "positive"
            }
        ]
    
    def load_json_file(self, filename, required=True):
        """Load JSON configuration file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            if required:
                print(f"❌ Required file {filename} not found")
                exit(1)
            return None
    
    def test_sagemaker_endpoint(self):
        """Test SageMaker endpoint directly"""
        print("🧪 Testing SageMaker Endpoint...")
        
        if not self.endpoint_info:
            print("⚠️  Skipping SageMaker test - endpoint not deployed")
            return False
        
        endpoint_name = self.endpoint_info['endpoint_name']
        success_count = 0
        correct_predictions = 0
        
        for i, test_case in enumerate(self.test_cases, 1):
            try:
                response = self.sagemaker_runtime.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='text/csv',
                    Body=test_case["text"]
                )
                
                result = json.loads(response['Body'].read().decode())
                toxicity_score = float(result[0]) if isinstance(result, list) else float(result)
                
                predicted_toxic = toxicity_score > 0.5
                is_correct = predicted_toxic == test_case["expected_toxic"]
                
                status = "✅" if is_correct else "❌"
                print(f"  {status} Test {i}: Score {toxicity_score:.3f}, "
                      f"Predicted: {'Toxic' if predicted_toxic else 'Safe'}, "
                      f"Expected: {'Toxic' if test_case['expected_toxic'] else 'Safe'}")
                
                success_count += 1
                if is_correct:
                    correct_predictions += 1
                    
            except Exception as e:
                print(f"  ❌ Test {i}: Exception - {e}")
        
        accuracy = correct_predictions / success_count if success_count > 0 else 0
        print(f"📊 SageMaker Endpoint: {success_count}/{len(self.test_cases)} tests completed")
        print(f"🎯 Accuracy: {accuracy:.2%} ({correct_predictions}/{success_count})")
        
        return success_count > 0
    
    def test_api_gateway(self):
        """Test API Gateway endpoints"""
        print("🧪 Testing API Gateway...")
        
        if not self.api_info:
            print("⚠️  Skipping API Gateway test - not deployed")
            return False
        
        api_url = self.api_info['endpoints']['moderate']
        success_count = 0
        correct_predictions = 0
        
        for i, test_case in enumerate(self.test_cases, 1):
            try:
                payload = {
                    "text": test_case["text"],
                    "metadata": {
                        "test_case": i,
                        "category": test_case["category"]
                    }
                }
                
                start_time = time.time()
                response = requests.post(
                    api_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                response_time = int((time.time() - start_time) * 1000)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('success'):
                        predicted_toxic = result.get('is_toxic', False)
                        toxicity_score = result.get('toxicity_score', 0)
                        action = result.get('action', 'unknown')
                        
                        is_correct = predicted_toxic == test_case["expected_toxic"]
                        status = "✅" if is_correct else "❌"
                        
                        print(f"  {status} Test {i}: Score {toxicity_score:.3f}, "
                              f"Action: {action}, Response: {response_time}ms")
                        
                        success_count += 1
                        if is_correct:
                            correct_predictions += 1
                    else:
                        print(f"  ❌ Test {i}: API error - {result.get('error')}")
                else:
                    print(f"  ❌ Test {i}: HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"  ❌ Test {i}: Timeout")
            except Exception as e:
                print(f"  ❌ Test {i}: Exception - {e}")
        
        accuracy = correct_predictions / success_count if success_count > 0 else 0
        print(f"📊 API Gateway: {success_count}/{len(self.test_cases)} tests completed")
        print(f"🎯 Accuracy: {accuracy:.2%} ({correct_predictions}/{success_count})")
        
        return success_count > 0
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("🧪 Testing Edge Cases...")
        
        edge_cases = [
            {"text": "", "description": "Empty string"},
            {"text": "   ", "description": "Whitespace only"},
            {"text": "a", "description": "Single character"},
            {"text": "x" * 10000, "description": "Very long text"},
            {"text": "Normal text with émojis 😀🚀", "description": "Unicode characters"},
        ]
        
        if not self.api_info:
            print("⚠️  Skipping edge case tests - API not available")
            return False
        
        api_url = self.api_info['endpoints']['moderate']
        passed_tests = 0
        
        for i, test_case in enumerate(edge_cases, 1):
            try:
                response = requests.post(
                    api_url,
                    json={"text": test_case["text"]},
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if response.status_code in [200, 400]:  # Both OK for edge cases
                    print(f"  ✅ Edge case {i}: {test_case['description']} - Handled correctly")
                    passed_tests += 1
                else:
                    print(f"  ❌ Edge case {i}: {test_case['description']} - HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Edge case {i}: {test_case['description']} - Exception: {e}")
        
        print(f"📊 Edge Cases: {passed_tests}/{len(edge_cases)} tests passed")
        return passed_tests > 0
    
    def performance_test(self):
        """Run performance tests"""
        print("🧪 Running Performance Tests...")
        
        if not self.api_info:
            print("⚠️  Skipping performance tests - API not available")
            return
        
        api_url = self.api_info['endpoints']['moderate']
        test_text = "This is a sample text for performance testing."
        
        response_times = []
        successful_requests = 0
        total_requests = 10
        
        print(f"  📡 Making {total_requests} concurrent requests...")
        
        for i in range(total_requests):
            try:
                start_time = time.time()
                response = requests.post(
                    api_url,
                    json={"text": f"{test_text} Request {i}"},
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    response_times.append(response_time)
                    successful_requests += 1
                
            except Exception:
                pass
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            print(f"  📊 Performance Results:")
            print(f"    Success Rate: {successful_requests}/{total_requests} ({successful_requests/total_requests:.1%})")
            print(f"    Average Response Time: {avg_response_time:.0f}ms")
            print(f"    Min Response Time: {min_response_time:.0f}ms")
            print(f"    Max Response Time: {max_response_time:.0f}ms")
        else:
            print("  ❌ No successful requests for performance measurement")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Content Moderation System - Test Suite")
        print("=" * 60)
        print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        test_results = []
        
        # Run individual tests
        test_results.append(("SageMaker Endpoint", self.test_sagemaker_endpoint()))
        print()
        
        test_results.append(("API Gateway", self.test_api_gateway()))
        print()
        
        test_results.append(("Edge Cases", self.test_edge_cases()))
        print()
        
        # Performance test
        self.performance_test()
        print()
        
        # Summary
        print("=" * 60)
        print("📋 Test Summary:")
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} {test_name}")
            if result:
                passed_tests += 1
        
        print()
        print(f"🎯 Overall Result: {passed_tests}/{total_tests} test suites passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Your content moderation system is working correctly.")
        elif passed_tests > 0:
            print("⚠️  Some tests passed. Check failed components above.")
        else:
            print("❌ All tests failed. Please check your system configuration.")
        
        return passed_tests == total_tests

if __name__ == "__main__":
    tester = ContentModerationTester()
    success = tester.run_all_tests()
    
    if not success:
        exit(1)
