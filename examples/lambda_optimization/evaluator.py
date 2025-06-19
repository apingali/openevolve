"""
Evaluator for AWS Lambda function cost optimization
Measures execution time, memory usage, API calls, and estimates costs
"""

import importlib.util
import json
import time
import tracemalloc
import psutil
import os
from unittest.mock import Mock, patch
from typing import Dict, Any
import tempfile
import uuid


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Evaluate Lambda function for cost optimization metrics
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of metrics including cost estimates and performance scores
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("lambda_function", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Create test data and environment
        test_event, test_context = create_test_environment()
        
        # Track metrics
        metrics = {}
        
        # Memory tracking setup
        tracemalloc.start()
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Mock AWS services to avoid actual calls
        with patch_aws_services() as (s3_mock, dynamodb_mock):
            
            # Execution timing
            start_time = time.time()
            
            # Execute the Lambda function
            result = module.lambda_handler(test_event, test_context)
            
            execution_time = time.time() - start_time
            
            # Memory usage
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            memory_used_mb = peak_memory / (1024 * 1024)  # Convert to MB
            
            # Count API calls from mocks
            s3_calls = len(s3_mock.get_object.call_args_list)
            dynamodb_puts = count_dynamodb_operations(dynamodb_mock)
            
            # AWS Cost calculations (based on actual AWS pricing)
            cost_metrics = calculate_aws_costs(
                execution_time, memory_used_mb, s3_calls, dynamodb_puts
            )
            
            # Functionality verification
            functionality_score = verify_functionality(result, test_event)
            
            # Performance scores (higher is better)
            execution_score = calculate_execution_score(execution_time)
            memory_score = calculate_memory_score(memory_used_mb)
            cost_score = calculate_cost_score(cost_metrics['total_cost'])
            api_efficiency_score = calculate_api_efficiency_score(s3_calls, dynamodb_puts)
            
            # Combined business value score
            # Prioritizes cost reduction while maintaining functionality
            combined_score = (
                0.35 * cost_score +           # Primary: cost optimization
                0.25 * execution_score +      # Performance matters for user experience
                0.20 * api_efficiency_score + # API call optimization
                0.15 * memory_score +         # Memory efficiency
                0.05 * functionality_score    # Must work correctly
            )
            
            return {
                # Raw metrics
                'execution_time_ms': execution_time * 1000,
                'memory_used_mb': memory_used_mb,
                's3_api_calls': s3_calls,
                'dynamodb_operations': dynamodb_puts,
                
                # Cost estimates
                'lambda_cost_per_invocation': cost_metrics['lambda_cost'],
                's3_cost_per_invocation': cost_metrics['s3_cost'],
                'dynamodb_cost_per_invocation': cost_metrics['dynamodb_cost'],
                'total_cost_per_invocation': cost_metrics['total_cost'],
                'monthly_cost_estimate_1m_invocations': cost_metrics['total_cost'] * 1_000_000,
                
                # Scores (higher is better)
                'execution_score': execution_score,
                'memory_score': memory_score,
                'cost_score': cost_score,
                'api_efficiency_score': api_efficiency_score,
                'functionality_score': functionality_score,
                'combined_score': combined_score,
                
                # Business metrics
                'cost_savings_vs_baseline': calculate_cost_savings(cost_metrics['total_cost']),
                'performance_improvement': calculate_performance_improvement(execution_time)
            }
            
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return {
            'execution_time_ms': 10000.0,  # Penalty for broken code
            'memory_used_mb': 1000.0,
            'total_cost_per_invocation': 1.0,
            'combined_score': 0.0,
            'functionality_score': 0.0,
            'error': str(e)
        }


def create_test_environment():
    """Create test S3 event and Lambda context"""
    
    # Sample S3 event with multiple records
    test_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'test-analytics-bucket'},
                    'object': {'key': f'user-data/batch-{i}.json'}
                }
            }
            for i in range(3)  # Test with multiple files
        ]
    }
    
    # Mock Lambda context
    test_context = Mock()
    test_context.aws_request_id = str(uuid.uuid4())
    test_context.function_name = 'test-analytics-processor'
    test_context.memory_limit_in_mb = 512
    test_context.remaining_time_in_millis = lambda: 30000
    
    return test_event, test_context


def patch_aws_services():
    """Mock AWS services to simulate operations without actual calls"""
    
    # Create sample user data for testing
    sample_data = {}
    for user_id in range(100):  # 100 users per file
        sessions = []
        for session in range(5):  # 5 sessions per user
            sessions.append({
                'duration': 120 + (session * 30),
                'page_views': 5 + session,
                'timestamp': 1640995200 + (session * 86400)  # Different days
            })
        
        sample_data[f'user_{user_id}'] = {'sessions': sessions}
    
    # Mock S3
    s3_mock = Mock()
    s3_mock.get_object.return_value = {
        'Body': Mock(read=lambda: json.dumps(sample_data).encode())
    }
    
    # Mock DynamoDB
    dynamodb_mock = Mock()
    table_mock = Mock()
    table_mock.put_item = Mock()
    table_mock.batch_writer = Mock()
    
    # Configure batch writer context manager
    batch_writer_mock = Mock()
    batch_writer_mock.put_item = Mock()
    table_mock.batch_writer.return_value.__enter__ = Mock(return_value=batch_writer_mock)
    table_mock.batch_writer.return_value.__exit__ = Mock(return_value=None)
    
    dynamodb_mock.Table.return_value = table_mock
    
    # Return context manager for patches
    return patch.multiple(
        'boto3',
        client=Mock(return_value=s3_mock),
        resource=Mock(return_value=dynamodb_mock)
    ), (s3_mock, dynamodb_mock)


def count_dynamodb_operations(dynamodb_mock):
    """Count DynamoDB operations from mock calls"""
    table_mock = dynamodb_mock.Table.return_value
    
    # Count individual put_item calls
    individual_puts = len(table_mock.put_item.call_args_list)
    
    # Count batch operations
    batch_operations = 0
    if table_mock.batch_writer.called:
        batch_writer = table_mock.batch_writer.return_value.__enter__.return_value
        batch_operations = len(batch_writer.put_item.call_args_list)
    
    return individual_puts + batch_operations


def calculate_aws_costs(execution_time_s, memory_mb, s3_calls, dynamodb_puts):
    """Calculate AWS costs based on actual pricing"""
    
    # Lambda pricing (as of 2024)
    # $0.0000166667 per GB-second
    # $0.0000002 per request
    memory_gb = memory_mb / 1024
    lambda_compute_cost = memory_gb * execution_time_s * 0.0000166667
    lambda_request_cost = 0.0000002
    lambda_cost = lambda_compute_cost + lambda_request_cost
    
    # S3 pricing
    # $0.0004 per 1,000 GET requests
    s3_cost = (s3_calls / 1000) * 0.0004
    
    # DynamoDB pricing
    # $0.25 per million write requests
    dynamodb_cost = (dynamodb_puts / 1_000_000) * 0.25
    
    total_cost = lambda_cost + s3_cost + dynamodb_cost
    
    return {
        'lambda_cost': lambda_cost,
        's3_cost': s3_cost,
        'dynamodb_cost': dynamodb_cost,
        'total_cost': total_cost
    }


def calculate_execution_score(execution_time_s):
    """Score execution time (higher is better)"""
    # Target: under 1 second gets full score
    target_time = 1.0
    if execution_time_s <= target_time:
        return 10.0
    else:
        # Exponential decay for longer times
        return max(0.1, 10.0 * (target_time / execution_time_s) ** 2)


def calculate_memory_score(memory_mb):
    """Score memory usage (higher is better)"""
    # Target: under 128MB gets full score
    target_memory = 128.0
    if memory_mb <= target_memory:
        return 10.0
    else:
        return max(0.1, 10.0 * (target_memory / memory_mb))


def calculate_cost_score(total_cost):
    """Score total cost (higher is better)"""
    # Target: under $0.00001 per invocation gets full score
    target_cost = 0.00001
    if total_cost <= target_cost:
        return 10.0
    else:
        return max(0.1, 10.0 * (target_cost / total_cost))


def calculate_api_efficiency_score(s3_calls, dynamodb_puts):
    """Score API call efficiency (higher is better)"""
    # Target: minimize API calls
    # 3 S3 calls (for 3 files) and 1 batch DynamoDB operation per 100 users = ideal
    target_s3 = 3
    target_dynamodb = 3  # Allow some individual puts, but prefer batching
    
    s3_efficiency = min(10.0, 10.0 * (target_s3 / max(s3_calls, 1)))
    dynamodb_efficiency = min(10.0, 10.0 * (target_dynamodb / max(dynamodb_puts, 1)))
    
    return (s3_efficiency + dynamodb_efficiency) / 2


def verify_functionality(result, test_event):
    """Verify the function produces correct output"""
    try:
        if not result or 'statusCode' not in result:
            return 0.0
        
        if result['statusCode'] != 200:
            return 0.5
        
        body = json.loads(result['body'])
        if 'results' not in body:
            return 0.7
        
        # Check if all files were processed
        expected_files = len(test_event['Records'])
        processed_files = len(body['results'])
        
        if processed_files == expected_files:
            return 10.0
        else:
            return 10.0 * (processed_files / expected_files)
            
    except Exception:
        return 0.0


def calculate_cost_savings(current_cost):
    """Calculate cost savings vs baseline (inefficient implementation)"""
    baseline_cost = 0.0001  # Baseline cost per invocation
    if current_cost < baseline_cost:
        savings_percent = ((baseline_cost - current_cost) / baseline_cost) * 100
        return min(100.0, savings_percent)
    else:
        return 0.0


def calculate_performance_improvement(current_execution_time):
    """Calculate performance improvement vs baseline"""
    baseline_time = 3.0  # Baseline execution time in seconds
    if current_execution_time < baseline_time:
        improvement_percent = ((baseline_time - current_execution_time) / baseline_time) * 100
        return min(100.0, improvement_percent)
    else:
        return 0.0