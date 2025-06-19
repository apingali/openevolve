# EVOLVE-BLOCK-START
"""AWS Lambda function for processing user analytics data - optimize for cost and performance"""
import json
import boto3
from typing import Dict, List

def process_user_data(event, context):
    """
    Process user analytics data from S3 and store aggregated metrics in DynamoDB
    Current implementation is inefficient and costly
    """
    s3 = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('user_analytics')
    
    # Process S3 event records
    records = event['Records']
    results = []
    
    for record in records:
        # Download file from S3
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        response = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(response['Body'].read())
        
        # Process user data (inefficient approach)
        processed_data = {}
        for user_id, user_data in data.items():
            # Calculate user metrics
            sessions = user_data.get('sessions', [])
            
            total_sessions = 0
            total_duration = 0
            page_views = 0
            
            for session in sessions:
                total_sessions += 1
                total_duration += session.get('duration', 0)
                page_views += session.get('page_views', 0)
            
            if total_sessions > 0:
                avg_duration = total_duration / total_sessions
                avg_page_views = page_views / total_sessions
            else:
                avg_duration = 0
                avg_page_views = 0
            
            # Find last active timestamp
            last_active = 0
            for session in sessions:
                timestamp = session.get('timestamp', 0)
                if timestamp > last_active:
                    last_active = timestamp
            
            processed_data[user_id] = {
                'total_sessions': total_sessions,
                'avg_duration': avg_duration,
                'avg_page_views': avg_page_views,
                'last_active': last_active,
                'total_page_views': page_views
            }
        
        # Store each user's data individually (inefficient)
        for user_id, metrics in processed_data.items():
            table.put_item(Item={
                'user_id': user_id,
                'metrics': metrics,
                'processed_at': context.aws_request_id,
                'file_processed': key
            })
        
        results.append({
            'file': key,
            'processed_users': len(processed_data)
        })
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Successfully processed user data',
            'results': results
        })
    }
# EVOLVE-BLOCK-END

def lambda_handler(event, context):
    """Entry point for Lambda function"""
    return process_user_data(event, context)