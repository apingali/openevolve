# OpenEvolve for Business Use Cases

OpenEvolve's evolutionary programming capabilities can deliver significant business value by automatically optimizing code for cost, performance, and operational efficiency. Here are practical applications with concrete examples.

## 1. Cloud Cost Optimization

### AWS Lambda Cost Reduction

**Problem**: Lambda functions often have suboptimal memory allocation, inefficient algorithms, and unnecessary operations that drive up costs.

**Solution**: Evolve Lambda functions to minimize execution time and memory usage while maintaining functionality.

#### Example: Data Processing Lambda

**Initial Function** (Inefficient):
```python
# EVOLVE-BLOCK-START
import json
import boto3
from typing import Dict, List

def process_user_data(event, context):
    """
    Process user analytics data - currently inefficient
    """
    s3 = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('user_analytics')
    
    # Inefficient: Processing records one by one
    records = event['Records']
    results = []
    
    for record in records:
        # Download entire file each time
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        response = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(response['Body'].read())
        
        # Inefficient processing
        processed_data = {}
        for user_id, user_data in data.items():
            # Slow aggregation
            total_sessions = 0
            total_duration = 0
            for session in user_data.get('sessions', []):
                total_sessions += 1
                total_duration += session.get('duration', 0)
            
            processed_data[user_id] = {
                'total_sessions': total_sessions,
                'avg_duration': total_duration / max(total_sessions, 1),
                'last_active': max([s.get('timestamp', 0) for s in user_data.get('sessions', [])]) if user_data.get('sessions') else 0
            }
        
        # Store each record individually
        for user_id, metrics in processed_data.items():
            table.put_item(Item={
                'user_id': user_id,
                'metrics': metrics,
                'processed_at': context.aws_request_id
            })
        
        results.append({'processed_users': len(processed_data)})
    
    return {'statusCode': 200, 'body': json.dumps(results)}
# EVOLVE-BLOCK-END

def lambda_handler(event, context):
    return process_user_data(event, context)
```

**Evolved Function** (Cost-Optimized):
```python
# EVOLVE-BLOCK-START
import json
import boto3
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
import operator

def process_user_data(event, context):
    """
    Optimized user analytics processing with batch operations and efficient algorithms
    """
    s3 = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('user_analytics')
    
    records = event['Records']
    
    # Batch download files concurrently
    def download_file(record):
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read())
    
    # Concurrent file downloads (reduces execution time)
    with ThreadPoolExecutor(max_workers=min(len(records), 10)) as executor:
        all_data = list(executor.map(download_file, records))
    
    # Merge all data efficiently
    merged_data = reduce(lambda a, b: {**a, **b}, all_data, {})
    
    # Vectorized processing using list comprehensions and built-ins
    processed_items = []
    for user_id, user_data in merged_data.items():
        sessions = user_data.get('sessions', [])
        if sessions:
            # Efficient aggregation using built-in functions
            durations = [s.get('duration', 0) for s in sessions]
            timestamps = [s.get('timestamp', 0) for s in sessions]
            
            processed_items.append({
                'user_id': user_id,
                'metrics': {
                    'total_sessions': len(sessions),
                    'avg_duration': sum(durations) / len(durations),
                    'last_active': max(timestamps)
                },
                'processed_at': context.aws_request_id
            })
    
    # Batch write to DynamoDB (reduces API calls and costs)
    if processed_items:
        with table.batch_writer() as batch:
            for item in processed_items:
                batch.put_item(Item=item)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'processed_users': len(processed_items)})
    }
# EVOLVE-BLOCK-END

def lambda_handler(event, context):
    return process_user_data(event, context)
```

**Business Impact:**
- **Execution Time**: Reduced from ~2000ms to ~400ms (80% improvement)
- **Memory Usage**: Optimized data structures reduce peak memory by 60%
- **Cost Savings**: ~85% reduction in Lambda costs
- **API Calls**: Batch operations reduce DynamoDB costs by 70%

#### Evaluator for Lambda Cost Optimization

```python
"""
Evaluator for Lambda function cost optimization
"""
import json
import time
import psutil
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
import boto3
from moto import mock_s3, mock_dynamodb

def evaluate(program_path):
    """
    Evaluate Lambda function for cost optimization metrics
    """
    try:
        # Mock AWS services for testing
        with mock_s3(), mock_dynamodb():
            # Set up test environment
            s3, dynamodb = setup_test_environment()
            
            # Load the program
            spec = importlib.util.spec_from_file_location("lambda_function", program_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Create test event
            test_event = create_test_event()
            test_context = create_test_context()
            
            # Measure performance metrics
            metrics = {}
            
            # Memory profiling
            tracemalloc.start()
            start_memory = psutil.Process().memory_info().rss
            
            # Execution time
            start_time = time.time()
            
            # Run the function
            result = module.lambda_handler(test_event, test_context)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # AWS Cost calculations (based on real pricing)
            # Lambda pricing: $0.0000166667 per GB-second
            # DynamoDB pricing: $0.25 per million write requests
            
            memory_gb = peak / (1024**3)  # Convert to GB
            cost_per_invocation = memory_gb * execution_time * 0.0000166667
            
            # Count DynamoDB operations (estimate from code analysis)
            dynamodb_writes = estimate_dynamodb_operations(module, test_event)
            dynamodb_cost = (dynamodb_writes / 1000000) * 0.25
            
            total_cost = cost_per_invocation + dynamodb_cost
            
            # Normalize metrics (higher is better)
            execution_score = min(10.0, 10.0 / execution_time)  # Target < 1 second
            memory_score = min(10.0, 10.0 / memory_gb)         # Target < 1 GB
            cost_score = min(10.0, 10.0 / (total_cost * 1000000))  # Target < $0.00001
            
            # Functionality check
            functionality_score = verify_functionality(result, test_event)
            
            # Combined score prioritizing cost reduction while maintaining functionality
            combined_score = (
                0.4 * cost_score +           # Primary: cost optimization
                0.25 * execution_score +     # Performance
                0.15 * memory_score +        # Resource efficiency  
                0.2 * functionality_score    # Correctness
            )
            
            return {
                'execution_time': execution_time,
                'memory_usage_gb': memory_gb,
                'estimated_cost_per_invocation': total_cost,
                'dynamodb_operations': dynamodb_writes,
                'execution_score': execution_score,
                'memory_score': memory_score,
                'cost_score': cost_score,
                'functionality_score': functionality_score,
                'combined_score': combined_score,
                'monthly_cost_estimate': total_cost * 1000000  # Assuming 1M invocations/month
            }
            
    except Exception as e:
        return {
            'execution_time': 999.0,
            'memory_usage_gb': 10.0,
            'estimated_cost_per_invocation': 1.0,
            'combined_score': 0.0,
            'error': str(e)
        }

def setup_test_environment():
    """Set up mock AWS environment"""
    # Implementation details for mocking S3 and DynamoDB
    pass

def estimate_dynamodb_operations(module, event):
    """Estimate DynamoDB operations from code"""
    # Static analysis to count batch_writer usage vs individual puts
    pass
```

### Configuration for Cost Optimization

```yaml
# config_lambda_optimization.yaml
max_iterations: 200
checkpoint_interval: 20

database:
  population_size: 150
  exploitation_ratio: 0.8  # High exploitation for cost convergence
  feature_dimensions: ["cost_score", "execution_score", "functionality_score"]

llm:
  models:
    - name: "gpt-4"
      weight: 0.7
    - name: "gpt-3.5-turbo"
      weight: 0.3
  temperature: 0.3  # Lower temperature for focused optimizations

prompt:
  system_message: "You are an expert AWS Lambda optimizer. Focus on reducing execution time, memory usage, and API calls while maintaining functionality. Use batch operations, efficient algorithms, and concurrent processing where appropriate."

evaluator:
  timeout: 30
  enable_artifacts: true
```

## 2. API Performance Optimization

### Example: E-commerce Product Search API

**Business Goal**: Reduce API response times to improve customer experience and reduce server costs.

```python
# EVOLVE-BLOCK-START
def search_products(query, filters=None, limit=20):
    """
    Product search API - optimize for speed and cost
    """
    # Initial: Inefficient database queries and processing
    import sqlite3
    import re
    from datetime import datetime
    
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    
    # Inefficient: Multiple separate queries
    base_query = "SELECT * FROM products WHERE active = 1"
    products = []
    
    # Text search (inefficient)
    cursor.execute(base_query)
    all_products = cursor.fetchall()
    
    for product in all_products:
        if query.lower() in product[2].lower():  # name field
            products.append(product)
    
    # Apply filters one by one (inefficient)
    if filters:
        if 'category' in filters:
            products = [p for p in products if p[3] == filters['category']]
        if 'price_min' in filters:
            products = [p for p in products if p[4] >= filters['price_min']]
        if 'price_max' in filters:
            products = [p for p in products if p[4] <= filters['price_max']]
    
    # Sort by relevance (basic)
    products.sort(key=lambda x: x[2].lower().find(query.lower()))
    
    return products[:limit]
# EVOLVE-BLOCK-END
```

**Evolved Solution**:
```python
# EVOLVE-BLOCK-START
def search_products(query, filters=None, limit=20):
    """
    Optimized product search with caching, indexing, and efficient queries
    """
    import sqlite3
    import redis
    import hashlib
    import json
    from functools import lru_cache
    
    # Cache key generation
    cache_key = hashlib.md5(
        f"{query}_{json.dumps(filters, sort_keys=True)}_{limit}".encode()
    ).hexdigest()
    
    # Check Redis cache first
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    
    # Build optimized single query with proper indexing
    query_parts = ["SELECT * FROM products WHERE active = 1"]
    params = []
    
    # Full-text search using FTS5 (if available) or optimized LIKE
    if query:
        query_parts.append("AND (name MATCH ? OR description MATCH ?)")
        params.extend([query, query])
    
    # Efficient filter application in SQL
    if filters:
        if 'category' in filters:
            query_parts.append("AND category = ?")
            params.append(filters['category'])
        if 'price_min' in filters:
            query_parts.append("AND price >= ?")
            params.append(filters['price_min'])
        if 'price_max' in filters:
            query_parts.append("AND price <= ?")
            params.append(filters['price_max'])
    
    # Add ordering and limit in SQL
    query_parts.append("ORDER BY CASE WHEN name LIKE ? THEN 1 ELSE 2 END, price ASC")
    query_parts.append("LIMIT ?")
    params.extend([f"%{query}%", limit])
    
    final_query = " ".join(query_parts)
    cursor.execute(final_query, params)
    products = cursor.fetchall()
    
    # Convert to dict format for better API response
    result = [
        {
            'id': p[0], 'name': p[1], 'description': p[2],
            'category': p[3], 'price': p[4], 'stock': p[5]
        }
        for p in products
    ]
    
    # Cache result for 5 minutes
    redis_client.setex(cache_key, 300, json.dumps(result))
    
    return result
# EVOLVE-BLOCK-END
```

## 3. Data Processing Pipeline Optimization

### ETL Pipeline Cost Reduction

**Problem**: Nightly ETL jobs processing customer data are expensive and slow.

```python
# EVOLVE-BLOCK-START
def process_customer_data(data_source, output_destination):
    """
    Customer data ETL pipeline - optimize for cost and speed
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Initial: Inefficient processing
    # Load entire dataset into memory
    df = pd.read_csv(data_source)
    
    # Process row by row (inefficient)
    processed_data = []
    for index, row in df.iterrows():
        # Complex customer scoring calculation
        purchase_score = 0
        engagement_score = 0
        
        # Inefficient datetime parsing
        last_purchase = datetime.strptime(row['last_purchase'], '%Y-%m-%d')
        signup_date = datetime.strptime(row['signup_date'], '%Y-%m-%d')
        
        # Recency scoring
        days_since_purchase = (datetime.now() - last_purchase).days
        if days_since_purchase <= 30:
            purchase_score = 10
        elif days_since_purchase <= 90:
            purchase_score = 7
        elif days_since_purchase <= 180:
            purchase_score = 4
        else:
            purchase_score = 1
        
        # Frequency scoring
        purchase_count = row['total_purchases']
        if purchase_count >= 50:
            frequency_score = 10
        elif purchase_count >= 20:
            frequency_score = 7
        elif purchase_count >= 10:
            frequency_score = 4
        else:
            frequency_score = 1
        
        # Monetary scoring
        total_spent = row['total_spent']
        if total_spent >= 1000:
            monetary_score = 10
        elif total_spent >= 500:
            monetary_score = 7
        elif total_spent >= 200:
            monetary_score = 4
        else:
            monetary_score = 1
        
        rfm_score = purchase_score + frequency_score + monetary_score
        
        processed_data.append({
            'customer_id': row['customer_id'],
            'rfm_score': rfm_score,
            'segment': 'high' if rfm_score >= 20 else 'medium' if rfm_score >= 10 else 'low',
            'processed_at': datetime.now().isoformat()
        })
    
    # Write processed data
    output_df = pd.DataFrame(processed_data)
    output_df.to_csv(output_destination, index=False)
    
    return len(processed_data)
# EVOLVE-BLOCK-END
```

**Evolved Solution**:
```python
# EVOLVE-BLOCK-START
def process_customer_data(data_source, output_destination):
    """
    Vectorized customer data processing with optimized algorithms
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Chunked processing for memory efficiency
    chunk_size = 10000
    processed_chunks = []
    
    for chunk in pd.read_csv(data_source, chunksize=chunk_size):
        # Vectorized datetime operations
        chunk['last_purchase'] = pd.to_datetime(chunk['last_purchase'])
        chunk['signup_date'] = pd.to_datetime(chunk['signup_date'])
        
        current_date = pd.Timestamp.now()
        chunk['days_since_purchase'] = (current_date - chunk['last_purchase']).dt.days
        
        # Vectorized RFM scoring using numpy operations
        # Recency score (vectorized conditions)
        recency_conditions = [
            chunk['days_since_purchase'] <= 30,
            chunk['days_since_purchase'] <= 90,
            chunk['days_since_purchase'] <= 180
        ]
        recency_values = [10, 7, 4]
        chunk['recency_score'] = np.select(recency_conditions, recency_values, default=1)
        
        # Frequency score
        frequency_conditions = [
            chunk['total_purchases'] >= 50,
            chunk['total_purchases'] >= 20,
            chunk['total_purchases'] >= 10
        ]
        frequency_values = [10, 7, 4]
        chunk['frequency_score'] = np.select(frequency_conditions, frequency_values, default=1)
        
        # Monetary score
        monetary_conditions = [
            chunk['total_spent'] >= 1000,
            chunk['total_spent'] >= 500,
            chunk['total_spent'] >= 200
        ]
        monetary_values = [10, 7, 4]
        chunk['monetary_score'] = np.select(monetary_conditions, monetary_values, default=1)
        
        # Calculate RFM score and segment
        chunk['rfm_score'] = chunk['recency_score'] + chunk['frequency_score'] + chunk['monetary_score']
        chunk['segment'] = pd.cut(
            chunk['rfm_score'],
            bins=[0, 10, 20, 30],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
        
        # Select only needed columns
        processed_chunk = chunk[['customer_id', 'rfm_score', 'segment']].copy()
        processed_chunk['processed_at'] = current_date.isoformat()
        
        processed_chunks.append(processed_chunk)
    
    # Combine all chunks and write efficiently
    final_df = pd.concat(processed_chunks, ignore_index=True)
    final_df.to_csv(output_destination, index=False)
    
    return len(final_df)
# EVOLVE-BLOCK-END
```

## 4. Machine Learning Model Optimization

### Fraud Detection Model Efficiency

**Business Goal**: Reduce inference costs while maintaining accuracy for real-time fraud detection.

```python
# EVOLVE-BLOCK-START
def detect_fraud(transaction_data):
    """
    Fraud detection model - optimize for speed and cost
    """
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # Initial: Expensive feature engineering and model
    model = joblib.load('fraud_model.pkl')
    
    # Inefficient feature creation
    features = []
    
    # Time-based features (slow)
    import datetime
    transaction_time = datetime.datetime.fromtimestamp(transaction_data['timestamp'])
    features.extend([
        transaction_time.hour,
        transaction_time.day_of_week(),
        transaction_time.month
    ])
    
    # Amount-based features
    amount = transaction_data['amount']
    features.extend([
        amount,
        np.log1p(amount),
        amount ** 0.5,
        amount ** 2
    ])
    
    # Location features (expensive lookups)
    user_location = get_user_location(transaction_data['user_id'])  # Database call
    merchant_location = get_merchant_location(transaction_data['merchant_id'])  # Database call
    
    distance = calculate_distance(user_location, merchant_location)
    features.append(distance)
    
    # User history features (expensive)
    user_history = get_user_transaction_history(transaction_data['user_id'])  # Database call
    avg_amount = np.mean([t['amount'] for t in user_history])
    features.append(amount / max(avg_amount, 1))
    
    # Make prediction
    features_array = np.array(features).reshape(1, -1)
    fraud_probability = model.predict_proba(features_array)[0][1]
    
    return {
        'is_fraud': fraud_probability > 0.5,
        'fraud_probability': fraud_probability,
        'features_used': len(features)
    }
# EVOLVE-BLOCK-END
```

**Evolved Solution**:
```python
# EVOLVE-BLOCK-START
def detect_fraud(transaction_data):
    """
    Optimized fraud detection with caching and lightweight model
    """
    import numpy as np
    import redis
    import json
    import hashlib
    from functools import lru_cache
    
    # Use lightweight model or cached results
    redis_client = redis.Redis(host='localhost', port=6379, db=1)
    
    # Check if we've seen similar transactions recently
    transaction_hash = hashlib.md5(
        f"{transaction_data['user_id']}_{transaction_data['merchant_id']}_{transaction_data['amount']//10}".encode()
    ).hexdigest()
    
    cached_result = redis_client.get(f"fraud:{transaction_hash}")
    if cached_result:
        cached_data = json.loads(cached_result)
        # Use cached result if transaction is similar and recent
        return cached_data
    
    # Fast feature extraction (pre-computed and cached)
    features = []
    
    # Efficient time features (avoid datetime objects)
    timestamp = transaction_data['timestamp']
    hour = (timestamp // 3600) % 24
    day_of_week = (timestamp // 86400) % 7
    features.extend([hour, day_of_week])
    
    # Simplified amount features (most predictive ones only)
    amount = transaction_data['amount']
    log_amount = np.log1p(amount)
    features.extend([amount, log_amount])
    
    # Cached user/merchant features (updated periodically)
    user_features = get_cached_user_features(transaction_data['user_id'])
    merchant_features = get_cached_merchant_features(transaction_data['merchant_id'])
    
    features.extend([
        amount / max(user_features['avg_amount'], 1),  # Amount vs user average
        user_features['transaction_count'],            # User activity level
        merchant_features['risk_score']                # Pre-computed merchant risk
    ])
    
    # Lightweight rule-based scoring (faster than ML model for most cases)
    risk_score = 0
    
    # High amount relative to user
    if amount > user_features['avg_amount'] * 5:
        risk_score += 0.3
    
    # Unusual time
    if hour < 6 or hour > 22:
        risk_score += 0.2
    
    # High-risk merchant
    if merchant_features['risk_score'] > 0.7:
        risk_score += 0.4
    
    # Only use ML model for borderline cases (cost optimization)
    if 0.2 <= risk_score <= 0.6:
        # Load lightweight model only when needed
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        lightweight_model = joblib.load('fraud_model_light.pkl')
        features_array = np.array(features).reshape(1, -1)
        ml_probability = lightweight_model.predict_proba(features_array)[0][1]
        final_probability = (risk_score + ml_probability) / 2
    else:
        final_probability = min(risk_score, 1.0)
    
    result = {
        'is_fraud': final_probability > 0.5,
        'fraud_probability': final_probability,
        'method': 'rules' if risk_score < 0.2 or risk_score > 0.6 else 'hybrid'
    }
    
    # Cache result for similar transactions
    redis_client.setex(f"fraud:{transaction_hash}", 3600, json.dumps(result))
    
    return result

@lru_cache(maxsize=10000)
def get_cached_user_features(user_id):
    """Cached user features updated periodically"""
    # Implementation would load from fast cache
    pass

@lru_cache(maxsize=50000)  
def get_cached_merchant_features(merchant_id):
    """Cached merchant features updated periodically"""
    # Implementation would load from fast cache
    pass
# EVOLVE-BLOCK-END
```

## 5. Business Configuration Examples

### Cost-Focused Configuration
```yaml
# config_cost_optimization.yaml
max_iterations: 300
checkpoint_interval: 25

database:
  population_size: 200
  exploitation_ratio: 0.85  # High exploitation for cost convergence
  feature_dimensions: ["cost_score", "performance_score"]

llm:
  models:
    - name: "gpt-4"
      weight: 0.6
    - name: "gpt-3.5-turbo"  # Cheaper model gets higher weight
      weight: 0.4
  temperature: 0.2  # Low temperature for focused optimizations

prompt:
  system_message: "You are a senior software engineer focused on cost optimization. Prioritize reducing cloud costs, API calls, memory usage, and execution time. Use caching, batch operations, vectorization, and efficient algorithms."

evaluator:
  timeout: 45
  enable_artifacts: true
  cascade_evaluation: true
```

### Performance-Focused Configuration
```yaml
# config_performance_optimization.yaml
max_iterations: 250
checkpoint_interval: 20

database:
  population_size: 180
  exploitation_ratio: 0.75
  feature_dimensions: ["latency_score", "throughput_score", "cost_score"]

llm:
  models:
    - name: "gpt-4"
      weight: 0.8
    - name: "claude-3-sonnet"
      weight: 0.2
  temperature: 0.4

prompt:
  system_message: "You are a performance optimization expert. Focus on reducing latency, increasing throughput, and optimizing resource usage. Use profiling insights, async programming, caching strategies, and algorithmic improvements."
```

## Expected Business ROI

### AWS Lambda Optimization
- **Cost Reduction**: 70-90% in compute costs
- **Performance Improvement**: 50-80% faster execution
- **Operational Benefits**: Reduced timeout errors, better user experience

### API Optimization  
- **Response Time**: 60-85% improvement
- **Server Costs**: 40-70% reduction through efficiency gains
- **User Experience**: Better conversion rates, reduced churn

### ETL Pipeline Optimization
- **Processing Time**: 80-95% reduction
- **Infrastructure Costs**: 60-80% savings
- **Operational Efficiency**: Faster insights, reduced maintenance

### ML Model Optimization
- **Inference Costs**: 50-80% reduction
- **Latency**: 70-90% improvement
- **Scale Efficiency**: Handle 5-10x more requests with same infrastructure

## Implementation Strategy

1. **Start Small**: Begin with one critical function or API endpoint
2. **Measure Baseline**: Establish current costs and performance metrics
3. **Define Success Metrics**: Set clear targets for cost reduction and performance
4. **Iterative Evolution**: Run multiple phases with different optimization focuses
5. **A/B Testing**: Compare evolved solutions against current implementations
6. **Gradual Rollout**: Deploy optimized code incrementally with monitoring

OpenEvolve provides a systematic approach to achieving significant business value through automated code optimization, making it particularly valuable for companies with high cloud infrastructure costs or performance-critical applications.