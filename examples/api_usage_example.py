"""
Example script demonstrating how to use the Cocoa Price Prediction API.

This script shows how to:
1. Make prediction requests
2. Retrieve performance metrics
3. List available models
4. Trigger retraining (admin only)
"""

import requests
from datetime import datetime, timedelta
import json


# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "your_secret_key_here"  # Replace with your actual token
ADMIN_TOKEN = "admin_your_secret_key_here"  # Replace with your admin token

# Headers for authentication
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

admin_headers = {
    "Authorization": f"Bearer {ADMIN_TOKEN}",
    "Content-Type": "application/json"
}


def check_health():
    """Check API health status."""
    print("\n=== Checking API Health ===")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


def get_predictions():
    """Request price predictions."""
    print("\n=== Requesting Price Predictions ===")
    
    request_data = {
        "horizons": [1, 7, 30],
        "market": "ICE_London",
        "include_sentiment": True
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/v1/predict",
        headers=headers,
        json=request_data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Model Version: {data['model_version']}")
        print(f"Market: {data['market']}")
        print(f"Sentiment Score: {data.get('sentiment_score', 'N/A')}")
        print("\nPredictions:")
        for pred in data['predictions']:
            print(f"  Horizon {pred['horizon']} days:")
            print(f"    Price: ${pred['price']:.2f}")
            print(f"    Confidence Interval: [${pred['confidence_interval'][0]:.2f}, ${pred['confidence_interval'][1]:.2f}]")
            print(f"    Confidence Level: {pred['confidence_level']:.0%}")
    else:
        print(f"Error: {response.json()}")
    
    return response.json()


def get_performance_metrics():
    """Retrieve model performance metrics."""
    print("\n=== Retrieving Performance Metrics ===")
    
    # Get metrics for the last 7 days
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }
    
    response = requests.get(
        f"{API_BASE_URL}/api/v1/performance",
        headers=headers,
        params=params
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Model Version: {data['model_version']}")
        print(f"Number of Metrics: {len(data['metrics'])}")
        
        if data['metrics']:
            print("\nLatest Metrics:")
            latest = data['metrics'][0]
            print(f"  RMSE: {latest['rmse']:.4f}")
            print(f"  MAE: {latest['mae']:.4f}")
            print(f"  MAPE: {latest['mape']:.4f}")
            print(f"  Directional Accuracy: {latest['directional_accuracy']:.2%}")
            print(f"  Coverage Rate: {latest['coverage_rate']:.2%}")
            print(f"  Mean Interval Width: {latest['mean_interval_width']:.2f}")
    else:
        print(f"Error: {response.json()}")
    
    return response.json()


def list_models():
    """List available model versions."""
    print("\n=== Listing Available Models ===")
    
    response = requests.get(
        f"{API_BASE_URL}/api/v1/models",
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Current Production Version: {data.get('current_production_version', 'N/A')}")
        print(f"\nAvailable Models ({len(data['models'])}):")
        
        for model in data['models']:
            print(f"\n  {model['name']} v{model['version']}")
            print(f"    Stage: {model['stage']}")
            print(f"    Created: {model['created_at']}")
            if model.get('metrics'):
                print(f"    Metrics:")
                for metric, value in model['metrics'].items():
                    print(f"      {metric}: {value}")
    else:
        print(f"Error: {response.json()}")
    
    return response.json()


def trigger_retraining():
    """Trigger model retraining (admin only)."""
    print("\n=== Triggering Model Retraining ===")
    
    request_data = {
        "model_type": "all",
        "reason": "Manual retraining for testing"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/v1/retrain",
        headers=admin_headers,
        json=request_data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Message: {data['message']}")
        print(f"Job ID: {data['job_id']}")
        print(f"Estimated Completion: {data.get('estimated_completion', 'N/A')}")
    else:
        print(f"Error: {response.json()}")
    
    return response.json()


def main():
    """Run all examples."""
    print("=" * 60)
    print("Cocoa Price Prediction API - Usage Examples")
    print("=" * 60)
    
    try:
        # 1. Check API health
        check_health()
        
        # 2. Get predictions
        get_predictions()
        
        # 3. Get performance metrics
        get_performance_metrics()
        
        # 4. List models
        list_models()
        
        # 5. Trigger retraining (admin only)
        # Uncomment to test retraining
        # trigger_retraining()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API. Make sure the API is running at", API_BASE_URL)
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
