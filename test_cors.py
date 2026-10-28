#!/usr/bin/env python3
"""
Simple CORS test script to verify preflight requests work
Run this after starting the backend on port 8000
"""
import requests
import json

# Test configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URLS = [
    "http://localhost:3000",      # Local dev
    "http://127.0.0.1:3000",      # Local dev (alt)
    "http://localhost:8042",      # Local frontend
    "http://172.16.68.4:8042",    # Azure VM React frontend
]

def test_cors_preflight():
    """Test CORS preflight (OPTIONS) request"""
    print("=" * 60)
    print("Testing CORS Preflight (OPTIONS) Request")
    print("=" * 60)
    print("(Note: This has a 3 second timeout)\n")
    
    for frontend_url in FRONTEND_URLS:
        print(f"Testing from origin: {frontend_url}")
        
        headers = {
            "Origin": frontend_url,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
        
        try:
            # Short timeout for OPTIONS (should be fast)
            response = requests.options(
                f"{BACKEND_URL}/api/chat/",
                headers=headers,
                timeout=3
            )
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"  ✅ CORS Preflight OK")
                allowed_origins = response.headers.get('access-control-allow-origin', 'N/A')
                allowed_methods = response.headers.get('access-control-allow-methods', 'N/A')
                print(f"  Allowed Origin: {allowed_origins}")
                print(f"  Allowed Methods: {allowed_methods}")
            else:
                print(f"  ❌ CORS Preflight FAILED with status {response.status_code}")
                
        except requests.Timeout:
            print(f"  ❌ TIMEOUT - Request took too long (>3s)")
            print(f"     This suggests middleware is hanging")
        except requests.ConnectionError as e:
            print(f"  ❌ Connection Error: {str(e)}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
        
        print()

def test_health_endpoint():
    """Test basic health endpoint"""
    print("\n" + "=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print("✅ Health endpoint working")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_actual_post():
    """Test actual POST request with CORS"""
    print("\n" + "=" * 60)
    print("Testing Actual POST Request (with CORS)")
    print("=" * 60)
    
    headers = {
        "Content-Type": "application/json",
        "Origin": "http://localhost:3000",
    }
    
    payload = {
        "message": "Hello, test!",
        "session_id": "test_session_123",
        "user_id": "test_user_456"
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat/",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            print("✅ POST request working with CORS")
        else:
            print(f"Response: {response.text}")
            print(f"❌ POST request failed")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("\n🧪 CORS Testing Script")
    print("Make sure backend is running on http://localhost:8000\n")
    
    test_health_endpoint()
    test_cors_preflight()
    test_actual_post()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
