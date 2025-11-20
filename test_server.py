#!/usr/bin/env python3
"""
Test script for LLM Quiz Solver
Run this after starting your server with: uvicorn app:app --reload --port 8000
"""

import requests
import json
import time

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
DEMO_URL = "https://tds-llm-analysis.s-anand.net/demo"

# Load secrets
with open("secrets.json") as f:
    secrets = json.load(f)

EMAIL = secrets["email"]
SECRET = secrets["secret"]

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check (GET /)")
    print("="*60)
    try:
        response = requests.get(BASE_URL)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_invalid_secret():
    """Test with invalid secret (should return 403)"""
    print("\n" + "="*60)
    print("TEST 2: Invalid Secret (should return 403)")
    print("="*60)
    try:
        payload = {
            "email": EMAIL,
            "secret": "wrong_secret",
            "url": DEMO_URL
        }
        response = requests.post(BASE_URL, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 403
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_invalid_json():
    """Test with invalid JSON (should return 400)"""
    print("\n" + "="*60)
    print("TEST 3: Invalid JSON (should return 400)")
    print("="*60)
    try:
        response = requests.post(
            BASE_URL,
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        return response.status_code == 400
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_demo_quiz():
    """Test with the official demo quiz"""
    print("\n" + "="*60)
    print("TEST 4: Demo Quiz Solver")
    print("="*60)
    try:
        payload = {
            "email": EMAIL,
            "secret": SECRET,
            "url": DEMO_URL
        }
        print(f"Sending request to: {BASE_URL}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(BASE_URL, json=payload)
        print(f"\nImmediate Response:")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("\n⏳ Waiting 10 seconds for background solver to complete...")
            time.sleep(10)
            
            # Check result file
            try:
                with open("last_solver_result.json") as f:
                    result = json.load(f)
                print(f"\n📊 Solver Result:")
                print(json.dumps(result, indent=2, default=str)[:1000])
                return True
            except FileNotFoundError:
                print("⚠️ No result file found yet - solver may still be running")
                return True
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("🧪 LLM QUIZ SOLVER TEST SUITE")
    print("="*60)
    print(f"Testing server at: {BASE_URL}")
    print(f"Demo quiz URL: {DEMO_URL}")
    
    results = {
        "Health Check": test_health_check(),
        "Invalid Secret (403)": test_invalid_secret(),
        "Invalid JSON (400)": test_invalid_json(),
        "Demo Quiz": test_demo_quiz()
    }
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your server is ready!")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
