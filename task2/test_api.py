import requests
import pandas as pd
import os

# API configuration
BASE_URL = "http://127.0.0.1:8000"

def check_input_csv():
    """Check if survey.csv exists in input folder"""
    input_file = 'input/survey.csv'
    if os.path.exists(input_file):
        print(f"Found input file: {input_file}")
        # Display basic info about the CSV
        try:
            df = pd.read_csv(input_file)
            print(f"CSV contains {len(df)} rows and {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)}")
            return True
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False
    else:
        print(f"Input file not found: {input_file}")
        print("Please make sure 'survey.csv' exists in the 'input' folder")
        return False

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health/")
        if response.status_code == 200:
            print("Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"Health check failed: {response.status_code}")
    except Exception as e:
        print(f"Health check error: {e}")

def test_root_endpoint():
    """Test the root endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("Root endpoint test passed")
            print(f"Response: {response.json()}")
        else:
            print(f"Root endpoint test failed: {response.status_code}")
    except Exception as e:
        print(f"Root endpoint error: {e}")

def test_analyze_endpoint():
    """Test the analyze endpoint with survey.csv from input folder"""
    try:
        # Call the analyze endpoint (no file upload needed)
        response = requests.post(f"{BASE_URL}/analyze/")
        
        if response.status_code == 200:
            print("Analysis endpoint test passed")
            
            # Save the response (analyzed CSV)
            output_file = 'output/analysis_result.csv'
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Read and display results
            result_df = pd.read_csv(output_file)
            print(f"Analysis completed! Results saved to: {output_file}")
            print(f"Processed {len(result_df)} employees")
            
            # Show sample results
            print("\nSample Results:")
            print(result_df.head())
            
            # Show available columns
            print(f"\nOutput columns: {list(result_df.columns)}")
            
        else:
            print(f"Analysis endpoint test failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Analysis endpoint error: {e}")

def run_all_tests():
    """Run all tests"""
    print("Starting API Tests...")
    print("=" * 50)
    
    print("\n1. Checking Input File:")
    if not check_input_csv():
        print("Stopping tests - input file not found")
        return
    
    print("\n2. Testing Root Endpoint:")
    test_root_endpoint()
    
    print("\n3. Testing Health Check:")
    test_health_check()
    
    print("\n4. Testing Analysis Endpoint:")
    test_analyze_endpoint()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    print("Make sure your FastAPI server is running on http://127.0.0.1:8000")
    print("You can start it with: python main.py")
    print()
    
    # Ask user if server is running
    input("Press Enter when your server is running...")
    
    run_all_tests()