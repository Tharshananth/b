from google import genai

# paste your api key here
API_KEY = ""

import time


# 2. Comprehensive Flash / Flash-Lite test matrix 
MODELS_TO_TEST = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash"
]

def test_flash_matrix():
    if API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("❌ Error: Please replace 'YOUR_GEMINI_API_KEY_HERE' with your real API key.")
        return

    print("🚀 Initializing Google Gen AI Client...")
    client = genai.Client(api_key=API_KEY)
    
    test_prompt = "Verify connection. Reply with exactly two words: 'Model Active'."
    
    print("\n--- Starting Flash Model Matrix Test ---")
    for model_name in MODELS_TO_TEST:
        print(f"\n📡 Testing: {model_name}...")
        try:
            # Measure generation latency roughly
            start_time = time.time()
            
            response = client.models.generate_content(
                model=model_name,
                contents=test_prompt
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"✅ Success ({elapsed_time:.2f}s)")
            print(f"   ↳ Response: {response.text.strip()}")
            
        except Exception as e:
            print(f"❌ Failed to query {model_name}")
            print(f"   ↳ Error Details: {e}")
            
        # 1-second delay to preserve your Free Tier RPM limits
        time.sleep(1)

    print("\n----------------------------------------")
    print("📋 Test Complete.")

if __name__ == "__main__":
    test_flash_matrix()

