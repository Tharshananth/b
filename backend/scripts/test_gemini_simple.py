"""
Minimal Gemini API test - do we get a response or not?

    python scripts/test_gemini_simple.py

DO NOT COMMIT: this file contains an API key.
"""
from google import genai

API_KEY = "AQ.Ab8RN6IlePOeRwUGGPfu8xk_mcMRjkANxwteFDmj7kt186bwKA"
MODEL = "gemini-3.5-flash"
PROMPT = "Hello! Reply in one short sentence."

client = genai.Client(api_key=API_KEY)

try:
    response = client.models.generate_content(model=MODEL, contents=PROMPT)
    print("SUCCESS - got a response")
    print("Model   :", MODEL)
    print("Response:", response.text)
except Exception as e:
    print("FAILED - no response")
    print("Error   :", e)
