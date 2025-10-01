#!/usr/bin/env python3
import requests
import json
import os
import argparse
from dotenv import load_dotenv

# Load API key from .env file if it exists
load_dotenv()

def test_translation(mingrelian_text, api_key=None, target_language="english", model=None, provider=None, url="http://localhost:8000"):
    """
    Test the translation API with a given Mingrelian text.
    
    Args:
        mingrelian_text (str): Text to translate in Mingrelian
        api_key (str, optional): OpenAI API key. Defaults to None (will try to load from env).
        target_language (str, optional): Target language for translation. Defaults to "english".
        model (str, optional): AI model to use. Defaults to None (will use server default).
        provider (str, optional): LLM provider to use. Defaults to None (will use server default).
        url (str, optional): Base URL of the API. Defaults to "http://localhost:8000".
    
    Returns:
        dict: Translation response or error message
    """
    # If no API key provided, try to get from environment
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "No OpenAI API key provided or found in environment"}
    
    # Prepare the request data
    data = {
        "prompt": mingrelian_text,
        "api_key": api_key,
        "target_language": target_language
    }
    
    # Add optional parameters if provided
    if model:
        data["model"] = model
    if provider:
        data["provider"] = provider
    
    # Set the headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # Make the request
    try:
        response = requests.post(
            f"{url}/chat",
            data=json.dumps(data),
            headers=headers
        )
        
        # Parse the response
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {
                "error": f"API request failed with status code {response.status_code}",
                "details": response.text
            }
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

def display_results(results):
    """
    Display the translation results in a readable format.
    
    Args:
        results (dict): Translation results or error message
    """
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        if "details" in results:
            print(f"Details: {results['details']}")
        return
    
    print("\n✅ Translation successful!")
    print(f"📥 Mingrelian (latinized): {results['mingrelian_latinized']}")
    print(f"📥 Mingrelian (mkhedruli): {results['mingrelian_mkhedruli']}")
    print(f"📤 Georgian: {results['georgian']}")
    print(f"📤 English: {results['english']}")
    
    # Print full response if it's not too long
    if results.get('full_response') and len(results['full_response']) < 500:
        print("\n🔍 Full response:")
        print(results['full_response'])
    else:
        print("\n🔍 Full response available but not shown (too long)")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the Mingrelian Translator API")
    parser.add_argument("text", help="Mingrelian text to translate")
    parser.add_argument("--api-key", help="OpenAI API key (will use OPENAI_API_KEY env var if not provided)")
    parser.add_argument("--target", choices=["english", "georgian"], default="english",
                        help="Target language for translation (default: english)")
    parser.add_argument("--model", help="AI model to use (e.g., gpt-4o, gpt-5-2025-08-07)")
    parser.add_argument("--provider", choices=["openai", "anthropic"], 
                        help="LLM provider to use (default: uses server default)")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the API (default: http://localhost:8000)")
    
    args = parser.parse_args()
    
    # Run the test
    results = test_translation(
        args.text,
        api_key=args.api_key,
        target_language=args.target,
        model=args.model,
        provider=args.provider,
        url=args.url
    )
    
    # Display the results
    display_results(results)

if __name__ == "__main__":
    main() 