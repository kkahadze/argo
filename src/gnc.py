import requests

# Step 1: Obtain session ID
response = requests.get('http://gnc.gov.ge/gnc/parse-api', params={'command': 'get-session'})
session_info = response.json()
session_id = session_info['session-id']

print(f"Session ID: {session_id}")

# Step 2: Parse Georgian text (No manual encoding required!)
text_to_parse = 'გამარჯობა'

parse_response = requests.get(
    'http://gnc.gov.ge/gnc/parse-api',
    params={
        'command': 'parse',
        'session-id': session_id,
        'text': text_to_parse  # ← requests handles encoding automatically
    }
)

parsed_data = parse_response.json()

# Step 3: Print parsed data
print(parsed_data)
