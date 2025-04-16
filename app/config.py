"""
Configuration module for the AI assistant.
Loads environment variables and provides application settings.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Google API settings
GOOGLE_CREDENTIALS_FILE = os.getenv('GOOGLE_CREDENTIALS_FILE', 'credentials.json')
GOOGLE_TOKEN_FILE = os.getenv('GOOGLE_TOKEN_FILE', 'token.json')
GOOGLE_CREDENTIALS_PATH = BASE_DIR / GOOGLE_CREDENTIALS_FILE
GOOGLE_TOKEN_PATH = BASE_DIR / GOOGLE_TOKEN_FILE


# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = os.getenv('OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions')

# Google API scopes
GOOGLE_SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/contacts.readonly'
]

# Twilio settings
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')  # Should include 'whatsapp:' prefix
TWILIO_WEBHOOK_URL = os.getenv('TWILIO_WEBHOOK_URL')

# Application settings
APPLICATION_NAME = os.getenv('APPLICATION_NAME', 'AI Personal Assistant')
TIME_ZONE = os.getenv('TIME_ZONE', 'Asia/Kolkata')  # Changed from 'UTC' to 'Asia/Kolkata'
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', '5000'))

# Get default phone number for testing in CLI mode
DEFAULT_PHONE_NUMBER = os.getenv('DEFAULT_PHONE_NUMBER', '1234567890')

# Load Google credentials
def get_google_credentials():
    """Load Google API credentials from the credentials file."""
    try:
        if GOOGLE_CREDENTIALS_PATH.exists():
            with open(GOOGLE_CREDENTIALS_PATH, 'r') as file:
                return json.load(file)
        else:
            print(f"Credentials file not found at {GOOGLE_CREDENTIALS_PATH}")
            return None
    except Exception as e:
        print(f"Error loading credentials: {e}")
        return None