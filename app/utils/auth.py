"""
Authentication utilities for Google API services.
Handles OAuth2 flow for Gmail, Calendar, and Contacts APIs.
"""
import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.config import (
    GOOGLE_CREDENTIALS_PATH,
    GOOGLE_TOKEN_PATH,
    GOOGLE_SCOPES,
    APPLICATION_NAME
)

def get_google_service(api_name, api_version):
    """
    Authenticates with Google and returns a service object for the specified API.
    
    Args:
        api_name: The name of the Google API (e.g., 'gmail', 'calendar', 'people')
        api_version: The version of the API (e.g., 'v1', 'v3')
        
    Returns:
        A service object for the specified API or None if authentication fails
    """
    creds = None
    
    # Load credentials from token.json if it exists
    if os.path.exists(GOOGLE_TOKEN_PATH):
        with open(GOOGLE_TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials don't exist or are invalid, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
                print(f"Credentials file not found at {GOOGLE_CREDENTIALS_PATH}")
                return None
                
            flow = InstalledAppFlow.from_client_secrets_file(
                GOOGLE_CREDENTIALS_PATH, GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for future use
        with open(GOOGLE_TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)
    
    # Build and return the service
    try:
        service = build(api_name, api_version, credentials=creds)
        return service
    except Exception as e:
        print(f"Error building {api_name} service: {e}")
        return None

def get_gmail_service():
    """Returns an authenticated Gmail service."""
    return get_google_service('gmail', 'v1')

def get_calendar_service():
    """Returns an authenticated Calendar service."""
    return get_google_service('calendar', 'v3')

def get_contacts_service():
    """Returns an authenticated People (Contacts) service."""
    return get_google_service('people', 'v1')