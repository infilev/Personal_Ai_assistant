"""
LLM utilities for intent recognition and entity extraction using OpenRouter API.
"""
import json
import requests
from app.config import OPENROUTER_API_KEY, OPENROUTER_URL

class OpenRouterClient:
    def __init__(self):
        """Initialize the OpenRouter client for LLM access."""
        self.api_key = OPENROUTER_API_KEY
        self.api_url = OPENROUTER_URL or "https://openrouter.ai/api/v1/chat/completions"
        self.model = "openai/gpt-4o-mini"     # Using GPT-4o-mini through OpenRouter
        self.initialized = self.api_key is not None
        
        if not self.initialized:
            print("OpenRouter API key not found. LLM features will fall back to transformer models.")
    
    def recognize_intent(self, message):
        """Recognize intent using OpenRouter API."""
        if not self.initialized:
            return None
            
        try:
            prompt = f"""Classify this message into exactly one of these intents:
            - send_email: If the user wants to send an email
            - schedule_meeting: If the user wants to schedule a meeting or appointment
            - check_calendar: If the user wants to check events or see what's on their calendar
            - find_contact: If the user wants contact information like email, phone, etc.
            - check_free_slots: If the user wants to know when they are available
            
            Message: "{message}"
            
            Return ONLY the intent, nothing else."""
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 15
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            intent = result["choices"][0]["message"]["content"].strip().lower()
            
            # Map to our intent constants
            intent_mapping = {
                "send_email": "send_email",
                "schedule_meeting": "schedule_meeting",
                "check_calendar": "check_calendar", 
                "find_contact": "find_contact",
                "check_free_slots": "check_free_slots"
            }
            
            mapped_intent = intent_mapping.get(intent, "unknown")
            return {"intent": mapped_intent, "confidence": 0.95}
            
        except Exception as e:
            print(f"Error in OpenRouter intent recognition: {e}")
            return None
    
    def extract_entities(self, message, intent=None):
        """Extract entities using OpenRouter API."""
        if not self.initialized:
            return None
            
        try:
            # Build context based on intent
            context = ""
            if intent:
                context = f"The user wants to {intent.replace('_', ' ')}. "
            
            prompt = f"""{context}Extract the following entities from this message in JSON format:
            {{
                "person": [], # List of people mentioned (names)
                "date": null, # Date mentioned (in YYYY-MM-DD format)
                "time": null, # Time mentioned (in HH:MM format)
                "duration": null, # Duration in minutes (as a number)
                "email": [], # List of email addresses
                "subject": null, # Subject/topic mentioned
                "body": null, # Body/content of a message
                "location": null # Location mentioned
            }}
            
            Message: "{message}"
            
            Return ONLY valid JSON, nothing else."""
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 400
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            result_content = result["choices"][0]["message"]["content"].strip()
            
            # Clean up the response to ensure valid JSON
            start_idx = result_content.find('{')
            end_idx = result_content.rfind('}')
            if start_idx >= 0 and end_idx >= 0:
                result_content = result_content[start_idx:end_idx+1]
            
            try:
                entities = json.loads(result_content)
                return entities
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from OpenRouter: {e}")
                print(f"Raw response: {result_content}")
                return None
            
        except Exception as e:
            print(f"Error in OpenRouter entity extraction: {e}")
            return None