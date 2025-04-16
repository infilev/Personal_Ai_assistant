"""
Entity extraction for user messages using OpenAI and Hugging Face transformers.
"""
import re
import datetime
import dateparser
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from app.utils.helpers import get_current_time
from app.utils.llm import OpenRouterClient

class EntityExtractor:
    def __init__(self):
        """Initialize the entity extractor with OpenRouter and transformer models."""
        # Initialize OpenRouter client
        self.openrouter_client = OpenRouterClient()
        
        # Initialize transformer model as fallback
        try:
            # Use a pre-trained NER model
            model_name = "dslim/bert-base-NER"  # Good efficient NER model
            
            # Initialize NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple"  # Merge entities spanning multiple tokens
            )
            
            self.initialized = True
            print("Transformer model for entity extraction loaded successfully")
        except Exception as e:
            print(f"Error initializing transformer model for entity extraction: {e}")
            self.initialized = False
    
    def extract_entities(self, message, intent=None):
        """
        Extract entities from a user message, using OpenAI first.
        
        Args:
            message: The user message
            intent: Intent type (optional, for context-specific extraction)
            
        Returns:
            Dict containing extracted entities
        """
        print(f"Extracting entities from: '{message}'")
            
        # Try OpenRouter first
        if self.openrouter_client.initialized:
            openrouter_entities = self.openrouter_client.extract_entities(message, intent)
            if openrouter_entities:
                print(f"Using OpenRouter entity extraction: {openrouter_entities}")
                
                
                # Process dates if needed
                if  openrouter_entities.get("date") and isinstance( openrouter_entities["date"], str):
                    try:
                         openrouter_entities["date"] = datetime.datetime.strptime(
                             openrouter_entities["date"], "%Y-%m-%d").date()
                    except ValueError:
                        # Try to parse with dateparser if format is different
                        parsed_date = dateparser.parse( openrouter_entities["date"])
                        if parsed_date:
                             openrouter_entities["date"] = parsed_date.date()
                
                # Process times if needed
                if  openrouter_entities.get("time") and isinstance( openrouter_entities["time"], str):
                    try:
                         openrouter_entities["time"] = datetime.datetime.strptime(
                             openrouter_entities["time"], "%H:%M").time()
                    except ValueError:
                        # Try to parse with dateparser
                        parsed_time = dateparser.parse( openrouter_entities["time"])
                        if parsed_time:
                             openrouter_entities["time"] = parsed_time.time()
                
                return  openrouter_entities
        
        # Entities to extract
        entities = {
            "person": [],
            "date": None,
            "time": None,
            "duration": None,
            "email": [],
            "subject": None,
            "body": None,
            "location": None
        }
        
        # Extract with transformer if available
        if self.initialized:
            try:
                # Run NER pipeline
                ner_results = self.ner_pipeline(message)
                
                # Process NER results
                for entity in ner_results:
                    entity_text = entity["word"]
                    entity_type = entity["entity_group"]
                    
                    # Map entity types to our structure
                    if entity_type == "PER" or entity_type == "B-PER":
                        entities["person"].append(entity_text)
                        print(f"Found person entity: {entity_text}")
                    elif entity_type == "LOC" or entity_type == "B-LOC":
                        entities["location"] = entity_text
                        print(f"Found location entity: {entity_text}")
                    elif entity_type == "ORG" or entity_type == "B-ORG":
                        # Could be used for company/organization meetings
                        if not entities["location"]:
                            entities["location"] = entity_text
                            print(f"Found organization (as location) entity: {entity_text}")
                
            except Exception as e:
                print(f"Error in transformer entity extraction: {e}")
        
        # Extract dates and times
        datetime_entities = self._extract_datetime(message)
        if datetime_entities.get("date"):
            print(f"Found date entity: {datetime_entities['date']}")
        if datetime_entities.get("time"):
            print(f"Found time entity: {datetime_entities['time']}")
        
        entities.update(datetime_entities)
        
        # Extract emails
        emails = self._extract_emails(message)
        if emails:
            print(f"Found email entities: {emails}")
        entities["email"] = emails
        
        # Intent-specific extraction
        if intent == "send_email":
            email_entities = self._extract_email_entities(message)
            if email_entities.get("subject"):
                print(f"Found email subject: {email_entities['subject']}")
            if email_entities.get("body"):
                print(f"Found email body: {email_entities['body']}")
            entities.update(email_entities)
        elif intent == "schedule_meeting":
            meeting_entities = self._extract_meeting_entities(message)
            if meeting_entities.get("location"):
                print(f"Found meeting location: {meeting_entities['location']}")
            if meeting_entities.get("subject"):
                print(f"Found meeting subject: {meeting_entities['subject']}")
            entities.update(meeting_entities)
        
        # If no person entity was found but we're looking for contacts,
        # try to extract names more aggressively
        if not entities["person"] and intent == "find_contact":
            words = message.split()
            for i, word in enumerate(words):
                if word.lower() in ["for", "about", "contact", "information"]:
                    if i + 1 < len(words) and words[i+1][0].isupper():
                        name_parts = []
                        for j in range(i+1, len(words)):
                            if words[j][0].isupper():
                                name_parts.append(words[j])
                            else:
                                break
                        if name_parts:
                            name = " ".join(name_parts)
                            entities["person"].append(name)
                            print(f"Found potential person name: {name}")
                            break
        
        return entities
    
    def _extract_datetime(self, message):
        """
        Extract date and time information from a message.
        
        Args:
            message: The user message
            
        Returns:
            Dict containing date, time, and duration entities
        """
        entities = {
            "date": None,
            "time": None,
            "duration": None
        }
        
        # Current time for reference
        now = get_current_time()
        
        # Try to extract time expressions
        time_expressions = []
        
        # Look for specific time patterns
        time_pattern = r'\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?'
        time_matches = re.finditer(time_pattern, message)
        
        for match in time_matches:
            time_expressions.append(match.group())
        
        # Look for duration patterns
        duration_pattern = r'(\d+)\s*(hour|minute|min)s?'
        duration_match = re.search(duration_pattern, message)
        
        if duration_match:
            amount = int(duration_match.group(1))
            unit = duration_match.group(2)
            
            if unit.startswith('hour'):
                entities["duration"] = amount * 60  # Convert to minutes
            elif unit.startswith('min'):
                entities["duration"] = amount
        
        # Use dateparser for more complex date/time extraction
        parsed_date = dateparser.parse(
            message,
            settings={
                'RELATIVE_BASE': now,
                'PREFER_DATES_FROM': 'future'
            }
        )
        
        if parsed_date:
            # If we have specific time expressions, use the date from parsed_date
            # but keep the time separate
            if time_expressions:
                entities["date"] = parsed_date.date()
                
                # Try to parse the first time expression
                time_str = time_expressions[0]
                time_obj = dateparser.parse(time_str)
                
                if time_obj:
                    entities["time"] = time_obj.time()
            else:
                # Use the full datetime
                entities["date"] = parsed_date.date()
                entities["time"] = parsed_date.time()
        
        return entities
    
    def _extract_emails(self, message):
        """
        Extract email addresses from a message.
        
        Args:
            message: The user message
            
        Returns:
            List of email addresses
        """
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return re.findall(email_pattern, message)
    
    def _extract_email_entities(self, message):
        """
        Extract email-specific entities (subject, body).
        
        Args:
            message: The user message
            
        Returns:
            Dict containing email-specific entities
        """
        entities = {
            "subject": None,
            "body": None
        }
        
        # Look for subject patterns
        subject_patterns = [
            r'subject\s*(?:is|:)\s*["\'"]?(.*?)["\'"]?(?:\s|$)',
            r'about\s*["\'"]?(.*?)["\'"]?(?:\s|$)',
            r'regarding\s*["\'"]?(.*?)["\'"]?(?:\s|$)'
        ]
        
        for pattern in subject_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                entities["subject"] = match.group(1).strip()
                break
        
        # Look for body/content patterns
        body_patterns = [
            r'body\s*(?:is|:)\s*["\'"]?(.*?)["\'"]?(?:\s|$)',
            r'content\s*(?:is|:)\s*["\'"]?(.*?)["\'"]?(?:\s|$)',
            r'message\s*(?:is|:)\s*["\'"]?(.*?)["\'"]?(?:\s|$)'
        ]
        
        for pattern in body_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                entities["body"] = match.group(1).strip()
                break
        
        return entities
    
    def _extract_meeting_entities(self, message):
        """
        Extract meeting-specific entities (location, etc.).
        
        Args:
            message: The user message
            
        Returns:
            Dict containing meeting-specific entities
        """
        entities = {
            "location": None,
            "subject": None
        }
        
        # Look for location patterns
        location_patterns = [
            r'(?:at|in)\s+(?:the\s+)?["\'"]?([\w\s]+)["\'"]?(?:\s|$)',
            r'location\s*(?:is|:)\s*["\'"]?([\w\s]+)["\'"]?(?:\s|$)',
            r'place\s*(?:is|:)\s*["\'"]?([\w\s]+)["\'"]?(?:\s|$)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Exclude common time-related words that might be misidentified as locations
                if location.lower() not in ['today', 'tomorrow', 'morning', 'afternoon', 'evening', 'night']:
                    entities["location"] = location
                    break
        
        # Look for subject/title patterns
        subject_patterns = [
            r'about\s*["\'"]?(.*?)["\'"]?(?:\s|$)',
            r'regarding\s*["\'"]?(.*?)["\'"]?(?:\s|$)',
            r'title\s*(?:is|:)\s*["\'"]?(.*?)["\'"]?(?:\s|$)',
            r'subject\s*(?:is|:)\s*["\'"]?(.*?)["\'"]?(?:\s|$)'
        ]
        
        for pattern in subject_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                entities["subject"] = match.group(1).strip()
                break
        
        return entities