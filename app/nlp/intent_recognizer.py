"""
Intent recognition for user messages using OpenAI and transformer models.
"""
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.utils.llm import OpenRouterClient

class IntentRecognizer:
    # Define intent types
    INTENT_SEND_EMAIL = "send_email"
    INTENT_SCHEDULE_MEETING = "schedule_meeting"
    INTENT_CHECK_CALENDAR = "check_calendar"
    INTENT_FIND_CONTACT = "find_contact"
    INTENT_CHECK_FREE_SLOTS = "check_free_slots"
    INTENT_UNKNOWN = "unknown"
    
    def __init__(self):
        """Initialize intent recognizers with primary and fallback models."""
        # Initialize OpenRouter client
        self.openrouter_client = OpenRouterClient()
        
        # Initialize fallback transformer model
        try:
            # Use a model fine-tuned for intent classification
            model_name = "facebook/bart-large-mnli"  # Zero-shot classification model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Define intent labels for zero-shot classification
            self.intent_labels = [
                "sending an email",
                "scheduling a meeting",
                "checking calendar",
                "finding contact information",
                "checking availability"
            ]
            
            # Map intent labels to our intent types
            self.intent_mapping = {
                "sending an email": self.INTENT_SEND_EMAIL,
                "scheduling a meeting": self.INTENT_SCHEDULE_MEETING,
                "checking calendar": self.INTENT_CHECK_CALENDAR,
                "finding contact information": self.INTENT_FIND_CONTACT,
                "checking availability": self.INTENT_CHECK_FREE_SLOTS
            }
            
            # Initialize model
            self.model.eval()
            
            # Set initialization flag
            self.initialized = True
            print("Advanced transformer model for intent recognition loaded successfully")
        except Exception as e:
            print(f"Error initializing advanced transformer model: {e}")
            self.initialized = False
    
    def recognize_intent(self, message):
        """
        Recognize the intent from a user message, using OpenAI first.
        
        Args:
            message: The user message
            
        Returns:
            Dict containing intent type and confidence score
        """
        """Recognize the intent from a user message, using OpenRouter first."""
        # Tring OpenRouter first
        if self.openrouter_client.initialized:
            result = self.openrouter_client.recognize_intent(message)
            if result:
                print(f"OpenRouter predicted intent: {result['intent']} with confidence: {result['confidence']}")
                return result
        
        # Fall back to transformer if OpenAI is not available or fails
        if not self.initialized or not message.strip():
            return self._recognize_intent_rule_based(message)
            
        # Try transformer-based approach
        try:
            # First check if any keywords directly indicate an intent
            # This helps with very short queries
            result = self._check_quick_keywords(message)
            if result:
                return result
                
            # Zero-shot classification approach
            premise = message
            hypothesis_template = "This text is about {}."
            
            # Prepare inputs for the model
            inputs = self.tokenizer(
                [premise] * len(self.intent_labels),
                [hypothesis_template.format(label) for label in self.intent_labels],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
                entailment_idx = 2  # Index for entailment in MNLI model
                scores = predictions[:, entailment_idx]
                
            # Get the best intent and its confidence
            best_idx = torch.argmax(scores).item()
            confidence = scores[best_idx].item()
            best_intent_label = self.intent_labels[best_idx]
            intent = self.intent_mapping.get(best_intent_label)
            
            print(f"Zero-shot model predictions:")
            for i, label in enumerate(self.intent_labels):
                print(f"  - {label}: {scores[i].item():.4f}")
            
            # If confidence is too low, fall back to rule-based approach
            if confidence < 0.65:
                print(f"Low confidence ({confidence:.2f}) from transformer model, falling back to rule-based")
                return self._recognize_intent_rule_based(message)
                
            print(f"Transformer model predicted intent: {intent} with confidence: {confidence:.2f}")
            return {"intent": intent, "confidence": confidence}
            
        except Exception as e:
            print(f"Error in transformer intent recognition: {e}")
            # Fall back to rule-based approach
            return self._recognize_intent_rule_based(message)
    
    def _check_quick_keywords(self, message):
        """Check for strong keywords that clearly indicate an intent."""
        text = message.lower()
        
        # Direct keywords that strongly indicate intents
        direct_mappings = {
            "email": self.INTENT_SEND_EMAIL,
            "schedule": self.INTENT_SCHEDULE_MEETING,
            "meeting": self.INTENT_SCHEDULE_MEETING,
            "calendar": self.INTENT_CHECK_CALENDAR,
            "schedule": self.INTENT_CHECK_CALENDAR,
            "contact": self.INTENT_FIND_CONTACT,
            "free time": self.INTENT_CHECK_FREE_SLOTS,
            "availability": self.INTENT_CHECK_FREE_SLOTS
        }
        
        # Check specific patterns for calendar queries
        calendar_patterns = [
            r"what'?s\s+on\s+(?:my\s+)?calendar",
            r"what\s+is\s+on\s+(?:my\s+)?calendar",
            r"show\s+(?:my\s+)?calendar",
            r"check\s+(?:my\s+)?calendar"
        ]
        
        for pattern in calendar_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                print(f"Detected calendar intent via quick pattern match")
                return {"intent": self.INTENT_CHECK_CALENDAR, "confidence": 0.95}
        
        # Check for direct keyword matches
        for keyword, intent in direct_mappings.items():
            if keyword in text:
                print(f"Detected intent via direct keyword '{keyword}': {intent}")
                return {"intent": intent, "confidence": 0.9}
        
        # No quick matches found
        return None
    
    def _recognize_intent_rule_based(self, message):
        """
        Recognize intent using enhanced rule-based approach as fallback.
        
        Args:
            message: The user message
            
        Returns:
            Dict containing intent type and confidence score
        """
        print("Using rule-based intent recognition")
        
        if not message.strip():
            return {"intent": self.INTENT_UNKNOWN, "confidence": 0.0}
        
        # Convert to lowercase for matching
        text = message.lower()
        
        # Enhanced patterns for better matching
        
        # Calendar patterns - Check these first as they're common
        calendar_patterns = [
            r"what'?s\s+on\s+(?:my\s+)?calendar",
            r"what\s+is\s+on\s+(?:my\s+)?calendar",
            r"check\s+(?:my\s+)?calendar",
            r"show\s+(?:my\s+)?calendar",
            r"what\s+do\s+i\s+have\s+(?:on|for|scheduled)",
            r"calendar\s+for\s+today",
            r"today'?s\s+(?:events|calendar|schedule)",
            r"my\s+events",
            r"my\s+schedule",
            r"my\s+agenda",
            r"what\s+events",
            r"any\s+events",
            r"appointments\s+(?:today|tomorrow|this week)",
            r"meetings\s+(?:today|tomorrow|this week)",
        ]
        
        if any(re.search(pattern, text) for pattern in calendar_patterns):
            print("Detected calendar checking intent")
            return {"intent": self.INTENT_CHECK_CALENDAR, "confidence": 0.9}
        
        # Email patterns
        email_patterns = [
            r"send\s+(?:an\s+)?email",
            r"write\s+(?:an\s+)?email",
            r"email\s+to",
            r"compose\s+(?:an\s+)?email",
            r"send\s+(?:a\s+)?message\s+to"
        ]
        
        # Meeting patterns
        meeting_patterns = [
            r"schedule\s+(?:a\s+)?meeting",
            r"set\s+up\s+(?:a\s+)?meeting",
            r"book\s+(?:a\s+)?meeting",
            r"arrange\s+(?:a\s+)?meeting",
            r"plan\s+(?:a\s+)?meeting",
            r"set\s+(?:a\s+)?appointment"
        ]
        
        # Contact patterns
        contact_patterns = [
            r"find\s+contact",
            r"find\s+(?:the\s+)?email\s+(?:address\s+)?(?:for|of)",
            r"get\s+contact\s+(?:info|information)",
            r"look\s+up\s+contact",
            r"search\s+(?:for\s+)?contact",
            r"who\s+is",
            r"contact\s+information",
            r"contact\s+details"
        ]
        
        # Free slots patterns
        free_slots_patterns = [
            r"find\s+(?:a\s+)?free\s+(?:slot|time)",
            r"check\s+(?:my\s+)?availability",
            r"when\s+am\s+i\s+free",
            r"available\s+(?:slot|time)",
            r"open\s+(?:slot|time)",
            r"free\s+time"
        ]
        
        # Check for each intent with regex patterns
        if any(re.search(pattern, text) for pattern in email_patterns):
            print("Detected email intent")
            return {"intent": self.INTENT_SEND_EMAIL, "confidence": 0.9}
        
        if any(re.search(pattern, text) for pattern in meeting_patterns):
            print("Detected meeting intent")
            return {"intent": self.INTENT_SCHEDULE_MEETING, "confidence": 0.9}
        
        if any(re.search(pattern, text) for pattern in contact_patterns):
            print("Detected contact finding intent")
            return {"intent": self.INTENT_FIND_CONTACT, "confidence": 0.9}
        
        if any(re.search(pattern, text) for pattern in free_slots_patterns):
            print("Detected free slots checking intent")
            return {"intent": self.INTENT_CHECK_FREE_SLOTS, "confidence": 0.9}
        
        # Fall back to keyword matching
        keywords = {
            "email": self.INTENT_SEND_EMAIL,
            "mail": self.INTENT_SEND_EMAIL,
            "message": self.INTENT_SEND_EMAIL,
            "meeting": self.INTENT_SCHEDULE_MEETING,
            "schedule": self.INTENT_SCHEDULE_MEETING,
            "appointment": self.INTENT_SCHEDULE_MEETING,
            "calendar": self.INTENT_CHECK_CALENDAR,
            "events": self.INTENT_CHECK_CALENDAR,
            "agenda": self.INTENT_CHECK_CALENDAR,
            "contact": self.INTENT_FIND_CONTACT,
            "find": self.INTENT_FIND_CONTACT,
            "who is": self.INTENT_FIND_CONTACT,
            "availability": self.INTENT_CHECK_FREE_SLOTS,
            "free time": self.INTENT_CHECK_FREE_SLOTS,
            "when am i free": self.INTENT_CHECK_FREE_SLOTS,
        }
        
        for keyword, intent in keywords.items():
            if keyword in text:
                print(f"Detected intent via keyword '{keyword}': {intent}")
                return {"intent": intent, "confidence": 0.7}
        
        # Default to unknown intent with low confidence
        print("No intent detected, returning unknown")
        return {"intent": self.INTENT_UNKNOWN, "confidence": 0.3}