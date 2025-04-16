"""
WhatsApp client integration using Twilio's official API.
"""
import os
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from app.config import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    DEBUG
)

class TwilioWhatsAppClient:
    def __init__(self, on_message=None):
        """
        Initialize the Twilio WhatsApp client.
        
        Args:
            on_message: Callback function for handling incoming messages (used for testing)
        """
        self.account_sid = TWILIO_ACCOUNT_SID
        self.auth_token = TWILIO_AUTH_TOKEN
        self.phone_number = TWILIO_PHONE_NUMBER
        self.on_message_callback = on_message
        
        # Initialize Twilio client
        if self.account_sid and self.auth_token:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                self.initialized = True
                if DEBUG:
                    print("Twilio WhatsApp client initialized successfully")
            except Exception as e:
                print(f"Error initializing Twilio client: {e}")
                self.client = None
                self.initialized = False
        else:
            print("Twilio credentials not found. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables.")
            self.client = None
            self.initialized = False
    
    def send_message(self, to, message):
        """
        Send a WhatsApp message using Twilio API.
        
        Args:
            to: Phone number with country code but without '+' 
                (e.g., '1234567890' for a US number)
            message: Message text to send
            
        Returns:
            Dict containing success status and message SID if successful
        """
        if not self.initialized:
            return {"success": False, "error": "Twilio client not initialized"}
        
        try:
            # Format the 'to' number for WhatsApp
            # If it already has the whatsapp: prefix, use as is
            if to.startswith('whatsapp:'):
                to_formatted = to
            else:
                # Otherwise, add the whatsapp: prefix and + for country code
                to_formatted = f"whatsapp:+{to}"
            
            # Send the message
            message = self.client.messages.create(
                from_=self.phone_number,
                body=message,
                to=to_formatted
            )
            
            return {
                "success": True,
                "message_sid": message.sid,
                "status": message.status
            }
            
        except TwilioRestException as e:
            print(f"Twilio API error: {e}")
            return {"success": False, "error": f"Twilio API error: {e}"}
        except Exception as e:
            print(f"Error sending WhatsApp message: {e}")
            return {"success": False, "error": f"Error sending message: {e}"}
    
    def process_incoming_webhook(self, request_data):
        """
        Process incoming webhook data from Twilio.
        
        Args:
            request_data: The request data from Twilio webhook
            
        Returns:
            Dict containing message information
        """
        try:
            # Extract message details from the webhook data
            message_sid = request_data.get('MessageSid', '')
            from_number = request_data.get('From', '')
            body = request_data.get('Body', '')
            
            # Remove WhatsApp prefix from the phone number if present
            if from_number.startswith('whatsapp:'):
                from_number = from_number.replace('whatsapp:', '')
            
            # Remove the + from the phone number if present
            if from_number.startswith('+'):
                from_number = from_number[1:]
            
            # Call the message callback if provided (for testing)
            if self.on_message_callback:
                self.on_message_callback(from_number, body)
            
            return {
                "message_sid": message_sid,
                "from": from_number,
                "body": body
            }
            
        except Exception as e:
            print(f"Error processing webhook data: {e}")
            return None