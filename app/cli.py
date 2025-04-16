"""
Command-line interface for testing the AI Personal Assistant without WhatsApp.
"""
import sys

from app.nlp.intent_recognizer import IntentRecognizer
from app.nlp.entity_extractor import EntityExtractor
from app.services.email_service import EmailService
from app.services.calendar_service import CalendarService
from app.services.contacts_service import ContactsService

def main():
    """Run the assistant in CLI mode."""
    print("Starting AI Personal Assistant CLI mode...")
    print("Type 'exit' to quit.\n")
    
    # Initialize services
    intent_recognizer = IntentRecognizer()
    entity_extractor = EntityExtractor()
    email_service = EmailService()
    calendar_service = CalendarService()
    contacts_service = ContactsService()
    
    # Mock user state for multi-step conversations
    user_state = {}
    
    # Create a mock handler class that mimics the WhatsApp message handler
    class MockHandler:
        @staticmethod
        def send_response(message):
            print(f"\nAssistant: {message}\n")
    
    mock_handler = MockHandler()
    
    # Create a mock WhatsApp client
    class MockWhatsAppClient:
        def send_message(self, to, message):
            mock_handler.send_response(message)
    
    mock_client = MockWhatsAppClient()
    
    # Import message handler here to avoid circular imports
    from app.whatsapp.message_handler import MessageHandler
    message_handler = MessageHandler(mock_client)
    
    # Override send_response method to print to console
    message_handler._send_response = lambda _, message: mock_handler.send_response(message)
    
    # Set a mock phone number for state tracking
    MOCK_PHONE = "123456789"
    
    # Main interaction loop
    try:
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Exiting assistant CLI...")
                break
            
            # Process message
            message_handler.handle_message(MOCK_PHONE, user_input)
            
    except KeyboardInterrupt:
        print("\nExiting assistant CLI...")
    except Exception as e:
        print(f"Error in CLI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()