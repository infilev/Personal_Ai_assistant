"""
Main application entry point for the AI Personal Assistant.
Supports both Twilio WhatsApp API mode and CLI testing mode.
"""
import os
import signal
import sys
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.whatsapp.twilio_client import TwilioWhatsAppClient
from app.whatsapp.webhook_server import WebhookServer
from app.whatsapp.message_handler import MessageHandler
from app.config import DEBUG, DEFAULT_PHONE_NUMBER

def signal_handler(sig, frame):
    """Handle interrupt signals to exit gracefully."""
    print("\nExiting application...")
    sys.exit(0)

def start_cli_mode():
    """Start the assistant in CLI mode for testing."""
    print("Starting AI Personal Assistant in CLI mode...")
    print("Loading transformer models (this may take a moment)...")
    
    # Initialize a mock WhatsApp client for CLI
    class MockWhatsAppClient:
        def send_message(self, to, message):
            print(f"\nAssistant: {message}\n")
    
    mock_client = MockWhatsAppClient()
    
    # Initialize message handler with mock client
    message_handler = MessageHandler(mock_client)
    
    # Set a mock phone number for state tracking
    phone_number = DEFAULT_PHONE_NUMBER
    
    # Main interaction loop
    try:
        print("\nAI Personal Assistant CLI mode")
        print("Type 'exit' to quit")
        print("Type 'sync contacts' to sync contacts database")
        print("Type your message as if you were sending it on WhatsApp\n")
        
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Exiting assistant CLI...")
                break
            
            # In the start_cli_mode function

            # Sync contacts command
            if user_input.lower() == 'sync contacts':
                print("Syncing contacts to local database...")
                result = message_handler.contacts_db_service.sync_contacts(message_handler.contacts_service)
                
                if isinstance(result, dict) and result.get("success"):
                    if result.get("complete"):
                        print("Contacts synchronized successfully!")
                    else:
                        print(f"Partially synchronized {result.get('contacts_synced')} contacts.")
                        print("Run 'sync contacts' again after 1-2 minutes to continue.")
                else:
                    print("Failed to synchronize contacts. Check connection to Google services.")
                continue
            
            # Process message
            message_handler.handle_message(phone_number, user_input)
            
    except KeyboardInterrupt:
        print("\nExiting assistant CLI...")
    except Exception as e:
        print(f"Error in CLI mode: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()

def start_twilio_mode():
    """Start the assistant in Twilio WhatsApp API mode."""
    print("Starting AI Personal Assistant with Twilio WhatsApp API...")
    
    # Initialize Twilio WhatsApp client
    twilio_client = TwilioWhatsAppClient()
    
    if not twilio_client.initialized:
        print("Failed to initialize Twilio client. Please check your credentials.")
        print("Exiting...")
        return
    
    # Initialize message handler
    message_handler = MessageHandler(twilio_client)
    
    # Start webhook server
    webhook_server = WebhookServer(
        message_handler=lambda from_number, message: 
            message_handler.handle_message(from_number, message)
    )
    
    webhook_server.start()
    
    # Display instructions
    print("\nAI Personal Assistant is running with Twilio WhatsApp integration.")
    print("To use the assistant:")
    print("1. Configure your Twilio WhatsApp Sandbox or phone number")
    print("2. Set up a webhook to forward messages to this application")
    print("3. Send a message to your Twilio WhatsApp number")
    print("\nThe assistant can help with:")
    print("- Sending emails")
    print("- Scheduling meetings")
    print("- Checking calendar")
    print("- Finding contacts")
    print("- Checking availability")
    print("\nPress Ctrl+C to exit.")
    
    # Keep the application running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        webhook_server.stop()

def main():
    """Main application entry point with CLI parsing."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Personal Assistant')
    parser.add_argument(
        '--mode', 
        choices=['twilio', 'cli'], 
        default='twilio',
        help='Operating mode: twilio (WhatsApp API) or cli (command line interface)'
    )
    
    args = parser.parse_args()
    
    # Register signal handlers for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start in the selected mode
    if args.mode == 'cli':
        start_cli_mode()
    else:
        start_twilio_mode()

if __name__ == "__main__":
    main()