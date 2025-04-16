"""
Flask server for handling Twilio WhatsApp webhooks.
"""
import threading
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

from app.config import SERVER_HOST, SERVER_PORT, DEBUG

class WebhookServer:
    def __init__(self, message_handler):
        """
        Initialize the webhook server.
        
        Args:
            message_handler: Function to handle incoming messages
        """
        self.app = Flask(__name__)
        self.message_handler = message_handler
        self.server_thread = None
        self.running = False
        
        # Register routes
        self.app.route('/webhook', methods=['POST'])(self.handle_webhook)
        self.app.route('/health', methods=['GET'])(self.health_check)
    
    def handle_webhook(self):
        """
        Handle incoming webhook requests from Twilio.
        """
        # Create a TwiML response
        resp = MessagingResponse()
        
        try:
            # Process the incoming message
            form_data = request.form.to_dict()
            
            if DEBUG:
                print(f"Received webhook: {form_data}")
            
            # Extract message details
            from_number = form_data.get('From', '')
            message_body = form_data.get('Body', '')
            
            # Remove WhatsApp prefix from the phone number if present
            if from_number.startswith('whatsapp:'):
                from_number = from_number.replace('whatsapp:', '')
            
            # Remove the + from the phone number if present
            if from_number.startswith('+'):
                from_number = from_number[1:]
            
            # Pass to message handler
            if self.message_handler:
                # Process in a separate thread to avoid blocking the response
                threading.Thread(
                    target=self.message_handler,
                    args=(from_number, message_body)
                ).start()
            
        except Exception as e:
            print(f"Error handling webhook: {e}")
        
        # Return empty response (processing happens asynchronously)
        return str(resp)
    
    def health_check(self):
        """
        Health check endpoint.
        """
        return "OK", 200
    
    def start(self):
        """
        Start the webhook server in a separate thread.
        """
        if self.running:
            print("Webhook server is already running")
            return
        
        def run_server():
            self.app.run(
                host=SERVER_HOST,
                port=SERVER_PORT,
                debug=False,  # Always set to False in production
                use_reloader=False  # Disable reloader to avoid starting multiple threads
            )
        
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True  # Thread will exit when main thread exits
        self.server_thread.start()
        self.running = True
        
        print(f"Webhook server started at http://{SERVER_HOST}:{SERVER_PORT}")
        print(f"Configure your Twilio webhook URL to: http://YOUR_PUBLIC_IP:{SERVER_PORT}/webhook")
    
    def stop(self):
        """
        Stop the webhook server.
        """
        self.running = False
        # Note: Flask doesn't have a clean way to stop from another thread
        # The server will be terminated when the main thread exits
        print("Webhook server will stop when the application exits")