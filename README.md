**Personal AI Assistant:**

A WhatsApp-integrated personal assistant that helps manage emails, calendar events, and contacts through natural language commands. The assistant uses AI to understand requests and integrates with Google services to perform actions on your behalf.
Features

**WhatsApp Integration:**

Communicate with your assistant through WhatsApp
Email Management: Send emails and check your inbox
Calendar Management: Schedule meetings, check availability, and view upcoming events
Contact Management: Search and retrieve contact information
Natural Language Understanding: Simply describe what you need in natural language

**Prerequisites**

Python 3.8 or higher
Google account with Gmail, Calendar, and Contacts access
Twilio account (for WhatsApp integration)
OpenRouter API key (for AI processing)

**Installation**

Clone the repository:
bashgit clone https://github.com/yourusername/ai-assistant.git
cd ai-assistant

Create and activate a virtual environment:
bashpython -m venv ai_assistant

# On Windows
ai_assistant\Scripts\activate
# On macOS/Linux
source ai_assistant/bin/activate

**Install dependencies:**

bashpip install -r requirements.txt

Create a .env file in the project root with your API keys:

# Google API settings
GOOGLE_CREDENTIALS_FILE=credentials.json
GOOGLE_TOKEN_FILE=token.json

# Twilio settings
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=whatsapp:+14155238886
TWILIO_WEBHOOK_URL=https://your-domain.com/webhook

# OpenRouter API settings
OPENROUTER_API_KEY=your_openrouter_api_key

# Application settings
TIME_ZONE=Asia/Kolkata
DEBUG=False

**Set up Google API credentials:**

Go to the Google Cloud Console
Create a new project
Enable the Gmail API, Calendar API, and People API
Create OAuth 2.0 credentials
Download the credentials JSON file and save as credentials.json in the project root



# Usage

**CLI Mode (for testing)**

Run the assistant in CLI mode to test functionality without WhatsApp:
bashpython -m app.main --mode cli
You can interact with the assistant by typing messages in the terminal.
Special commands:

exit - Quit the assistant
sync contacts - Sync contacts from Google to local database

**WhatsApp Mode (with Twilio)**

Set up a Twilio WhatsApp sandbox:

Go to the Twilio Console
Set up the WhatsApp sandbox
Configure the webhook URL to point to your server's /webhook endpoint


Start the assistant in WhatsApp mode:
bashpython -m app.main --mode twilio

Expose your webhook endpoint using ngrok for testing:
bashngrok http 5000

Update your Twilio webhook URL with the ngrok URL.
Send a message to the WhatsApp number provided by Twilio.

**Example Commands**

"Schedule a meeting"
"Send an email to Sarah"
"What's on my calendar today?"
"Find contact information for Michael"
"When am I free tomorrow?"

# Troubleshooting

**Google API Issues**

If you encounter Google API authentication errors, delete the token.json file and restart the application to re-authenticate.
For rate limit errors, the system will automatically resume operations after waiting.

**WhatsApp Integration Issues**

Make sure your Twilio account is properly set up with WhatsApp sandbox.
Ensure your webhook URL is accessible from the internet.
Check that your webhook URL is correctly configured in Twilio.

**Contact Syncing Issues**

If contact syncing fails, run sync contacts again after 1-2 minutes.
Ensure your Google People API is enabled and properly authenticated.


License

Built with Twilio, Google APIs, and OpenRouter AI
Uses Hugging Face Transformers for fallback NLP processing
