# Setting Up Twilio WhatsApp Integration

This guide walks you through the process of setting up Twilio's WhatsApp API for your AI Personal Assistant.

## 1. Create a Twilio Account

1. Go to [Twilio's website](https://www.twilio.com/try-twilio) and sign up for an account
2. Complete the verification process (email verification and phone verification)
3. You'll receive some free credits to get started

## 2. Access the WhatsApp Sandbox

1. From your Twilio Console, navigate to:
   - Messaging > Try it out > Send a WhatsApp message

   OR
   
   - Direct link: [WhatsApp Sandbox](https://www.twilio.com/console/sms/whatsapp/sandbox)

2. You'll see instructions to connect your personal WhatsApp to the Twilio Sandbox:
   - Add the Twilio number to your contacts
   - Send the displayed code phrase to that number in WhatsApp
   - You should receive a confirmation message from Twilio

## 3. Get Your Twilio Credentials

1. In the Twilio Console, find your Account SID and Auth Token:
   - Look at the dashboard or go to [Account Settings](https://www.twilio.com/console/account/settings)
   - The Account SID is visible on the page
   - Click "Show" to reveal your Auth Token

2. Copy these credentials for your `.env` file:
   ```
   TWILIO_ACCOUNT_SID=your_account_sid
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_PHONE_NUMBER=whatsapp:+14155238886  # Default sandbox number
   ```

   Note: The default sandbox number is typically `whatsapp:+14155238886`, but use whatever number Twilio gave you in the sandbox. Make sure to include the `whatsapp:` prefix.

## 4. Set Up Your Webhook

Twilio needs to know where to send incoming WhatsApp messages. There are two main ways to set this up:

### Option A: Use ngrok for Development

1. Install ngrok:
   ```bash
   npm install -g ngrok
   ```

2. Start ngrok pointing to your Flask server port (default 5000):
   ```bash
   ngrok http 5000
   ```

3. Copy the HTTPS forwarding URL (e.g., `https://abc123.ngrok.io`)

4. In the Twilio WhatsApp Sandbox settings:
   - Set "WHEN A MESSAGE COMES IN" to `https://abc123.ngrok.io/webhook`
   - Click Save

### Option B: Deploy to a Public Server

1. Deploy your application to a server with a public IP address
2. Configure your server to expose the webhook endpoint (e.g., `https://yourdomain.com/webhook`)
3. In the Twilio WhatsApp Sandbox settings:
   - Set "WHEN A MESSAGE COMES IN" to `https://yourdomain.com/webhook`
   - Click Save

## 5. Update Your Environment Variables

Update your `.env` file with the webhook URL:

```
TWILIO_WEBHOOK_URL=https://yourdomain.com/webhook
```

Or for development with ngrok:

```
TWILIO_WEBHOOK_URL=https://abc123.ngrok.io/webhook
```

## 6. Start Your Application in Twilio Mode

```bash
python -m app.main --mode twilio
```

## 7. Test the Integration

1. Send a message from your WhatsApp to the Twilio sandbox number
2. Check your application logs to see if the message was received
3. You should get a response back from your AI assistant

## Common Issues and Troubleshooting

### Not Receiving Messages

- Make sure your webhook URL is publicly accessible
- Check that the URL is correctly set in the Twilio dashboard
- Verify that your application is running and listening for incoming requests
- Look at the Twilio Console Debugger for error messages

### Authentication Failures

- Double-check your Account SID and Auth Token
- Ensure there are no extra spaces in your `.env` file
- If you've regenerated your Auth Token, update it in your `.env` file

### Rate Limiting

- The Twilio WhatsApp Sandbox has limitations on the number of unique users you can message
- Consider upgrading to a production WhatsApp sender if you need more capacity

### Webhook Validation

For added security, you can implement webhook validation using Twilio's signature verification:

```python
from twilio.request_validator import RequestValidator

validator = RequestValidator(TWILIO_AUTH_TOKEN)

# Inside your Flask route
def handle_webhook():
    # Get the request data and headers
    request_data = request.form
    signature = request.headers.get('X-Twilio-Signature', '')
    url = request.url

    # Validate the request
    if validator.validate(url, request_data, signature):
        # Process the message
        # ...
    else:
        # Invalid request
        return "Invalid request", 403
```

## Moving to Production

The WhatsApp Sandbox is great for development but has limitations. To move to production:

1. Apply for a WhatsApp Business Profile through Twilio
2. Wait for approval (this can take several days or weeks)
3. Once approved, you'll get a dedicated WhatsApp number
4. Update your `TWILIO_PHONE_NUMBER` in the `.env` file to use this new number

For more details, see [Twilio's WhatsApp Production Documentation](https://www.twilio.com/docs/whatsapp/tutorial/connect-number-business-profile).