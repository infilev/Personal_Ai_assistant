"""
Email service for sending and retrieving emails using Gmail API.
"""
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googleapiclient.errors import HttpError

from app.utils.auth import get_gmail_service
from app.utils.helpers import summarize_text, is_valid_email

class EmailService:
    def __init__(self):
        """Initialize the email service with Gmail API."""
        self.service = get_gmail_service()
        if not self.service:
            print("Failed to initialize Gmail service")
    
    def send_email(self, to, subject, body, cc=None, bcc=None):
        """
        Send an email using Gmail API.
        
        Args:
            to: Recipient email address or list of addresses
            subject: Email subject
            body: Email body (HTML or plain text)
            cc: Carbon copy recipients (optional)
            bcc: Blind carbon copy recipients (optional)
            
        Returns:
            Dict containing success status and message ID if successful
        """
        if not self.service:
            return {"success": False, "error": "Gmail service not initialized"}
        
        # Convert single email to list
        if isinstance(to, str):
            to = [to]
        
        # Validate email addresses
        for email in to:
            if not is_valid_email(email):
                return {"success": False, "error": f"Invalid email address: {email}"}
        
        try:
            # Create message
            message = MIMEMultipart()
            message['to'] = ', '.join(to)
            message['subject'] = subject
            
            if cc:
                if isinstance(cc, str):
                    cc = [cc]
                message['cc'] = ', '.join(cc)
                
            if bcc:
                if isinstance(bcc, str):
                    bcc = [bcc]
                message['bcc'] = ', '.join(bcc)
            
            # Attach body
            message.attach(MIMEText(body, 'html'))
            
            # Encode message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Send message
            sent_message = self.service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            return {
                "success": True,
                "message_id": sent_message.get('id'),
                "thread_id": sent_message.get('threadId')
            }
            
        except HttpError as error:
            return {"success": False, "error": f"Gmail API error: {error}"}
        except Exception as e:
            return {"success": False, "error": f"Error sending email: {e}"}
    
    def get_recent_emails(self, max_results=10):
        """
        Retrieve recent emails from the inbox.
        
        Args:
            max_results: Maximum number of emails to retrieve
            
        Returns:
            List of email objects with basic information
        """
        if not self.service:
            return []
            
        try:
            # Get list of messages
            results = self.service.users().messages().list(
                userId='me',
                labelIds=['INBOX'],
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            # Get details for each message
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me', 
                    id=message['id']
                ).execute()
                
                # Extract headers
                headers = msg['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
                date = next((h['value'] for h in headers if h['name'].lower() == 'date'), 'Unknown Date')
                
                # Extract snippet
                snippet = msg.get('snippet', '')
                
                emails.append({
                    'id': msg['id'],
                    'thread_id': msg['threadId'],
                    'subject': subject,
                    'sender': sender,
                    'date': date,
                    'snippet': snippet
                })
                
            return emails
            
        except HttpError as error:
            print(f"Gmail API error: {error}")
            return []
        except Exception as e:
            print(f"Error getting emails: {e}")
            return []
    
    def get_email_content(self, message_id):
        """
        Get the content of a specific email.
        
        Args:
            message_id: The ID of the email to retrieve
            
        Returns:
            Dict containing email details and content
        """
        if not self.service:
            return None
            
        try:
            # Get message
            message = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()
            
            # Extract headers
            headers = message['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
            to = next((h['value'] for h in headers if h['name'].lower() == 'to'), 'Unknown Recipient')
            date = next((h['value'] for h in headers if h['name'].lower() == 'date'), 'Unknown Date')
            
            # Extract content
            content = self._get_email_body(message['payload'])
            
            return {
                'id': message['id'],
                'thread_id': message['threadId'],
                'subject': subject,
                'sender': sender,
                'to': to,
                'date': date,
                'content': content
            }
            
        except HttpError as error:
            print(f"Gmail API error: {error}")
            return None
        except Exception as e:
            print(f"Error getting email content: {e}")
            return None
    
    def _get_email_body(self, payload):
        """
        Extract email body from the payload.
        
        Args:
            payload: The message payload from Gmail API
            
        Returns:
            The email body as plain text
        """
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif 'parts' in part:
                    return self._get_email_body(part)
        elif 'body' in payload and 'data' in payload['body']:
            return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return ""
    
    def summarize_email(self, email_content):
        """
        Summarize the content of an email.
        
        Args:
            email_content: The email content to summarize
            
        Returns:
            A summary of the email content
        """
        if isinstance(email_content, dict) and 'content' in email_content:
            return summarize_text(email_content['content'])
        return summarize_text(email_content)