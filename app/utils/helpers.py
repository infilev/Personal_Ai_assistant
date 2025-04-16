"""
Helper utilities for the AI assistant.
Provides common functions used across the application.
"""
import datetime
import pytz
import re
from app.config import TIME_ZONE

def get_current_time():
    """Returns the current time in the configured timezone."""
    return datetime.datetime.now(pytz.timezone(TIME_ZONE))

def format_datetime(dt, format_str="%Y-%m-%d %H:%M:%S"):
    """Formats a datetime object as a string."""
    if isinstance(dt, str):
        return dt
    return dt.strftime(format_str)

def format_date(dt, format_str="%Y-%m-%d"):
    """Formats a date as a string."""
    if isinstance(dt, datetime.datetime):
        return dt.strftime(format_str)
    elif isinstance(dt, datetime.date):
        return dt.strftime(format_str)
    return dt

def format_time(dt, format_str="%H:%M"):
    """Formats a time as a string."""
    if isinstance(dt, datetime.datetime):
        return dt.strftime(format_str)
    elif isinstance(dt, datetime.time):
        return dt.strftime(format_str)
    elif callable(dt):  # Check if it's a function
        return str(dt)  # Convert function to string instead of calling .upper()
    return str(dt)  # Convert any other type to string

def is_valid_email(email):
    """
    Validates an email address format with enhanced error detection.
    
    Returns:
        tuple: (is_valid, error_message, suggestion)
    """
    # Basic format check with regex
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    is_valid = bool(re.match(email_regex, email))
    
    if is_valid:
        return True, None, None
    
    # Check for common errors and provide suggestions
    error_message = "Invalid email format."
    suggestion = None
    
    # Check for missing @ symbol
    if '@' not in email:
        error_message = "Missing '@' symbol in email address."
        parts = email.split('.')
        if len(parts) >= 2:
            domain_part = parts[-2] if len(parts) > 2 else parts[0]
            if 'gmail' in domain_part or 'gmaill' in domain_part or 'gamail' in domain_part:
                username = email.split('.')[0] if '.' in email else email
                suggestion = f"{username}@gmail.com"
            elif 'yahoo' in domain_part:
                username = email.split('.')[0] if '.' in email else email
                suggestion = f"{username}@yahoo.com"
            elif 'hotmail' in domain_part:
                username = email.split('.')[0] if '.' in email else email
                suggestion = f"{username}@hotmail.com"
            else:
                # Generic suggestion
                username = parts[0]
                domain = '.'.join(parts[1:])
                suggestion = f"{username}@{domain}"
    
    # Check for missing dot in domain
    elif '.' not in email.split('@')[1]:
        error_message = "Missing '.' in domain part of email."
        username, domain = email.split('@')
        
        if 'gmail' in domain or 'gmaill' in domain or 'gamail' in domain:
            suggestion = f"{username}@gmail.com"
        elif 'yahoo' in domain:
            suggestion = f"{username}@yahoo.com"
        elif 'hotmail' in domain:
            suggestion = f"{username}@hotmail.com"
        else:
            suggestion = f"{username}@{domain}.com"
    
    # Check for comma instead of dot
    elif ',' in email:
        error_message = "Email contains a comma (,) which should be a period (.)."
        suggestion = email.replace(',', '.')
    
    # Check for common domain typos
    elif '@' in email:
        username, domain = email.split('@')
        domain_parts = domain.split('.')
        
        if 'gamail' in domain_parts[0]:
            error_message = "Did you mean 'gmail' instead of 'gamail'?"
            domain_parts[0] = domain_parts[0].replace('gamail', 'gmail')
            suggestion = f"{username}@{'.'.join(domain_parts)}"
        elif 'gmaill' in domain_parts[0]:
            error_message = "Did you mean 'gmail' instead of 'gmaill'?"
            domain_parts[0] = domain_parts[0].replace('gmaill', 'gmail')
            suggestion = f"{username}@{'.'.join(domain_parts)}"
        elif 'gmal' in domain_parts[0]:
            error_message = "Did you mean 'gmail' instead of 'gmal'?"
            domain_parts[0] = domain_parts[0].replace('gmal', 'gmail')
            suggestion = f"{username}@{'.'.join(domain_parts)}"
    
    return False, error_message, suggestion

def normalize_name(name):
    """Normalizes a name for search purposes."""
    if not name:
        return ""
    # Convert to lowercase and remove extra spaces
    return " ".join(name.lower().split())

def create_time_slot_range(start_time, end_time, duration_minutes=30):
    """
    Creates a list of time slots between start_time and end_time.
    
    Args:
        start_time: The start time as a datetime object
        end_time: The end time as a datetime object
        duration_minutes: The duration of each slot in minutes
        
    Returns:
        A list of (start, end) tuples for each time slot
    """
    slots = []
    slot_start = start_time
    
    while slot_start < end_time:
        slot_end = slot_start + datetime.timedelta(minutes=duration_minutes)
        if slot_end <= end_time:
            slots.append((slot_start, slot_end))
        slot_start = slot_end
        
    return slots

def get_weekday_name(date):
    """Returns the name of the weekday for a given date."""
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return weekdays[date.weekday()]

def summarize_text(text, max_length=200):
    """
    Creates a summary of the given text.
    Very basic implementation - for production use a more sophisticated algorithm.
    
    Args:
        text: The text to summarize
        max_length: Maximum length of the summary
        
    Returns:
        A summary of the text
    """
    # Simple implementation - just return the first few sentences
    if len(text) <= max_length:
        return text
        
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    summary = ""
    for sentence in sentences:
        if len(summary) + len(sentence) + 1 <= max_length:
            summary += sentence + " "
        else:
            break
            
    return summary.strip()