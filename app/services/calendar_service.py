"""
Calendar service for managing events using Google Calendar API.
"""
import datetime
import pytz
from googleapiclient.errors import HttpError

from app.utils.auth import get_calendar_service
from app.utils.helpers import (
    format_datetime, 
    format_date, 
    format_time,
    create_time_slot_range,
    get_weekday_name,
    get_current_time
)
from app.config import TIME_ZONE

class CalendarService:
    def __init__(self):
        """Initialize the calendar service with Google Calendar API."""
        self.service = get_calendar_service()
        self.timezone = pytz.timezone(TIME_ZONE)
        if not self.service:
            print("Failed to initialize Calendar service")
    
    def create_event(self, summary, start_time, end_time, description=None, 
                     location=None, attendees=None, send_notifications=True):
        """
        Create a calendar event.
        
        Args:
            summary: Event title
            start_time: Start time (datetime object or ISO format string)
            end_time: End time (datetime object or ISO format string)
            description: Event description (optional)
            location: Event location (optional)
            attendees: List of attendee email addresses (optional)
            send_notifications: Whether to send notifications to attendees
            
        Returns:
            Dict containing success status and event details if successful
        """
        if not self.service:
            return {"success": False, "error": "Calendar service not initialized"}
        
        # Convert datetime objects to strings if needed
        if isinstance(start_time, datetime.datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime.datetime):
            end_time = end_time.isoformat()
        
        # Prepare event data
        event = {
            'summary': summary,
            'start': {
                'dateTime': start_time,
                'timeZone': TIME_ZONE,
            },
            'end': {
                'dateTime': end_time,
                'timeZone': TIME_ZONE,
            }
        }
        
        if description:
            event['description'] = description
            
        if location:
            event['location'] = location
            
        if attendees:
            if isinstance(attendees, str):
                attendees = [attendees]
            event['attendees'] = [{'email': email} for email in attendees]
            
        try:
            event = self.service.events().insert(
                calendarId='primary',
                body=event,
                sendUpdates='all' if send_notifications else 'none'
            ).execute()
            
            return {
                "success": True,
                "event_id": event.get('id'),
                "html_link": event.get('htmlLink')
            }
            
        except HttpError as error:
            return {"success": False, "error": f"Calendar API error: {error}"}
        except Exception as e:
            return {"success": False, "error": f"Error creating event: {e}"}
    
    def get_events(self, start_date=None, end_date=None, max_results=10):
        """
        Get events from the calendar.
        
        Args:
            start_date: Start date (datetime object, date object, or ISO format string)
            end_date: End date (datetime object, date object, or ISO format string)
            max_results: Maximum number of events to retrieve
            
        Returns:
            List of event objects
        """
        if not self.service:
            return []
        
        # Set default dates if not provided
        if not start_date:
            start_date = get_current_time().date()
        
        if not end_date:
            if isinstance(start_date, datetime.date) and not isinstance(start_date, datetime.datetime):
                end_date = start_date + datetime.timedelta(days=1)
            elif isinstance(start_date, datetime.datetime):
                end_date = start_date + datetime.timedelta(minutes=30)
            else:
                # Assume it's an ISO string
                try:
                    # Try to parse it
                    dt = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    end_date = (dt + datetime.timedelta(minutes=30)).isoformat()
                except Exception:
                    # Fall back to 1 day from now
                    end_date = (get_current_time() + datetime.timedelta(days=1)).isoformat()
        
        # Format dates for API
        if isinstance(start_date, datetime.date) and not isinstance(start_date, datetime.datetime):
            time_min = datetime.datetime.combine(start_date, datetime.time.min)
            time_min = self.timezone.localize(time_min).isoformat()
        elif isinstance(start_date, datetime.datetime):
            # Make sure it's timezone-aware
            if start_date.tzinfo is None:
                time_min = self.timezone.localize(start_date).isoformat()
            else:
                time_min = start_date.isoformat()
        else:
            # Assume it's already in ISO format
            time_min = start_date
        
        if isinstance(end_date, datetime.date) and not isinstance(end_date, datetime.datetime):
            time_max = datetime.datetime.combine(end_date, datetime.time.max)
            time_max = self.timezone.localize(time_max).isoformat()
        elif isinstance(end_date, datetime.datetime):
            # Make sure it's timezone-aware
            if end_date.tzinfo is None:
                time_max = self.timezone.localize(end_date).isoformat()
            else:
                time_max = end_date.isoformat()
        else:
            # Assume it's already in ISO format
            time_max = end_date
        
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                formatted_events.append({
                    'id': event['id'],
                    'summary': event.get('summary', 'No Title'),
                    'start': start,
                    'end': end,
                    'description': event.get('description', ''),
                    'location': event.get('location', ''),
                    'link': event.get('htmlLink', '')
                })
                
            return formatted_events
            
        except HttpError as error:
            print(f"Calendar API error: {error}")
            return []
        except Exception as e:
            print(f"Error getting events: {e}")
            return []
    
    def get_free_slots(self, date, start_time=None, end_time=None, duration_minutes=30):
        """
        Find free time slots on a specific date.
        
        Args:
            date: The date to check (date object or ISO format string)
            start_time: Start of working hours (time object or string, default: 9:00)
            end_time: End of working hours (time object or string, default: 17:00)
            duration_minutes: Duration of each slot in minutes
            
        Returns:
            List of available time slots as (start, end) datetime tuples
        """
        if not self.service:
            return []
        
        # Set default times if not provided
        if not start_time:
            start_time = datetime.time(9, 0)  # 9:00 AM
        elif isinstance(start_time, str):
            # Parse time string (format: HH:MM)
            hour, minute = map(int, start_time.split(':'))
            start_time = datetime.time(hour, minute)
            
        if not end_time:
            end_time = datetime.time(17, 0)  # 5:00 PM
        elif isinstance(end_time, str):
            # Parse time string (format: HH:MM)
            hour, minute = map(int, end_time.split(':'))
            end_time = datetime.time(hour, minute)
        
        # Convert date string to date object if needed
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        
        # Create datetime objects for start and end of the day
        day_start = datetime.datetime.combine(date, start_time)
        day_start = self.timezone.localize(day_start)
        
        day_end = datetime.datetime.combine(date, end_time)
        day_end = self.timezone.localize(day_end)
        
        # Get all slots for the day
        all_slots = create_time_slot_range(day_start, day_end, duration_minutes)
        
        # Get events for the day
        events = self.get_events(
            start_date=day_start,
            end_date=day_end
        )
        
        # Mark busy slots
        busy_slots = []
        for event in events:
            event_start = event['start']
            event_end = event['end']
            
            # Convert to datetime objects if they are strings
            if isinstance(event_start, str):
                event_start = datetime.datetime.fromisoformat(event_start.replace('Z', '+00:00'))
                event_start = event_start.astimezone(self.timezone)
                
            if isinstance(event_end, str):
                event_end = datetime.datetime.fromisoformat(event_end.replace('Z', '+00:00'))
                event_end = event_end.astimezone(self.timezone)
            
            # Add to busy slots
            busy_slots.append((event_start, event_end))
        
        # Find free slots
        free_slots = []
        for slot_start, slot_end in all_slots:
            is_free = True
            
            for busy_start, busy_end in busy_slots:
                # Check if slot overlaps with any busy period
                if (slot_start < busy_end and slot_end > busy_start):
                    is_free = False
                    break
            
            if is_free:
                free_slots.append((slot_start, slot_end))
        
        return free_slots
    
    def format_free_slots(self, free_slots):
        """
        Format free slots for display.
        
        Args:
            free_slots: List of (start, end) datetime tuples
            
        Returns:
            List of formatted time slot strings
        """
        formatted_slots = []
        
        for start, end in free_slots:
            start_time = format_time(start)
            end_time = format_time(end)
            formatted_slots.append(f"{start_time} - {end_time}")
        
        return formatted_slots
    
    def get_next_event(self):
        """
        Get the next upcoming event.
        
        Returns:
            Dict containing the next event details or None if no upcoming events
        """
        if not self.service:
            return None
            
        now = get_current_time().isoformat()
        
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=1,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                return None
                
            event = events[0]
            start = event['start'].get('dateTime', event['start'].get('date'))
            
            # Convert to datetime object if it's a string
            if isinstance(start, str):
                if 'T' in start:  # It's a datetime
                    start_dt = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
                else:  # It's a date
                    start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
                    
                start_dt = start_dt.astimezone(self.timezone)
            else:
                start_dt = start
                
            # Format date and time
            event_date = format_date(start_dt)
            event_time = format_time(start_dt)
            weekday = get_weekday_name(start_dt)
            
            return {
                'id': event['id'],
                'summary': event.get('summary', 'No Title'),
                'date': event_date,
                'time': event_time,
                'weekday': weekday,
                'location': event.get('location', ''),
                'description': event.get('description', ''),
                'link': event.get('htmlLink', '')
            }
            
        except HttpError as error:
            print(f"Calendar API error: {error}")
            return None
        except Exception as e:
            print(f"Error getting next event: {e}")
            return None