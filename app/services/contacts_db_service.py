"""
Local database service for cached contact searches.
"""
import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path
from app.config import BASE_DIR
from app.utils.helpers import normalize_name
from requests.exceptions import HTTPError
import time

class ContactsDBService:
    def __init__(self):
        """Initialize the local contacts database."""
        self.db_path = os.path.join(BASE_DIR, 'contacts.db')
        self.initialized = self._initialize_db()
        
    def _initialize_db(self):
        """Create the database and tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create contacts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS contacts (
                id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                phone TEXT,
                address TEXT,
                organization TEXT,
                raw_data TEXT,
                last_updated TEXT
            )
            ''')
            
            # Create search index
            cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS contacts_index USING fts5(
                id, name, email, organization
            )
            ''')
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error initializing contacts database: {e}")
            return False
    
    def sync_contacts(self, google_contacts_service, resume_from=None):
        """
        Sync contacts from Google to local database with resumption capability.
        
        Args:
            google_contacts_service: Google Contacts service instance
            resume_from: Optional sync token or position to resume from
        
        Returns:
            bool: Success status
        """
        import time
        
        if not self.initialized or not google_contacts_service:
            return False
        
        try:
            # First, check if we have a sync state
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sync state table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_state (
                id TEXT PRIMARY KEY,
                next_page_token TEXT,
                last_resource_name TEXT,
                last_updated TEXT
            )
            ''')
            
            # Get last sync state
            cursor.execute("SELECT next_page_token, last_resource_name FROM sync_state WHERE id = 'contacts_sync'")
            sync_state = cursor.fetchone()
            
            next_page_token = None
            last_resource_name = None
            
            if resume_from:
                # Use provided resume point
                if isinstance(resume_from, str):
                    next_page_token = resume_from
                elif isinstance(resume_from, dict):
                    next_page_token = resume_from.get('next_page_token')
                    last_resource_name = resume_from.get('last_resource_name')
            elif sync_state:
                # Use stored sync state
                next_page_token = sync_state[0]
                last_resource_name = sync_state[1]
                print(f"Resuming sync from page token: {next_page_token}")
                
            # Get all contacts from Google with pagination
            connections = []
            
            print("Fetching contacts for database sync...")
            while True:
                results = google_contacts_service.service.people().connections().list(
                    resourceName='people/me',
                    pageSize=500,  # Reduced page size to avoid hitting rate limits
                    personFields='names,emailAddresses,phoneNumbers,addresses,organizations',
                    pageToken=next_page_token,
                    sortOrder='LAST_MODIFIED_DESCENDING'
                ).execute()
                
                batch = results.get('connections', [])
                if batch:
                    connections.extend(batch)
                    print(f"  - Fetched batch of {len(batch)} contacts")
                
                next_page_token = results.get('nextPageToken')
                if not next_page_token or not batch:
                    break
                
            print(f"Total contacts to sync: {len(connections)}")
            
            # Begin transaction
            conn.execute("BEGIN TRANSACTION")
            
            # If this is a fresh sync (not resuming), clear existing indexes
            if not resume_from and not sync_state:
                cursor.execute("DELETE FROM contacts_index")
            
            contact_count = 0
            resume_point = None
            
            # Determine starting point if resuming
            start_index = 0
            if last_resource_name:
                for i, person in enumerate(connections):
                    if person.get('resourceName') == last_resource_name:
                        start_index = i + 1
                        print(f"Resuming from contact {start_index} of {len(connections)}")
                        break
            
            for i in range(start_index, len(connections)):
                person_resource = connections[i]
                
                try:
                    # Add a delay between requests to avoid rate limiting (0.7 seconds = ~85 requests per minute)
                    time.sleep(0.7)
                    
                    # Get full details
                    detail = google_contacts_service.get_contact_details(person_resource['resourceName'])
                    if not detail:
                        continue
                    
                    # Insert or update contact
                    cursor.execute('''
                    INSERT OR REPLACE INTO contacts 
                    (id, name, email, phone, address, organization, raw_data, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        detail['resource_name'],
                        detail.get('name', ''),
                        detail.get('email', ''),
                        detail.get('phone', ''),
                        detail.get('address', ''),
                        detail.get('organization', ''),
                        json.dumps(detail),
                        datetime.now().isoformat()
                    ))
                    
                    # Update search index
                    cursor.execute('''
                    INSERT INTO contacts_index (id, name, email, organization)
                    VALUES (?, ?, ?, ?)
                    ''', (
                        detail['resource_name'],
                        detail.get('name', ''),
                        detail.get('email', '') or '',
                        detail.get('organization', '') or ''
                    ))
                    
                    # Update sync state after each successful contact
                    cursor.execute('''
                    INSERT OR REPLACE INTO sync_state 
                    (id, next_page_token, last_resource_name, last_updated)
                    VALUES (?, ?, ?, ?)
                    ''', (
                        'contacts_sync',
                        next_page_token,
                        detail['resource_name'],
                        datetime.now().isoformat()
                    ))
                    
                    # Save the current position for resuming if needed
                    resume_point = {
                        'next_page_token': next_page_token,
                        'last_resource_name': detail['resource_name']
                    }
                    
                    contact_count += 1
                    if contact_count % 20 == 0:
                        # Commit every 20 contacts to avoid losing progress
                        conn.commit()
                        print(f"  - Synced {contact_count} contacts...")
                    
                except HTTPError as e:
                    if "RATE_LIMIT_EXCEEDED" in str(e):
                        print(f"Rate limit exceeded after syncing {contact_count} contacts.")
                        print(f"You can resume syncing after 1-2 minutes by running 'sync contacts' again.")
                        
                        # Commit what we have so far
                        conn.commit()
                        
                        # Store sync state for resuming
                        if resume_point:
                            return {"success": True, "contacts_synced": contact_count, "resume_from": resume_point}
                        else:
                            return {"success": True, "contacts_synced": contact_count}
                    else:
                        raise
            
            # Commit final transaction
            conn.commit()
            
            # If we've completed the sync, clear the resume point
            if contact_count == len(connections) - start_index:
                cursor.execute("DELETE FROM sync_state WHERE id = 'contacts_sync'")
                conn.commit()
            
            conn.close()
            
            print(f"Successfully synced {contact_count} contacts to local database")
            return {"success": True, "contacts_synced": contact_count, "complete": True}
        
        except Exception as e:
            print(f"Error syncing contacts: {e}")
            try:
                # Try to store current progress
                if 'conn' in locals() and conn and resume_point:
                    conn.commit()
            except:
                pass
            return {"success": False, "error": str(e)}
        
    def search_contacts(self, query, max_results=10):
        """Search contacts in local database."""
        if not self.initialized:
            return []
            
        try:
            print(f"Searching local contacts for: '{query}'")
            
            # Normalize query
            normalized_query = normalize_name(query)
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # First try exact substring matching
            cursor.execute('''
            SELECT c.* FROM contacts c
            WHERE c.name LIKE ? 
            LIMIT ?
            ''', (f'%{query}%', max_results))
            
            exact_matches = [dict(row) for row in cursor.fetchall()]
            
            # If no exact matches, try fuzzy search using FTS
            if not exact_matches and len(normalized_query) > 2:
                cursor.execute('''
                SELECT c.* FROM contacts_index i
                JOIN contacts c ON i.id = c.id
                WHERE contacts_index MATCH ?
                LIMIT ?
                ''', (f'{normalized_query}*', max_results))
                
                fuzzy_matches = [dict(row) for row in cursor.fetchall()]
                results = fuzzy_matches
            else:
                results = exact_matches
            
            # Format results
            formatted_results = []
            for result in results:
                contact_data = {
                    'resource_name': result['id'],
                    'name': result['name'],
                    'match_quality': 'database match'
                }
                
                if result['email']: contact_data['email'] = result['email']
                if result['phone']: contact_data['phone'] = result['phone']
                
                formatted_results.append(contact_data)
            
            conn.close()
            return formatted_results
            
        except Exception as e:
            print(f"Error searching local contacts: {e}")
            return []
    
    def get_contact_by_name(self, name):
        """Get a contact by name from local database."""
        print(f"Looking up contact by name in local DB: {name}")
        contacts = self.search_contacts(name, max_results=1)
        return contacts[0] if contacts else None
    
    def get_contact_details(self, resource_name):
        """Get detailed contact information from local database."""
        if not self.initialized:
            return None
            
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM contacts WHERE id = ?', (resource_name,))
            result = cursor.fetchone()
            
            if not result:
                return None
                
            conn.close()
            
            # Convert to dict
            return {
                'resource_name': result['id'],
                'name': result['name'],
                'email': result['email'],
                'phone': result['phone'],
                'address': result['address'],
                'organization': result['organization']
            }
            
        except Exception as e:
            print(f"Error getting contact details from local DB: {e}")
            return None