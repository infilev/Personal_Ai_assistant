"""
Contacts service for accessing Google Contacts.
"""
from googleapiclient.errors import HttpError

from app.utils.auth import get_contacts_service
from app.utils.helpers import normalize_name

class ContactsService:
    def __init__(self):
        """Initialize the contacts service with Google People API."""
        self.service = get_contacts_service()
        if not self.service:
            print("Failed to initialize Contacts service")
    
    def search_contacts(self, query, max_results=10):
        """Improved contact search with strict matching and pagination support"""
        if not self.service:
            return []
            
        try:
            print(f"Searching contacts for: '{query}'")
            original_query = query  # Store original query for exact matching
            normalized_query = normalize_name(query)
            
            # Get ALL contacts with pagination
            connections = []
            next_page_token = None
            
            print("Fetching contacts with pagination...")
            while True:
                results = self.service.people().connections().list(
                    resourceName='people/me',
                    pageSize=1000,  # Maximum page size
                    personFields='names,emailAddresses,phoneNumbers',
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
            
            print(f"Total contacts found: {len(connections)}")
            
            # Create three tiers of matches
            exact_matches = []      # Perfect matches or exact substrings
            close_matches = []      # High similarity matches
            
            for person in connections:
                names = person.get('names', [])
                emails = person.get('emailAddresses', [])
                phones = person.get('phoneNumbers', [])
                
                # Skip contacts without names
                if not names:
                    continue
                
                # Get all name variations
                display_names = []
                original_display_names = []
                
                for name in names:
                    if name.get('displayName'):
                        display_name = name.get('displayName')
                        display_names.append(display_name)
                        original_display_names.append(display_name)
                        
                        # Add first/last names as separate entries
                        if name.get('givenName'):
                            display_names.append(name.get('givenName'))
                            original_display_names.append(name.get('givenName'))
                        if name.get('familyName'):
                            display_names.append(name.get('familyName'))
                            original_display_names.append(name.get('familyName'))
                
                # 1. Direct substring matching with original query (case insensitive)
                if any(original_query.lower() == name.lower() for name in original_display_names):
                    match_tier = exact_matches
                    match_quality = "exact match"
                    print(f"Exact match: '{original_display_names[0]}' matches '{original_query}'")
                
                # 2. Substring containing the query
                elif any(original_query.lower() in name.lower() for name in original_display_names):
                    match_tier = exact_matches
                    match_quality = "contains match"
                    print(f"Contains match: '{original_display_names[0]}' contains '{original_query}'")
                
                # 3. Name is contained in query
                elif any(name.lower() in original_query.lower() for name in original_display_names):
                    if len(original_query) - len(min(original_display_names, key=len)) <= 5:
                        # Only if the length difference is small
                        match_tier = close_matches
                        match_quality = "partial match"
                        print(f"Partial match: '{original_display_names[0]}' is within '{original_query}'")
                    else:
                        continue
                else:
                    # No good match
                    continue
                
                # Extract contact details
                primary_name = next((name.get('displayName') for name in names 
                                   if name.get('metadata', {}).get('primary', False)), 
                                  names[0].get('displayName') if names else 'No Name')
                
                primary_email = next((email.get('value') for email in emails 
                                    if email.get('metadata', {}).get('primary', False)), 
                                   emails[0].get('value') if emails else None)
                
                primary_phone = next((phone.get('value') for phone in phones 
                                    if phone.get('metadata', {}).get('primary', False)), 
                                   phones[0].get('value') if phones else None)
                
                contact_data = {
                    'resource_name': person.get('resourceName'),
                    'name': primary_name,
                    'match_quality': match_quality
                }
                
                if primary_email: 
                    contact_data['email'] = primary_email
                
                if primary_phone: 
                    contact_data['phone'] = primary_phone
                
                if primary_email or primary_phone:
                    match_tier.append(contact_data)
            
            # Combine results with exact matches first
            all_matches = exact_matches + close_matches
            filtered_contacts = all_matches[:max_results]
            
            print(f"Found matches: {len(filtered_contacts)} (Exact: {len(exact_matches)}, Close: {len(close_matches)})")
            return filtered_contacts
            
        except Exception as e:
            print(f"Error searching contacts: {e}")
            return []
    
    def get_contact_by_name(self, name):
        """
        Get a contact by name.
        
        Args:
            name: Contact name to search for
            
        Returns:
            Contact object or None if not found
        """
        print(f"Looking up contact by name: {name}")
        contacts = self.search_contacts(name, max_results=1)
        return contacts[0] if contacts else None
    
    def get_contact_details(self, resource_name):
        """
        Get detailed information for a specific contact.
        
        Args:
            resource_name: The resource name of the contact
            
        Returns:
            Dict containing contact details
        """
        if not self.service:
            return None
            
        try:
            person = self.service.people().get(
                resourceName=resource_name,
                personFields='names,emailAddresses,phoneNumbers,addresses,organizations'
            ).execute()
            
            # Extract information
            names = person.get('names', [])
            emails = person.get('emailAddresses', [])
            phones = person.get('phoneNumbers', [])
            addresses = person.get('addresses', [])
            organizations = person.get('organizations', [])
            
            # Get primary values
            name = next((name.get('displayName') for name in names 
                          if name.get('metadata', {}).get('primary', False)), 
                         names[0].get('displayName') if names else 'No Name')
            
            # Always include all emails, not just primary
            all_emails = [email.get('value') for email in emails if email.get('value')]
            email = all_emails[0] if all_emails else None
            
            # Always include all phone numbers, not just primary
            all_phones = [phone.get('value') for phone in phones if phone.get('value')]
            phone = all_phones[0] if all_phones else None
            
            address = next((address.get('formattedValue') for address in addresses 
                             if address.get('metadata', {}).get('primary', False)), 
                            addresses[0].get('formattedValue') if addresses else None)
            
            organization = next((org.get('name') for org in organizations 
                                 if org.get('metadata', {}).get('primary', False)), 
                                organizations[0].get('name') if organizations else None)
            
            # Include all emails and phones in the result
            result = {
                'resource_name': person.get('resourceName'),
                'name': name,
                'email': email,
                'phone': phone,
                'address': address,
                'organization': organization
            }
            
            # Add all emails and phones
            if len(all_emails) > 1:
                result['all_emails'] = all_emails
            
            if len(all_phones) > 1:
                result['all_phones'] = all_phones
                
            return result
            
        except HttpError as error:
            print(f"People API error: {error}")
            return None
        except Exception as e:
            print(f"Error getting contact details: {e}")
            return None