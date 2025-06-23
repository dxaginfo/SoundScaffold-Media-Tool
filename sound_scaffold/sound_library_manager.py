import os
import json
import uuid
import datetime
from google.cloud import storage
from google.cloud import firestore
from google.cloud import language_v1

class SoundLibraryManager:
    """Sound library management for SoundScaffold.
    
    This class provides functionality to manage, categorize, and search audio files
    in a sound library for media production.
    """
    
    def __init__(self, storage_client=None):
        """Initialize the SoundLibraryManager.
        
        Args:
            storage_client (google.cloud.storage.Client, optional): Storage client. Defaults to None.
        """
        self.storage_client = storage_client or storage.Client()
        self.db = self._initialize_db()
        self.language_client = language_v1.LanguageServiceClient()
    
    def _initialize_db(self):
        """Initialize Firestore database connection.
        
        Returns:
            google.cloud.firestore.Client: Firestore client.
        """
        return firestore.Client()
    
    def add_sound(self, sound_file, metadata=None):
        """Add a sound file to the library with metadata.
        
        Args:
            sound_file (str): Path to the sound file.
            metadata (dict, optional): Metadata for the sound file. Defaults to None.
            
        Returns:
            str: ID of the added sound in the library.
        """
        try:
            # Generate a unique ID for the sound
            sound_id = str(uuid.uuid4())
            
            # Set up metadata with defaults if not provided
            meta = metadata or {}
            file_name = os.path.basename(sound_file)
            
            # Set default metadata fields if not provided
            if 'name' not in meta:
                meta['name'] = os.path.splitext(file_name)[0]
            
            if 'category' not in meta:
                # Try to infer category from filename or path
                meta['category'] = self._infer_category(sound_file)
            
            if 'tags' not in meta:
                meta['tags'] = []
            
            if 'description' not in meta:
                meta['description'] = f"Sound file: {file_name}"
            
            # Add system metadata
            meta['file_name'] = file_name
            meta['upload_date'] = datetime.datetime.now().isoformat()
            meta['sound_id'] = sound_id
            
            # Upload file to Cloud Storage
            bucket_name = 'sound-scaffold-assets'  # Default bucket
            bucket = self.storage_client.bucket(bucket_name)
            blob_name = f"sound_library/{sound_id}/{file_name}"
            blob = bucket.blob(blob_name)
            
            # Upload file
            blob.upload_from_filename(sound_file)
            
            # Add file metadata to Cloud Storage object
            blob.metadata = {
                'sound_id': sound_id,
                'category': meta['category'],
                'name': meta['name']
            }
            blob.patch()
            
            # Add storage info to metadata
            meta['storage_bucket'] = bucket_name
            meta['storage_path'] = blob_name
            meta['content_type'] = blob.content_type
            meta['size_bytes'] = blob.size
            meta['md5_hash'] = blob.md5_hash
            
            # Store metadata in Firestore
            doc_ref = self.db.collection('sound_library').document(sound_id)
            doc_ref.set(meta)
            
            # Auto-categorize if enabled
            if metadata and metadata.get('auto_categorize', False):
                self.categorize_sound(sound_id)
            
            return sound_id
        
        except Exception as e:
            print(f"Error adding sound to library: {e}")
            return None
    
    def search_sounds(self, criteria):
        """Search the sound library using provided criteria.
        
        Args:
            criteria (dict): Search criteria such as category, tags, etc.
            
        Returns:
            list: List of matching sound metadata.
        """
        try:
            # Start with base query
            query = self.db.collection('sound_library')
            
            # Apply filters based on criteria
            if 'category' in criteria:
                query = query.where('category', '==', criteria['category'])
            
            if 'tags' in criteria and criteria['tags']:
                # If multiple tags, search for any matching tag
                if isinstance(criteria['tags'], list) and len(criteria['tags']) > 0:
                    query = query.where('tags', 'array_contains_any', criteria['tags'])
                else:
                    query = query.where('tags', 'array_contains', criteria['tags'])
            
            if 'name' in criteria:
                # Filter by name (simple equality, not substring)
                query = query.where('name', '==', criteria['name'])
            
            if 'text_search' in criteria and criteria['text_search']:
                # This is a simplified approach - Firestore doesn't support true full-text search
                # In a real implementation, consider using Algolia, Elasticsearch, or similar
                # For now, we'll do a simple client-side filter
                results = query.get()
                filtered_results = []
                search_term = criteria['text_search'].lower()
                
                for doc in results:
                    data = doc.to_dict()
                    # Search in name, description, and tags
                    if (
                        search_term in data.get('name', '').lower() or
                        search_term in data.get('description', '').lower() or
                        any(search_term in tag.lower() for tag in data.get('tags', []))
                    ):
                        filtered_results.append(data)
                
                return filtered_results
            
            # Apply duration filter if provided
            if 'duration_min' in criteria or 'duration_max' in criteria:
                results = query.get()
                filtered_results = []
                
                for doc in results:
                    data = doc.to_dict()
                    duration = data.get('duration', 0)
                    
                    # Check minimum duration
                    if 'duration_min' in criteria and duration < criteria['duration_min']:
                        continue
                    
                    # Check maximum duration
                    if 'duration_max' in criteria and duration > criteria['duration_max']:
                        continue
                    
                    filtered_results.append(data)
                
                return filtered_results
            
            # Execute query and gather results
            results = query.get()
            return [doc.to_dict() for doc in results]
        
        except Exception as e:
            print(f"Error searching sound library: {e}")
            return []
    
    def categorize_sound(self, sound_id):
        """Auto-categorize a sound file using Gemini API.
        
        Args:
            sound_id (str): ID of the sound to categorize.
            
        Returns:
            dict: Updated metadata with categorization results.
        """
        try:
            # Get sound metadata from Firestore
            doc_ref = self.db.collection('sound_library').document(sound_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                print(f"Sound with ID {sound_id} not found")
                return None
            
            metadata = doc.to_dict()
            
            # Get the sound file from Cloud Storage
            bucket_name = metadata.get('storage_bucket')
            storage_path = metadata.get('storage_path')
            
            if not bucket_name or not storage_path:
                print(f"Storage information missing for sound {sound_id}")
                return metadata
            
            # For this example, we'll use a simple text-based categorization approach
            # In a real implementation, this would use actual audio analysis with Gemini API
            
            # Extract text to analyze (name, description, etc.)
            text_to_analyze = f"{metadata.get('name', '')} {metadata.get('description', '')}"
            
            # Use Google Cloud Natural Language API for entity and category detection
            document = language_v1.Document(
                content=text_to_analyze,
                type_=language_v1.Document.Type.PLAIN_TEXT
            )
            
            # Analyze entities
            entities_response = self.language_client.analyze_entities(document=document)
            
            # Extract relevant entities as tags
            new_tags = []
            for entity in entities_response.entities:
                if entity.salience > 0.1:  # Only include significant entities
                    new_tags.append(entity.name.lower())
            
            # Determine sound category based on name and existing tags
            sound_categories = {
                'ambient': ['ambient', 'background', 'environment', 'nature', 'atmosphere'],
                'sfx': ['effect', 'sfx', 'sound effect', 'impact', 'explosion', 'crash'],
                'music': ['music', 'melody', 'song', 'tune', 'rhythm', 'beat'],
                'voice': ['voice', 'speech', 'dialog', 'vocal', 'talking', 'conversation'],
                'foley': ['foley', 'footstep', 'cloth', 'movement', 'practical']
            }
            
            # Count category matches in name, description, and tags
            category_scores = {category: 0 for category in sound_categories}
            
            for category, keywords in sound_categories.items():
                for keyword in keywords:
                    if keyword in metadata.get('name', '').lower():
                        category_scores[category] += 3  # Name matches are weighted higher
                    
                    if keyword in metadata.get('description', '').lower():
                        category_scores[category] += 2  # Description matches
                    
                    for tag in metadata.get('tags', []) + new_tags:
                        if keyword in tag.lower():
                            category_scores[category] += 1  # Tag matches
            
            # Determine best category
            best_category = max(category_scores.items(), key=lambda x: x[1])
            
            # Only update if we have a reasonable match or no category set
            if best_category[1] > 0 or 'category' not in metadata:
                metadata['category'] = best_category[0]
            
            # Add new tags, avoiding duplicates
            existing_tags = set(tag.lower() for tag in metadata.get('tags', []))
            for tag in new_tags:
                if tag.lower() not in existing_tags:
                    metadata.setdefault('tags', []).append(tag)
            
            # Update metadata in Firestore
            doc_ref.update({
                'category': metadata['category'],
                'tags': metadata['tags'],
                'auto_categorized': True,
                'categorization_date': datetime.datetime.now().isoformat()
            })
            
            return metadata
        
        except Exception as e:
            print(f"Error categorizing sound: {e}")
            return None
    
    def _infer_category(self, sound_file):
        """Infer category from filename or path.
        
        Args:
            sound_file (str): Path to the sound file.
            
        Returns:
            str: Inferred category.
        """
        # Simple category inference based on filename or path
        file_path = sound_file.lower()
        file_name = os.path.basename(file_path)
        
        # Check for category hints in path or filename
        if any(x in file_path for x in ['ambient', 'background', 'atmosphere', 'env']):
            return 'ambient'
        elif any(x in file_path for x in ['effect', 'sfx', 'impact', 'explosion']):
            return 'sfx'
        elif any(x in file_path for x in ['music', 'song', 'melody', 'beat']):
            return 'music'
        elif any(x in file_path for x in ['voice', 'dialog', 'speech', 'vocal']):
            return 'voice'
        elif any(x in file_path for x in ['foley', 'footstep', 'cloth', 'movement']):
            return 'foley'
        else:
            # Default category
            return 'uncategorized'
    
    def get_sound_details(self, sound_id):
        """Get detailed information about a sound.
        
        Args:
            sound_id (str): ID of the sound.
            
        Returns:
            dict: Sound metadata and details.
        """
        try:
            # Get sound metadata from Firestore
            doc_ref = self.db.collection('sound_library').document(sound_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                print(f"Sound with ID {sound_id} not found")
                return None
            
            metadata = doc.to_dict()
            
            # Get download URL if available
            if 'storage_bucket' in metadata and 'storage_path' in metadata:
                bucket = self.storage_client.bucket(metadata['storage_bucket'])
                blob = bucket.blob(metadata['storage_path'])
                
                # Generate a signed URL that expires in 1 hour
                url = blob.generate_signed_url(
                    version='v4',
                    expiration=datetime.timedelta(hours=1),
                    method='GET'
                )
                
                metadata['download_url'] = url
            
            return metadata
        
        except Exception as e:
            print(f"Error getting sound details: {e}")
            return None
    
    def update_metadata(self, sound_id, metadata_updates):
        """Update metadata for a sound.
        
        Args:
            sound_id (str): ID of the sound.
            metadata_updates (dict): Metadata fields to update.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Get reference to document
            doc_ref = self.db.collection('sound_library').document(sound_id)
            
            # Check if document exists
            if not doc_ref.get().exists:
                print(f"Sound with ID {sound_id} not found")
                return False
            
            # Update metadata fields
            doc_ref.update(metadata_updates)
            
            return True
        
        except Exception as e:
            print(f"Error updating metadata: {e}")
            return False
    
    def delete_sound(self, sound_id):
        """Delete a sound from the library.
        
        Args:
            sound_id (str): ID of the sound to delete.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Get sound metadata from Firestore
            doc_ref = self.db.collection('sound_library').document(sound_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                print(f"Sound with ID {sound_id} not found")
                return False
            
            metadata = doc.to_dict()
            
            # Delete file from Cloud Storage if available
            if 'storage_bucket' in metadata and 'storage_path' in metadata:
                bucket = self.storage_client.bucket(metadata['storage_bucket'])
                blob = bucket.blob(metadata['storage_path'])
                blob.delete()
            
            # Delete metadata from Firestore
            doc_ref.delete()
            
            return True
        
        except Exception as e:
            print(f"Error deleting sound: {e}")
            return False
