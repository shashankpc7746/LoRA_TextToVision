#!/usr/bin/env python3
"""
Storage and Delivery System for AnimateDiff
- Manages video storage, organization, and delivery
- Provides access paths for UI integration
- Handles metadata export
- Implements file cleanup policies
"""

import os
import json
import shutil
import datetime
import hashlib
import requests
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("storage_delivery")

# Storage configuration
STORAGE_CONFIG = {
    'base_storage_dir': 'storage',
    'max_files_to_keep': 5,  # Keep only the 5 most recent files
    'production_endpoint': 'http://192.168.0.121:8001/receive-video',
    'metadata_format': 'json',
    'organize_by_date': True,
    'organize_by_style': True,
    'organize_by_lesson': True
}

class StorageDeliverySystem:
    """Manages video storage, organization, and delivery"""
    
    def __init__(self, config: Dict = None):
        """Initialize storage system with configuration"""
        self.config = config or STORAGE_CONFIG
        self.base_dir = self.config['base_storage_dir']
        
        # Create base storage directory
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"Storage system initialized at: {self.base_dir}")
        
        # Track stored files
        self.stored_files = []
    
    def generate_storage_path(self, video_path: str, metadata: Dict) -> str:
        """Generate organized storage path based on metadata"""
        # Extract components for organization
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        style = metadata.get('style', 'default')
        lesson_id = metadata.get('lesson_id', 'unknown')
        lesson_title = metadata.get('title', '').replace(' ', '_')[:30]
        
        # Create path components
        components = []
        
        if self.config['organize_by_date']:
            components.append(timestamp)
        
        if self.config['organize_by_style'] and style:
            components.append(f"style_{style}")
        
        if self.config['organize_by_lesson'] and lesson_id:
            components.append(f"lesson_{lesson_id}_{lesson_title}")
        
        # Create directory path
        storage_dir = os.path.join(self.base_dir, *components)
        os.makedirs(storage_dir, exist_ok=True)
        
        # Generate unique filename
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        
        # Add hash for uniqueness
        file_hash = hashlib.md5(f"{name}_{timestamp}".encode()).hexdigest()[:8]
        unique_name = f"{name}_{file_hash}{ext}"
        
        return os.path.join(storage_dir, unique_name)
    
    def store_video(self, video_path: str, metadata: Dict) -> Dict:
        """Store video with metadata and return access information"""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {'success': False, 'error': 'Video file not found'}
        
        # Generate storage path
        storage_path = self.generate_storage_path(video_path, metadata)
        
        try:
            # Copy video to storage
            shutil.copy2(video_path, storage_path)
            logger.info(f"Video stored at: {storage_path}")
            
            # Save metadata
            metadata_path = f"{os.path.splitext(storage_path)[0]}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Add to stored files
            self.stored_files.append({
                'original_path': video_path,
                'storage_path': storage_path,
                'metadata_path': metadata_path,
                'timestamp': datetime.datetime.now().isoformat(),
                'metadata': metadata
            })
            
            # Clean up old files if needed
            self.cleanup_old_files()
            
            # Generate access information
            access_info = {
                'success': True,
                'storage_path': storage_path,
                'metadata_path': metadata_path,
                'relative_path': os.path.relpath(storage_path, self.base_dir),
                'access_url': f"/videos/{os.path.relpath(storage_path, self.base_dir)}",
                'filename': os.path.basename(storage_path)
            }
            
            return access_info
            
        except Exception as e:
            logger.error(f"Error storing video: {e}")
            return {'success': False, 'error': str(e)}
    
    def cleanup_old_files(self) -> None:
        """Clean up old files, keeping only the most recent ones"""
        max_files = self.config['max_files_to_keep']
        
        if len(self.stored_files) <= max_files:
            return
        
        # Sort by timestamp (newest first)
        sorted_files = sorted(
            self.stored_files, 
            key=lambda x: x['timestamp'], 
            reverse=True
        )
        
        # Remove oldest files
        files_to_remove = sorted_files[max_files:]
        for file_info in files_to_remove:
            try:
                if os.path.exists(file_info['storage_path']):
                    os.remove(file_info['storage_path'])
                
                if os.path.exists(file_info['metadata_path']):
                    os.remove(file_info['metadata_path'])
                
                logger.info(f"Cleaned up old file: {file_info['storage_path']}")
            except Exception as e:
                logger.error(f"Error cleaning up file: {e}")
        
        # Update stored files list
        self.stored_files = sorted_files[:max_files]
    
    def deliver_to_production(self, video_path: str, metadata: Dict) -> Dict:
        """Deliver video to production system via API"""
        endpoint = self.config['production_endpoint']
        
        try:
            # Prepare multipart form data
            files = {
                'video': (os.path.basename(video_path), open(video_path, 'rb'), 'video/mp4')
            }
            
            # Add metadata
            data = {
                'subject': metadata.get('subject', 'Unknown'),
                'topic': metadata.get('title', 'Unknown'),
                'prompt': metadata.get('text', ''),
                'style': metadata.get('style', 'anime'),
                'level': metadata.get('level', 'Beginner')
            }
            
            # Send POST request
            response = requests.post(endpoint, files=files, data=data)
            
            if response.status_code == 200:
                logger.info(f"Video delivered to production system: {endpoint}")
                return {
                    'success': True,
                    'status_code': response.status_code,
                    'response': response.json() if response.headers.get('content-type') == 'application/json' else response.text
                }
            else:
                logger.error(f"Error delivering video: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': response.text
                }
                
        except Exception as e:
            logger.error(f"Error delivering video to production: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                file_count += 1
        
        return {
            'total_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'stored_videos': len([f for f in self.stored_files if f['storage_path'].endswith('.mp4')]),
            'base_directory': self.base_dir
        }

# Initialize global instance
storage_system = StorageDeliverySystem()
