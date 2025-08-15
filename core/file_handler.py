"""
File handling utilities for AI Excel Assistant.
"""

import os
import tempfile
import logging
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class FileHandler:
    """Handles file operations and utilities."""
    
    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """Get basic file information."""
        try:
            if not os.path.exists(file_path):
                return {}
            
            stat = os.stat(file_path)
            file_size = stat.st_size
            file_size_mb = file_size / (1024 * 1024)
            
            return {
                'name': os.path.basename(file_path),
                'path': file_path,
                'size_bytes': file_size,
                'size_mb': round(file_size_mb, 2),
                'extension': os.path.splitext(file_path)[1].lower(),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'created': datetime.fromtimestamp(stat.st_ctime)
            }
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {}
    
    @staticmethod
    def create_temp_file(prefix: str = "ai_excel_", suffix: str = ".csv") -> str:
        """Create a temporary file and return its path."""
        try:
            temp_file = tempfile.NamedTemporaryFile(
                prefix=prefix,
                suffix=suffix,
                delete=False
            )
            temp_path = temp_file.name
            temp_file.close()
            logger.info(f"Created temporary file: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Error creating temp file: {e}")
            return ""
    
    @staticmethod
    def cleanup_temp_file(file_path: str) -> bool:
        """Clean up a temporary file."""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {e}")
            return False
    
    @staticmethod
    def validate_file_format(file_path: str) -> Tuple[bool, str]:
        """Validate if the file format is supported."""
        supported_formats = ['.csv', '.xlsx', '.xls']
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in supported_formats:
            return False, f"Unsupported file format: {file_ext}. Supported formats: {', '.join(supported_formats)}"
        
        return True, "File format is supported"
    
    @staticmethod
    def generate_export_filename(original_name: str, suffix: str = "_processed") -> str:
        """Generate a filename for exported data."""
        try:
            name, ext = os.path.splitext(original_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{name}{suffix}_{timestamp}{ext}"
        except Exception as e:
            logger.error(f"Error generating export filename: {e}")
            return f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> bool:
        """Ensure a directory exists, create if it doesn't."""
        try:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                logger.info(f"Created directory: {directory_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating directory: {e}")
            return False
    
    @staticmethod
    def get_file_size_display(size_bytes: int) -> str:
        """Convert file size to human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
