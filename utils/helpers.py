"""
Utility functions and helpers for AI Excel Assistant.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_excel_assistant.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_logging(log_file: str = "ai_excel_assistant.log", level: str = "INFO"):
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def format_number(number: float) -> str:
    """Format large numbers with commas."""
    if isinstance(number, (int, float)):
        return f"{number:,.2f}" if number % 1 != 0 else f"{int(number):,}"
    return str(number)

def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary."""
    try:
        return data.get(key, default)
    except (KeyError, TypeError, AttributeError):
        return default

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate if a DataFrame is valid and not empty."""
    if df is None:
        return False
    if not isinstance(df, pd.DataFrame):
        return False
    if df.empty:
        return False
    return True

def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive information about a DataFrame."""
    if not validate_dataframe(df):
        return {}
    
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'null_counts': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
        'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
    }
    
    return info

def create_sample_data() -> pd.DataFrame:
    """Create sample data for testing purposes."""
    import numpy as np
    
    # Create sample sales data
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'Date': pd.date_range('2024-01-01', periods=n_rows, freq='D'),
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor'], n_rows),
        'Category': np.random.choice(['Electronics', 'Computers', 'Mobile'], n_rows),
        'Sales_Amount': np.random.uniform(100, 5000, n_rows),
        'Quantity': np.random.randint(1, 10, n_rows),
        'Customer_ID': np.random.randint(1000, 9999, n_rows),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
        'Rating': np.random.uniform(1, 5, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 10), 'Rating'] = np.nan
    df.loc[np.random.choice(df.index, 5), 'Region'] = np.nan
    
    return df

def export_sample_data(file_path: str, format: str = 'csv') -> bool:
    """Export sample data to a file for testing."""
    try:
        df = create_sample_data()
        
        if format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format.lower() == 'excel':
            df.to_excel(file_path, index=False, engine='openpyxl')
        else:
            logger.error(f"Unsupported format: {format}")
            return False
        
        logger.info(f"Sample data exported to: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting sample data: {e}")
        return False

def get_application_info() -> Dict[str, str]:
    """Get application information."""
    return {
        'name': 'AI Excel Assistant',
        'version': '2.0.0',
        'description': 'Intelligent data analysis and transformation tool',
        'author': 'Local AI',
        'python_version': f"{pd.__version__}",
        'pandas_version': f"{pd.__version__}",
        'build_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available."""
    dependencies = {
        'pandas': False,
        'numpy': False,
        'openpyxl': False,
        'PyQt5': False
    }
    
    try:
        import pandas
        dependencies['pandas'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    try:
        import openpyxl
        dependencies['openpyxl'] = True
    except ImportError:
        pass
    
    try:
        import PyQt5
        dependencies['PyQt5'] = True
    except ImportError:
        pass
    
    return dependencies

def get_system_info() -> Dict[str, str]:
    """Get system information."""
    import platform
    
    return {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation()
    }

def create_backup_file(file_path: str) -> Optional[str]:
    """Create a backup of a file."""
    try:
        if not os.path.exists(file_path):
            return None
        
        # Create backup filename
        base_name = os.path.splitext(file_path)[0]
        extension = os.path.splitext(file_path)[1]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{base_name}_backup_{timestamp}{extension}"
        
        # Copy file
        import shutil
        shutil.copy2(file_path, backup_path)
        
        logger.info(f"Backup created: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return None

def cleanup_temp_files(temp_files: List[str]) -> int:
    """Clean up temporary files and return count of cleaned files."""
    cleaned_count = 0
    
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                cleaned_count += 1
                logger.debug(f"Cleaned up temp file: {temp_file}")
        except Exception as e:
            logger.warning(f"Could not clean up temp file {temp_file}: {e}")
    
    return cleaned_count

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def validate_file_path(file_path: str) -> bool:
    """Validate if a file path is valid and accessible."""
    try:
        if not file_path or not file_path.strip():
            return False
        
        # Check if file exists
        if not os.path.exists(file_path):
            return False
        
        # Check if it's a file (not directory)
        if not os.path.isfile(file_path):
            return False
        
        # Check if readable
        if not os.access(file_path, os.R_OK):
            return False
        
        return True
        
    except Exception:
        return False

def get_file_extension(file_path: str) -> str:
    """Get file extension from file path."""
    try:
        return os.path.splitext(file_path)[1].lower()
    except Exception:
        return ""

def is_supported_file_format(file_path: str) -> bool:
    """Check if file format is supported."""
    supported_formats = ['.csv', '.xlsx', '.xls']
    file_ext = get_file_extension(file_path)
    return file_ext in supported_formats
