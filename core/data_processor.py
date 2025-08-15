"""
Core data processing module for AI Excel Assistant.
Handles data loading, analysis, transformations, and AI operations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import re
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processing class for handling Excel/CSV data."""
    
    def __init__(self):
        self.df_original = None
        self.df_transformed = None
        self.analysis = None
        self.transformation_steps = []
        
    def load_file(self, file_path: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """Load Excel or CSV file with error handling."""
        try:
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                return False, "Unsupported file format. Please use CSV, XLSX, or XLS files.", None
            
            # Basic data cleaning
            df = df.dropna(how='all')  # Remove completely empty rows
            df = df.dropna(axis=1, how='all')  # Remove completely empty columns
            
            self.df_original = df.copy()
            self.df_transformed = df.copy()
            
            logger.info(f"Successfully loaded file: {file_path}")
            return True, f"Successfully loaded {len(df)} rows × {len(df.columns)} columns", df
            
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None
    
    def analyze_data_structure(self) -> Dict[str, Any]:
        """Analyze data structure and provide insights."""
        if self.df_transformed is None:
            return {}
            
        df = self.df_transformed
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'date_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Detect potential data quality issues
        analysis['quality_issues'] = []
        
        # Check for mixed data types
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                    analysis['quality_issues'].append(f"Column '{col}' contains numeric data but is stored as text")
                except:
                    pass
        
        # Check for inconsistent date formats
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col], errors='raise')
                    analysis['quality_issues'].append(f"Column '{col}' contains date data but is stored as text")
                except:
                    pass
        
        self.analysis = analysis
        return analysis
    
    def suggest_transformations(self) -> List[Dict[str, str]]:
        """Suggest data transformations based on analysis."""
        if not self.analysis:
            return []
            
        suggestions = []
        
        # Data type conversions
        for col in self.analysis['quality_issues']:
            if 'numeric' in col:
                suggestions.append({
                    'type': 'convert_numeric',
                    'column': col.split("'")[1],
                    'description': f"Convert '{col.split("'")[1]}' to numeric type",
                    'code': f"df['{col.split("'")[1]}'] = pd.to_numeric(df['{col.split("'")[1]}'], errors='coerce')"
                })
            elif 'date' in col:
                suggestions.append({
                    'type': 'convert_date',
                    'column': col.split("'")[1],
                    'description': f"Convert '{col.split("'")[1]}' to datetime type",
                    'code': f"df['{col.split("'")[1]}'] = pd.to_datetime(df['{col.split("'")[1]}'], errors='coerce')"
                })
        
        # Handle missing values
        for col, missing_count in self.analysis['missing_values'].items():
            if missing_count > 0:
                if col in self.analysis['numeric_columns']:
                    suggestions.append({
                        'type': 'fill_missing_numeric',
                        'column': col,
                        'description': f"Fill missing values in '{col}' with median",
                        'code': f"df['{col}'].fillna(df['{col}'].median(), inplace=True)"
                    })
                elif col in self.analysis['categorical_columns']:
                    suggestions.append({
                        'type': 'fill_missing_categorical',
                        'column': col,
                        'description': f"Fill missing values in '{col}' with mode",
                        'code': f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)"
                    })
        
        # Remove duplicates
        if self.analysis['duplicates'] > 0:
            suggestions.append({
                'type': 'remove_duplicates',
                'column': 'all',
                'description': f"Remove {self.analysis['duplicates']} duplicate rows",
                'code': "df.drop_duplicates(inplace=True)"
            })
        
        return suggestions
    
    def apply_transformation(self, transformation: Dict[str, str]) -> bool:
        """Apply a single transformation to the DataFrame."""
        try:
            if self.df_transformed is None:
                return False
                
            df = self.df_transformed
            
            if transformation['type'] == 'convert_numeric':
                df[transformation['column']] = pd.to_numeric(df[transformation['column']], errors='coerce')
            elif transformation['type'] == 'convert_date':
                df[transformation['column']] = pd.to_datetime(df[transformation['column']], errors='coerce')
            elif transformation['type'] == 'fill_missing_numeric':
                df[transformation['column']].fillna(df[transformation['column']].median(), inplace=True)
            elif transformation['type'] == 'fill_missing_categorical':
                df[transformation['column']].fillna(df[transformation['column']].mode()[0], inplace=True)
            elif transformation['type'] == 'remove_duplicates':
                df.drop_duplicates(inplace=True)
            
            self.df_transformed = df
            self.transformation_steps.append(transformation)
            logger.info(f"Applied transformation: {transformation['description']}")
            return True
            
        except Exception as e:
            logger.error(f"Transformation error: {e}")
            return False
    
    def get_data_preview(self, rows: int = 10) -> str:
        """Get a preview of the current data."""
        if self.df_transformed is None:
            return "No data loaded"
        
        df = self.df_transformed
        preview = f"Data Preview ({len(df)} rows × {len(df.columns)} columns):\n\n"
        preview += df.head(rows).to_string()
        return preview
    
    def export_data(self, file_path: str, format: str = 'csv') -> bool:
        """Export the transformed data to a file."""
        try:
            if self.df_transformed is None:
                return False
            
            if format.lower() == 'csv':
                self.df_transformed.to_csv(file_path, index=False)
            elif format.lower() == 'excel':
                self.df_transformed.to_excel(file_path, index=False, engine='openpyxl')
            else:
                return False
            
            logger.info(f"Data exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            return False
    
    def reset_transformations(self):
        """Reset to original data."""
        if self.df_original is not None:
            self.df_transformed = self.df_original.copy()
            self.transformation_steps = []
            logger.info("Reset to original data")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the current data."""
        if self.df_transformed is None:
            return {}
        
        df = self.df_transformed
        stats = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'date_columns': len(df.select_dtypes(include=['datetime64']).columns)
        }
        
        return stats
