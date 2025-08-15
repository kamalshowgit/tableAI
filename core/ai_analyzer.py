"""
AI analysis module for natural language processing and Excel operations.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """AI-powered analysis for natural language Excel operations."""
    
    def __init__(self):
        self.excel_operations = {
            'filter': ['filter', 'show only', 'where', 'condition', 'criteria', 'above', 'below', 'greater than', 'less than'],
            'sort': ['sort', 'order', 'arrange', 'highest', 'lowest', 'top', 'bottom', 'ascending', 'descending'],
            'calculate': ['sum', 'average', 'mean', 'count', 'total', 'calculate', 'formula', 'maximum', 'minimum'],
            'group': ['group by', 'pivot', 'categorize', 'by category', 'by group', 'aggregate'],
            'find': ['find', 'search', 'locate', 'which', 'what', 'contains'],
            'compare': ['compare', 'difference', 'vs', 'versus', 'between', 'against'],
            'trend': ['trend', 'pattern', 'over time', 'growth', 'decline', 'trending'],
            'outlier': ['outlier', 'unusual', 'extreme', 'anomaly', 'exception', 'unusual']
        }
    
    def understand_question(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Understand natural language question and determine Excel operations needed."""
        question_lower = question.lower()
        operations_needed = []
        
        # Determine operations needed
        for operation, keywords in self.excel_operations.items():
            if any(keyword in question_lower for keyword in keywords):
                operations_needed.append(operation)
        
        # Extract column names mentioned
        columns_mentioned = []
        for col in df.columns:
            if col.lower() in question_lower:
                columns_mentioned.append(col)
        
        # Extract values mentioned
        values_mentioned = re.findall(r'\d+(?:\.\d+)?', question)
        
        return {
            'operations': operations_needed,
            'columns': columns_mentioned,
            'values': values_mentioned,
            'question_type': 'filter' if 'filter' in operations_needed else 'analysis'
        }
    
    def execute_excel_operation(self, df: pd.DataFrame, operation_info: Dict[str, Any], question: str) -> Tuple[pd.DataFrame, List[str], str]:
        """Execute Excel operations based on natural language question."""
        try:
            result_df = df.copy()
            operation_log = []
            
            # Handle filtering operations
            if 'filter' in operation_info['operations']:
                result_df, logs = self._apply_filters(result_df, operation_info, question)
                operation_log.extend(logs)
            
            # Handle sorting operations
            if 'sort' in operation_info['operations']:
                result_df, logs = self._apply_sorting(result_df, operation_info, question)
                operation_log.extend(logs)
            
            # Handle calculations
            if 'calculate' in operation_info['operations']:
                result_df, logs = self._apply_calculations(result_df, operation_info, question)
                operation_log.extend(logs)
            
            # Handle grouping operations
            if 'group' in operation_info['operations']:
                result_df, logs = self._apply_grouping(result_df, operation_info, question)
                operation_log.extend(logs)
            
            # Handle find operations
            if 'find' in operation_info['operations']:
                result_df, logs = self._apply_find_operations(result_df, operation_info, question)
                operation_log.extend(logs)
            
            # Handle comparison operations
            if 'compare' in operation_info['operations']:
                result_df, logs = self._apply_comparisons(result_df, operation_info, question)
                operation_log.extend(logs)
            
            # Handle trend analysis
            if 'trend' in operation_info['operations']:
                result_df, logs = self._apply_trend_analysis(result_df, operation_info, question)
                operation_log.extend(logs)
            
            # Handle outlier detection
            if 'outlier' in operation_info['operations']:
                result_df, logs = self._apply_outlier_detection(result_df, operation_info, question)
                operation_log.extend(logs)
            
            return result_df, operation_log, "success"
            
        except Exception as e:
            logger.error(f"Error executing Excel operation: {e}")
            return df, [f"Error: {str(e)}"], "error"
    
    def _apply_filters(self, df: pd.DataFrame, operation_info: Dict[str, Any], question: str) -> Tuple[pd.DataFrame, List[str]]:
        """Apply filtering operations."""
        logs = []
        result_df = df.copy()
        
        for col in operation_info['columns']:
            if col in df.columns:
                # Look for comparison operators
                if any(op in question.lower() for op in ['>', '<', '>=', '<=']):
                    numbers = re.findall(r'\d+(?:\.\d+)?', question)
                    if numbers:
                        if '>' in question:
                            result_df = result_df[result_df[col] > float(numbers[0])]
                            logs.append(f"Filtered {col} > {numbers[0]}")
                        elif '<' in question:
                            result_df = result_df[result_df[col] < float(numbers[0])]
                            logs.append(f"Filtered {col} < {numbers[0]}")
                        elif '>=' in question:
                            result_df = result_df[result_df[col] >= float(numbers[0])]
                            logs.append(f"Filtered {col} >= {numbers[0]}")
                        elif '<=' in question:
                            result_df = result_df[result_df[col] <= float(numbers[0])]
                            logs.append(f"Filtered {col} <= {numbers[0]}")
                else:
                    # Text-based filtering
                    text_values = re.findall(r'"([^"]+)"', question)
                    if text_values:
                        result_df = result_df[result_df[col].str.contains(text_values[0], case=False, na=False)]
                        logs.append(f"Filtered {col} containing '{text_values[0]}'")
        
        return result_df, logs
    
    def _apply_sorting(self, df: pd.DataFrame, operation_info: Dict[str, Any], question: str) -> Tuple[pd.DataFrame, List[str]]:
        """Apply sorting operations."""
        logs = []
        result_df = df.copy()
        
        for col in operation_info['columns']:
            if col in df.columns:
                if any(word in question.lower() for word in ['highest', 'top', 'descending', 'desc']):
                    result_df = result_df.sort_values(col, ascending=False)
                    logs.append(f"Sorted {col} in descending order")
                elif any(word in question.lower() for word in ['lowest', 'bottom', 'ascending', 'asc']):
                    result_df = result_df.sort_values(col, ascending=True)
                    logs.append(f"Sorted {col} in ascending order")
        
        return result_df, logs
    
    def _apply_calculations(self, df: pd.DataFrame, operation_info: Dict[str, Any], question: str) -> Tuple[pd.DataFrame, List[str]]:
        """Apply calculation operations."""
        logs = []
        result_df = df.copy()
        
        for col in operation_info['columns']:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                if 'sum' in question.lower():
                    total = result_df[col].sum()
                    logs.append(f"Sum of {col}: {total:,.2f}")
                elif 'average' in question.lower() or 'mean' in question.lower():
                    avg = result_df[col].mean()
                    logs.append(f"Average of {col}: {avg:,.2f}")
                elif 'count' in question.lower():
                    count = result_df[col].count()
                    logs.append(f"Count of {col}: {count:,}")
                elif 'max' in question.lower() or 'highest' in question.lower():
                    max_val = result_df[col].max()
                    logs.append(f"Maximum of {col}: {max_val:,.2f}")
                elif 'min' in question.lower() or 'lowest' in question.lower():
                    min_val = result_df[col].min()
                    logs.append(f"Minimum of {col}: {min_val:,.2f}")
        
        return result_df, logs
    
    def _apply_grouping(self, df: pd.DataFrame, operation_info: Dict[str, Any], question: str) -> Tuple[pd.DataFrame, List[str]]:
        """Apply grouping operations."""
        logs = []
        result_df = df.copy()
        
        for col in operation_info['columns']:
            if col in df.columns:
                if df[col].dtype == 'object':  # Categorical column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        # Create pivot table
                        pivot_df = df.groupby(col)[numeric_cols].agg(['sum', 'mean', 'count']).round(2)
                        pivot_df.columns = [f"{col}_{agg}" for col in numeric_cols for agg in ['sum', 'mean', 'count']]
                        pivot_df = pivot_df.reset_index()
                        result_df = pivot_df
                        logs.append(f"Created pivot table grouped by {col}")
                    else:
                        # Simple count grouping
                        grouped = df.groupby(col).size().reset_index(name='count')
                        result_df = grouped
                        logs.append(f"Grouped by {col} and counted occurrences")
        
        return result_df, logs
    
    def _apply_find_operations(self, df: pd.DataFrame, operation_info: Dict[str, Any], question: str) -> Tuple[pd.DataFrame, List[str]]:
        """Apply find/search operations."""
        logs = []
        result_df = df.copy()
        
        for col in operation_info['columns']:
            if col in df.columns:
                text_values = re.findall(r'"([^"]+)"', question)
                if text_values:
                    search_value = text_values[0]
                    found_rows = result_df[result_df[col].str.contains(search_value, case=False, na=False)]
                    if len(found_rows) > 0:
                        result_df = found_rows
                        logs.append(f"Found {len(found_rows)} rows containing '{search_value}' in {col}")
                    else:
                        logs.append(f"No rows found containing '{search_value}' in {col}")
        
        return result_df, logs
    
    def _apply_comparisons(self, df: pd.DataFrame, operation_info: Dict[str, Any], question: str) -> Tuple[pd.DataFrame, List[str]]:
        """Apply comparison operations."""
        logs = []
        result_df = df.copy()
        
        if len(operation_info['columns']) >= 2:
            col1, col2 = operation_info['columns'][:2]
            if col1 in df.columns and col2 in df.columns:
                if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
                    comparison_df = pd.DataFrame({
                        'Column': [col1, col2],
                        'Sum': [df[col1].sum(), df[col2].sum()],
                        'Average': [df[col1].mean(), df[col2].mean()],
                        'Count': [df[col1].count(), df[col2].count()],
                        'Min': [df[col1].min(), df[col2].min()],
                        'Max': [df[col1].max(), df[col2].max()]
                    })
                    result_df = comparison_df
                    logs.append(f"Created comparison summary between {col1} and {col2}")
        
        return result_df, logs
    
    def _apply_trend_analysis(self, df: pd.DataFrame, operation_info: Dict[str, Any], question: str) -> Tuple[pd.DataFrame, List[str]]:
        """Apply trend analysis operations."""
        logs = []
        result_df = df.copy()
        
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df_copy = df.copy()
                df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                df_copy = df_copy.sort_values(date_col)
                
                # Calculate moving averages
                for num_col in numeric_cols[:3]:
                    df_copy[f'{num_col}_trend'] = df_copy[num_col].rolling(window=3, min_periods=1).mean()
                
                result_df = df_copy
                logs.append(f"Added trend analysis with moving averages")
        
        return result_df, logs
    
    def _apply_outlier_detection(self, df: pd.DataFrame, operation_info: Dict[str, Any], question: str) -> Tuple[pd.DataFrame, List[str]]:
        """Apply outlier detection operations."""
        logs = []
        result_df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outlier_df = df.copy()
            for col in numeric_cols[:3]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    outlier_df[f'{col}_is_outlier'] = df[col].apply(
                        lambda x: 'Yes' if x < lower_bound or x > upper_bound else 'No'
                    )
                    logs.append(f"Detected {len(outliers)} outliers in {col}")
            
            result_df = outlier_df
            logs.append("Added outlier detection columns")
        
        return result_df, logs
    
    def generate_business_insight(self, df: pd.DataFrame, question: str, operation_log: List[str]) -> str:
        """Generate business insights based on the analysis."""
        try:
            insight = f"**Business Insight for: '{question}'**\n\n"
            
            if operation_log:
                insight += "**Operations Performed:**\n"
                for log in operation_log:
                    insight += f"• {log}\n"
                insight += "\n"
            
            # Add data summary
            insight += f"**Data Summary:**\n"
            insight += f"• Total records analyzed: {len(df):,}\n"
            
            if len(df.columns) <= 5:
                insight += f"• Columns: {', '.join(df.columns)}\n"
            else:
                insight += f"• Columns: {len(df.columns)} columns\n"
            
            # Add specific insights based on data types
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                insight += f"• Numeric columns: {', '.join(numeric_cols)}\n"
                for col in numeric_cols[:3]:
                    if col in df.columns:
                        insight += f"  - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, avg={df[col].mean():.2f}\n"
            
            # Add recommendations
            insight += "\n**Recommendations:**\n"
            if len(df) == 0:
                insight += "• No data matches your criteria. Consider relaxing the filters.\n"
            elif len(df) < 10:
                insight += "• Small dataset - results may not be statistically significant.\n"
            else:
                insight += "• Sufficient data for analysis.\n"
            
            return insight
            
        except Exception as e:
            return f"Error generating insight: {str(e)}"
