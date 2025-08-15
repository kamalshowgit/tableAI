import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
import uuid
import logging
from typing import Any, Tuple, Optional
import warnings
import traceback
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="DataFlow AI - Excel Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glass-morphism design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Glass-morphism container */
.glass-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.glass-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

/* Animated gradient background */
.main-background {
    background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    min-height: 100vh;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Smooth animations */
.fade-in {
    animation: fadeIn 0.8s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.slide-in-left {
    animation: slideInLeft 0.6s ease-out;
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}

.slide-in-right {
    animation: slideInRight 0.6s ease-out;
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(30px); }
    to { opacity: 1; transform: translateX(0); }
}

/* Custom buttons */
.glass-button {
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    color: white;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.glass-button:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
}

/* Data transformation steps */
.step-container {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid #667eea;
    transition: all 0.3s ease;
}

.step-container:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: scale(1.02);
}

/* Progress bar */
.progress-container {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    height: 8px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 10px;
    transition: width 0.8s ease;
    animation: progressGlow 2s ease-in-out infinite alternate;
}

@keyframes progressGlow {
    from { box-shadow: 0 0 10px rgba(102, 126, 234, 0.5); }
    to { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }
}

/* Data preview cards */
.data-card {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
}

.data-card:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.5);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_transformed' not in st.session_state:
    st.session_state.df_transformed = None
if 'transformation_steps' not in st.session_state:
    st.session_state.transformation_steps = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0



def show_loading_animation():
    """Show an elegant loading animation."""
    st.markdown("""
    <div class="glass-container" style="text-align: center;">
        <div style="font-size: 2rem; margin-bottom: 1rem;">✨</div>
        <div style="font-size: 1.2rem; font-weight: 500; margin-bottom: 1rem;">
            Processing your data...
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width: 100%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def load_excel_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """Load Excel file with robust error handling."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload CSV, XLSX, or XLS files.")
            return None, ""
        
        # Basic data cleaning
        df = df.dropna(how='all')  # Remove completely empty rows
        df = df.dropna(axis=1, how='all')  # Remove completely empty columns
        
        return df, file_extension
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        logger.error(f"File loading error: {e}")
        return None, ""

def analyze_data_structure(df: pd.DataFrame) -> dict:
    """Analyze data structure and provide insights."""
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
    
    return analysis

def suggest_transformations(analysis: dict) -> list:
    """Suggest data transformations based on analysis."""
    suggestions = []
    
    # Data type conversions
    for col in analysis['quality_issues']:
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
    for col, missing_count in analysis['missing_values'].items():
        if missing_count > 0:
            if col in analysis['numeric_columns']:
                suggestions.append({
                    'type': 'fill_missing_numeric',
                    'column': col,
                    'description': f"Fill missing values in '{col}' with median",
                    'code': f"df['{col}'].fillna(df['{col}'].median(), inplace=True)"
                })
            elif col in analysis['categorical_columns']:
                suggestions.append({
                    'type': 'fill_missing_categorical',
                    'column': col,
                    'description': f"Fill missing values in '{col}' with mode",
                    'code': f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)"
                })
    
    # Remove duplicates
    if analysis['duplicates'] > 0:
        suggestions.append({
            'type': 'remove_duplicates',
            'column': 'all',
            'description': f"Remove {analysis['duplicates']} duplicate rows",
            'code': "df.drop_duplicates(inplace=True)"
        })
    
    return suggestions

def apply_transformation(df: pd.DataFrame, transformation: dict) -> pd.DataFrame:
    """Apply a single transformation to the DataFrame."""
    try:
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
        
        return df
    except Exception as e:
        logger.error(f"Transformation error: {e}")
        return df

def create_data_visualization(df: pd.DataFrame, analysis: dict) -> go.Figure:
    """Create comprehensive data visualizations."""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Data Types Distribution', 'Missing Values', 'Numeric Data Distribution', 'Categorical Data'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    # Data types distribution
    dtypes_counts = df.dtypes.value_counts()
    fig.add_trace(
        go.Pie(labels=dtypes_counts.index, values=dtypes_counts.values, name="Data Types"),
        row=1, col=1
    )
    
    # Missing values
    missing_data = df.isnull().sum()
    fig.add_trace(
        go.Bar(x=missing_data.index, y=missing_data.values, name="Missing Values"),
        row=1, col=2
    )
    
    # Numeric data distribution (first numeric column)
    if analysis['numeric_columns']:
        numeric_col = analysis['numeric_columns'][0]
        fig.add_trace(
            go.Histogram(x=df[numeric_col].dropna(), name=f"{numeric_col} Distribution"),
            row=2, col=1
        )
    
    # Categorical data (first categorical column)
    if analysis['categorical_columns']:
        cat_col = analysis['categorical_columns'][0]
        cat_counts = df[cat_col].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=cat_counts.index, y=cat_counts.values, name=f"{cat_col} Counts"),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Data Analysis Overview",
        title_x=0.5
    )
    
    return fig

def generate_insights(df: pd.DataFrame, analysis: dict) -> list:
    """Generate insights about the data."""
    insights = []
    
    # Basic statistics
    insights.append(f"📊 Dataset contains {analysis['shape'][0]:,} rows and {analysis['shape'][1]} columns")
    insights.append(f"💾 Memory usage: {analysis['memory_usage']:.2f} MB")
    
    # Data quality insights
    total_missing = sum(analysis['missing_values'].values())
    if total_missing > 0:
        missing_percentage = (total_missing / (analysis['shape'][0] * analysis['shape'][1])) * 100
        insights.append(f"⚠️ {total_missing:,} missing values ({missing_percentage:.1f}% of total data)")
    
    if analysis['duplicates'] > 0:
        insights.append(f"🔄 {analysis['duplicates']:,} duplicate rows detected")
    
    # Column type insights
    if analysis['numeric_columns']:
        insights.append(f"🔢 {len(analysis['numeric_columns'])} numeric columns for quantitative analysis")
    
    if analysis['categorical_columns']:
        insights.append(f"📝 {len(analysis['categorical_columns'])} categorical columns for grouping and filtering")
    
    if analysis['date_columns']:
        insights.append(f"📅 {len(analysis['date_columns'])} date columns for time-based analysis")
    
    # Data quality score
    quality_score = 100
    if total_missing > 0:
        quality_score -= min(30, (total_missing / (analysis['shape'][0] * analysis['shape'][1])) * 100)
    if analysis['duplicates'] > 0:
        quality_score -= min(20, (analysis['duplicates'] / analysis['shape'][0]) * 100)
    if analysis['quality_issues']:
        quality_score -= len(analysis['quality_issues']) * 5
    
    quality_score = max(0, quality_score)
    insights.append(f"⭐ Data quality score: {quality_score:.0f}/100")
    
    return insights

def main():
    """Main application function."""
    
    # Header with glass-morphism design
    st.markdown("""
    <div class="main-background">
        <div style="text-align: center; padding: 3rem 0;">
            <h1 style="font-size: 3.5rem; font-weight: 700; color: white; margin-bottom: 1rem; text-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                📊 DataFlow AI
            </h1>
            <p style="font-size: 1.3rem; color: rgba(255,255,255,0.9); margin-bottom: 2rem;">
                Intelligent Excel Analysis with Step-by-Step Data Transformation
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    if not st.session_state.data_loaded:
        # File upload section
        st.markdown("""
            <div class="glass-container fade-in">
                <h2 style="color: white; margin-bottom: 1.5rem;">🚀 Upload Your Data</h2>
                <p style="color: rgba(255,255,255,0.9); margin-bottom: 2rem;">
                    Upload your Excel, CSV, or other tabular data files to begin intelligent analysis
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, XLSX, XLS"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading and analyzing your data..."):
                # Load file
                df, file_type = load_excel_file(uploaded_file)
                
                if df is not None:
                    st.session_state.df_original = df.copy()
                    st.session_state.df_transformed = df.copy()
                    st.session_state.data_loaded = True
                    st.session_state.file_type = file_type
                    st.rerun()
    
    else:
        # Data analysis and transformation section
        df = st.session_state.df_transformed
        
        # Analysis overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
                <div class="glass-container fade-in">
                    <h2 style="color: white; margin-bottom: 1rem;">📈 Data Overview</h2>
                    <div class="data-card">
                        <strong>File Type:</strong> {st.session_state.file_type.upper()}<br>
                        <strong>Dimensions:</strong> {df.shape[0]:,} rows × {df.shape[1]} columns<br>
                        <strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🔄 Upload New File", use_container_width=True):
                st.session_state.data_loaded = False
                st.session_state.df_original = None
                st.session_state.df_transformed = None
                st.session_state.transformation_steps = []
                st.session_state.analysis_complete = False
                st.rerun()
        
        # Data analysis
        if st.button("🔍 Analyze Data Structure", type="primary", use_container_width=True):
            with st.spinner("Analyzing data structure..."):
                analysis = analyze_data_structure(df)
                st.session_state.analysis = analysis
                st.session_state.current_step = 1
                st.rerun()
        
        # Show analysis results
        if 'analysis' in st.session_state:
            analysis = st.session_state.analysis
            
            # Insights
            st.markdown("""
                <div class="glass-container fade-in">
                    <h2 style="color: white; margin-bottom: 1.5rem;">💡 Data Insights</h2>
                </div>
            """, unsafe_allow_html=True)
            
            insights = generate_insights(df, analysis)
            for insight in insights:
                st.markdown(f"""
                    <div class="step-container slide-in-left">
                        <p style="color: white; margin: 0; font-size: 1.1rem;">{insight}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Data visualization
            st.markdown("""
                <div class="glass-container fade-in">
                    <h2 style="color: white; margin-bottom: 1.5rem;">📊 Data Visualization</h2>
                </div>
            """, unsafe_allow_html=True)
            
            fig = create_data_visualization(df, analysis)
            st.plotly_chart(fig, use_container_width=True)
            
            # Transformation suggestions
            st.markdown("""
                <div class="glass-container fade-in">
                    <h2 style="color: white; margin-bottom: 1.5rem;">🛠️ Suggested Transformations</h2>
                </div>
            """, unsafe_allow_html=True)
            
            suggestions = suggest_transformations(analysis)
            
            if suggestions:
                for i, suggestion in enumerate(suggestions):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                            <div class="step-container slide-in-right">
                                <h4 style="color: white; margin-bottom: 0.5rem;">{suggestion['description']}</h4>
                                <code style="background: rgba(0,0,0,0.3); padding: 0.5rem; border-radius: 8px; color: #00ff88;">
                                    {suggestion['code']}
                                </code>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button(f"Apply {i+1}", key=f"apply_{i}"):
                            df = apply_transformation(df, suggestion)
                            st.session_state.df_transformed = df
                            st.session_state.transformation_steps.append(suggestion)
                            st.rerun()
                    
                    with col3:
                        if st.button(f"Preview {i+1}", key=f"preview_{i}"):
                            st.session_state.preview_transformation = suggestion
                            st.rerun()
                
                # Apply all transformations button
                if st.button("🚀 Apply All Transformations", type="primary", use_container_width=True):
                    with st.spinner("Applying all transformations..."):
                        for suggestion in suggestions:
                            df = apply_transformation(df, suggestion)
                        st.session_state.df_transformed = df
                        st.session_state.transformation_steps = suggestions
                        st.session_state.analysis_complete = True
                        st.rerun()
            else:
                st.markdown("""
                    <div class="step-container">
                        <p style="color: white; text-align: center; font-size: 1.1rem;">
                            🎉 Your data is already well-structured! No transformations needed.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Show transformation preview
            if 'preview_transformation' in st.session_state:
                preview = st.session_state.preview_transformation
                st.markdown(f"""
                    <div class="glass-container fade-in">
                        <h3 style="color: white; margin-bottom: 1rem;">🔍 Transformation Preview</h3>
                        <div class="step-container">
                            <h4 style="color: white;">{preview['description']}</h4>
                            <p style="color: rgba(255,255,255,0.8);">Code: <code>{preview['code']}</code></p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Show final transformed data
            if st.session_state.analysis_complete:
                st.markdown("""
                    <div class="glass-container fade-in">
                        <h2 style="color: white; margin-bottom: 1.5rem;">✨ Transformation Complete!</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Download transformed data
                csv = st.session_state.df_transformed.to_csv(index=False)
                st.download_button(
                    label="📥 Download Transformed Data (CSV)",
                    data=csv,
                    file_name=f"transformed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Show final data preview
                st.markdown("""
                    <h3 style="color: white; margin-bottom: 1rem;">📋 Final Data Preview</h3>
                """, unsafe_allow_html=True)
                
                st.dataframe(
                    st.session_state.df_transformed,
                    use_container_width=True,
                    height=400
                )
        
        # Data preview tabs
        if st.session_state.data_loaded:
            st.markdown("""
                <div class="glass-container fade-in">
                    <h2 style="color: white; margin-bottom: 1.5rem;">📋 Data Preview</h2>
                </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["📊 Original Data", "🔄 Transformed Data", "📈 Statistics"])
            
            with tab1:
                st.dataframe(
                    st.session_state.df_original,
                    use_container_width=True,
                    height=400
                )
            
            with tab2:
                st.dataframe(
                    st.session_state.df_transformed,
                    use_container_width=True,
                    height=400
                )
            
            with tab3:
                st.write("### Numeric Columns Statistics")
                numeric_cols = st.session_state.df_transformed.select_dtypes(include=[np.number])
                if not numeric_cols.empty:
                    st.dataframe(numeric_cols.describe(), use_container_width=True)
                
                st.write("### Categorical Columns Summary")
                cat_cols = st.session_state.df_transformed.select_dtypes(include=['object'])
                if not cat_cols.empty:
                    for col in cat_cols.columns:
                        st.write(f"**{col}:** {cat_cols[col].nunique()} unique values")
                        st.dataframe(cat_cols[col].value_counts().head(10), use_container_width=True)

if __name__ == "__main__":
    main()
    