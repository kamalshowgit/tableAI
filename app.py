
# --- Standard Library Imports ---
import os
import traceback
import tempfile
import uuid
import contextlib
import io
import logging
import gc
import psutil
from typing import Optional, Any, Tuple, List
import warnings

# --- Third-Party Imports ---
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Enhanced Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('tableai_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Memory Management and Performance ---
def get_memory_usage():
    """Get current memory usage for monitoring."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def optimize_dataframe(df: pd.DataFrame, max_memory_mb: int = 500) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage and handle large datasets.
    Returns optimized DataFrame and chunk size for processing.
    """
    try:
        # Check memory usage
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        logger.info(f"DataFrame memory usage: {memory_usage:.2f} MB")
        
        # Optimize dtypes for memory efficiency
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to category if low cardinality
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'int64':
                # Downcast integers
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'float64':
                # Downcast floats
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Calculate chunk size for large datasets
        chunk_size = len(df)
        if memory_usage > max_memory_mb:
            chunk_size = int(len(df) * (max_memory_mb / memory_usage))
            chunk_size = max(1000, min(chunk_size, 10000))  # Between 1000 and 10000
            logger.info(f"Large dataset detected. Using chunk size: {chunk_size}")
        
        return df, chunk_size
        
    except Exception as e:
        logger.warning(f"DataFrame optimization failed: {e}")
        return df, len(df)

def process_large_dataset(df: pd.DataFrame, chunk_size: int, operation_func) -> pd.DataFrame:
    """
    Process large datasets in chunks to avoid memory issues.
    """
    if len(df) <= chunk_size:
        return operation_func(df)
    
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        result_chunk = operation_func(chunk)
        results.append(result_chunk)
        
        # Force garbage collection
        del chunk
        gc.collect()
    
    # Combine results
    if results:
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        else:
            return results[0] if len(results) == 1 else results
    
    return df

# --- Enhanced AI Code Generation ---
def generate_robust_ai_prompt(schema: str, sample_rows: str, user_question: str, 
                             data_source_type: str, df_info: dict) -> Tuple[str, str]:
    """
    Generate robust AI prompts that ensure accurate code generation.
    """
    if data_source_type == 'db':
        system_prompt = (
            "You are an expert SQL data analyst with deep knowledge of database optimization and query performance. "
            "Given the following database schema and sample data, generate ONLY SQL code to answer the user's question. "
            "IMPORTANT RULES:\n"
            "1. Return ONLY the SQL query, no explanations or markdown\n"
            "2. Use proper SQL syntax for the detected database type\n"
            "3. For aggregations, always use appropriate GROUP BY clauses\n"
            "4. For sorting, use ORDER BY with LIMIT for top/bottom queries\n"
            "5. Use table aliases for complex queries\n"
            "6. Ensure the query will execute without errors\n"
            f"Database schema:\n{schema}\n"
            f"Sample data: {sample_rows}\n"
            f"User question: {user_question}\n"
            "Return ONLY the SQL code."
        )
        ai_instruction = (
            "Generate ONLY SQL code to answer the question. "
            "Use the table name 'data' and ensure the query is syntactically correct. "
            "No explanations, just the SQL code."
        )
    else:
        # Enhanced Python prompt for file-based data
        system_prompt = (
            "You are an expert Python data analyst specializing in pandas, numpy, and data visualization. "
            "Given the following data schema and sample, generate ONLY Python code to answer the user's question. "
            "IMPORTANT RULES:\n"
            "1. Return ONLY Python code, no explanations or markdown\n"
            "2. Use the DataFrame variable 'df' (already loaded)\n"
            "3. Handle missing values appropriately (df.dropna() or df.fillna())\n"
            "4. Use efficient pandas operations (vectorized operations over loops)\n"
            "5. For aggregations, use groupby() with proper aggregation functions\n"
            "6. For sorting, use sort_values() with ascending/descending parameters\n"
            "7. For visualizations, use matplotlib.pyplot as plt\n"
            "8. Always check data types before operations\n"
            "9. Use df.info() or df.dtypes to understand data structure\n"
            f"Data schema:\n{schema}\n"
            f"Sample data: {sample_rows}\n"
            f"DataFrame info: {df_info}\n"
            f"User question: {user_question}\n"
            "Return ONLY the Python code."
        )
        ai_instruction = (
            "Generate ONLY Python code using pandas to answer the question. "
            "Use the DataFrame 'df' and ensure the code handles the data efficiently. "
            "No explanations, just the Python code."
        )
    
    return system_prompt, ai_instruction

def validate_and_clean_code(code: str, language: str, df: pd.DataFrame) -> str:
    """
    Enhanced code validation and cleaning for better execution.
    """
    if not code or not isinstance(code, str):
        return ""
    
    # Clean the code
    code = clean_ai_code(code)
    
    if language == 'sql':
        # Ensure SQL ends with semicolon
        if not code.strip().endswith(';'):
            code = code.strip() + ';'
        
        # Basic SQL validation
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'ORDER BY', 'GROUP BY', 'HAVING', 'JOIN', 'UNION']
        if not any(keyword.lower() in code.lower() for keyword in sql_keywords):
            code = f"SELECT * FROM data WHERE 1=0; -- Invalid SQL generated"
    
    elif language == 'python':
        # Python-specific enhancements
        code_lines = code.strip().split('\n')
        enhanced_lines = []
        
        for line in code_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Add data validation
            if 'df.' in line and not any(x in line for x in ['df.info()', 'df.dtypes', 'df.head()', 'df.describe()']):
                # Add safety checks for operations
                if 'groupby' in line and 'agg(' not in line and 'sum(' not in line and 'mean(' not in line:
                    line = f"# {line}  # WARNING: groupby without aggregation may not show results"
                elif 'sort_values' in line and 'ascending=' not in line:
                    line = f"{line}, ascending=False"  # Default to descending for better UX
            
            enhanced_lines.append(line)
        
        # Add result assignment if missing
        if enhanced_lines and not any('result =' in line for line in enhanced_lines):
            last_line = enhanced_lines[-1]
            if not last_line.startswith('#') and '=' not in last_line:
                enhanced_lines.append("result = " + last_line)
                enhanced_lines[-2] = "# " + last_line  # Comment out the original line
        
        code = '\n'.join(enhanced_lines)
    
    return code

# --- Enhanced Data Loading ---
def load_dataframe_from_temp_robust(temp_path: str, max_size_mb: int = 100) -> Tuple[pd.DataFrame, int]:
    """
    Robust DataFrame loading with size validation and optimization.
    """
    try:
        # Check file size
        file_size_mb = os.path.getsize(temp_path) / 1024 / 1024
        if file_size_mb > max_size_mb:
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Processing may take longer.")
        
        ext = temp_path.lower()
        df = None
        
        if ext.endswith(".csv"):
            # Use chunking for large CSV files
            if file_size_mb > 50:
                chunks = []
                chunk_size = 10000  # Process 10k rows at a time
                for chunk in pd.read_csv(temp_path, chunksize=chunk_size):
                    chunks.append(chunk)
                    if len(chunks) * chunk_size > 100000:  # Stop at 100k rows for preview
                        break
                df = pd.concat(chunks, ignore_index=True)
                st.info(f"📊 Large CSV file loaded. Showing first {len(df)} rows for analysis.")
            else:
                df = pd.read_csv(temp_path)
                
        elif ext.endswith(".tsv"):
            df = pd.read_csv(temp_path, sep='\t')
        elif ext.endswith(".xlsx") or ext.endswith(".xls"):
            # Excel files can be memory-intensive
            if file_size_mb > 20:
                st.info("📊 Large Excel file detected. Loading with optimization...")
                df = pd.read_excel(temp_path, engine='openpyxl')
            else:
                df = pd.read_excel(temp_path)
        elif ext.endswith(".parquet"):
            df = pd.read_parquet(temp_path)
        elif ext.endswith(".json"):
            df = pd.read_json(temp_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        if df is None or df.empty:
            raise pd.errors.EmptyDataError("File is empty or contains no data")
        
        # Optimize the DataFrame
        df, chunk_size = optimize_dataframe(df)
        
        # Validate data quality
        missing_pct = (df.isnull().sum() / len(df)) * 100
        if missing_pct.max() > 50:
            st.warning(f"⚠️ Column '{missing_pct.idxmax()}' has {missing_pct.max():.1f}% missing values.")
        
        return df, chunk_size
        
    except pd.errors.EmptyDataError:
        logger.error("File is empty or contains no data")
        st.error("❌ File is empty or contains no data. Please check your file.")
        return pd.DataFrame(), 1000
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing file: {e}")
        st.error(f"❌ Error parsing file: {e}. Please check your file format.")
        return pd.DataFrame(), 1000
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        st.error(f"❌ Error loading file: {e}")
        return pd.DataFrame(), 1000

# --- Enhanced Database Connection ---
def connect_database_robust(db_uri: str, db_type: str) -> Tuple[Any, List[str], str]:
    """
    Robust database connection with better error handling and validation.
    """
    try:
        import sqlalchemy
        from sqlalchemy import text
        
        # Validate URI format
        if not db_uri.strip():
            raise ValueError("Empty database URI")
        
        # Fix common SQLite URI issues
        if db_type == "SQLite" and db_uri.startswith("sqlite://"):
            if not db_uri.startswith("sqlite:///"):
                db_uri = db_uri.replace("sqlite://", "sqlite:///")
        
        engine = sqlalchemy.create_engine(db_uri, pool_pre_ping=True)
        
        # Test connection
        with engine.connect() as conn:
            # Get table information based on database type
            if db_type == "SQLite":
                tables_query = text("SELECT name FROM sqlite_master WHERE type='table';")
            elif db_type in ["PostgreSQL", "MariaDB", "MySQL"]:
                tables_query = text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            elif db_type == "SQL Server":
                tables_query = text("SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE';")
            elif db_type == "Oracle":
                tables_query = text("SELECT table_name FROM user_tables")
            elif db_type == "DuckDB":
                tables_query = text("SHOW TABLES")
            else:
                # Try generic information_schema
                tables_query = text("SELECT table_name FROM information_schema.tables")
            
            tables_result = conn.execute(tables_query)
            table_names = [t[0] for t in tables_result.fetchall()]
            
            if not table_names:
                st.warning("⚠️ Connected successfully, but no tables found in the database.")
                return engine, [], db_uri
            
            # Get sample data from first table for schema analysis
            sample_table = table_names[0]
            sample_query = text(f"SELECT * FROM {sample_table} LIMIT 5")
            sample_data = conn.execute(sample_query).fetchall()
            
            return engine, table_names, db_uri
            
    except sqlalchemy.exc.OperationalError as oe:
        logger.error(f"Database connection error: {oe}")
        st.error(f"❌ Database connection failed: {oe}. Please check your connection details.")
        raise
    except sqlalchemy.exc.ArgumentError as ae:
        logger.error(f"Invalid database URI: {ae}")
        st.error(f"❌ Invalid database URI: {ae}. Please check your connection string.")
        raise
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        st.error(f"❌ Unexpected database error: {e}")
        raise

# --- Enhanced Code Execution ---
def execute_code_robust(code: str, language: str, df: pd.DataFrame, 
                        db_engine=None, table_name=None) -> Tuple[Any, str]:
    """
    Robust code execution with better error handling and fallbacks.
    """
    try:
        if language == 'sql':
            return execute_sql_robust(code, db_engine, table_name)
        else:
            return execute_python_robust(code, df)
    except Exception as e:
        logger.error(f"Code execution failed: {e}")
        return None, f"Execution error: {e}"

def execute_sql_robust(sql_code: str, db_engine, table_name: str) -> Tuple[Any, str]:
    """
    Robust SQL execution with validation and error handling.
    """
    try:
        import sqlalchemy
        from sqlalchemy import text
        
        # Clean and validate SQL
        sql_code = sql_code.strip()
        if not sql_code.endswith(';'):
            sql_code += ';'
        
        # Replace 'data' with actual table name
        sql_code = sql_code.replace('data', table_name)
        
        # Execute query
        with db_engine.connect() as conn:
            result_df = pd.read_sql_query(text(sql_code), conn)
            
            if result_df.empty:
                return "No results found for your query.", ""
            elif len(result_df) == 1 and result_df.shape[1] == 1:
                return result_df.iloc[0, 0], ""
            else:
                return result_df, ""
                
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return None, f"SQL execution failed: {e}"

def execute_python_robust(code: str, df: pd.DataFrame) -> Tuple[Any, str]:
    """
    Robust Python execution with memory management and error handling.
    """
    try:
        # Prepare execution environment
        local_vars = {"df": df, "pd": pd, "np": None}
        
        # Try to import numpy if available
        try:
            import numpy as np
            local_vars["np"] = np
        except ImportError:
            pass
        
        # Execute code with safety measures
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")
            
            # Execute the code
            exec(code, {}, local_vars)
            
            # Get result
            result = local_vars.get("result", None)
            
            # Check for matplotlib figures
            if result is None:
                try:
                    import matplotlib.pyplot as plt
                    fig = plt.gcf()
                    if fig and fig.get_axes():
                        result = fig
                except:
                    pass
            
            return result, ""
            
    except Exception as e:
        logger.error(f"Python execution error: {e}")
        return None, f"Python execution failed: {e}"

# --- Core AI Analysis Function ---
def ask_analyst():
    """Main function to handle AI analysis requests with robust error handling."""
    user_question = st.session_state.get("analyst_chat_input", "").strip()
    
    # Validate input
    if not user_question:
        return
    
    # Check for gibberish or random characters
    import re
    gibberish_pattern = r'^[^a-zA-Z0-9]+$|^(?:[a-zA-Z]{1,2}){4,}$|^[a-zA-Z]{6,}$'
    if (len(user_question) < 8 and re.match(r'^[^a-zA-Z0-9]+$', user_question)) or \
       re.match(r'^(?:[a-zA-Z]{1,2}){4,}$', user_question) or \
       (len(user_question) > 6 and re.match(r'^[a-zA-Z]{6,}$', user_question)):
        st.session_state['analyst_last_output'] = (
            "❌ <b>Error:</b> Your question appears to be random characters or a typo.<br>"
            "<b>Tip:</b> Please enter a meaningful question about your data for the analyst to help you."
        )
        st.session_state['analyst_code_loading'] = False
        st.session_state['analyst_output_loading'] = False
        return
    
    st.session_state['analyst_last_question'] = user_question
    st.session_state['analyst_code_loading'] = True
    st.session_state['analyst_output_loading'] = True
    
    try:
        # Get the optimized DataFrame
        if st.session_state.get('data_source_type') == 'file':
            file_path = st.session_state.get('data_file_path')
            if file_path:
                df, chunk_size = load_dataframe_from_temp_robust(file_path, max_size_mb=200)
        elif st.session_state.get('data_source_type') == 'db':
            db_uri = st.session_state.get('data_db_uri')
            table = st.session_state.get('data_db_table')
            if db_uri and table:
                import sqlalchemy
                engine = sqlalchemy.create_engine(db_uri)
                with engine.connect() as conn:
                    sample_query = f"SELECT * FROM {table} LIMIT 10000"  # Limit for analysis
                    df = pd.read_sql_query(sample_query, conn)
                df, chunk_size = optimize_dataframe(df)
        else:
            df = st.session_state.get('df_optimized', st.session_state.get('df_preview'))
            chunk_size = st.session_state.get('df_chunk_size', 1000)
        
        if df is None or df.empty:
            st.session_state['analyst_last_output'] = "No data available for analysis. Please check your file or database connection."
            st.session_state['analyst_code_loading'] = False
            st.session_state['analyst_output_loading'] = False
            return
        
        # Prepare data schema and sample
        schema = "\n".join([f"- {col}: {str(dtype)}" for col, dtype in zip(df.columns, df.dtypes)])
        sample_rows = df.head(3).to_dict(orient="records")
        
        # Generate robust AI prompts
        if st.session_state.get('data_source_type') == 'db':
            system_prompt, ai_instruction = generate_robust_ai_prompt(schema, str(sample_rows), user_question, 'db', df.info())
        else:
            system_prompt, ai_instruction = generate_robust_ai_prompt(schema, str(sample_rows), user_question, 'file', df.info())
        
        try:
            with st.spinner("AI is generating code and answer..."):
                llm = get_llm()
                embed_model = get_embed_model()
                doc_text = (
                    f"Data schema:\n{schema}\nSample rows: {sample_rows}\n"
                    f"User question: {user_question}\n"
                    f"{ai_instruction}"
                )
                
                # Create temporary file for document
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as f:
                    f.write(doc_text)
                    doc_path = f.name
                
                try:
                    documents = SimpleDirectoryReader(input_files=[doc_path]).load_data()
                    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
                    query_engine = index.as_query_engine(llm=llm)
                    
                    # AI query
                    ai_response = query_engine.query(user_question)
                        
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(doc_path)
                    except:
                        pass
            
            # Validate and extract code from AI response
            code, code_lang = validate_ai_response(ai_response)
            
            st.session_state['analyst_last_code_language'] = code_lang
            st.session_state['analyst_last_code'] = code or str(ai_response)
            st.session_state['analyst_code_loading'] = False
            logger.info(f"AI generated code for question: {user_question}")
            
            # Check if we have valid code
            if not code or code.strip() == "":
                # Try fallback for common queries
                fallback_result, fallback_msg = handle_common_queries_fallback(user_question, df)
                if fallback_result is not None:
                    st.session_state['analyst_last_output'] = f"🔄 {fallback_msg}<br><br>{fallback_result}"
                    st.session_state['analyst_output_loading'] = False
                    return
                else:
                    st.session_state['analyst_last_output'] = "❌ <b>Error:</b> No valid code generated by AI.<br><b>Tip:</b> Please try rephrasing your question."
                    st.session_state['analyst_output_loading'] = False
                    return
            
            # Execute code using robust execution
            if code_lang == 'sql':
                # Use robust SQL execution
                db_uri = st.session_state.get('data_db_uri')
                table = st.session_state.get('data_db_table')
                if db_uri and table:
                    import sqlalchemy
                    engine = sqlalchemy.create_engine(db_uri)
                    result, error = execute_sql_robust(code, engine, table)
                    if error:
                        st.session_state['analyst_last_output'] = f"❌ <b>Error:</b> {error}<br><b>Query:</b> <pre>{code}</pre>"
                    else:
                        st.session_state['analyst_last_output'] = result
                    st.session_state['analyst_output_loading'] = False
                    return
                else:
                    st.session_state['analyst_last_output'] = "Database connection info missing. Please reconnect and try again."
                    st.session_state['analyst_output_loading'] = False
                    return
            else:
                # Use robust Python execution
                result, error = execute_python_robust(code, df)
                if error:
                    # Try fallback for common queries
                    fallback_result, fallback_msg = handle_common_queries_fallback(user_question, df)
                    if fallback_result is not None:
                        st.session_state['analyst_last_output'] = f"🔄 {fallback_msg}<br><br>{fallback_result}"
                    else:
                        st.session_state['analyst_last_output'] = f"❌ <b>Error:</b> {error}<br><b>Code:</b> <pre>{code}</pre>"
                else:
                    st.session_state['analyst_last_output'] = result
                st.session_state['analyst_output_loading'] = False
                return
                
        except Exception as e:
            logger.error(f"Error in AI processing: {e}\n{traceback.format_exc()}")
            # Try fallback for common queries
            fallback_result, fallback_msg = handle_common_queries_fallback(user_question, df)
            if fallback_result is not None:
                st.session_state['analyst_last_output'] = f"🔄 {fallback_msg}<br><br>{fallback_result}"
            else:
                st.session_state['analyst_last_output'] = f"❌ <b>Error:</b> {e}<br><b>Tip:</b> Please try again or rephrase your question."
            st.session_state['analyst_code_loading'] = False
            st.session_state['analyst_output_loading'] = False
            
    except Exception as e:
        logger.error(f"Error in ask_analyst: {e}\n{traceback.format_exc()}")
        st.session_state['analyst_last_code'] = None
        st.session_state['analyst_last_output'] = f"❌ <b>Error:</b> {e}<br><b>Tip:</b> Please try again or rephrase your question."
        st.session_state['analyst_code_loading'] = False
        st.session_state['analyst_output_loading'] = False

# --- Model/Embedding Caching ---
@st.cache_resource
def get_llm():
    """Get cached LLM instance for better performance."""
    return Ollama(model=OLLAMA_MODEL, request_timeout=120)

@st.cache_resource 
def get_embed_model():
    """Get cached embedding model instance for better performance."""
    return OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL)

def cleanup_session_state():
    """Clean up session state variables."""
    keys_to_remove = [
        'df_preview', 'df_columns', 'df_source', 'data_file_path', 'data_source_type',
        'data_db_uri', 'data_db_table', 'show_analyst', 'analyst_last_question',
        'analyst_last_code', 'analyst_last_output', 'analyst_code_loading', 'analyst_output_loading',
        'analyst_last_code_language', 'df_chunk_size', 'df_optimized'
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

# LlamaIndex/Ollama imports
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding

# --- Configurable Model Names (from .env or fallback) ---
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')
OLLAMA_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'mistral')

# --- Startup: Check for required packages ---
missing_packages = []
try:
    import matplotlib.pyplot as plt
except ImportError:
    missing_packages.append('matplotlib')
try:
    import plotly.graph_objs as go
    plotly_available = True
except ImportError:
    missing_packages.append('plotly')
    plotly_available = False
try:
    import sqlalchemy
except ImportError:
    missing_packages.append('sqlalchemy')

# --- Production Ready UI ---
st.set_page_config(page_title="TableAI - Secure Data Analyst", layout="wide")
if missing_packages:
    st.warning(f"The following required packages are missing: {', '.join(missing_packages)}. Please install them using pip for full functionality.")

# --- Placeholder: Add Streamlit resource/data caching for expensive operations ---
# Example: @st.cache_resource or @st.cache_data for model loading, embedding, etc.

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"]  {
    font-family: 'Inter', 'JetBrains Mono', system-ui, Arial, sans-serif !important;
    font-size: 1.09rem;
    color: #e3f2fd;
    background: #181c24;
    letter-spacing: 0.01em;
}
.stApp {
    background: linear-gradient(120deg, #181c24 0%, #232a36 100%);
    min-height: 100vh;
}
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2.5rem;
    animation: fadeInUp 1.2s cubic-bezier(.23,1.01,.32,1) both;
}
.stButton > button {
    background: linear-gradient(90deg, #1976d2 60%, #42a5f5 100%);
    color: #fff;
    border: none;
    border-radius: 0.5em;
    font-weight: 700;
    font-size: 1.09rem;
    padding: 0.7em 1.7em;
    box-shadow: 0 2px 12px #0005;
    transition: background 0.3s, transform 0.2s, box-shadow 0.2s;
    outline: none;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1565c0 60%, #1976d2 100%);
    color: #fff !important;
    transform: translateY(-3px) scale(1.06);
    box-shadow: 0 4px 18px #0007;
}
.stTextInput > div > input {
    border-radius: 0.5em;
    border: 1.5px solid #90caf9;
    font-size: 1.09rem;
    padding: 0.6em 1.1em;
    background: #232a36;
    color: #e3f2fd;
    transition: border 0.2s, box-shadow 0.2s;
    font-family: 'JetBrains Mono', 'Inter', monospace;
}
.stTextInput > div > input:focus {
    border: 2px solid #1976d2;
    background: #232a36;
    box-shadow: 0 0 0 2px #1976d2aa;
}
.stDataFrame, .stTable {
    font-size: 1.06rem;
    border-radius: 0.5em;
    background: #232a36;
    color: #e3f2fd;
    box-shadow: 0 2px 8px #0004;
}
.stCodeBlock, .stCode, pre {
    font-family: 'JetBrains Mono', 'Fira Mono', 'Menlo', monospace !important;
    font-size: 1.03rem !important;
    background: #181c24 !important;
    color: #90caf9 !important;
    border-radius: 0.5em !important;
    padding: 1em !important;
    margin-bottom: 1em !important;
    box-shadow: 0 2px 8px #0004;
}
.output-card, .analyst-output {
    background: #232a36;
    border-radius: 0.7em;
    box-shadow: 0 2px 18px #0008;
    padding: 1.5em 1.5em 1em 1.5em;
    margin-bottom: 1.5em;
    animation: fadeInUp 1.1s cubic-bezier(.23,1.01,.32,1) both;
    transition: box-shadow 0.2s;
}
.output-card:hover, .analyst-output:hover {
    box-shadow: 0 6px 32px #000b;
}
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(40px); }
    100% { opacity: 1; transform: none; }
}
.analyst-chat {
    animation: fadeInChat 1.2s cubic-bezier(.23,1.01,.32,1) both;
    background: #232a36;
    border-radius: 0.7em;
    padding: 1.2em 1.5em 1em 1.5em;
    margin-bottom: 1.5em;
    box-shadow: 0 2px 18px #0008;
    border-left: 4px solid #1976d2;
    position: relative;
}
@keyframes fadeInChat {
    0% { opacity: 0; transform: translateX(-40px); }
    100% { opacity: 1; transform: none; }
}
.analyst-bubble {
    display: inline-block;
    background: linear-gradient(90deg, #1976d2 60%, #42a5f5 100%);
    color: #fff;
    border-radius: 1.2em 1.2em 1.2em 0.2em;
    padding: 0.7em 1.3em;
    margin-bottom: 0.7em;
    font-size: 1.08rem;
    font-family: 'Inter', 'JetBrains Mono', sans-serif;
    box-shadow: 0 2px 8px #0004;
    animation: fadeInBubble 0.7s cubic-bezier(.23,1.01,.32,1) both;
}
@keyframes fadeInBubble {
    0% { opacity: 0; transform: scale(0.8); }
    100% { opacity: 1; transform: none; }
}
.analyst-bubble.user {
    background: linear-gradient(90deg, #232a36 60%, #1976d2 100%);
    color: #90caf9;
    border-radius: 1.2em 1.2em 0.2em 1.2em;
    margin-left: 1.5em;
}
.stSidebar {
    background: #181c24 !important;
}
.stSidebar .stTitle, .stSidebar .stMarkdown {
    color: #e3f2fd !important;
}
.stSidebar .stInfo {
    background: #232a36 !important;
    color: #e3f2fd !important;
}
.home-btn {
    position: fixed;
    top: 1.1rem;
    left: 1.1rem;
    z-index: 9999;
}
/* Terminal style for output window */
.terminal-output {
    background: #181c24 !important;
    color: #00ff5f !important;
    font-family: 'JetBrains Mono', 'Fira Mono', 'Menlo', monospace !important;
    font-size: 1.08rem !important;
    border-radius: 0.5em !important;
    border: 2px solid #222 !important;
    box-shadow: 0 2px 12px #0007;
    padding: 1.2em 1.3em 1.1em 1.3em !important;
    margin-bottom: 1.2em !important;
    margin-top: 0.5em !important;
    min-height: 120px;
    max-height: 400px;
    overflow-x: auto;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# --- Home Button (top left, persistent) ---
home_clicked = False
with st.container():
    col_home, _ = st.columns([0.1, 0.9])
    with col_home:
        if st.button("Home", key="home_btn", help="Return to main page", use_container_width=True):
            home_clicked = True
if home_clicked:
    st.session_state.clear()
    st.rerun()
if not ('show_analyst' in st.session_state and st.session_state['show_analyst']):
    with st.sidebar:
        st.title("TableAI Help & Info")
        st.markdown("""
        **How to use:**
        1. Upload a tabular file or connect to a database.
        2. Preview your data and click 'Meet an analyst'.
        3. Ask questions about your data. The AI will generate code and show the result.
        
        **Security:**
        - All code runs locally. No data leaves your machine.
        - Powered by Ollama (Mistral) and LlamaIndex.
        
        **Supported file types:** CSV, XLSX, XLS, TSV, Parquet, JSON
        **Supported databases:** SQLite, PostgreSQL, MySQL, MariaDB, SQL Server, Oracle, DuckDB
        """)
        st.info("For best results, use clean tabular data. If you see errors, check your file format and column names.")


def clean_ai_code(code: str) -> str:
    """
    Post-process AI-generated code to fix common mistakes and improve safety.
    - Fix common matplotlib argument errors (labels -> label)
    - Remove dangerous code patterns (already handled, but double check)
    - Strip trailing whitespace
    """
    import re
    
    if not code or not isinstance(code, str):
        return ""
    
    # Fix common matplotlib argument typo: labels= -> label=
    code = re.sub(r'labels\s*=', 'label=', code)
    
    # Remove any double semicolons
    code = code.replace(';;', ';')
    
    # Remove dangerous code patterns (defensive)
    forbidden = [
        r'os\.', r'subprocess', r'open\(', r'exec\(', r'eval\(', r'import sys', r'import os',
        r'import shutil', r'import socket', r'import requests', r'import urllib', r'del ',
        r'remove\(', r'rmdir\(', r'system\(', r'exit\(', r'kill\(', r'__import__', r'globals\(',
        r'locals\(', r'compile\(', r'input\(', r'raw_input\(', r'file\(', r'reload\('
    ]
    
    for pattern in forbidden:
        code = re.sub(pattern, '# BLOCKED', code)
    
    # Remove any import statements that could be dangerous
    dangerous_imports = [
        r'import\s+os\s*$', r'import\s+sys\s*$', r'import\s+subprocess\s*$',
        r'from\s+os\s+import\s*', r'from\s+sys\s+import\s*', r'from\s+subprocess\s+import\s*'
    ]
    
    for pattern in dangerous_imports:
        code = re.sub(pattern, '# BLOCKED IMPORT', code, flags=re.MULTILINE)
    
    # Log code cleaning for audit
    logger.debug("AI code cleaned for safety.")
    return code.strip()

def save_and_get_temp_path(uploaded_file: Any) -> str:
    """Save uploaded file to temporary path with error handling."""
    try:
    ext = os.path.splitext(uploaded_file.name)[-1]
    temp_path = os.path.join(tempfile.gettempdir(), f"st_{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"Saved uploaded file to temp path: {temp_path}")
    return temp_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        st.error(f"❌ Error saving uploaded file: {e}")
        raise

def validate_ai_response(ai_response: Any) -> tuple[Optional[str], str]:
    """
    Validate and extract code from AI response.
    Returns (code, language) tuple.
    """
    if not ai_response:
        return None, 'python'
    
    response_str = str(ai_response)
    
    # Clean the response - remove any markdown formatting
    response_str = response_str.replace('```python', '').replace('```sql', '').replace('```', '').strip()
    
    # Detect code block or plain SQL
    if "```python" in str(ai_response):
        code = str(ai_response).split("```python")[1].split("```", 1)[0].strip()
        return code, 'python'
    elif "```sql" in str(ai_response):
        code = str(ai_response).split("```sql")[1].split("```", 1)[0].strip()
        return code, 'sql'
        else:
        # Try to detect if the response is a plain SQL query
        import re
        sql_pattern = r"^\s*SELECT .* FROM .*;?\s*$"
        if re.match(sql_pattern, response_str, re.IGNORECASE):
            return response_str.strip(), 'sql'
        else:
            # Check if it looks like SQL (contains SQL keywords)
            sql_keywords = ['SELECT', 'FROM', 'WHERE', 'ORDER BY', 'GROUP BY', 'HAVING', 'JOIN', 'UNION']
            if any(keyword.lower() in response_str.lower() for keyword in sql_keywords):
                return response_str.strip(), 'sql'
            else:
                return response_str.strip(), 'python'

# --- Output Rendering Helper ---
def render_analyst_output(ai_output: Any) -> None:
    """Render AI output in a clean, consistent way."""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    if ai_output is None:
        st.markdown("<div class='terminal-output'><span style='font-size:0.98rem; color:#00ff5f; font-style:italic;'>-Your output will be generated here.-</span></div>", unsafe_allow_html=True)
        return
    
    # Handle DataFrame output
    if isinstance(ai_output, pd.DataFrame):
        max_rows = 1000
        if len(ai_output) > max_rows:
            st.warning(f"Output DataFrame has {len(ai_output)} rows. Showing only the first {max_rows} rows.")
            st.dataframe(ai_output.head(max_rows), use_container_width=True)
        else:
            st.dataframe(ai_output, use_container_width=True)
        return
    
    # Handle string output
    if isinstance(ai_output, str):
        if 'File read detected:' in ai_output:
        st.markdown(f"<div class='terminal-output'>{ai_output}</div>", unsafe_allow_html=True)
        else:
        st.markdown(f"<div class='terminal-output'>{ai_output}</div>", unsafe_allow_html=True)
        return
    
    # Handle numeric output
    if isinstance(ai_output, (int, float)):
        st.markdown(f"<div class='terminal-output'>{ai_output}</div>", unsafe_allow_html=True)
        return
    
    # Handle matplotlib figures
    if hasattr(ai_output, 'figure') or hasattr(ai_output, 'show'):
        try:
            st.pyplot(ai_output)
        except Exception as e:
            logger.warning(f"Failed to render matplotlib figure: {e}")
            if plotly_available:
                try:
                    st.plotly_chart(ai_output)
                except Exception as e2:
                    logger.warning(f"Failed to render plotly chart: {e2}")
                    st.markdown("<div class='terminal-output'>[Graph output could not be rendered]</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='terminal-output'>[Plotly not installed. Graph output could not be rendered]</div>", unsafe_allow_html=True)
        return
    
    # Handle plot dictionary
    if isinstance(ai_output, dict) and ai_output.get('type') == 'plot':
        fig = ai_output.get('figure')
        if fig is not None:
            try:
                st.pyplot(fig)
            except Exception as e:
                logger.warning(f"Failed to render plot figure: {e}")
                if plotly_available:
                    try:
                        st.plotly_chart(fig)
                    except Exception as e2:
                        logger.warning(f"Failed to render plotly plot: {e2}")
                        st.markdown("<div class='terminal-output'>[Graph output could not be rendered]</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='terminal-output'>[Plotly not installed. Graph output could not be rendered]</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='terminal-output'>[No figure found in output]</div>", unsafe_allow_html=True)
        return
    
    # Handle other objects with to_dict method
    if hasattr(ai_output, 'to_dict') and not isinstance(ai_output, str):
        st.write(ai_output)
        return
    
    # Default case - convert to string
        st.markdown(f"<div class='terminal-output'>{str(ai_output)}</div>", unsafe_allow_html=True)

# --- Enhanced Data Preview with Memory Management ---
def render_data_preview_robust(df: pd.DataFrame, chunk_size: int):
    """
    Render data preview with memory-efficient handling for large datasets.
    """
    try:
        # Show basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory", f"{memory_mb:.1f} MB")
        
        # Show data types and missing values
        st.subheader("Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.write("**Sample Data:**")
            # Use chunking for large datasets
            if len(df) > chunk_size:
                preview_df = df.head(chunk_size // 2)
                st.info(f"📊 Large dataset detected. Showing first {len(preview_df)} rows.")
            else:
                preview_df = df.head(10)
            st.dataframe(preview_df, use_container_width=True)
        
        # Show statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.subheader("Numeric Statistics")
            stats_df = df[numeric_cols].describe()
            st.dataframe(stats_df, use_container_width=True)
        
        # Show value counts for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.subheader("Categorical Overview")
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                if df[col].nunique() <= 20:  # Only show if not too many unique values
                    st.write(f"**{col}:**")
                    value_counts = df[col].value_counts().head(10)
                    st.bar_chart(value_counts)
        
    except Exception as e:
        logger.error(f"Error rendering data preview: {e}")
        st.error(f"❌ Error rendering data preview: {e}")

# --- Enhanced Error Handling and Fallbacks ---
def handle_common_queries_fallback(user_question: str, df: pd.DataFrame) -> Optional[Any]:
    """
    Handle common query patterns with intelligent fallbacks.
    """
    question_lower = user_question.lower()
    
    try:
        # Salary/Income related queries
        if any(word in question_lower for word in ['salary', 'income', 'pay', 'wage']):
            if any(word in question_lower for word in ['highest', 'maximum', 'top', 'best']):
                # Find salary column
                salary_cols = [col for col in df.columns if 'salary' in col.lower() or 'income' in col.lower() or 'pay' in col.lower()]
                if salary_cols:
                    salary_col = salary_cols[0]
                    name_cols = [col for col in df.columns if 'name' in col.lower() or 'employee' in col.lower() or 'person' in col.lower()]
                    if name_cols:
                        name_col = name_cols[0]
                        result = df[[name_col, salary_col]].nlargest(1, salary_col)
                        return result, f"Fallback: Found highest {salary_col} using {name_col}"
        
        # Count queries
        elif any(word in question_lower for word in ['count', 'how many', 'total number']):
            if 'rows' in question_lower or 'records' in question_lower:
                return len(df), "Fallback: Total number of rows"
            else:
                # Try to count non-null values in a specific column
                for col in df.columns:
                    if col.lower() in question_lower:
                        count = df[col].count()
                        return count, f"Fallback: Count of non-null values in {col}"
        
        # Average queries
        elif any(word in question_lower for word in ['average', 'mean', 'avg']):
            for col in df.columns:
                if col.lower() in question_lower and df[col].dtype in ['int64', 'float64']:
                    avg = df[col].mean()
                    return avg, f"Fallback: Average of {col}"
        
        # Sum queries
        elif any(word in question_lower for word in ['sum', 'total', 'add up']):
            for col in df.columns:
                if col.lower() in question_lower and df[col].dtype in ['int64', 'float64']:
                    total = df[col].sum()
                    return total, f"Fallback: Sum of {col}"
        
        # Group by queries
        elif any(word in question_lower for word in ['group', 'by', 'category']):
            # Try to identify grouping column and aggregation
            for col in df.columns:
                if col.lower() in question_lower:
                    if df[col].dtype in ['object', 'category']:
                        # Group by categorical column
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            result = df.groupby(col)[numeric_cols[0]].agg(['count', 'mean', 'sum']).round(2)
                            return result, f"Fallback: Grouped by {col}, aggregated {numeric_cols[0]}"
        
        return None, None
        
    except Exception as e:
        logger.warning(f"Fallback query failed: {e}")
        return None, None

# --- Main Application Logic with Enhanced Robustness ---
if 'df_preview' not in st.session_state:
    st.title("TableAI Loader")
    st.write("Upload any tabular file (csv, xlsx, xls, tsv, parquet, json) or connect to a SQL database.")
    tab1, tab2 = st.tabs(["Upload File", "Connect SQL DB"])

    with tab1:
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "tsv", "parquet", "json"])
        if uploaded_file:
            try:
                # Validate file size (max 200MB for robust handling)
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB
                    st.error("❌ File too large. Maximum file size is 200MB.")
                    st.stop()
                
                temp_path = save_and_get_temp_path(uploaded_file)
                df, chunk_size = load_dataframe_from_temp_robust(temp_path, max_size_mb=200)
                
                if df.empty:
                    st.error("❌ No data loaded. Please check your file format.")
                elif len(df.columns) == 0:
                    st.error("❌ File contains no columns. Please check your file format.")
                elif len(df) == 0:
                    st.error("❌ File contains no rows. Please check your file.")
                else:
                    st.success(f"✅ Successfully loaded {uploaded_file.name} with {len(df):,} rows and {len(df.columns)} columns")
                    st.session_state['df_preview'] = df.head()
                    st.session_state['df_columns'] = list(df.columns)
                    st.session_state['df_source'] = f"File: {uploaded_file.name}"
                    st.session_state['data_file_path'] = temp_path
                    st.session_state['data_source_type'] = 'file'
                    st.session_state['df_chunk_size'] = chunk_size
                    st.session_state['df_optimized'] = df
                    st.rerun()
            except Exception as e:
                logger.error(f"Error processing uploaded file: {e}")
                st.error(f"❌ Error processing file: {e}")

    with tab2:
        st.write("Connect to a tabular database (SQLite, PostgreSQL, MySQL, MariaDB, SQL Server, Oracle, DuckDB, etc.)")
        db_type = st.selectbox(
            "Database Type",
            [
                "SQLite",
                "PostgreSQL",
                "MySQL",
                "MariaDB",
                "SQL Server",
                "Oracle",
                "DuckDB",
                "Other"
            ]
        )
        default_uris = {
            "SQLite": "sqlite:///your.db",
            "PostgreSQL": "postgresql://user:password@localhost:5432/dbname",
            "MySQL": "mysql+pymysql://user:password@localhost:3306/dbname",
            "MariaDB": "mariadb+pymysql://user:password@localhost:3306/dbname",
            "SQL Server": "mssql+pyodbc://user:password@localhost:1433/dbname?driver=ODBC+Driver+17+for+SQL+Server",
            "Oracle": "oracle+cx_oracle://user:password@localhost:1521/dbname",
            "DuckDB": "duckdb:///your.duckdb",
            "Other": ""
        }
        db_uri = st.text_input("Database URI/Path", default_uris[db_type])
        connect_btn = st.button("Connect to Database")
        if connect_btn and db_uri:
            try:
                engine, table_names, db_uri = connect_database_robust(db_uri, db_type)
                
                if not table_names:
                    st.warning("⚠️ Connected successfully, but no tables found in the database.")
                    st.stop()
                
                st.success(f"✅ Connected successfully! Found {len(table_names)} table(s): {', '.join(table_names)}")
                    logger.info(f"Connected to DB: {db_type}, Tables: {table_names}")
                
                        table = st.selectbox("Select Table", table_names)
                        if table:
                    try:
                        # Load table data efficiently
                        import sqlalchemy
                        with engine.connect() as conn:
                            # Get sample data for preview
                            sample_query = f"SELECT * FROM {table} LIMIT 1000"
                            df = pd.read_sql_query(sample_query, conn)
                        
                            if df.empty:
                            st.warning("⚠️ Table is empty. Please select a different table.")
                            else:
                            # Optimize DataFrame
                            df, chunk_size = optimize_dataframe(df)
                            
                                st.session_state['df_preview'] = df.head()
                                st.session_state['df_columns'] = list(df.columns)
                                st.session_state['df_source'] = f"DB Table: {table}"
                                st.session_state['data_db_uri'] = db_uri
                                st.session_state['data_db_table'] = table
                                st.session_state['data_source_type'] = 'db'
                            st.session_state['df_chunk_size'] = chunk_size
                            st.session_state['df_optimized'] = df
                                st.rerun()
                    except Exception as table_error:
                        logger.error(f"Error reading table {table}: {table_error}")
                        st.error(f"❌ Error reading table '{table}': {table_error}")
                        
            except Exception as e:
                logger.error(f"Database connection error: {e}")
                st.error(f"❌ Database connection failed: {e}")

# --- Data Preview Section ---
if 'df_preview' in st.session_state:
    st.title("TableAI Data Preview")
    
    # Get the optimized DataFrame
                if st.session_state.get('data_source_type') == 'file':
                    file_path = st.session_state.get('data_file_path')
                    if file_path:
            df, chunk_size = load_dataframe_from_temp_robust(file_path, max_size_mb=200)
                elif st.session_state.get('data_source_type') == 'db':
                    db_uri = st.session_state.get('data_db_uri')
                    table = st.session_state.get('data_db_table')
                    if db_uri and table:
                        import sqlalchemy
                        engine = sqlalchemy.create_engine(db_uri)
            with engine.connect() as conn:
                sample_query = f"SELECT * FROM {table} LIMIT 10000"  # Limit for preview
                df = pd.read_sql_query(sample_query, conn)
            df, chunk_size = optimize_dataframe(df)
                else:
                    df = st.session_state.get('df_preview')
        chunk_size = st.session_state.get('df_chunk_size', 1000)
    
    if df is not None and not df.empty:
        st.write(f"**Source:** {st.session_state.get('df_source', 'Unknown')}")
        
        # Render enhanced data preview
        render_data_preview_robust(df, chunk_size)
        
        # Meet Analyst button
        if st.button("Meet an Analyst", type="primary", use_container_width=True):
            st.session_state['show_analyst'] = True
            st.rerun()
        
        # Back button
        if st.button("Back", use_container_width=True):
            cleanup_session_state()
            st.rerun()

# --- Analyst Section ---
if st.session_state.get('show_analyst'):
    st.title("TableAI Analyst Chatbot")
    st.write("Ask questions about your data and get AI-generated code and answers.")
    
    # Back button
    if st.button("← Back to Data", use_container_width=True):
        cleanup_session_state()
        st.rerun()
    
    # Get the optimized DataFrame
    if st.session_state.get('data_source_type') == 'file':
        file_path = st.session_state.get('data_file_path')
        if file_path:
            df, chunk_size = load_dataframe_from_temp_robust(file_path, max_size_mb=200)
    elif st.session_state.get('data_source_type') == 'db':
                        db_uri = st.session_state.get('data_db_uri')
                        table = st.session_state.get('data_db_table')
                            if db_uri and table:
                                import sqlalchemy
                                engine = sqlalchemy.create_engine(db_uri)
                                with engine.connect() as conn:
                sample_query = f"SELECT * FROM {table} LIMIT 10000"  # Limit for analysis
                df = pd.read_sql_query(sample_query, conn)
            df, chunk_size = optimize_dataframe(df)
                                else:
        df = st.session_state.get('df_optimized', st.session_state.get('df_preview'))
        chunk_size = st.session_state.get('df_chunk_size', 1000)
    
    if df is not None and not df.empty:
        # Show data summary
        st.info(f"📊 Analyzing data with {len(df):,} rows and {len(df.columns)} columns")
        
        # Analyst Chatbot Form
        with st.form(key="analyst_chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question about your data", key="analyst_chat_input", 
                                        placeholder="e.g., Show the average of column X grouped by column Y")
            code_loading = st.session_state.get('analyst_code_loading', False)
            output_loading = st.session_state.get('analyst_output_loading', False)
            submitted = st.form_submit_button("Submit", disabled=code_loading or output_loading)
            
            # Show progress indicators
            if code_loading or output_loading:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simple progress indicator
                progress_bar.progress(50)
                status_text.text("🤖 AI is thinking and generating code...")
                
                # Clear after a short delay
                import time
                time.sleep(0.1)
                progress_bar.empty()
                status_text.empty()
            
            if submitted:
                ask_analyst()

        # Placeholder for user's question and previous response
        if st.session_state.get("analyst_last_question"):
            st.markdown(f"**Last question:** {st.session_state['analyst_last_question']}")

        # --- Code and Output in two columns ---
        col1, col2 = st.columns([1, 1])
        
        # --- Left: Code ---
        with col1:
            st.markdown("<span style='font-size:1.13rem; font-weight:700;'>AI Generated Code</span>", unsafe_allow_html=True)
            user_q = st.session_state.get('analyst_last_question', '')
            code_loading = st.session_state.get('analyst_code_loading', False)
            ai_code = st.session_state.get('analyst_last_code', None)
            code_lang = st.session_state.get('analyst_last_code_language', 'python')
            if code_loading:
                # Context-aware filler for code
                if user_q:
                    code_filler = f"# AI is writing code to answer: '{user_q}'\n# Please wait..."
                else:
                    code_filler = "# AI is writing code...\n# Please wait..."
                st.markdown(f"""
<div style='font-family: JetBrains Mono, monospace; font-size:1.03rem; color:#90caf9; background:#181c24; border-radius:0.5em; border:2px solid #222; padding:1em; min-height:120px; max-height:400px; margin-bottom:1em; box-shadow:0 2px 8px #0004;'>
<span style='color:#00ff5f;'>AI is writing code for: <span style="color:#fff;">{user_q}</span><span class='blinking-cursor'>|</span></span>
</div>
<style>
@keyframes blink {{
  0% {{ opacity: 1; }}
  50% {{ opacity: 0; }}
  100% {{ opacity: 1; }}
}}
.blinking-cursor {{
  animation: blink 1s step-end infinite;
}}
</style>
""", unsafe_allow_html=True)
                st.code(code_filler, language=code_lang)
            elif not ai_code:
                st.code('***Your AI-generated code will be displayed here.***', language=code_lang)
            else:
                st.code(ai_code, language=code_lang)

        # --- Right: Output ---
        with col2:
            st.markdown("<span style='font-size:1.13rem; font-weight:700;'>Output</span>", unsafe_allow_html=True)
            user_q = st.session_state.get('analyst_last_question', '')
            output_loading = st.session_state.get('analyst_output_loading', False)
            ai_output = st.session_state.get('analyst_last_output', None)
            import matplotlib.pyplot as plt
            if output_loading:
                # Context-aware filler for output
                if user_q:
                    output_filler = f"AI is preparing output for: '{user_q}'\nPlease wait while the answer is generated..."
                else:
                    output_filler = "AI is preparing output...\nPlease wait..."
                st.markdown(f"""
<div class='terminal-output'><span style='font-size:0.98rem; color:#00ff5f;'>AI is preparing output for: <span style='color:#fff'>{user_q}</span><span class='blinking-cursor'>|</span></span></div>
<style>
@keyframes blink2 {{
  0% {{ opacity: 1; }}
  50% {{ opacity: 0; }}
  100% {{ opacity: 1; }}
}}
.blinking-cursor {{
  animation: blink2 1s step-end infinite;
}}
</style>
""", unsafe_allow_html=True)
            else:
                render_analyst_output(ai_output)


