
# --- Standard Library Imports ---
import os
import traceback
import tempfile
import uuid
import contextlib
import io
import logging
from typing import Optional, Any

# --- Third-Party Imports ---
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('tableai_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        'analyst_last_code_language'
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

def load_dataframe_from_temp(temp_path: str) -> pd.DataFrame:
    """Load DataFrame from temporary file with robust error handling."""
    ext = temp_path.lower()
    try:
        if ext.endswith(".csv"):
            return pd.read_csv(temp_path)
        elif ext.endswith(".tsv"):
            return pd.read_csv(temp_path, sep='\t')
        elif ext.endswith(".xlsx") or ext.endswith(".xls"):
            return pd.read_excel(temp_path)
        elif ext.endswith(".parquet"):
            return pd.read_parquet(temp_path)
        elif ext.endswith(".json"):
            return pd.read_json(temp_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except pd.errors.EmptyDataError:
        logger.error("File is empty or contains no data")
        st.error("❌ File is empty or contains no data. Please check your file.")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing file: {e}")
        st.error(f"❌ Error parsing file: {e}. Please check your file format.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        st.error(f"❌ Error loading file: {e}")
        return pd.DataFrame()

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

# --- State: Only show preview after load ---
if 'df_preview' not in st.session_state:
    st.title("TableAI Loader")
    st.write("Upload any tabular file (csv, xlsx, xls, tsv, parquet, json) or connect to a SQL database.")
    tab1, tab2 = st.tabs(["Upload File", "Connect SQL DB"])

    with tab1:
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "tsv", "parquet", "json"])
        if uploaded_file:
            try:
                # Validate file size (max 100MB)
                if uploaded_file.size > 100 * 1024 * 1024:  # 100MB
                    st.error("❌ File too large. Maximum file size is 100MB.")
                    st.stop()
                
                temp_path = save_and_get_temp_path(uploaded_file)
                df = load_dataframe_from_temp(temp_path)
                
                if df.empty:
                    st.error("❌ No data loaded. Please check your file format.")
                elif len(df.columns) == 0:
                    st.error("❌ File contains no columns. Please check your file format.")
                elif len(df) == 0:
                    st.error("❌ File contains no rows. Please check your file.")
                else:
                    st.success(f"✅ Successfully loaded {uploaded_file.name} with {len(df)} rows and {len(df.columns)} columns")
                    st.session_state['df_preview'] = df.head()
                    st.session_state['df_columns'] = list(df.columns)
                    st.session_state['df_source'] = f"File: {uploaded_file.name}"
                    st.session_state['data_file_path'] = temp_path
                    st.session_state['data_source_type'] = 'file'
                    st.rerun()
            except Exception as e:
                logger.error(f"Error processing uploaded file: {e}")
                st.error(f"❌ Error processing file: {e}")
        # ...no back button...

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
                import sqlalchemy
                # Validate URI format
                if not db_uri.strip():
                    st.error("❌ Please enter a valid database URI")
                    st.stop()
                
                engine = sqlalchemy.create_engine(db_uri)
                with engine.connect() as conn:
                    # Table listing SQL for each DB type
                    try:
                        if db_type == "SQLite":
                            tables = conn.execute(sqlalchemy.text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()
                        elif db_type in ["PostgreSQL", "MariaDB", "MySQL"]:
                            tables = conn.execute(sqlalchemy.text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")).fetchall()
                        elif db_type == "SQL Server":
                            tables = conn.execute(sqlalchemy.text("SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE';")).fetchall()
                        elif db_type == "Oracle":
                            tables = conn.execute(sqlalchemy.text("SELECT table_name FROM user_tables")).fetchall()
                        elif db_type == "DuckDB":
                            tables = conn.execute(sqlalchemy.text("SHOW TABLES")).fetchall()
                        else:
                            # Try generic information_schema
                            tables = conn.execute(sqlalchemy.text("SELECT table_name FROM information_schema.tables")).fetchall()
                        
                        table_names = [t[0] for t in tables]
                        if not table_names:
                            st.warning("⚠️ Connected successfully, but no tables found in the database.")
                        else:
                            st.success(f"✅ Connected successfully! Found {len(table_names)} table(s): {', '.join(table_names)}")
                            logger.info(f"Connected to DB: {db_type}, Tables: {table_names}")
                            
                            table = st.selectbox("Select Table", table_names)
                            if table:
                                try:
                                    df = pd.read_sql_table(table, engine)
                                    if df.empty:
                                        st.warning("⚠️ Table is empty. Please select a different table.")
                                    else:
                                        st.session_state['df_preview'] = df.head()
                                        st.session_state['df_columns'] = list(df.columns)
                                        st.session_state['df_source'] = f"DB Table: {table}"
                                        st.session_state['data_db_uri'] = db_uri
                                        st.session_state['data_db_table'] = table
                                        st.session_state['data_source_type'] = 'db'
                                        st.rerun()
                                except Exception as table_error:
                                    logger.error(f"Error reading table {table}: {table_error}")
                                    st.error(f"❌ Error reading table '{table}': {table_error}")
                                    
                    except sqlalchemy.exc.ProgrammingError as pe:
                        logger.error(f"Database query error: {pe}")
                        st.error(f"❌ Database query error: {pe}. Please check your database permissions.")
                    except Exception as query_error:
                        logger.error(f"Error querying database: {query_error}")
                        st.error(f"❌ Error querying database: {query_error}")
                        
            except sqlalchemy.exc.OperationalError as oe:
                logger.error(f"Database connection error: {oe}")
                st.error(f"❌ Database connection failed: {oe}. Please check your connection details.")
            except sqlalchemy.exc.ArgumentError as ae:
                logger.error(f"Invalid database URI: {ae}")
                st.error(f"❌ Invalid database URI: {ae}. Please check your connection string.")
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                st.error(f"❌ Error connecting to database: {e}")
        # ...no back button...
else:
    if 'show_analyst' not in st.session_state:
        st.session_state['show_analyst'] = False
    # Always show preview if data is loaded and analyst is not shown
    if not st.session_state['show_analyst']:
        st.title("TableAI Preview")
        st.write(f"**Source:** {st.session_state['df_source']}")
        st.write(f"**Columns:** {', '.join(st.session_state['df_columns'])}")
        st.dataframe(st.session_state['df_preview'], use_container_width=True)
        col_preview1, col_preview2 = st.columns([1, 1])
        with col_preview1:
            if st.button("Back to Loader", key="back_to_loader_btn"):
                cleanup_session_state()
                st.rerun()
        with col_preview2:
            if st.button("Ask the analyst", key="meet_analyst_btn"):
                st.session_state['show_analyst'] = True
                st.rerun()
    else:
        # --- Data Preview Full Width ---
        st.subheader("Data Preview")
        # Only display the first 3 rows of data
        if 'df_preview' in st.session_state and not st.session_state['df_preview'].empty:
            preview_df = st.session_state['df_preview'].head(3)
            st.dataframe(preview_df, use_container_width=True)

        # --- Analyst Chatbot Full Width ---
        st.subheader("Analyst Chatbot")

        # ...removed 'Back to Preview' button from chatbot screen...

        # Create the form for user input (so Enter submits)
        if "analyst_last_question" not in st.session_state:
            st.session_state["analyst_last_question"] = ""
        if "analyst_last_code" not in st.session_state:
            st.session_state["analyst_last_code"] = ""
        if "analyst_last_output" not in st.session_state:
            st.session_state["analyst_last_output"] = ""

        def ask_analyst():
            user_question = st.session_state.get("analyst_chat_input", "").strip()
            # Detect if the question is likely random characters or a typo
            import re
            if not user_question:
                return
            # If the question is mostly non-word characters or gibberish, show a message and do not proceed
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
            # Start typing animation for both code and output immediately
            st.session_state['analyst_code_loading'] = True
            st.session_state['analyst_output_loading'] = True
            # --- Reload DataFrame from source before each query ---
            df = None
            try:
                if st.session_state.get('data_source_type') == 'file':
                    file_path = st.session_state.get('data_file_path')
                    if file_path:
                        df = load_dataframe_from_temp(file_path)
                elif st.session_state.get('data_source_type') == 'db':
                    db_uri = st.session_state.get('data_db_uri')
                    table = st.session_state.get('data_db_table')
                    if db_uri and table:
                        import sqlalchemy
                        engine = sqlalchemy.create_engine(db_uri)
                        df = pd.read_sql_table(table, engine)
                else:
                    df = st.session_state.get('df_preview')
            except Exception as e:
                st.session_state['analyst_last_output'] = f"Error loading data for analysis: {e}\n{traceback.format_exc()}"
                df = None
            if df is None or df.empty:
                st.session_state['analyst_last_output'] = "No data available for analysis. Please check your file or database connection."
                st.stop()
            schema = "\n".join([f"- {col}: {str(dtype)}" for col, dtype in zip(df.columns, df.dtypes)])
            sample_rows = df.head(3).to_dict(orient="records")
            # Set prompt and instruction based on data source type
            if st.session_state.get('data_source_type') == 'db':
                system_prompt = (
                    "You are a professional data analyst. "
                    "Given the following data schema and sample rows, answer the user's question by generating SQL code to get the answer. "
                    "Then, execute the SQL query and return the result, along with the query used. "
                    "If the user wants to modify the query, accept their English instructions and update the SQL accordingly. "
                    "Data schema:\n" + schema +
                    f"\nSample rows: {sample_rows}\n"
                    "Always use a table named 'data' in your SQL. "
                    "IMPORTANT: Return ONLY the SQL code, no explanations or other text."
                )
                ai_instruction = (
                    "Instruction: Write SQL code to answer the question using a table named 'data'. "
                    "Return only the SQL code and a brief explanation."
                )
            else:
                system_prompt = (
                    "You are a professional data analyst. "
                    "Given the following data schema and sample rows, answer the user's question by generating Python (pandas) code to get the answer. "
                    "Then, execute the code and return the result, along with the code used. "
                    "If the user wants to modify the code, accept their English instructions and update the code accordingly. "
                    "Data schema:\n" + schema +
                    f"\nSample rows: {sample_rows}\n"
                    "Always use the DataFrame variable 'df' for Python code. "
                    "IMPORTANT: Return ONLY the Python code, no explanations or other text."
                )
                ai_instruction = (
                    "Instruction: Write Python (pandas) code to answer the question using the DataFrame 'df'. "
                    "Return only the code and a brief explanation."
                )
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
                        
                        # Simple AI query without signal-based timeout
                        ai_response = query_engine.query(user_question)
                            
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(doc_path)
                        except:
                            pass
                # Validate and extract code from AI response
                code, code_lang = validate_ai_response(ai_response)
                warning_no_code = not code or code == str(ai_response)
                
                st.session_state['analyst_last_code_language'] = code_lang
                st.session_state['analyst_last_code'] = code or str(ai_response)
                st.session_state['analyst_code_loading'] = False
                logger.info(f"AI generated code for question: {user_question}")
                
                if warning_no_code and code_lang == 'python':
                    st.warning("⚠️ No code block detected in the AI response. Please try rephrasing your question for a better result.")
                
                # --- Execute code ---
                # Start output animation until output is ready
                result = None
                exec_output = ""
                
                # Ensure we have valid code to execute
                if not code or code.strip() == "":
                    st.session_state['analyst_last_output'] = "❌ <b>Error:</b> No valid code generated by AI.<br><b>Tip:</b> Please try rephrasing your question."
                    st.session_state['analyst_output_loading'] = False
                    return
                
                if code:
                    st.session_state['analyst_output_loading'] = True
                    if code_lang == 'sql':
                        # Extract only the actual SQL statement (ignore explanations)
                        import re
                        sql_lines = [l.strip() for l in code.splitlines() if re.match(r'^(SELECT|UPDATE|DELETE|INSERT)', l.strip(), re.IGNORECASE)]
                        if sql_lines:
                            sql_code = sql_lines[0]
                        else:
                            # Fallback: try to find first line ending with semicolon
                            sql_lines = [l.strip() for l in code.splitlines() if l.strip().endswith(';')]
                            sql_code = sql_code if sql_lines else code.strip()
                        
                        # Replace 'data' with actual table name if needed
                        db_uri = st.session_state.get('data_db_uri')
                        table = st.session_state.get('data_db_table')
                        try:
                            if db_uri and table:
                                import sqlalchemy
                                engine = sqlalchemy.create_engine(db_uri)
                                sql_code = sql_code.replace('data', table)
                                with engine.connect() as conn:
                                    result_df = pd.read_sql_query(sql_code, conn)
                                if result_df.empty:
                                    st.session_state['analyst_last_output'] = "No results found for your query. Please check your question or table data."
                                elif len(result_df) == 1 and result_df.shape[1] == 1:
                                    # Single value
                                    st.session_state['analyst_last_output'] = result_df.iloc[0, 0]
                                else:
                                    st.session_state['analyst_last_output'] = result_df
                                st.session_state['analyst_output_loading'] = False
                                return
                            else:
                                st.session_state['analyst_last_output'] = "Database connection info missing. Please reconnect and try again."
                                st.session_state['analyst_output_loading'] = False
                                return
                        except Exception as e:
                            st.session_state['analyst_last_output'] = f"❌ <b>Error:</b> SQL query execution failed.<br><b>Details:</b> {e}<br><b>Query:</b> <pre>{sql_code}</pre>"
                            st.session_state['analyst_output_loading'] = False
                            return
                    
                    # --- Clean and post-process AI code for Python ---
                    st.session_state['analyst_output_loading'] = True
                    code = clean_ai_code(code)
                    import re
                    code_lines = code.strip().split('\n')
                    new_code_lines = []
                    columns_warning = None
                    for line in code_lines:
                        # Remove lines that assign to df (e.g., df = ...)
                        if re.match(r"^\s*df\s*=", line):
                            continue
                        # Check for 'df.columns = [...]' and validate length
                        col_assign = re.match(r"^\s*df\.columns\s*=\s*\[(.*)\]", line)
                        if col_assign:
                            try:
                                assigned_cols = eval(f'[{col_assign.group(1)}]')
                                if isinstance(assigned_cols, list) and len(assigned_cols) != len(df.columns):
                                    columns_warning = f"⚠️ <b>Warning:</b> The generated code tried to set DataFrame columns to {len(assigned_cols)} names, but your data has {len(df.columns)} columns. This line was skipped."
                                    continue
                            except Exception:
                                columns_warning = "⚠️ <b>Warning:</b> The generated code tried to set DataFrame columns, but the assignment was invalid and was skipped."
                                continue
                        new_code_lines.append(line)
                    code = '\n'.join(new_code_lines)
                    # Now handle file reads if present
                    if "pd.read_csv" in code or "pd.read_excel" in code:
                        # Always use the latest df loaded above
                        csv_matches = re.findall(r"pd\\.read_csv\\(['\"]([^'\"]+)['\"]\\)", code)
                        excel_matches = re.findall(r"pd\\.read_excel\\(['\"]([^'\"]+)['\"]\\)", code)
                        file_map = {}
                        for fname in csv_matches:
                            fpath = os.path.join(tempfile.gettempdir(), fname)
                            if os.path.exists(fpath):
                                base, ext = os.path.splitext(fname)
                                new_fname = f"{base}_{uuid.uuid4().hex[:6]}{ext or '.csv'}"
                                new_fpath = os.path.join(tempfile.gettempdir(), new_fname)
                                df.to_csv(new_fpath, index=False)
                                file_map[fname] = new_fname
                            else:
                                df.to_csv(fpath, index=False)
                                file_map[fname] = fname
                        for fname in excel_matches:
                            fpath = os.path.join(tempfile.gettempdir(), fname)
                            if os.path.exists(fpath):
                                base, ext = os.path.splitext(fname)
                                new_fname = f"{base}_{uuid.uuid4().hex[:6]}{ext or '.xlsx'}"
                                new_fpath = os.path.join(tempfile.gettempdir(), new_fname)
                                df.to_excel(new_fpath, index=False)
                                file_map[fname] = new_fname
                            else:
                                df.to_excel(fpath, index=False)
                                file_map[fname] = fname
                        for old, new in file_map.items():
                            if old != new:
                                code = code.replace(old, new)
                    # Now run the code
                    local_vars = {"df": df}
                    code_lines = code.strip().split('\n')
                    last_line = code_lines[-1] if code_lines else ''
                    assigns_result = any(l.strip().startswith("result =") for l in code_lines)
                    import io, contextlib, warnings
                    import matplotlib.pyplot as plt
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")
                            if not assigns_result and last_line and not last_line.strip().startswith("#"):
                                exec_code = '\n'.join(code_lines[:-1])
                                with contextlib.redirect_stdout(io.StringIO()) as f:
                                    if exec_code.strip():
                                        exec(exec_code, {}, local_vars)
                                    last_line_strip = last_line.strip()
                                    if "=" in last_line_strip and not "==" in last_line_strip and not last_line_strip.startswith("#"):
                                        try:
                                            exec(last_line_strip, {}, local_vars)
                                            var_name = last_line_strip.split("=")[0].strip()
                                            result = local_vars.get(var_name, None)
                                        except SyntaxError as se:
                                            st.session_state['analyst_last_output'] = (
                                                "❌ <b>Error:</b> The AI-generated code could not be executed due to invalid syntax. "
                                                "<br><b>Details:</b> The question may be unclear or nonsensical.<br>"
                                                "<b>Tip:</b> Please clarify your question or provide more specific details about your data or desired analysis. "
                                                f"<br><b>AI code:</b> <pre>{last_line_strip}</pre>"
                                            )
                                            st.session_state['analyst_output_loading'] = False
                                            return
                                    else:
                                        try:
                                            result = eval(last_line_strip, {}, local_vars)
                                        except SyntaxError as se:
                                            st.session_state['analyst_last_output'] = (
                                                "❌ <b>Error:</b> The AI-generated code could not be executed due to invalid syntax. "
                                                "<br><b>Details:</b> The question may be unclear or nonsensical.<br>"
                                                "<b>Tip:</b> Please clarify your question or provide more specific details about your data or desired analysis. "
                                                f"<br><b>AI code:</b> <pre>{last_line_strip}</pre>"
                                            )
                                            st.session_state['analyst_output_loading'] = False
                                            return
                                exec_output = f.getvalue()
                            else:
                                with contextlib.redirect_stdout(io.StringIO()) as f:
                                    exec(code, {}, local_vars)
                                exec_output = f.getvalue()
                                result = local_vars.get("result", None)
                            # If result is None, but a matplotlib figure was created, use it    
                            if result is None:
                                fig = plt.gcf()
                                # Only use if a figure was actually created (has axes)
                                if fig and fig.get_axes():
                                    result = fig
                            # Add columns warning if present
                            if columns_warning:
                                st.session_state['analyst_last_output'] = columns_warning + "<br>" + (str(result) if result is not None else exec_output)
                            else:
                                st.session_state['analyst_last_output'] = result if result is not None else exec_output
                            st.session_state['analyst_output_loading'] = False
                    except ValueError as ve:
                        error_msg = "❌ <b>Error:</b> "
                        if "Length mismatch" in str(ve):
                            error_msg += (
                                "The generated code tried to set DataFrame columns or index with the wrong number of labels. "
                                f"<br><b>Details:</b> {ve}<br>"
                                "<b>Tip:</b> Your data has a different number of columns or rows than the code expects. Please check your question or try again."
                            )
                        elif "got an unexpected keyword argument" in str(ve):
                            error_msg += (
                                "The generated code used an invalid argument in a plotting function. "
                                f"<br><b>Details:</b> {ve}<br>"
                                "<b>Tip:</b> This is often caused by using 'labels' instead of 'label' in matplotlib. Please try again or rephrase your question."
                            )
                        else:
                            error_msg += f"Value error: {ve}"
                        st.session_state['analyst_last_output'] = error_msg
                        st.session_state['analyst_output_loading'] = False
                        
                    except TypeError as te:
                        error_msg = "❌ <b>Error:</b> "
                        if "agg function failed" in str(te) or "could not convert" in str(te):
                            error_msg += (
                                "The generated code tried to aggregate non-numeric columns. "
                                f"<br><b>Details:</b> {te}<br>"
                                "<b>Tip:</b> Only numeric columns can be aggregated (e.g., mean, sum). Please specify a numeric column or let the AI know to use only numeric columns. "
                                "<br><b>Detected column types:</b> "
                                f"{local_vars['df'].dtypes.to_dict() if 'df' in local_vars else ''}"
                            )
                        else:
                            error_msg += f"Type error: {te}"
                        st.session_state['analyst_last_output'] = error_msg
                        st.session_state['analyst_output_loading'] = False
                        
                    except Exception as e:
                        logger.error(f"Error executing AI code: {e}\n{traceback.format_exc()}")
                        error_msg = "❌ <b>Error:</b> "
                        if "got an unexpected keyword argument" in str(e):
                            error_msg += (
                                "The generated code used an invalid argument in a plotting function. "
                                f"<br><b>Details:</b> {e}<br>"
                                "<b>Tip:</b> This is often caused by using 'labels' instead of 'label' in matplotlib. Please try again or rephrase your question."
                            )
                        else:
                            error_msg += f"Unexpected error: {e}"
                        
                        # Try to provide a helpful fallback based on the question
                        try:
                            if "highest" in user_question.lower() or "maximum" in user_question.lower():
                                if 'salary' in user_question.lower() or 'income' in user_question.lower():
                                    fallback_code = "df[['name', 'salary']].nlargest(1, 'salary')"
                                    fallback_result = eval(fallback_code, {"df": df})
                                    error_msg += f"<br><br>🔄 <b>Fallback result:</b> {fallback_result.to_string()}"
                                    logger.info("Applied fallback for highest salary query")
                        except:
                            pass
                        
                        st.session_state['analyst_last_output'] = error_msg
                        st.session_state['analyst_output_loading'] = False
                else:
                    st.session_state['analyst_last_output'] = str(ai_response)
                    # Input will be cleared by clear_on_submit=True in the form
            except Exception as e:
                logger.error(f"Error in AI processing: {e}\n{traceback.format_exc()}")
                st.session_state['analyst_last_code'] = None
                st.session_state['analyst_last_output'] = f"❌ <b>Error:</b> {e}<br><b>Tip:</b> Please try again or rephrase your question."
                st.session_state['analyst_code_loading'] = False
                st.session_state['analyst_output_loading'] = False

        # --- Analyst Chatbot Form ---
        with st.form(key="analyst_chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question about your data", key="analyst_chat_input", placeholder="e.g., Show the average of column X grouped by column Y")
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


