
import streamlit as st
import pandas as pd
import io
import base64
import time
import os
import asyncio
from typing import Optional
import traceback
import json
from datetime import datetime

# Import the DataAnalystAgent from the provided code
try:
    from paste import DataAnalystAgent, DataAnalysisState
except ImportError:
    st.error("Please ensure the DataAnalystAgent code is available as 'paste.py'")
    st.stop()

# Configure page settings
st.set_page_config(
    page_title="AI Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
def load_css():
    st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Card styling */
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed #667eea;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9ff;
    }

    /* Error and success styling */
    .stAlert {
        border-radius: 8px;
    }

    /* Visualization container */
    .viz-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-success { background-color: #28a745; }
    .status-error { background-color: #dc3545; }
    .status-processing { background-color: #ffc107; animation: pulse 1.5s infinite; }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .analysis-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

# Data upload and validation
def handle_file_upload():
    st.markdown("### 📁 Data Upload")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your dataset for analysis. Maximum file size: 200MB"
    )

    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df

            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{df.shape[0]:,}</h3>
                    <p>Rows</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{df.shape[1]}</h3>
                    <p>Columns</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB</h3>
                    <p>Memory</p>
                </div>
                """, unsafe_allow_html=True)

            # Data preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Data quality summary
            st.markdown("#### Data Quality Summary")
            missing_data = df.isnull().sum()
            if missing_data.any():
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Values': missing_data.values,
                    'Missing %': (missing_data.values / len(df) * 100).round(2)
                }).query('`Missing Values` > 0')

                if not missing_df.empty:
                    st.warning(f"Found missing values in {len(missing_df)} columns")
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("✅ No missing values detected")
            else:
                st.success("✅ No missing values detected")

            return True

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return False

    return False

# Analysis configuration
def analysis_configuration():
    st.markdown("### ⚙️ Analysis Configuration")

    # API Key input
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.groq_api_key,
        help="Enter your Groq API key for AI-powered analysis"
    )

    if api_key:
        st.session_state.groq_api_key = api_key
        st.success("✅ API Key configured")
        return True
    else:
        st.warning("⚠️ Please enter your Groq API key to proceed")
        return False

# Run analysis
def run_analysis():
    if not st.session_state.groq_api_key:
        st.error("Please configure your Groq API key first")
        return

    if st.session_state.data is None:
        st.error("Please upload a dataset first")
        return

    try:
        # Initialize the agent
        agent = DataAnalystAgent(st.session_state.groq_api_key)

        # Create progress tracking
        progress_container = st.container()
        status_container = st.container()

        with progress_container:
            st.markdown("### 🔄 Analysis Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()

        # Update progress function
        def update_progress(step, total_steps, message):
            progress = step / total_steps
            progress_bar.progress(progress)
            status_text.text(f"Step {step}/{total_steps}: {message}")

        # Run analysis with progress updates
        update_progress(1, 6, "Initializing analysis...")
        time.sleep(0.5)

        update_progress(2, 6, "Preparing data preview...")
        time.sleep(0.5)

        update_progress(3, 6, "Analyzing data types...")
        time.sleep(0.5)

        update_progress(4, 6, "Identifying relationships...")
        time.sleep(1)

        update_progress(5, 6, "Creating visualizations...")

        # Run the actual analysis
        with st.spinner("Running AI-powered analysis..."):
            results = agent.run_analysis(st.session_state.data)

        update_progress(6, 6, "Analysis complete!")

        # Store results
        st.session_state.analysis_results = results

        # Add to history
        analysis_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_shape': st.session_state.data.shape,
            'visualizations_count': len(results.get('visualization_outputs', [])),
            'success': True
        }
        st.session_state.analysis_history.append(analysis_entry)

        st.success("🎉 Analysis completed successfully!")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.exception(e)

# Display results
def display_results():
    if st.session_state.analysis_results is None:
        return

    results = st.session_state.analysis_results

    st.markdown("## 📊 Analysis Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    viz_outputs = results.get('visualization_outputs', [])
    successful_viz = sum(1 for viz in viz_outputs if viz.get('success', False))

    with col1:
        st.metric("Total Visualizations", len(viz_outputs))
    with col2:
        st.metric("Successful", successful_viz)
    with col3:
        st.metric("Failed", len(viz_outputs) - successful_viz)
    with col4:
        success_rate = (successful_viz / len(viz_outputs) * 100) if viz_outputs else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")

    # Display visualizations
    if viz_outputs:
        st.markdown("### 📈 Generated Visualizations")

        for i, viz in enumerate(viz_outputs):
            with st.expander(f"Visualization {i+1}: {viz.get('chart_type', 'Unknown').title()}", expanded=True):
                col1, col2 = st.columns([2, 1])

                with col1:
                    if viz.get('success', False) and viz.get('image_base64'):
                        # Display the image
                        image_data = base64.b64decode(viz['image_base64'])
                        st.image(image_data, caption=f"{viz.get('chart_type', 'Visualization').title()}")
                    else:
                        st.error("❌ Visualization generation failed")
                        if viz.get('insights'):
                            st.text(viz['insights'])

                with col2:
                    # Visualization metadata
                    st.markdown("**Details:**")
                    st.write(f"**Type:** {viz.get('chart_type', 'Unknown')}")
                    st.write(f"**Variables:** {', '.join(viz.get('variables', []))}")
                    st.write(f"**Priority:** {viz.get('priority', 'Unknown')}")

                    if viz.get('success'):
                        st.markdown('<span class="status-indicator status-success"></span>**Success**', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="status-indicator status-error"></span>**Failed**', unsafe_allow_html=True)

                # Insights
                if viz.get('explanation'):
                    st.markdown("**Purpose:**")
                    st.write(viz['explanation'])

                if viz.get('insights') and viz.get('success'):
                    st.markdown("**Insights:**")
                    st.write(viz['insights'])

                # Technical details (collapsible)
                if st.checkbox(f"Show technical details for visualization {i+1}", key=f"tech_{i}"):
                    if viz.get('generated_code'):
                        st.markdown("**Generated Code:**")
                        st.code(viz['generated_code'], language='python')

                    if viz.get('healing_history'):
                        st.markdown("**Healing History:**")
                        st.json(viz['healing_history'])

    # LLM Feedback
    if results.get('feedback'):
        st.markdown("### 🤖 AI Analysis Summary")
        st.markdown(results['feedback'])

    # Data insights summary
    if results.get('variable_relationships'):
        st.markdown("### 🔗 Identified Relationships")
        relationships_df = pd.DataFrame(results['variable_relationships'])
        st.dataframe(relationships_df, use_container_width=True)

# Analysis history
def show_analysis_history():
    if st.session_state.analysis_history:
        st.markdown("### 📚 Analysis History")
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df, use_container_width=True)

# Main app
def main():
    load_css()
    initialize_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Data Analysis Dashboard</h1>
        <p>Upload your data and get instant AI-powered insights with beautiful visualizations</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")

        # API Configuration
        api_configured = analysis_configuration()

        st.markdown("---")

        # Analysis controls
        st.markdown("## 🚀 Analysis")

        if st.button("🔍 Run Analysis", disabled=not api_configured, key="run_analysis"):
            st.session_state.analysis_running = True
            st.rerun()

        if st.button("🗑️ Clear Results", key="clear_results"):
            st.session_state.analysis_results = None
            st.session_state.data = None
            st.rerun()

        st.markdown("---")

        # Settings
        st.markdown("## ⚙️ Settings")

        show_technical_details = st.checkbox("Show technical details", value=False)
        show_code = st.checkbox("Show generated code", value=False)

        st.markdown("---")

        # Info
        st.markdown("## ℹ️ About")
        st.markdown("""
        This dashboard uses AI to automatically:
        - Analyze your data structure
        - Identify relationships
        - Generate appropriate visualizations
        - Provide insights and recommendations

        **Powered by:**
        - LangGraph for workflow orchestration
        - Groq for AI processing
        - Streamlit for the interface
        """)

    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📈 Results", "📚 History"])

    with tab1:
        # File upload
        file_uploaded = handle_file_upload()

        if file_uploaded and api_configured:
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### Ready to analyze!")
                st.write("Your data is loaded and API is configured. Click 'Run Analysis' in the sidebar to begin.")
            with col2:
                if st.button("🚀 Start Analysis", key="start_analysis_main"):
                    st.session_state.analysis_running = True
                    st.rerun()

    with tab2:
        if st.session_state.analysis_running:
            run_analysis()
            st.session_state.analysis_running = False
            st.rerun()

        display_results()

    with tab3:
        show_analysis_history()

if __name__ == "__main__":
    main()
