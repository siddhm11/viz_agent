
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import time
import os
import asyncio
from typing import Optional, Dict, List
import traceback
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Enhanced imports for better UI
try:
    import streamlit_shadcn_ui as ui
    SHADCN_AVAILABLE = True
except ImportError:
    SHADCN_AVAILABLE = False

# Import the DataAnalystAgent from the provided code
try:
    from paste import DataAnalystAgent, DataAnalysisState
except ImportError:
    st.error("Please ensure the DataAnalystAgent code is available as 'paste.py'")
    st.stop()

from config import get_config

# Configure page settings
config = get_config()
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Enhanced CSS with animations and modern styling
def load_enhanced_css():
    st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* CSS Variables for easy theming */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --error-color: #dc3545;
        --background-color: #fafbfc;
        --card-background: #ffffff;
        --text-color: #2c3e50;
        --border-color: #e1e5e9;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        --shadow-hover: 0 8px 25px rgba(0, 0, 0, 0.15);
        --border-radius: 12px;
    }

    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
        background-color: var(--background-color);
    }

    /* Enhanced header with gradient animation */
    .main-header {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(1px);
    }

    .main-header > * {
        position: relative;
        z-index: 1;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.95;
        font-weight: 400;
    }

    /* Enhanced card styling with hover effects */
    .analysis-card {
        background: var(--card-background);
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .analysis-card:hover {
        box-shadow: var(--shadow-hover);
        transform: translateY(-4px);
    }

    .analysis-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }

    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        text-align: center;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: default;
    }

    .metric-card:hover {
        transform: scale(1.05);
    }

    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }

    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-weight: 500;
    }

    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Sidebar enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid var(--border-color);
    }

    /* Progress bar enhancements */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 10px;
    }

    /* File uploader enhancements */
    .stFileUploader > div {
        border: 2px dashed var(--primary-color);
        border-radius: var(--border-radius);
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #fff8f8 100%);
        transition: all 0.3s ease;
    }

    .stFileUploader > div:hover {
        border-color: var(--secondary-color);
        background: linear-gradient(135deg, #f0f4ff 0%, #fff0f0 100%);
    }

    /* Enhanced alerts */
    .stAlert {
        border-radius: var(--border-radius);
        border: none;
        box-shadow: var(--shadow);
    }

    /* Visualization container */
    .viz-container {
        background: var(--card-background);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }

    .viz-container:hover {
        box-shadow: var(--shadow-hover);
    }

    /* Status indicators with animation */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem 0;
    }

    .status-success {
        background-color: rgba(40, 167, 69, 0.1);
        color: var(--success-color);
        border: 1px solid rgba(40, 167, 69, 0.2);
    }

    .status-error {
        background-color: rgba(220, 53, 69, 0.1);
        color: var(--error-color);
        border: 1px solid rgba(220, 53, 69, 0.2);
    }

    .status-processing {
        background-color: rgba(255, 193, 7, 0.1);
        color: var(--warning-color);
        border: 1px solid rgba(255, 193, 7, 0.2);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* Code block styling */
    .stCode {
        background-color: #1e1e1e !important;
        border-radius: var(--border-radius) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow);
    }

    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        .analysis-card {
            padding: 1.5rem;
        }
        .metric-card h3 {
            font-size: 1.5rem;
        }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-color);
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced session state management
class SessionManager:
    @staticmethod
    def initialize():
        defaults = {
            'analysis_results': None,
            'data': None,
            'analysis_running': False,
            'groq_api_key': config.get_groq_api_key(),
            'analysis_history': [],
            'current_tab': 0,
            'show_advanced_options': False,
            'export_format': 'png',
            'theme_preference': 'auto'
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def clear_analysis():
        """Clear analysis-related session state"""
        keys_to_clear = ['analysis_results', 'data', 'analysis_running']
        for key in keys_to_clear:
            if key in st.session_state:
                st.session_state[key] = None

    @staticmethod
    def add_to_history(entry: Dict):
        """Add entry to analysis history"""
        st.session_state.analysis_history.append(entry)

# Enhanced data upload with more file types and validation
class DataUploader:
    @staticmethod
    def handle_upload():
        st.markdown("### 📁 Data Upload")

        # File uploader with multiple types
        uploaded_file = st.file_uploader(
            "Choose your data file",
            type=config.ALLOWED_FILE_TYPES,
            help=f"Supported formats: {', '.join(config.ALLOWED_FILE_TYPES)}. Maximum file size: {config.MAX_UPLOAD_SIZE_MB}MB"
        )

        if uploaded_file is not None:
            try:
                # Read file based on extension
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    # CSV with encoding detection
                    try:
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error(f"Unsupported file type: {file_extension}")
                    return False

                # Data validation
                if df.empty:
                    st.error("The uploaded file is empty.")
                    return False

                if df.shape[0] > 100000:
                    st.warning("Large dataset detected. Analysis may take longer.")

                st.session_state.data = df

                # Enhanced data preview
                DataUploader._display_data_overview(df)

                return True

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return False

        return False

    @staticmethod
    def _display_data_overview(df: pd.DataFrame):
        """Display comprehensive data overview"""
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)

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
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.markdown(f"""
            <div class="metric-card">
                <h3>{memory_mb:.1f}MB</h3>
                <p>Memory</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>{completeness:.1f}%</h3>
                <p>Complete</p>
            </div>
            """, unsafe_allow_html=True)

        # Data types summary
        st.markdown("#### 📋 Column Information")

        # Create column info DataFrame
        column_info = []
        for col in df.columns:
            info = {
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null': df[col].count(),
                'Null %': f"{(df[col].isnull().sum() / len(df) * 100):.1f}%",
                'Unique': df[col].nunique()
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                info['Min'] = f"{df[col].min():.2f}" if not pd.isna(df[col].min()) else "N/A"
                info['Max'] = f"{df[col].max():.2f}" if not pd.isna(df[col].max()) else "N/A"
            else:
                info['Min'] = "N/A"
                info['Max'] = "N/A"

            column_info.append(info)

        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True)

        # Data preview
        st.markdown("#### 👀 Data Preview")

        # Preview options
        col1, col2 = st.columns([1, 3])
        with col1:
            preview_rows = st.selectbox("Rows to show", [5, 10, 20, 50], index=1)

        st.dataframe(df.head(preview_rows), use_container_width=True)

        # Data quality issues
        DataUploader._check_data_quality(df)

    @staticmethod
    def _check_data_quality(df: pd.DataFrame):
        """Check and report data quality issues"""
        issues = []

        # Missing values
        missing_cols = df.isnull().sum()
        high_missing = missing_cols[missing_cols > len(df) * 0.5]
        if not high_missing.empty:
            issues.append(f"High missing values (>50%): {list(high_missing.index)}")

        # Duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate rows found")

        # Constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols}")

        # Display issues or success
        if issues:
            st.markdown("#### ⚠️ Data Quality Issues")
            for issue in issues:
                st.warning(issue)
        else:
            st.success("✅ No major data quality issues detected")

# Enhanced analysis runner with better progress tracking
class AnalysisRunner:
    def __init__(self):
        self.steps = [
            "Initializing AI agent...",
            "Preparing data preview...",
            "Analyzing data types...",
            "Identifying relationships...",
            "Generating visualizations...",
            "Completing analysis..."
        ]

    def run(self):
        if not st.session_state.groq_api_key:
            st.error("🔑 Please configure your Groq API key first")
            return

        if st.session_state.data is None:
            st.error("📊 Please upload a dataset first")
            return

        try:
            # Initialize the agent
            agent = DataAnalystAgent(st.session_state.groq_api_key)

            # Create progress tracking UI
            progress_container = st.container()

            with progress_container:
                st.markdown("### 🔄 Analysis in Progress")

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_estimate = st.empty()

                # Step indicators
                step_container = st.container()

                start_time = time.time()

                for i, step in enumerate(self.steps):
                    # Update progress
                    progress = (i + 1) / len(self.steps)
                    progress_bar.progress(progress)

                    # Update status
                    status_text.markdown(f"""
                    <div class="status-indicator status-processing">
                        <span class="loading-spinner"></span>
                        <span style="margin-left: 10px;">{step}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Estimate time remaining
                    if i > 0:
                        elapsed = time.time() - start_time
                        estimated_total = elapsed / progress
                        remaining = estimated_total - elapsed
                        time_estimate.text(f"Estimated time remaining: {remaining:.0f} seconds")

                    # Simulate processing time (replace with actual steps)
                    if i < len(self.steps) - 1:
                        time.sleep(1)

                # Run the actual analysis
                with st.spinner("🤖 Running AI-powered analysis..."):
                    results = agent.run_analysis(st.session_state.data)

                # Complete progress
                progress_bar.progress(1.0)
                status_text.markdown("""
                <div class="status-indicator status-success">
                    ✅ Analysis completed successfully!
                </div>
                """, unsafe_allow_html=True)
                time_estimate.empty()

                # Store results
                st.session_state.analysis_results = results

                # Add to history
                analysis_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'dataset_name': 'uploaded_data',
                    'dataset_shape': st.session_state.data.shape,
                    'visualizations_count': len(results.get('visualization_outputs', [])),
                    'success': True,
                    'processing_time': f"{time.time() - start_time:.1f}s"
                }
                SessionManager.add_to_history(analysis_entry)

                st.balloons()  # Celebration animation

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            if config.DEBUG:
                st.exception(e)

# Enhanced results display with export options
class ResultsDisplay:
    @staticmethod
    def show():
        if st.session_state.analysis_results is None:
            st.info("🔍 No analysis results yet. Upload data and run analysis to see results here.")
            return

        results = st.session_state.analysis_results

        st.markdown("## 📊 Analysis Results")

        # Summary dashboard
        ResultsDisplay._show_summary_dashboard(results)

        # Visualizations
        ResultsDisplay._show_visualizations(results)

        # AI insights
        ResultsDisplay._show_ai_insights(results)

        # Export options
        ResultsDisplay._show_export_options(results)

    @staticmethod
    def _show_summary_dashboard(results):
        """Show analysis summary dashboard"""
        st.markdown("### 📈 Analysis Summary")

        viz_outputs = results.get('visualization_outputs', [])
        successful_viz = sum(1 for viz in viz_outputs if viz.get('success', False))

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Visualizations", 
                len(viz_outputs),
                help="Total number of visualizations generated"
            )
        with col2:
            st.metric(
                "Successful", 
                successful_viz,
                delta=successful_viz if successful_viz > 0 else None,
                delta_color="normal"
            )
        with col3:
            failed_count = len(viz_outputs) - successful_viz
            st.metric(
                "Failed", 
                failed_count,
                delta=-failed_count if failed_count > 0 else None,
                delta_color="inverse"
            )
        with col4:
            success_rate = (successful_viz / len(viz_outputs) * 100) if viz_outputs else 0
            st.metric(
                "Success Rate", 
                f"{success_rate:.1f}%",
                delta=f"{success_rate:.1f}%" if success_rate > 0 else None
            )

    @staticmethod
    def _show_visualizations(results):
        """Display all generated visualizations"""
        viz_outputs = results.get('visualization_outputs', [])

        if not viz_outputs:
            st.warning("No visualizations were generated.")
            return

        st.markdown("### 📈 Generated Visualizations")

        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_successful = st.checkbox("Show successful only", value=True)
        with col2:
            show_failed = st.checkbox("Show failed", value=False)
        with col3:
            sort_by = st.selectbox("Sort by", ["Priority", "Type", "Success"])

        # Filter and sort visualizations
        filtered_viz = []
        for viz in viz_outputs:
            if show_successful and viz.get('success', False):
                filtered_viz.append(viz)
            elif show_failed and not viz.get('success', False):
                filtered_viz.append(viz)

        if not filtered_viz:
            st.info("No visualizations match the current filters.")
            return

        # Display visualizations
        for i, viz in enumerate(filtered_viz):
            with st.expander(
                f"📊 {viz.get('chart_type', 'Unknown').title()} - {', '.join(viz.get('variables', []))}", 
                expanded=True
            ):
                ResultsDisplay._display_single_visualization(viz, i)

    @staticmethod
    def _display_single_visualization(viz, index):
        """Display a single visualization with all details"""
        col1, col2 = st.columns([2, 1])

        with col1:
            if viz.get('success', False) and viz.get('image_base64'):
                # Display the image
                try:
                    image_data = base64.b64decode(viz['image_base64'])
                    st.image(
                        image_data, 
                        caption=f"{viz.get('chart_type', 'Visualization').title()}",
                        use_column_width=True
                    )

                    # Download button for image
                    st.download_button(
                        label="💾 Download Image",
                        data=image_data,
                        file_name=f"visualization_{index+1}_{viz.get('chart_type', 'chart')}.png",
                        mime="image/png",
                        key=f"download_viz_{index}"
                    )
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
            else:
                st.error("❌ Visualization generation failed")
                if viz.get('insights'):
                    st.code(viz['insights'], language='text')

        with col2:
            # Visualization metadata
            st.markdown("**📋 Details**")

            details_data = {
                "Type": viz.get('chart_type', 'Unknown'),
                "Variables": ', '.join(viz.get('variables', [])),
                "Priority": viz.get('priority', 'Unknown'),
                "Status": "✅ Success" if viz.get('success') else "❌ Failed"
            }

            for key, value in details_data.items():
                st.text(f"{key}: {value}")

            # Priority indicator
            priority = viz.get('priority', 'unknown').lower()
            if priority == 'high':
                st.markdown('<div class="status-indicator" style="background: #dc3545; color: white;">🔥 High Priority</div>', unsafe_allow_html=True)
            elif priority == 'medium':
                st.markdown('<div class="status-indicator" style="background: #ffc107; color: #212529;">⚡ Medium Priority</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-indicator" style="background: #6c757d; color: white;">📋 Low Priority</div>', unsafe_allow_html=True)

        # Insights and explanations
        if viz.get('explanation'):
            st.markdown("**🎯 Purpose**")
            st.info(viz['explanation'])

        if viz.get('insights') and viz.get('success'):
            st.markdown("**💡 Insights**")
            st.success(viz['insights'])

        # Technical details (collapsible)
        show_technical = st.checkbox(
            f"🔧 Show technical details", 
            key=f"tech_details_{index}",
            help="Show generated code and debugging information"
        )

        if show_technical:
            if viz.get('generated_code'):
                st.markdown("**🐍 Generated Code**")
                st.code(viz['generated_code'], language='python')

            if viz.get('healing_history'):
                st.markdown("**🔄 Healing History**")
                st.json(viz['healing_history'])

            if viz.get('generation_stats'):
                st.markdown("**📊 Generation Statistics**")
                st.json(viz['generation_stats'])

    @staticmethod
    def _show_ai_insights(results):
        """Display AI-generated insights and feedback"""
        if results.get('feedback'):
            st.markdown("### 🤖 AI Analysis Summary")

            # Parse and display the feedback in a structured way
            feedback = results['feedback']

            # Display in an attractive format
            st.markdown(f"""
            <div class="analysis-card">
                {feedback}
            </div>
            """, unsafe_allow_html=True)

        # Display identified relationships
        if results.get('variable_relationships'):
            st.markdown("### 🔗 Identified Data Relationships")

            relationships = results['variable_relationships']

            # Create a nice display for relationships
            for i, rel in enumerate(relationships):
                with st.expander(f"Relationship {i+1}: {rel.get('relationship_type', 'Unknown').title()}", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Variables:**", ', '.join(rel.get('variables', [])))
                        st.write("**Type:**", rel.get('relationship_type', 'Unknown'))

                    with col2:
                        st.write("**Priority:**", rel.get('priority', 'Unknown'))

                    if rel.get('hypothesis'):
                        st.write("**Hypothesis:**", rel['hypothesis'])

    @staticmethod
    def _show_export_options(results):
        """Show export options for results"""
        st.markdown("### 💾 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Export All Visualizations", key="export_all_viz"):
                ResultsDisplay._export_all_visualizations(results)

        with col2:
            if st.button("📋 Export Analysis Report", key="export_report"):
                ResultsDisplay._export_analysis_report(results)

        with col3:
            if st.button("💾 Export Raw Results", key="export_raw"):
                ResultsDisplay._export_raw_results(results)

    @staticmethod
    def _export_all_visualizations(results):
        """Export all successful visualizations as a ZIP file"""
        # This would create a ZIP file with all images
        st.success("📊 Visualizations export feature coming soon!")

    @staticmethod
    def _export_analysis_report(results):
        """Export a comprehensive analysis report"""
        # This would generate a PDF or HTML report
        st.success("📋 Report export feature coming soon!")

    @staticmethod
    def _export_raw_results(results):
        """Export raw analysis results as JSON"""
        results_json = json.dumps(results, default=str, indent=2)
        st.download_button(
            label="💾 Download Results JSON",
            data=results_json,
            file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Enhanced history view
class HistoryView:
    @staticmethod
    def show():
        st.markdown("### 📚 Analysis History")

        if not st.session_state.analysis_history:
            st.info("🔍 No analysis history yet. Complete an analysis to see it here.")
            return

        # History summary
        total_analyses = len(st.session_state.analysis_history)
        successful_analyses = sum(1 for entry in st.session_state.analysis_history if entry.get('success', False))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyses", total_analyses)
        with col2:
            st.metric("Successful", successful_analyses)
        with col3:
            success_rate = (successful_analyses / total_analyses * 100) if total_analyses > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

        # History table
        history_df = pd.DataFrame(st.session_state.analysis_history)

        # Add some formatting
        if not history_df.empty:
            history_df['Timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('Timestamp', ascending=False)

            st.dataframe(
                history_df.drop('timestamp', axis=1), 
                use_container_width=True,
                column_config={
                    "Timestamp": st.column_config.DatetimeColumn(
                        "Timestamp",
                        format="DD/MM/YYYY HH:mm:ss"
                    ),
                    "success": st.column_config.CheckboxColumn("Success")
                }
            )

        # Clear history option
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.analysis_history = []
            st.rerun()

def main():
    """Main application function"""
    # Load CSS and initialize
    load_enhanced_css()
    SessionManager.initialize()

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
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.groq_api_key,
            help="Enter your Groq API key for AI-powered analysis"
        )

        if api_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = api_key

        api_configured = bool(st.session_state.groq_api_key)

        if api_configured:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter your Groq API key")

        st.markdown("---")

        # Analysis controls
        st.markdown("### 🚀 Analysis Controls")

        if st.button("🔍 Run Analysis", disabled=not api_configured, key="run_analysis_sidebar"):
            st.session_state.analysis_running = True
            st.rerun()

        if st.button("🗑️ Clear Results", key="clear_results_sidebar"):
            SessionManager.clear_analysis()
            st.rerun()

        st.markdown("---")

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            st.session_state.show_advanced_options = st.checkbox("Show technical details", value=st.session_state.show_advanced_options)
            st.session_state.export_format = st.selectbox("Export format", ['png', 'svg', 'pdf'], index=0)
            st.session_state.theme_preference = st.selectbox("Theme", ['auto', 'light', 'dark'], index=0)

        st.markdown("---")

        # Info section
        st.markdown("### ℹ️ About")
        st.markdown("""
        This dashboard uses advanced AI to:

        🔍 **Analyze** your data structure automatically

        🔗 **Identify** hidden relationships and patterns

        📊 **Generate** appropriate visualizations

        💡 **Provide** actionable insights and recommendations

        **Powered by:**
        - 🤖 LangGraph for workflow orchestration
        - ⚡ Groq for lightning-fast AI processing
        - 🎨 Streamlit for beautiful interfaces
        """)

        # Version info
        st.markdown("---")
        st.caption("Dashboard v2.0 | Enhanced Edition")

    # Main content with enhanced tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Upload & Analyze", "📈 Results", "📚 History", "⚙️ Settings"])

    with tab1:
        # Enhanced upload and analysis tab
        st.markdown("## 📊 Data Upload & Analysis")

        # File upload section
        file_uploaded = DataUploader.handle_upload()

        if file_uploaded and api_configured:
            st.markdown("---")

            # Ready to analyze section
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### 🚀 Ready to Analyze!")
                st.write("Your data is loaded and API is configured. Click the button to begin AI-powered analysis.")

                # Show data shape
                if st.session_state.data is not None:
                    shape = st.session_state.data.shape
                    st.info(f"📊 Dataset: {shape[0]:,} rows × {shape[1]} columns")

            with col2:
                if st.button("🚀 Start Analysis", key="start_analysis_main", type="primary"):
                    st.session_state.analysis_running = True
                    st.rerun()

    with tab2:
        # Results tab with enhanced display
        if st.session_state.analysis_running:
            runner = AnalysisRunner()
            runner.run()
            st.session_state.analysis_running = False
            st.rerun()

        ResultsDisplay.show()

    with tab3:
        # Enhanced history tab
        HistoryView.show()

    with tab4:
        # Settings and configuration tab
        st.markdown("## ⚙️ Settings & Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎨 Display Settings")
            # Theme settings would go here
            st.info("Display customization options coming soon!")

        with col2:
            st.markdown("### 🔧 Analysis Settings")
            # Analysis parameter settings would go here
            st.info("Advanced analysis options coming soon!")

        # System info
        st.markdown("### 📊 System Information")
        system_info = {
            "Streamlit Version": st.__version__,
            "Python Version": f"{3}.{8}+",  # Simplified
            "Session ID": str(id(st.session_state))[:8],
            "Cache Status": "Active"
        }

        info_df = pd.DataFrame(list(system_info.items()), columns=['Component', 'Version/Status'])
        st.dataframe(info_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
