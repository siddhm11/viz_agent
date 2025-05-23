from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from typing import Dict, List, TypedDict, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
import io
import base64
import logging
import sys
import traceback
import time
import datetime
import os
import sklearn
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Set up logging configuration
os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_analysis_agent1.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

@dataclass
class DataAnalysisState:
    data: pd.DataFrame
    data_preview: str
    column_types: Dict[str, str]
    variable_relationships: List[Dict]
    suggested_visualizations: List[Dict]
    messages: List
    current_step: str
    visualization_outputs: List[Dict]
    feedback: Optional[str] = None

class DataAnalystAgent:
    def __init__(self, groq_api_key: str):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing DataAnalystAgent")
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="deepseek-r1-distill-llama-70b",
            temperature=0.1
        )
        self.logger.info(f"Initialized LLM with model: deepseek-r1-distill-llama-70b")
        
        self.graph = self._create_workflow()
        self.logger.info("Created workflow graph")
    
    def prepare_data_preview(self, state: DataAnalysisState) -> DataAnalysisState:
        """Generate a preview of the data including head() and info()"""
        try:
            self.logger.info("Starting data preview preparation")
            start_time = time.time()
            
            buffer = io.StringIO()
            
            # Log basic DataFrame info
            self.logger.info(f"DataFrame shape: {state.data.shape}")
            self.logger.info(f"DataFrame columns: {list(state.data.columns)}")
            
            # Capture DataFrame.head()
            buffer.write("## Data Preview (First 15 rows)\n")
            buffer.write(state.data.head(2).to_string())
            buffer.write("\n\n")
            
            # Capture DataFrame.info() 
            buffer.write("## Data Information\n")
            buffer.write("Shape: " + str(state.data.shape) + "\n")
            buffer.write("Columns: " + str(list(state.data.columns)) + "\n")
            buffer.write("Data types:\n")
            for col, dtype in state.data.dtypes.items():
                buffer.write(f"- {col}: {dtype}\n")
            
            # Add basic statistics
            buffer.write("\n## Basic Statistics\n")
            buffer.write(state.data.describe().to_string())
            
            # Store the preview in the state
            state.data_preview = buffer.getvalue()
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Completed data preview preparation in {elapsed_time:.2f} seconds")
            return state
            
        except Exception as e:
            self.logger.error(f"Error in prepare_data_preview: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        
    def analyze_data_types(self, state: DataAnalysisState) -> DataAnalysisState:
        """Analyze and categorize data types of each column"""
        try:
            self.logger.info("Starting data type analysis")
            start_time = time.time()
            
            df = state.data
            self.logger.info(f"Analyzing data types for {len(df.columns)} columns")
            
            analysis_prompt = f"""
            <thinking>
            You are a data analysis expert analyzing a dataset to categorize column types.

            First, examine the dataset preview and data types:
            Dataset Preview:
            {state.data_preview}

            Pandas Data Types:
            {df.dtypes.to_dict()}

            Now, analyze each column systematically:
            1. Look at the pandas dtype
            2. Examine sample values in the preview
            3. Consider the column name for context
            4. Determine the most appropriate category

            Categories to use:
            - "numerical_continuous": Float values, measurements, ratios
            - "numerical_discrete": Integer counts, rankings, IDs
            - "categorical_nominal": Unordered categories (colors, names, types)
            - "categorical_ordinal": Ordered categories (ratings, grades, sizes)
            - "temporal": Dates, times, timestamps
            - "text": Free-form text data
            - "boolean": True/False, Yes/No, binary values
            </thinking>

            Based on your analysis, return ONLY a Python dictionary mapping each column name to its category.
            Example format: {{"column1": "numerical_continuous", "column2": "categorical_nominal"}}

            Dictionary:
            """
            
            self.logger.info("Sending analyze_data_types prompt to LLM")
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            self.logger.info("Received response from LLM for data type analysis")
            
            try:
                # More robust parsing
                import re
                import ast
                
                # Try to extract dictionary from response
                dict_pattern = r'\{[^{}]*\}'
                dict_match = re.search(dict_pattern, response.content)
                
                if dict_match:
                    dict_str = dict_match.group(0)
                    state.column_types = ast.literal_eval(dict_str)
                    self.logger.info(f"Parsed column types: {state.column_types}")
                else:
                    raise ValueError("No dictionary found in response")
                    
            except Exception as parse_error:
                self.logger.error(f"Error parsing column types: {str(parse_error)}")
                self.logger.error(f"Raw LLM response: {response.content}")
                # Create a basic fallback categorization
                state.column_types = {}
                for col, dtype in df.dtypes.items():
                    if 'int' in str(dtype) or 'float' in str(dtype):
                        state.column_types[col] = 'numerical_continuous'
                    elif 'datetime' in str(dtype):
                        state.column_types[col] = 'temporal'
                    else:
                        state.column_types[col] = 'categorical_nominal'
                self.logger.info(f"Using fallback column types: {state.column_types}")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Completed data type analysis in {elapsed_time:.2f} seconds")
            return state
            
        except Exception as e:
            self.logger.error(f"Error in analyze_data_types: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def identify_relationships(self, state: DataAnalysisState) -> DataAnalysisState:
        """Identify potential relationships between variables"""
        try:
            self.logger.info("Starting relationship identification")
            start_time = time.time()
            
            relationship_prompt = f"""
            <thinking>
            As a data analysis expert, I need to identify meaningful relationships in this dataset.

            Dataset Information:
            {state.data_preview}

            Column Types:
            {state.column_types}

            Let me systematically identify relationships:

            1. Numerical-Numerical correlations: Look for pairs that might be correlated
            2. Categorical-Numerical comparisons: Categories that might explain numerical variations
            3. Temporal patterns: Time-based trends or seasonal effects
            4. Grouping effects: How categories might segment the data
            5. Causal relationships: Variables that might influence others

            For each potential relationship, I'll consider:
            - Statistical significance likelihood
            - Business/domain logic
            - Data exploration value
            </thinking>

            Return a JSON list of relationship dictionaries. Each should have:
            - "variables": [list of 2-3 column names]
            - "relationship_type": one of ["correlation", "comparison", "temporal_trend", "grouping", "distribution_analysis"]
            - "hypothesis": concise explanation of expected relationship
            - "priority": "high", "medium", or "low" based on likely insights

            JSON Array:
            """
            
            self.logger.info("Sending identify_relationships prompt to LLM")
            response = self.llm.invoke([HumanMessage(content=relationship_prompt)])
            self.logger.info("Received response from LLM for relationship identification")
            
            # Extract the list from the response
            import re
            import json
            
            try:
                # Try to find list content in the response
                list_match = re.search(r'\[\s*\{.*\}\s*\]', response.content, re.DOTALL)
                if list_match:
                    state.variable_relationships = json.loads(list_match.group(0))
                    self.logger.info(f"Identified {len(state.variable_relationships)} potential relationships")
                else:
                    # Fallback if parsing fails
                    self.logger.warning("Could not extract relationships list from LLM response. Using empty list.")
                    self.logger.debug(f"Raw LLM response: {response.content}")
                    state.variable_relationships = []
            except Exception as parse_error:
                self.logger.error(f"Error parsing relationships: {str(parse_error)}")
                self.logger.error(f"Raw LLM response: {response.content}")
                state.variable_relationships = []
            
            # Log identified relationships
            for i, rel in enumerate(state.variable_relationships):
                self.logger.info(f"Relationship {i+1}: {rel.get('variables', [])} - {rel.get('relationship_type', 'unknown')}")
                
            elapsed_time = time.time() - start_time
            self.logger.info(f"Completed relationship identification in {elapsed_time:.2f} seconds")
            return state
            
        except Exception as e:
            self.logger.error(f"Error in identify_relationships: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def suggest_visualizations(self, state: DataAnalysisState) -> DataAnalysisState:
        """Suggest appropriate visualizations based on data types and relationships"""
        try:
            self.logger.info("Starting visualization suggestions")
            start_time = time.time()
            
            viz_prompt = f"""
            <thinking>
            I need to suggest optimal visualizations based on the data characteristics.

            Data Preview:
            {state.data_preview}

            Column Types: {state.column_types}
            Identified Relationships: {state.variable_relationships}

            Let me apply visualization selection logic:

            For each variable type combination:
            - Single numerical → histogram (distribution), boxplot (outliers), density plot
            - Single categorical → bar chart (frequency), pie chart (if <8 categories)
            - Numerical vs Numerical → scatter plot (correlation), line plot (if temporal)
            - Numerical vs Categorical → box plot (distributions), violin plot (detailed distributions)
            - Categorical vs Categorical → stacked bar chart, heatmap (if not too sparse)
            - Temporal data → time series plot, seasonal decomposition

            Priority factors:
            1. Data size and complexity
            2. Likely insights from relationships identified
            3. Statistical appropriateness
            4. Visual clarity
            </thinking>

            Return a JSON array of visualization suggestions:
            [
            {{
                "chart_type": "specific_chart_name",
                "variables_to_use": ["column1", "column2"],
                "reasoning": "why this chart reveals important insights",
                "priority": "high/medium/low",
                "expected_insight": "what pattern or relationship we expect to see"
            }}
            ]

            JSON Array:
            """

            response = self.llm.invoke([HumanMessage(content=viz_prompt)])
            
            # Parse the response to extract valid JSON
            try:
                import re
                import json
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response.content, re.DOTALL)
                if json_match:
                    state.suggested_visualizations = json.loads(json_match.group(0))
                    self.logger.info(f"Generated {len(state.suggested_visualizations)} visualization suggestions")
                else:
                    # Fallback to a simple visualization if parsing fails
                    numerical_cols = [col for col, dtype in state.column_types.items() 
                                    if 'numerical' in dtype]
                    categorical_cols = [col for col, dtype in state.column_types.items() 
                                      if 'categorical' in dtype]
                    
                    fallback_viz = []
                    if numerical_cols:
                        fallback_viz.append({
                            "chart_type": "histogram",
                            "variables_to_use": [numerical_cols[0]],
                            "reasoning": "Fallback visualization - distribution analysis",
                            "priority": "medium"
                        })
                    if categorical_cols:
                        fallback_viz.append({
                            "chart_type": "bar_chart",
                            "variables_to_use": [categorical_cols[0]],
                            "reasoning": "Fallback visualization - frequency analysis",
                            "priority": "medium"
                        })
                    
                    state.suggested_visualizations = fallback_viz or [{
                        "chart_type": "histogram",
                        "variables_to_use": [list(state.data.columns)[0]],
                        "reasoning": "Fallback visualization due to parsing error",
                        "priority": "low"
                    }]
                    
            except Exception as e:
                self.logger.error(f"Error parsing visualization suggestions: {e}")
                state.suggested_visualizations = [{
                    "chart_type": "histogram",
                    "variables_to_use": [list(state.data.columns)[0]],
                    "reasoning": "Fallback visualization due to parsing error",
                    "priority": "low"
                }]
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Completed visualization suggestions in {elapsed_time:.2f} seconds")
            return state
            
        except Exception as e:
            self.logger.error(f"Error in suggest_visualizations: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    


    def create_visualization(self, state: DataAnalysisState) -> DataAnalysisState:
        """Create visualizations by having the LLM dynamically generate the plotting code"""
        self.logger.info("="*60)
        self.logger.info("STARTING DYNAMIC LLM-BASED VISUALIZATION CREATION")
        self.logger.info("="*60)
        
        state.visualization_outputs = []

        # First, create a comprehensive data summary
        try:
            self.logger.info("Creating comprehensive data summary...")
            data_summary = self._create_enhanced_data_summary(state.data, state.column_types)
            self.logger.info(f"Data summary created successfully with {len(data_summary['columns'])} columns")
            self.logger.debug(f"Data summary: {data_summary}")
        except Exception as e:
            self.logger.error(f"Failed to create data summary: {str(e)}")
            self.logger.error(traceback.format_exc())
            return state

        self.logger.info(f"Processing {len(state.suggested_visualizations)} suggested visualizations")
        
        for viz_idx, viz in enumerate(state.suggested_visualizations):
            self.logger.info("-" * 50)
            self.logger.info(f"PROCESSING VISUALIZATION {viz_idx + 1}/{len(state.suggested_visualizations)}")
            self.logger.info("-" * 50)
            
            chart_type = viz.get('chart_type', 'unknown').lower()
            variables = viz.get('variables_to_use', [])
            reasoning = viz.get('reasoning', 'No reasoning provided')
            priority = viz.get('priority', 'unknown')
            
            self.logger.info(f"Chart Type: {chart_type}")
            self.logger.info(f"Variables: {variables}")
            self.logger.info(f"Priority: {priority}")
            self.logger.info(f"Reasoning: {reasoning}")
            
            # Validate variables exist in dataset
            missing_vars = [var for var in variables if var not in state.data.columns]
            if missing_vars:
                self.logger.error(f"Missing variables in dataset: {missing_vars}")
                self.logger.error(f"Available columns: {list(state.data.columns)}")
                error_result = self._create_error_result(
                    f"Variables {missing_vars} not found in dataset", 
                    chart_type, variables
                )
                state.visualization_outputs.append(error_result)
                continue

            try:
                # 1. Get visualization explanation
                self.logger.info("Step 1: Getting visualization explanation...")
                explanation = self._get_visualization_explanation(chart_type, variables, reasoning)
                self.logger.info(f"Explanation generated: {explanation[:100]}...")
                
                # 2. Generate and execute visualization with healing
                self.logger.info("Step 2: Generating visualization with self-healing...")
                result = self._generate_visualization_with_healing(
                    chart_type, variables, data_summary, state.data, viz_idx
                )
                
                # 3. Compile final result
                final_result = {
                    'chart_type': chart_type,
                    'variables': variables,
                    'reasoning': reasoning,
                    'priority': priority,
                    'explanation': explanation,
                    'insights': result.get('insights', 'No insights generated'),
                    'image_base64': result.get('image_base64'),
                    'generated_code': result.get('final_code'),
                    'healing_history': result.get('healing_history', []),
                    'success': result.get('success', False),
                    'generation_stats': result.get('stats', {}),
                    'viz_index': viz_idx + 1
                }
                
                state.visualization_outputs.append(final_result)
                
                # Log success/failure
                if final_result['success']:
                    self.logger.info(f"✅ Visualization {viz_idx + 1} completed successfully")
                    if final_result['healing_history']:
                        self.logger.info(f"⚕️  Required {len(final_result['healing_history'])} healing attempts")
                else:
                    self.logger.error(f"❌ Visualization {viz_idx + 1} failed")
                    self.logger.error(f"Final error: {final_result['insights']}")
                    
            except Exception as e:
                self.logger.error(f"💥 Critical error processing visualization {viz_idx + 1}: {str(e)}")
                self.logger.error(traceback.format_exc())
                
                error_result = self._create_error_result(
                    f"Critical error: {str(e)}", chart_type, variables
                )
                error_result.update({
                    'viz_index': viz_idx + 1,
                    'priority': priority,
                    'reasoning': reasoning
                })
                state.visualization_outputs.append(error_result)

        # Final summary
        successful = sum(1 for viz in state.visualization_outputs if viz.get('success', False))
        total = len(state.visualization_outputs)
        
        self.logger.info("="*60)
        self.logger.info("VISUALIZATION CREATION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total visualizations processed: {total}")
        self.logger.info(f"Successful visualizations: {successful}")
        self.logger.info(f"Failed visualizations: {total - successful}")
        self.logger.info(f"Success rate: {(successful/total*100):.1f}%" if total > 0 else "0%")
        
        return state

    def _create_enhanced_data_summary(self, data, column_types):
        """Create a comprehensive data summary for the LLM"""
        self.logger.info("Creating enhanced data summary...")
        
        try:
            summary = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "column_types": column_types,
                "memory_usage": data.memory_usage(deep=True).sum(),
                "missing_data": data.isnull().sum().to_dict(),
                "sample_values": {},
                "unique_counts": {},
                "numeric_ranges": {},
                "categorical_info": {}
            }
            
            # Enhanced sample values and statistics
            for col in data.columns:
                try:
                    col_data = data[col].dropna()
                    
                    if len(col_data) == 0:
                        summary["sample_values"][col] = []
                        summary["unique_counts"][col] = 0
                        continue
                        
                    # Sample values
                    sample_size = min(5, len(col_data))
                    summary["sample_values"][col] = col_data.sample(sample_size).tolist()
                    summary["unique_counts"][col] = col_data.nunique()
                    
                    # Type-specific information
                    if pd.api.types.is_numeric_dtype(col_data):
                        summary["numeric_ranges"][col] = {
                            "min": float(col_data.min()),
                            "max": float(col_data.max()),
                            "mean": float(col_data.mean()),
                            "std": float(col_data.std()) if len(col_data) > 1 else 0
                        }
                    elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
                        value_counts = col_data.value_counts().head(10)
                        summary["categorical_info"][col] = {
                            "top_values": value_counts.to_dict(),
                            "unique_count": col_data.nunique()
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Error processing column {col}: {str(e)}")
                    summary["sample_values"][col] = ["Error getting samples"]
                    summary["unique_counts"][col] = -1
            
            self.logger.info(f"Enhanced data summary created with {len(summary)} main sections")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced data summary: {str(e)}")
            # Fallback to basic summary
            return {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "column_types": column_types
            }

    def _generate_visualization_with_healing(self, chart_type, variables, data_summary, data, viz_index):
        """Generate visualization with comprehensive self-healing capabilities"""
        self.logger.info(f"Starting visualization generation with healing for {chart_type}")
        
        generation_start_time = time.time()
        stats = {
            "attempts": 0,
            "healing_attempts": 0,
            "total_time": 0,
            "code_generation_time": 0,
            "execution_time": 0
        }
        
        # Generate initial code with detailed logging
        self.logger.info("Phase 1: Generating initial plotting code...")
        code_gen_start = time.time()
        
        initial_code = self._generate_plotting_code_with_logging(
            chart_type, variables, data_summary, viz_index
        )
        
        stats["code_generation_time"] = time.time() - code_gen_start
        
        if not initial_code:
            self.logger.error("❌ Failed to generate initial plotting code")
            stats["total_time"] = time.time() - generation_start_time
            return self._create_error_result(
                "Failed to generate initial plotting code", 
                chart_type, variables, stats
            )
        
        self.logger.info("✅ Initial code generated successfully")
        self.logger.debug(f"Generated code length: {len(initial_code)} characters")
        
        # Execute with healing
        self.logger.info("Phase 2: Executing code with self-healing...")
        exec_start = time.time()
        
        result = self._execute_with_enhanced_healing(initial_code, data, chart_type, stats)
        
        stats["execution_time"] = time.time() - exec_start
        stats["total_time"] = time.time() - generation_start_time
        
        result["stats"] = stats
        
        self.logger.info(f"Visualization generation completed in {stats['total_time']:.2f}s")
        self.logger.info(f"Total attempts: {stats['attempts']}, Healing attempts: {stats['healing_attempts']}")
        
        return result

    def _generate_plotting_code_with_logging(self, chart_type, variables, data_summary, viz_index):
        """Generate initial plotting code with detailed logging"""
        self.logger.info(f"Generating code for {chart_type} visualization of {variables}")
        
        # Log data context for the LLM
        self.logger.debug("Data context for LLM:")
        self.logger.debug(f"  - Data shape: {data_summary['shape']}")
        self.logger.debug(f"  - Variables to plot: {variables}")
        self.logger.debug(f"  - Variable types: {[data_summary['column_types'].get(var) for var in variables]}")
        self.logger.debug(f"  - Missing data: {[data_summary['missing_data'].get(var, 0) for var in variables]}")
        
        code_prompt = self._create_enhanced_code_prompt(chart_type, variables, data_summary)
        
        try:
            self.logger.info("Sending code generation prompt to LLM...")
            
            # Log the prompt for debugging
            log_file = f"logs/code_prompts_viz_{viz_index}.log"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("=== CODE GENERATION PROMPT ===\n")
                f.write(f"Chart Type: {chart_type}\n")
                f.write(f"Variables: {variables}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")
                f.write("PROMPT:\n")
                f.write(code_prompt)
                f.write("\n\n")
            
            response = self.llm.invoke([HumanMessage(content=code_prompt)])
            
            # Log the response
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("=== LLM RESPONSE ===\n")
                f.write(response.content)
                f.write("\n\n")
            
            self.logger.info("✅ Received response from LLM")
            
            extracted_code = self._extract_code_from_response(response.content)
            
            if extracted_code:
                # Log the extracted code
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("=== EXTRACTED CODE ===\n")
                    f.write(extracted_code)
                    f.write("\n\n")
                
                self.logger.info("✅ Successfully extracted code from LLM response")
                self.logger.debug(f"Code preview: {extracted_code[:200]}...")
                return extracted_code
            else:
                self.logger.error("❌ Failed to extract code from LLM response")
                self.logger.debug(f"Raw response: {response.content[:500]}...")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error generating code: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _create_enhanced_code_prompt(self, chart_type, variables, data_summary):
        """Create a comprehensive code generation prompt"""
        
        # Get variable-specific information
        var_info = []
        for var in variables:
            info = {
                "name": var,
                "type": data_summary['column_types'].get(var, 'unknown'),
                "dtype": data_summary['dtypes'].get(var, 'unknown'),
                "missing": data_summary['missing_data'].get(var, 0),
                "unique": data_summary['unique_counts'].get(var, 0),
                "samples": data_summary['sample_values'].get(var, [])
            }
            
            if var in data_summary.get('numeric_ranges', {}):
                info.update(data_summary['numeric_ranges'][var])
            elif var in data_summary.get('categorical_info', {}):
                info.update(data_summary['categorical_info'][var])
                
            var_info.append(info)
        
        prompt = f"""<thinking>
    I need to generate robust, error-free Python code for a {chart_type} visualization.

    Dataset Context:
    - Shape: {data_summary['shape']}
    - Total columns: {len(data_summary['columns'])}

    Variables to visualize: {variables}
    Variable details:
    {chr(10).join([f"  - {var['name']}: {var['type']} ({var['dtype']}) - {var['missing']} missing, {var['unique']} unique" for var in var_info])}

    Sample values:
    {chr(10).join([f"  - {var['name']}: {var['samples']}" for var in var_info])}

    Critical Requirements:
    1. Handle ALL edge cases (empty data, missing values, wrong types)
    2. Use ONLY standard libraries (matplotlib, seaborn, pandas, numpy)
    3. NO deprecated parameters (use 'errorbar=None' not 'ci=None')
    4. NO external libraries like sklearn
    5. Proper data type validation and conversion
    6. Clear, publication-ready visualization
    7. Meaningful statistical insights
    8. Robust error handling
    9. Modern styling and appropriate figure size

    Chart-specific considerations for {chart_type}:
    - Validate that variables exist and have appropriate data types
    - Handle categorical variables with many levels appropriately
    - Ensure numerical variables are actually numeric
    - Apply appropriate statistical transformations if needed
    - Consider data distribution and outliers
    </thinking>

    Generate a complete, robust Python function for a {chart_type} visualization.

    STRICT REQUIREMENTS:
    - Function signature: def generate_plot(data):
    - Return: (base64_image_string, insights_text, success_boolean)
    - Include comprehensive data validation
    - Handle edge cases gracefully
    - Use modern matplotlib/seaborn (avoid deprecated parameters)
    - Generate meaningful insights based on what's visible in the plot
    - Professional styling with appropriate figure size

    Variables: {variables}
    Data types: {[var['dtype'] for var in var_info]}
    Column types: {[var['type'] for var in var_info]}

    RETURN ONLY THE COMPLETE PYTHON FUNCTION:

    ```python
    def generate_plot(data):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        import io
        import base64
        import warnings
        warnings.filterwarnings('ignore')
        
        try:
            # Comprehensive data validation
            required_cols = {variables}
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                return None, f"Missing columns: {{missing_cols}}", False
            
            # Data preparation with type checking
            # ... your complete implementation here ...
            
            # Create the visualization
            # ... plotting code here ...
            
            # Generate insights
            insights = "Detailed analysis of patterns observed in the visualization..."
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close()
            
            return img_str, insights, True
            
        except Exception as e:
            import traceback
            error_details = f"Error: {{str(e)}}\\nTraceback: {{traceback.format_exc()}}"
            return None, error_details, False
    ```"""
        
        return prompt

    def _execute_with_enhanced_healing(self, code, data, chart_type, stats, max_attempts=4):
        """Execute code with enhanced self-healing capabilities"""
        self.logger.info(f"Starting enhanced healing execution (max {max_attempts} attempts)")
        
        healing_history = []
        current_code = code
        
        for attempt in range(max_attempts):
            stats["attempts"] += 1
            attempt_start = time.time()
            
            self.logger.info(f"🔄 Attempt {attempt + 1}/{max_attempts} for {chart_type}")
            
            try:
                # Execute the code with detailed logging
                result = self._safe_execute_code_with_logging(current_code, data, attempt + 1)
                
                attempt_time = time.time() - attempt_start
                self.logger.info(f"⏱️ Attempt {attempt + 1} completed in {attempt_time:.2f}s")
                
                if result['success']:
                    self.logger.info(f"✅ SUCCESS on attempt {attempt + 1}")
                    return {
                        'image_base64': result['image_base64'],
                        'insights': result['insights'],
                        'success': True,
                        'final_code': current_code,
                        'healing_history': healing_history
                    }
                else:
                    # Code executed but reported failure
                    error_msg = result['error']
                    self.logger.warning(f"⚠️ Code executed but reported failure: {error_msg}")
                    
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                traceback_str = traceback.format_exc()
                
                self.logger.error(f"💥 Exception on attempt {attempt + 1}: {error_msg}")
                self.logger.debug(f"Full traceback:\n{traceback_str}")
                
            # If we're not on the last attempt, try healing
            if attempt < max_attempts - 1:
                stats["healing_attempts"] += 1
                
                self.logger.info(f"🔧 Starting healing attempt {stats['healing_attempts']}...")
                
                # Record healing attempt
                healing_entry = {
                    'attempt': attempt + 1,
                    'error': error_msg,
                    'traceback': traceback_str if 'traceback_str' in locals() else "No traceback",
                    'code_before': current_code,
                    'healing_start_time': time.time()
                }
                
                # Generate corrected code
                corrected_code = self._get_corrected_code_with_logging(
                    current_code, error_msg, 
                    traceback_str if 'traceback_str' in locals() else None,
                    attempt + 1
                )
                
                healing_entry['healing_time'] = time.time() - healing_entry['healing_start_time']
                
                if corrected_code and corrected_code != current_code:
                    current_code = corrected_code
                    healing_entry['code_after'] = current_code
                    healing_entry['healing_success'] = True
                    healing_history.append(healing_entry)
                    
                    self.logger.info(f"✅ Generated corrected code (healing attempt {stats['healing_attempts']})")
                    self.logger.debug(f"Code diff length: {abs(len(corrected_code) - len(healing_entry['code_before']))}")
                else:
                    healing_entry['healing_success'] = False
                    healing_history.append(healing_entry)
                    
                    self.logger.error(f"❌ Failed to generate corrected code (healing attempt {stats['healing_attempts']})")
                    break
        
        # If we've reached max attempts without success
        self.logger.error(f"❌ Failed to create {chart_type} visualization after {max_attempts} attempts")
        self.logger.error(f"Total healing attempts: {stats['healing_attempts']}")
        
        return self._create_error_result(
            f"Failed after {max_attempts} attempts and {stats['healing_attempts']} healing attempts", 
            chart_type, variables, stats, healing_history
        )

    def _safe_execute_code_with_logging(self, code, data, attempt_num):
        """Safely execute the plotting code with comprehensive logging"""
        self.logger.debug(f"Executing code for attempt {attempt_num}")
        
        try:
            # Create isolated namespace with comprehensive imports
            namespace = {
                '__builtins__': __builtins__,
                'matplotlib': __import__('matplotlib'),
                'plt': plt,
                'sns': sns,
                'pd': pd,
                'np': __import__('numpy'),
                'io': io,
                'base64': base64,
                'warnings': __import__('warnings'),
                'traceback': traceback,
                'datetime': datetime
            }
            
            # Execute the function definition
            self.logger.debug("Executing function definition...")
            exec(code, namespace)
            
            # Validate function exists
            if 'generate_plot' not in namespace:
                raise ValueError("Function 'generate_plot' not found in executed code")
            
            # Call the function with logging
            self.logger.debug("Calling generate_plot function...")
            function_start = time.time()
            
            img_str, insights, success = namespace['generate_plot'](data)
            
            function_time = time.time() - function_start
            self.logger.debug(f"Function execution completed in {function_time:.2f}s")
            
            # Validate return values
            if success:
                if not img_str:
                    self.logger.warning("Success reported but no image string returned")
                    success = False
                elif not isinstance(img_str, str):
                    self.logger.warning(f"Image string is not a string: {type(img_str)}")
                    success = False
                
                if not insights:
                    insights = "Visualization created successfully but no insights generated"
                    self.logger.warning("No insights returned")
            
            self.logger.debug(f"Function returned: success={success}, img_len={len(img_str) if img_str else 0}, insights_len={len(insights) if insights else 0}")
            
            return {
                'image_base64': img_str,
                'insights': insights,
                'success': success,
                'error': insights if not success else None,
                'execution_time': function_time
            }
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Exception during code execution: {error_msg}")
            self.logger.debug(f"Exception type: {type(e).__name__}")
            
            return {
                'image_base64': None,
                'insights': None,
                'success': False,
                'error': error_msg,
                'exception_type': type(e).__name__
            }

    def _get_corrected_code_with_logging(self, code, error_msg, traceback_str, attempt_num):
        """Get corrected code from LLM with comprehensive logging"""
        self.logger.info(f"Requesting code correction for attempt {attempt_num}")
        self.logger.debug(f"Error to fix: {error_msg}")
        
        # Create enhanced healing prompt
        healing_prompt = f"""URGENT: Fix this broken Python visualization code.

    ATTEMPT NUMBER: {attempt_num}
    ERROR ENCOUNTERED: {error_msg}

    {f'FULL TRACEBACK:\n{traceback_str}\n' if traceback_str else ''}

    PROBLEMATIC CODE:
    ```python
    {code}
    ```

    COMMON FIXES NEEDED:
    1. Seaborn 'ci' parameter → use 'errorbar=None' instead
    2. Missing imports → add required imports
    3. Data type issues → add proper validation and conversion
    4. Empty data → add data checks
    5. Column access errors → validate column existence
    6. Plotting parameter errors → use correct parameter names
    7. Memory/resource issues → add proper cleanup

    REQUIREMENTS:
    - Fix the specific error mentioned above
    - Maintain the same function signature: def generate_plot(data):
    - Return: (base64_string, insights_text, success_boolean)
    - Keep all existing functionality that works
    - Add better error handling
    - Ensure robust data validation

    RETURN ONLY THE CORRECTED PYTHON FUNCTION:

    ```python
    def generate_plot(data):
        # Fixed code here
    ```"""
        
        try:
            # Log the healing prompt
            log_file = f"logs/healing_attempt_{attempt_num}.log"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("=== HEALING PROMPT ===\n")
                f.write(f"Attempt: {attempt_num}\n")
                f.write(f"Error: {error_msg}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")
                f.write("PROMPT:\n")
                f.write(healing_prompt)
                f.write("\n\n")
            
            self.logger.info(f"Sending healing prompt to LLM (attempt {attempt_num})...")
            response = self.llm.invoke([HumanMessage(content=healing_prompt)])
            
            # Log the response
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("=== LLM HEALING RESPONSE ===\n")
                f.write(response.content)
                f.write("\n\n")
            
            corrected_code = self._extract_code_from_response(response.content)
            
            if corrected_code:
                # Log the corrected code
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("=== CORRECTED CODE ===\n")
                    f.write(corrected_code)
                    f.write("\n\n")
                
                self.logger.info(f"✅ Successfully generated corrected code (attempt {attempt_num})")
                return corrected_code
            else:
                self.logger.error(f"❌ Failed to extract corrected code (attempt {attempt_num})")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting corrected code (attempt {attempt_num}): {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _create_error_result(self, error_message, chart_type, variables, stats=None, healing_history=None):
        """Create comprehensive error result with visualization"""
        self.logger.info(f"Creating error result: {error_message}")
        
        try:
            # Create informative error visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Error message
            ax.text(0.5, 0.7, "Visualization Generation Failed", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=20, fontweight='bold', color='darkred',
                    transform=ax.transAxes)
            
            ax.text(0.5, 0.5, f"Chart Type: {chart_type}\nVariables: {', '.join(variables)}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14, color='black',
                    transform=ax.transAxes)
            
            ax.text(0.5, 0.3, f"Error: {error_message}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red', wrap=True,
                    transform=ax.transAxes)
            
            if stats:
                stats_text = f"Attempts: {stats.get('attempts', 0)} | Healing: {stats.get('healing_attempts', 0)} | Time: {stats.get('total_time', 0):.1f}s"
                ax.text(0.5, 0.1, stats_text, 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=10, color='gray',
                        transform=ax.transAxes)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Save as base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close()
            
            return {
                'image_base64': img_str,
                'insights': f"Error: {error_message}",
                'success': False,
                'final_code': None,
                'healing_history': healing_history or [],
                'stats': stats or {},
                'chart_type': chart_type,
                'variables': variables
            }
            
        except Exception as e:
            self.logger.error(f"Error creating error visualization: {str(e)}")
            return {
                'image_base64': None,
                'insights': f"Critical Error: {error_message}",
                'success': False,
                'final_code': None,
                'healing_history': healing_history or [],
                'stats': stats or {},
                'chart_type': chart_type,
                'variables': variables
            }

            
    def _get_visualization_explanation(self, chart_type, variables, reasoning):
        """Get explanation of visualization purpose from LLM"""
        explain_prompt = f"""
        <thinking>
        I need to explain the analytical purpose of this {chart_type} visualization using variables {variables}.

        Context:
        - Chart type: {chart_type}
        - Variables: {variables}
        - Reasoning: {reasoning}

        What insights can this visualization provide:
        1. What patterns might be visible?
        2. What questions does it answer?
        3. What business/analytical value does it offer?
        4. What should viewers focus on?
        </thinking>

        As a data visualization expert, explain the analytical purpose of this {chart_type} chart using {variables}.

        Address:
        1. What specific insights this visualization reveals
        2. What patterns or relationships viewers should look for
        3. How this contributes to overall data understanding

        Keep response to 2-3 clear, actionable sentences focused on analytical value.

        Explanation:
        """
        try:
            explanation_resp = self.llm.invoke([HumanMessage(content=explain_prompt)])
            explanation = explanation_resp.content.strip()
            
            # Log explanation
            log_path = "logs/explanations.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("=== New Explanation ===\n")
                f.write("Prompt:\n")
                f.write(explain_prompt.strip() + "\n\n")
                f.write("Explanation:\n")
                f.write(explanation + "\n\n")
            
            return explanation
        except Exception as e:
            self.logger.error(f"Error getting explanation: {str(e)}")
            return f"This {chart_type} visualization shows relationships between {', '.join(variables)}."

    def _extract_code_from_response(self, response_content):
        """Extract Python code from LLM response"""
        import re
        
        # Try to find code block
        patterns = [
            r'```python\s*(.*?)\s*```',
            r'```\s*(def generate_plot.*?)\s*```',
            r'(def generate_plot\(data\):.*?)(?=\n\n|\n```|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_content, re.DOTALL)
            if match:
                code = match.group(1).strip()
                
                # Ensure function definition is present
                if not code.startswith("def generate_plot"):
                    code = "def generate_plot(data):\n" + code
                
                return code
        
        # If no pattern matches, try to extract everything after "def generate_plot"
        if "def generate_plot" in response_content:
            start_idx = response_content.find("def generate_plot")
            code = response_content[start_idx:].strip()
            
            # Clean up common artifacts
            code = re.sub(r'```.*$', '', code, flags=re.MULTILINE).strip()
            return code
        
        return None

    def _execute_with_healing(self, initial_code, data, chart_type, max_attempts=3):
        """Execute code with self-healing capabilities"""
        healing_history = []
        current_code = initial_code
        
        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Attempt {attempt + 1} for {chart_type} visualization")
                
                # Execute the code
                result = self._safe_execute_code(current_code, data)
                
                if result['success']:
                    return {
                        'image_base64': result['image_base64'],
                        'insights': result['insights'],
                        'success': True,
                        'final_code': current_code,
                        'healing_history': healing_history
                    }
                else:
                    # Code executed but reported failure
                    error_msg = result['error']
                    self.logger.warning(f"Code reported failure on attempt {attempt + 1}: {error_msg}")
                    
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                traceback_str = traceback.format_exc()
                self.logger.error(f"Exception on attempt {attempt + 1}: {error_msg}")
                
                # Record healing attempt
                healing_entry = {
                    'attempt': attempt + 1,
                    'error': error_msg,
                    'traceback': traceback_str,
                    'code_before': current_code
                }
                
                # Generate corrected code
                corrected_code = self._get_corrected_code(current_code, error_msg, traceback_str)
                
                if corrected_code:
                    current_code = corrected_code
                    healing_entry['code_after'] = current_code
                    healing_history.append(healing_entry)
                else:
                    self.logger.error("Failed to generate corrected code")
                    break
        
        # If we've reached max attempts without success
        self.logger.error(f"Failed to create {chart_type} visualization after {max_attempts} attempts")
        return self._create_error_result(f"Failed after {max_attempts} healing attempts", healing_history)

    def _safe_execute_code(self, code, data):
        """Safely execute the plotting code"""
        try:
            # Create isolated namespace
            import numpy as np
            
            namespace = {
                '__builtins__': __builtins__,
                'matplotlib': __import__('matplotlib'),
                'plt': plt,
                'sns': sns,
                'pd': pd,
                'np': np,
                'io': io,
                'base64': base64,
                'warnings': __import__('warnings')
            }
            
            # Log the code that is about to be executed
            self.logger.debug(f"Attempting to execute code:\n{code}")
            
            # Execute the function definition
            exec(code, namespace)
            
            # Call the function
            img_str, insights, success = namespace['generate_plot'](data)
            
            return {
                'image_base64': img_str,
                'insights': insights,
                'success': success,
                'error': insights if not success else None
            }
            
        except Exception as e:
            # Log the code that caused the exception
            self.logger.error(f"Exception during _safe_execute_code. Error: {str(e)}. Problematic code:\n{code}")
            return {
                'image_base64': None,
                'insights': None,
                'success': False,
                'error': str(e)
            }

    def _get_corrected_code(self, code, error_msg, traceback_str=None):
        """Get corrected code from LLM based on error"""
        healing_prompt = f"""
        Fix this Python visualization code that encountered an error.
        
        ERROR: {error_msg}
        
        {f'TRACEBACK:\n{traceback_str}\n' if traceback_str else ''}
        
        PROBLEMATIC CODE:
        ```python
        {code}
        ```
        
        Common fixes needed:
        1. For 'ci' parameter errors: Replace 'ci=None' with 'errorbar=None' in seaborn plots
        2. For import errors: Remove or replace unavailable libraries (like sklearn , I have sklearn now tho )
        3. For data type errors: Add proper type checking and conversion
        4. For empty data errors: Add data validation checks
        5. For 'PathCollection.set()' errors: This usually means passing wrong parameters to scatter plots
        
        Return ONLY the corrected Python function code without explanations:
        
        ```python
        def generate_plot(data):
            # corrected code here
        ```
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=healing_prompt)])
            return self._extract_code_from_response(response.content)
        except Exception as e:
            self.logger.error(f"Error getting corrected code: {str(e)}")
            return None

    def _log_healing_process(self, chart_type, healing_history):
        """Log the healing process for analysis"""
        try:
            log_path = "logs/code_healing_history.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("=== New Code Healing Entry ===\n")
                f.write(f"Timestamp: {datetime.datetime.utcnow().isoformat()}Z\n")
                f.write(f"Chart Type: {chart_type}\n\n")
                
                for i, attempt in enumerate(healing_history):
                    f.write(f"--- Attempt {attempt['attempt']} ---\n")
                    f.write(f"Error: {attempt['error']}\n")
                    if 'traceback' in attempt:
                        f.write(f"Traceback:\n{attempt['traceback']}\n")
                    f.write("\nCode Before:\n")
                    f.write("```python\n")
                    f.write(attempt['code_before'].strip() + "\n")
                    f.write("```\n\n")
                    if 'code_after' in attempt:
                        f.write("Code After:\n")
                        f.write("```python\n")
                        f.write(attempt['code_after'].strip() + "\n")
                        f.write("```\n\n")
                f.write("\n" + "="*50 + "\n\n")
        except Exception as e:
            self.logger.error(f"Error logging healing process: {str(e)}")
            
    def evaluate_results(self, state: DataAnalysisState) -> DataAnalysisState:
        """Have the LLM evaluate visualization results and suggest improvements"""
        
        # Create a description of the visualizations for the LLM
        viz_descriptions = []
        for i, viz in enumerate(state.visualization_outputs):
            desc = f"Visualization {i+1}: {viz['chart_type'].title()} of {', '.join(viz['variables'])}"
            if 'error' in viz:
                desc += f" (Error: {viz['error']})"
            else:
                desc += f" (Reasoning: {viz['reasoning']})"
            viz_descriptions.append(desc)
        
        evaluation_prompt = f"""
        <thinking>
        I need to evaluate the visualization results comprehensively.

        Dataset context:
        {state.data_preview}

        Column types: {state.column_types}
        Relationships identified: {state.variable_relationships}

        Visualizations created:
        {chr(10).join(viz_descriptions)}

        Evaluation framework:
        1. Coverage: Do visualizations address key data aspects?
        2. Appropriateness: Are chart types optimal for data types?
        3. Insights: What patterns are revealed or missed?
        4. Gaps: What additional analysis would be valuable?
        5. Quality: Are there technical or interpretive issues?
        </thinking>

        ## Data Visualization Analysis Report

        ### Executive Summary
        Provide a 2-3 sentence overview of the visualization quality and key findings.

        ### Visualization Assessment
        For each visualization created:
        1. **Effectiveness**: How well does it serve its analytical purpose?
        2. **Technical Quality**: Any issues with implementation or display?
        3. **Insights Revealed**: What patterns or relationships are visible?

        ### Gap Analysis
        1. **Missing Visualizations**: What important aspects weren't covered?
        2. **Alternative Approaches**: Better ways to reveal the same insights?
        3. **Additional Recommendations**: Next steps for deeper analysis?

        ### Key Conclusions
        Summarize the most important findings and actionable insights from the visualizations.

        ### Recommended Actions
        Suggest 2-3 specific next steps for further analysis.

        Report:
        """
        
        response = self.llm.invoke([HumanMessage(content=evaluation_prompt)])
        state.feedback = response.content
        print("LLM Feedback:")
        print(state.feedback)
        # Set current_step to complete to end the workflow
        state.current_step = "complete"
        return state

    def _create_workflow(self) -> StateGraph:
        """Create the workflow graph with the improved flow"""
        # Define the state schema based on DataAnalysisState
        graph = StateGraph(state_schema=DataAnalysisState)
        
        # Add nodes
        graph.add_node("prepare_preview", self.prepare_data_preview)
        graph.add_node("analyze", self.analyze_data_types)
        graph.add_node("identify_relationships", self.identify_relationships)
        graph.add_node("suggest", self.suggest_visualizations)
        graph.add_node("visualize", self.create_visualization)
        graph.add_node("evaluate", self.evaluate_results)
        
        # Add edges between nodes
        graph.add_edge(START, "prepare_preview")
        graph.add_edge("prepare_preview", "analyze")
        graph.add_edge("analyze", "identify_relationships")
        graph.add_edge("identify_relationships", "suggest")
        graph.add_edge("suggest", "visualize")
        graph.add_edge("visualize", "evaluate")
        
        def should_continue(state):
            return END if state.current_step == "complete" else "prepare_preview"
        
        graph.add_conditional_edges("evaluate", should_continue)
        
        return graph.compile()
        

    def run_analysis(self, data: pd.DataFrame):
        """Run the complete analysis workflow"""
        self.logger.info("Starting analysis workflow")
        start_time = time.time()
        
        try:
            self.logger.info(f"Input data shape: {data.shape}")
            
            initial_state = DataAnalysisState(
                data=data,
                data_preview="",  # Will be populated in the workflow
                column_types={},
                variable_relationships=[],
                suggested_visualizations=[],
                visualization_outputs=[],
                messages=[],
                current_step="start",
                feedback=None
            )
            
            self.logger.info("Invoking workflow graph")
            result = self.graph.invoke(initial_state)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Analysis workflow completed in {elapsed_time:.2f} seconds")
            
            # Access result attributes using dictionary-style access
            if 'visualization_outputs' in result:
                self.logger.info(f"Result contains {len(result['visualization_outputs'])} visualizations")
            else:
                self.logger.info("No visualization outputs found in result")
                
            if 'feedback' in result and result['feedback']:
                self.logger.info("LLM feedback was generated")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in run_analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
