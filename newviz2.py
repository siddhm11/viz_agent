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
import sklearn
import scipy


# Set up logging configuration
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
        self.logger.info(f"Initialized LLM with model: llama3-70b-8192")
        
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
            buffer.write(state.data.head(15).to_string())
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
                state.column_types = eval(response.content)
                self.logger.info(f"Parsed column types: {state.column_types}")
            except Exception as parse_error:
                self.logger.error(f"Error parsing column types: {str(parse_error)}")
                self.logger.error(f"Raw LLM response: {response.content}")
                # Create a basic fallback categorization
                state.column_types = {}
                for col, dtype in df.dtypes.items():
                    if 'int' in str(dtype) or 'float' in str(dtype):
                        state.column_types[col] = 'numerical'
                    elif 'datetime' in str(dtype):
                        state.column_types[col] = 'temporal'
                    else:
                        state.column_types[col] = 'categorical'
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
            else:
                # Fallback to a simple visualization if parsing fails
                state.suggested_visualizations = [
                    {
                        "chart_type": "histogram",
                        "variables_to_use": [list(state.column_types.get("numerical", [""]))[0] if state.column_types.get("numerical", []) else list(state.data.columns)[0]],
                        "reasoning": "Fallback visualization due to parsing error"
                    }
                ]
        except Exception as e:
            print(f"Error parsing visualization suggestions: {e}")
            state.suggested_visualizations = [
                {
                    "chart_type": "histogram",
                    "variables_to_use": [list(state.data.columns)[0]],
                    "reasoning": "Fallback visualization due to parsing error"
                }
            ]
        
        return state


    def create_visualization(self, state: DataAnalysisState) -> DataAnalysisState:
        """Create visualizations by having the LLM dynamically generate the plotting code"""
        self.logger.info("Starting dynamic LLM-based visualization creation")
        state.visualization_outputs = []

        # First, create a data summary that the LLM can use
        data_summary = {
            "shape": state.data.shape,
            "columns": list(state.data.columns),
            "dtypes": {col: str(dtype) for col, dtype in state.data.dtypes.items()},
            "sample_values": {col: state.data[col].dropna().sample(min(3, len(state.data))).tolist() 
                            for col in state.data.columns},
            "column_types": state.column_types
        }
        
        self.logger.info(f"Created data summary for LLM with {len(data_summary['columns'])} columns")

        for viz in state.suggested_visualizations:
            chart_type = viz['chart_type'].lower()
            variables = viz['variables_to_use']
            reasoning = viz.get('reasoning', '')
            
            self.logger.info(f"Requesting code for {chart_type} visualization of {variables}")

            # 1. First ask LLM to explain visualization purpose
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
            explanation_resp = self.llm.invoke([HumanMessage(content=explain_prompt)])
            explanation = explanation_resp.content.strip()
            log_path = "logs/explanations.log"   # or wherever you want the file
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("=== New Explanation ===\n")
                f.write("Prompt:\n")
                f.write(explain_prompt.strip() + "\n\n")
                f.write("Explanation:\n")
                f.write(explanation + "\n\n")

            # 2. Now ask LLM to generate the plotting code
            code_prompt = f"""
            <thinking>
            I need to generate robust Python code for a {chart_type} visualization of {variables}.

            Dataset characteristics:
            - Shape: {data_summary['shape']}
            - Column types: {data_summary['column_types']}
            - Data types: {data_summary['dtypes']}
            - Sample values: {{{", ".join([f"'{v}': {data_summary['sample_values'][v]}" for v in variables if v in data_summary['sample_values']])}}}

            Code requirements:
            1. Handle missing values appropriately
            2. Check data types and convert if needed
            3. Apply appropriate statistical transformations
            4. Create clear, publication-ready visualization
            5. Include meaningful insights detection
            6. Robust error handling for edge cases
            </thinking>

            Generate a complete Python function that creates a {chart_type} visualization.

            Requirements:
            - Function signature: def generate_plot(data):
            - Return: (base64_image_string, insights_text, success_boolean)
            - Handle all edge cases (empty data, wrong types, etc.)
            - Include statistical insights where relevant
            - Use modern matplotlib/seaborn styling
            - Appropriate figure size and DPI for clarity

            ```python
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            import numpy as np
            import io
            import base64
            from scipy import stats
            import warnings
            warnings.filterwarnings('ignore')

            def generate_plot(data):
                try:
                    # Your complete implementation here
                    # Include data validation, cleaning, plotting, and insight generation
                    pass
                except Exception as e:
                    return None, f"Error: {{str(e)}}", False
            ```

            Complete Python code:
            """
            code_response = self.llm.invoke([HumanMessage(content=code_prompt)])
            self.logger.info("Received code generation response from LLM")
            
            # Extract the code from the LLM response
            import re
            code_match = re.search(r'```python\s*(.*?)\s*```', code_response.content, re.DOTALL)
            if not code_match:
                code_match = re.search(r'def generate_plot\(data\):(.*?)(?:$|```)', code_response.content, re.DOTALL)
            
            if code_match:
                plot_code = code_match.group(1)
                # If the match doesn't include the function definition, add it
                if not plot_code.strip().startswith("def generate_plot"):
                    plot_code = "def generate_plot(data):" + plot_code
                    
                self.logger.info(f"Extracted plotting code of length {len(plot_code)}")
                
                # Execute the generated code in a safe context
                try:
    # Extract the code from the LLM response
                    plot_code = extract_code_from_response(code_response.content)
                    
                    if plot_code:
                        # Execute with self-healing capabilities
                        (img_str, insights, success), final_code, healing_history = execute_with_healing(plot_code, state.data)
                        
                        # Log the healing process
                        self.logger.info(f"Code execution completed after {len(healing_history) + 1} attempts")
                        
                        # Store the visualization output
                        state.visualization_outputs.append({
                            'chart_type': chart_type,
                            'variables': variables,
                            'reasoning': reasoning,
                            'explanation': explanation,
                            'insights': insights if insights else "No specific insights detected.",
                            'image_base64': img_str,
                            'generated_code': final_code,  # Store the final healed code
                            'healing_history': healing_history  # Store the healing history for analysis
                        })
                        
                        # Log the healing process for analysis
                        log_healing_process(chart_type, healing_history)
                    else:
                        # Handle case where code extraction failed
                        self.logger.error("Could not extract plotting code from LLM response")
                        # Create error visualization as in the original code
                        
                except Exception as e:
                    self.logger.error(f"Unrecoverable error in self-healing process: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    # Create error visualization as in the original code

                
    
    def execute_with_healing(code, data, max_attempts=3):
        """Execute code with self-healing capabilities."""
        healing_history = []
        current_code = code
        
        for attempt in range(max_attempts):
            try:
                # Create local namespace for execution
                local_vars = {}
                
                # Add necessary imports to local namespace
                exec("import matplotlib.pyplot as plt", local_vars)
                exec("import seaborn as sns", local_vars)
                exec("import pandas as pd", local_vars)
                exec("import io", local_vars)
                exec("import base64", local_vars)
                exec("import numpy as np", local_vars)
                
                # Execute the generated plotting function
                exec(current_code, local_vars)
                
                # Call the function with our data
                img_str, insights, success = local_vars['generate_plot'](data)
                
                if success:
                    return (img_str, insights, success), current_code, healing_history
                else:
                    # Code executed but reported failure
                    error_msg = f"Code reported failure: {insights}"
                    healing_history.append({
                        'attempt': attempt + 1,
                        'error': error_msg,
                        'code_before': current_code
                    })
                    
                    # Generate healing prompt
                    healing_prompt = create_healing_prompt(current_code, error_msg)
                    
                    # Get corrected code from LLM
                    corrected_code = get_corrected_code_from_llm(healing_prompt)
                    
                    # Update current code
                    current_code = corrected_code
                    healing_history[-1]['code_after'] = current_code
                    
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                traceback_str = traceback.format_exc()
                
                healing_history.append({
                    'attempt': attempt + 1,
                    'error': error_msg,
                    'traceback': traceback_str,
                    'code_before': current_code
                })
                
                # Generate healing prompt
                healing_prompt = create_healing_prompt(current_code, error_msg, traceback_str)
                
                # Get corrected code from LLM
                corrected_code = get_corrected_code_from_llm(healing_prompt)
                
                # Update current code
                current_code = corrected_code
                healing_history[-1]['code_after'] = current_code
        
        # If we've reached max attempts without success
        return create_error_visualization(f"Failed after {max_attempts} healing attempts"), current_code, healing_history

    def create_healing_prompt(code, error_msg, traceback_str=None):
        """Create a prompt for the LLM to fix the code based on error information."""
        prompt = f"""
        I need you to fix the following Python code that encountered an error.
        
        ERROR INFORMATION:
        {error_msg}
        
        {'TRACEBACK:\n' + traceback_str if traceback_str else ''}
        
        ORIGINAL CODE:
        ```
        {code}
        ```
        
        Please analyze the error and provide a corrected version of the code. Focus on:
        1. Fixing the specific error mentioned
        2. Ensuring robust error handling
        3. Maintaining the original functionality
        4. Improving code quality where possible
        
        Return ONLY the corrected code without explanations.
        """
        
        return prompt

    def get_corrected_code_from_llm(healing_prompt):
        """Get corrected code from LLM based on the healing prompt."""
        # Send the healing prompt to the LLM
        correction_response = self.llm.invoke([HumanMessage(content=healing_prompt)])
        
        # Extract the code from the LLM response
        import re
        code_match = re.search(r'``````', correction_response.content, re.DOTALL)
        
        if not code_match:
            code_match = re.search(r'``````', correction_response.content, re.DOTALL)
        
        if code_match:
            corrected_code = code_match.group(1)
            # If the match doesn't include the function definition, add it
            if not corrected_code.strip().startswith("def generate_plot"):
                corrected_code = "def generate_plot(data):" + corrected_code
            
            return corrected_code
        else:
            # Fallback if no code block is found
            return correction_response.content

    def create_error_visualization(error_message):
        """Create an error visualization image with the provided message."""
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, error_message, 
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='red')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close()
        
        return img_str, error_message, False


    def log_healing_process(chart_type, healing_history):
        """Log the healing process for analysis."""
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
                f.write("```")
                f.write(attempt['code_before'].strip() + "\n")
                f.write("```\n\n")
                f.write("Code After:\n")
                f.write("```")
                f.write(attempt['code_after'].strip() + "\n")
                f.write("```\n\n")



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
