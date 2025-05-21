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

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_analysis_agent.log"),
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
            buffer.write("## Data Preview (First 5 rows)\n")
            buffer.write(state.data.head().to_string())
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
            You are a data analysis expert. 
            
            Here's a preview of the dataset:
            {state.data_preview}
            
            Analyze these columns and their data types carefully:
            {df.dtypes.to_dict()}
            
            Categorize each as:
            - numerical (continuous/discrete)
            - categorical (ordinal/nominal)
            - temporal
            
            Return ONLY a Python dictionary without any additional text.
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
            You are a data analysis expert.
            
            Here's a preview of the dataset:
            {state.data_preview}
            
            And here are the column types you identified:
            {state.column_types}
            
            Based on the column names and types, identify potential relationships worth exploring.
            Consider correlations between numerical variables, group comparisons, and time-based patterns.
            
            Return ONLY a list of dictionaries with these keys:
            - variables: list of column names that might be related
            - relationship_type: (correlation, comparison, time_series, etc.)
            - hypothesis: brief explanation of the potential relationship
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
        Given the data preview:
        {state.data_preview}
        
        These column types: {state.column_types}
        And these relationships: {state.variable_relationships}
        
        Suggest appropriate visualizations following these rules:
        1. For single numerical variables: histogram, boxplot, or density plot
        2. For categorical variables: bar charts or pie charts
        3. For numerical vs numerical: scatter plots or line plots
        4. For numerical vs categorical: box plots or violin plots
        5. For temporal data: time series plots
        
        Return a JSON list of visualization suggestions with:
        - chart_type
        - variables_to_use
        - reasoning
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
            You are a data visualization expert.
            Before generating the chart, explain what this {chart_type} using variables {variables} is meant to reveal.
            Reasoning: {reasoning}
            Provide a 2-3 sentence interpretation.
            """
            explanation_resp = self.llm.invoke([HumanMessage(content=explain_prompt)])
            explanation = explanation_resp.content.strip()
            
            # 2. Now ask LLM to generate the plotting code
            code_prompt = f"""
            Generate Python code to create a {chart_type} visualization for the variables {variables}.
            
            Here's information about the dataset:
            - Shape: {data_summary['shape']}
            - Column types identified: {data_summary['column_types']}
            - Data types from pandas: {data_summary['dtypes']}
            - Sample values for relevant columns: 
            {{{", ".join([f"'{v}': {data_summary['sample_values'][v]}" for v in variables if v in data_summary['sample_values']])}}}
            
            The code should:
            1. Create a matplotlib figure with appropriate size
            2. Generate the visualization using seaborn or matplotlib
            3. Set proper title and labels
            4. Handle any necessary data cleaning or transformation
            5. Include proper error handling
            6. Save the figure to a BytesIO buffer and encode as base64
            
            Return only the executable Python code block without any explanation or markdown. 
            Include these imports in your code:
            ```python
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            import io
            import base64
            ```
            
            The code should start with "def generate_plot(data):" and return three values:
            1. The base64-encoded image string
            2. Any detected insights or warnings (string)
            3. Success status (boolean)
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
                    exec(plot_code, local_vars)
                    
                    # Call the function with our data
                    img_str, insights, success = local_vars['generate_plot'](state.data)
                    
                    self.logger.info(f"Successfully executed plotting code. Success status: {success}")
                    
                    if success:
                        state.visualization_outputs.append({
                            'chart_type': chart_type,
                            'variables': variables,
                            'reasoning': reasoning,
                            'explanation': explanation,
                            'insights': insights if insights else "No specific insights detected.",
                            'image_base64': img_str,
                            'generated_code': plot_code  # Store the generated code for reference
                        })
                    else:
                        # If the code reported failure but didn't raise an exception
                        self.logger.warning(f"Code executed but reported failure: {insights}")
                        # Create error visualization
                        plt.figure(figsize=(10, 4))
                        plt.text(0.5, 0.5, f"Error creating {chart_type} visualization:\n{insights}", 
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=12, color='red')
                        plt.axis('off')
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        img_str = base64.b64encode(buf.read()).decode()
                        buf.close()
                        plt.close()
                        
                        state.visualization_outputs.append({
                            'chart_type': chart_type,
                            'variables': variables,
                            'reasoning': reasoning,
                            'explanation': explanation,
                            'error': insights,
                            'image_base64': img_str,
                            'generated_code': plot_code
                        })
                    
                except Exception as e:
                    self.logger.error(f"Error executing generated code: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    
                    # Create error visualization with the full error message
                    plt.figure(figsize=(10, 4))
                    plt.text(0.5, 0.5, f"Error executing generated code for {chart_type}:\n{str(e)}", 
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=12, color='red')
                    plt.axis('off')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode()
                    buf.close()
                    plt.close()
                    
                    state.visualization_outputs.append({
                        'chart_type': chart_type,
                        'variables': variables,
                        'reasoning': reasoning,
                        'explanation': explanation,
                        'error': str(e),
                        'image_base64': img_str,
                        'generated_code': plot_code
                    })
            else:
                self.logger.error("Could not extract plotting code from LLM response")
                self.logger.error(f"Raw LLM response: {code_response.content}")
                
                # Create error visualization
                plt.figure(figsize=(10, 4))
                plt.text(0.5, 0.5, f"Failed to generate code for {chart_type} visualization.", 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12, color='red')
                plt.axis('off')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode()
                buf.close()
                plt.close()
                
                state.visualization_outputs.append({
                    'chart_type': chart_type,
                    'variables': variables,
                    'reasoning': reasoning,
                    'explanation': explanation,
                    'error': "Failed to generate plotting code",
                    'image_base64': img_str
                })
        
        self.logger.info(f"Completed creation of {len(state.visualization_outputs)} visualizations")
        return state
            
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
        You are a data visualization expert.
        
        Here's the dataset preview:
        {state.data_preview}
        
        Here are the column types identified:
        {state.column_types}
        
        Here are the relationships identified:
        {state.variable_relationships}
        
        The following visualizations were created:
        {chr(10).join(viz_descriptions)}
        
        Please evaluate the visualizations and provide:
        1. Are there any issues with the current visualizations?
        2. What additional visualizations would be helpful?
        3. Are there any alternative approaches that might reveal insights better?
        4. What conclusions can be drawn from the current visualizations?
        
        Format your response as a report that could be shown to a data analyst.
        """
        
        response = self.llm.invoke([HumanMessage(content=evaluation_prompt)])
        state.feedback = response.content
        
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
# Example usage:
# agent = DataAnalystAgent(groq_api_key="your-groq-api-key")
# result = agent.run_analysis(your_dataframe)
# 
# # To display visualizations in a notebook:
# from IPython.display import Image, display
# import base64
# 
# for viz in result.visualization_outputs:
#     if 'image_base64' in viz:
#         display(Image(data=base64.b64decode(viz['image_base64'])))
#         print(f"Chart: {viz['chart_type']} of {viz['variables']}")
#         print(f"Reasoning: {viz['reasoning']}")
#         print("---")
# 
# # Display LLM feedback and evaluation
# print("\nAnalysis Feedback:")
# print(result.feedback)