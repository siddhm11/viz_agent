from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from typing import Dict, List ,TypedDict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class DataAnalysisState:
    data: pd.DataFrame
    column_types: Dict[str, str]  # numerical, categorical, temporal
    variable_relationships: List[Dict]
    suggested_visualizations: List[Dict]
    messages: List
    current_step: str


class DataAnalystAgent:
    def __init__(self ,groq_api_key: str):# Initialize with Groq's LLM (using Mixtral model)
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama3-70b-8192",  # or "llama2-70b-4096"
            temperature=0.1
        )
        self.graph = self._create_workflow()
        
    def analyze_data_types(self, state: DataAnalysisState) -> DataAnalysisState:
        """Analyze and categorize data types of each column"""
        df = state.data
        analysis_prompt = f"""
        You are a data analysis expert. 
        Analyze these columns and their data types carefully:
        {df.dtypes.to_dict()}
        
        Categorize each as:
        - numerical (continuous/discrete)
        - categorical (ordinal/nominal)
        - temporal
        
        Return ONLY a Python dictionary without any additional text.
        """
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        state.column_types = eval(response.content)
        return state

    
    def suggest_visualizations(self, state: DataAnalysisState) -> DataAnalysisState:
        """Suggest appropriate visualizations based on data types"""
        viz_prompt = f"""
        Given these column types: {state.column_types}
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
        
        # Instead of using eval directly, parse the response to extract valid JSON
        try:
            # Try to find JSON-like content in the response
            import re
            import json
            # Look for content that might be a list of dictionaries
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response.content, re.DOTALL)
            if json_match:
                state.suggested_visualizations = json.loads(json_match.group(0))
            else:
                # Fallback to a simple visualization if parsing fails
                state.suggested_visualizations = [
                    {
                        "chart_type": "histogram",
                        "variables_to_use": [state.column_types.get("numerical", [])[0] if state.column_types.get("numerical", []) else list(state.data.columns)[0]],
                        "reasoning": "Fallback visualization due to parsing error"
                    }
                ]
        except Exception as e:
            # Handle any parsing errors with a fallback visualization
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
        """Create the suggested visualizations"""
        for viz in state.suggested_visualizations:
            plt.figure(figsize=(10, 6))
            
            if viz['chart_type'] == 'histogram':
                sns.histplot(data=state.data, x=viz['variables_to_use'][0])
            elif viz['chart_type'] == 'scatter':
                sns.scatterplot(data=state.data, 
                              x=viz['variables_to_use'][0],
                              y=viz['variables_to_use'][1])
            # Add more visualization types as needed
            
            plt.title(f"{viz['chart_type'].title()} of {', '.join(viz['variables_to_use'])}")
            plt.show()
        
        # Set current_step to complete to prevent infinite loop
        state.current_step = "complete"
        return state

    def _create_workflow(self) -> StateGraph:
        """Create the workflow graph"""
        # Define the state schema based on DataAnalysisState
        graph = StateGraph(state_schema=DataAnalysisState)
        
        # Add nodes
        graph.add_node("analyze", self.analyze_data_types)
        graph.add_node("suggest", self.suggest_visualizations)
        graph.add_node("visualize", self.create_visualization)
        
                
        # Add edges between nodes
        graph.add_edge(START, "analyze")
        graph.add_edge("analyze", "suggest")
        graph.add_edge("suggest", "visualize")
        
        def should_continue(state):
            return END if state.current_step == "complete" else "analyze"
        
        graph.add_conditional_edges("visualize", should_continue)
        
        return graph.compile()

    def run_analysis(self, data: pd.DataFrame):
        """Run the complete analysis workflow"""
        initial_state = DataAnalysisState(
            data=data,
            column_types={},
            variable_relationships=[],
            suggested_visualizations=[],
            messages=[],
            current_step="start"
        )
        
        return self.graph.invoke(initial_state)