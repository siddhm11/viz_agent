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

explain_prompt = f"""
            You are a data visualization expert.
            Before generating the chart, explain what this {chart_type} using variables {variables} is meant to reveal.
            Reasoning: {reasoning}
            Provide a 2-3 sentence interpretation.
            """

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