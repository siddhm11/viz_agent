# Enhanced prompts optimized for DeepSeek-R1-Distill-Llama-70B

# 1. ANALYSIS PROMPT - Enhanced with reasoning steps
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

# 2. RELATIONSHIP PROMPT - Enhanced with structured thinking
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

# 3. VISUALIZATION PROMPT - Enhanced with decision tree logic
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

# 4. CODE GENERATION PROMPT - Enhanced with error handling focus
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

# 5. EXPLANATION PROMPT - Enhanced with context
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

# 6. EVALUATION PROMPT - Enhanced with structured assessment
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