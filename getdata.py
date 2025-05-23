import os
import pandas as pd
import base64
from newviz3 import DataAnalystAgent

# Set your Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Load your data
df = pd.read_csv("random_datasets/1000_ml_jobs_us.csv")

# Create and run the agent with Groq
agent = DataAnalystAgent(groq_api_key=groq_api_key)
result = agent.run_analysis(df)

# Save the visualizations
for idx, viz in enumerate(result['visualization_outputs']):
    img_data = base64.b64decode(viz['image_base64'])
    filename = f"visualization_{idx+1}_{viz['chart_type']}.png"
    with open(filename, "wb") as f:
        f.write(img_data)
    print(f"Saved {filename}")
