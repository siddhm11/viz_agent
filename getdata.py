import os
import pandas as pd
from data_analyst_agent import DataAnalystAgent

# Set your Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Load your data
df = pd.read_csv("your_data.csv")

# Create and run the agent with Groq
agent = DataAnalystAgent(groq_api_key=groq_api_key)
result = agent.run_analysis(df)