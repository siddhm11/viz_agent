# Create a requirements.txt file for dependencies
requirements_content = '''streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
scipy>=1.7.0
scikit-learn>=1.0.0
langchain-groq>=0.1.0
langgraph>=0.0.40
langchain-core>=0.1.0
python-dotenv>=0.19.0
streamlit-shadcn-ui>=0.1.0
streamlit-option-menu>=0.3.0
'''

with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

print("✅ Requirements file created: requirements.txt")