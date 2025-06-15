# 🚀 Quick Start Guide

## Get started with your AI Data Analysis Dashboard in 5 minutes!

### Step 1: Setup (2 minutes) ⚙️

```bash
# Clone or download all the files to a folder
# Open terminal/command prompt in that folder

# Quick setup
python setup.py

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# Install dependencies
python setup.py install
```

### Step 2: Configure API Key (1 minute) 🔑

1. Get your Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys)
2. Edit the `.env` file and replace:
   ```
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

### Step 3: Add Your Code (30 seconds) 📄

Save your `DataAnalystAgent` code as `paste.py` in the project folder.

### Step 4: Launch! (30 seconds) 🚀

```bash
# Option 1: Use the launcher
python launch.py

# Option 2: Direct launch
streamlit run enhanced_app.py
```

### Step 5: Test with Sample Data (1 minute) 📊

1. Open your browser to [http://localhost:8501](http://localhost:8501)
2. Upload one of the sample datasets from the `data/` folder:
   - `test_data.csv` (quick test)
   - `sample_sales_data.csv` (full demo)
   - `employee_data.csv` (HR analytics)
3. Click "Run Analysis" and watch the magic happen! ✨

## 🎯 What You'll See

- **Beautiful Interface**: Modern UI with gradient headers and smooth animations
- **Real-time Progress**: Watch your analysis progress with live updates
- **AI-Generated Visualizations**: Charts created automatically based on your data
- **Smart Insights**: AI-powered analysis and recommendations
- **Interactive Results**: Explore your data with professional visualizations

## 🔧 Troubleshooting

### Common Issues:

**"Module not found" error**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

**"API key not configured"**
- Check your `.env` file
- Make sure the key starts with `gsk_` (Groq keys)
- No quotes around the key value

**"DataAnalystAgent not found"**
- Save your original code as `paste.py`
- Make sure it includes the complete class definition

## 🎨 Two Versions Available

### Basic Version (`streamlit_app.py`)
- Clean, simple interface
- Essential features only
- Perfect for quick analysis

### Enhanced Version (`enhanced_app.py`) - **Recommended**
- Modern UI with animations
- Advanced progress tracking
- Export capabilities
- Analysis history
- Professional styling

## 📱 Browser Access

Once running, access your dashboard at:
- **Local**: [http://localhost:8501](http://localhost:8501)
- **Network**: `http://YOUR-IP:8501` (for other devices)

## 💡 Pro Tips

1. **Start with test data**: Use `data/test_data.csv` for your first run
2. **Check the logs**: Look in `logs/` folder if something goes wrong
3. **Use the enhanced version**: It has better error handling and UI
4. **Try different datasets**: Each dataset will show different AI capabilities

## 🆘 Need Help?

1. **Check the README.md** for detailed documentation
2. **Look at the logs** in the `logs/` folder
3. **Verify your setup** with `python launch.py --check`
4. **Test with sample data** first before using your own

## 🎉 You're Ready!

Your AI Data Analysis Dashboard is now ready to transform your data into insights. Upload any CSV file and let the AI do the heavy lifting!

---

**Happy Analyzing! 📊✨**