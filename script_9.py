# Create the data directory first
import os
os.makedirs('data', exist_ok=True)

# Create a sample dataset for testing the application
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducible data
np.random.seed(42)

# Generate sample sales data
n_records = 1000
start_date = datetime.now() - timedelta(days=365)

# Create sample data
sample_data = {
    'date': [start_date + timedelta(days=x) for x in range(n_records)],
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports'], n_records, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    'sales_amount': np.random.lognormal(4, 1, n_records).round(2),
    'quantity_sold': np.random.poisson(5, n_records) + 1,
    'customer_age': np.random.normal(35, 12, n_records).astype(int).clip(18, 80),
    'customer_satisfaction': np.random.normal(4.2, 0.8, n_records).clip(1, 5).round(1),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
    'sales_channel': np.random.choice(['Online', 'In-Store', 'Phone'], n_records, p=[0.6, 0.35, 0.05]),
    'promotion_used': np.random.choice([True, False], n_records, p=[0.3, 0.7]),
    'shipping_cost': np.random.uniform(5, 25, n_records).round(2)
}

# Add some correlations to make the data more interesting
for i in range(n_records):
    # Electronics tend to have higher sales amounts
    if sample_data['product_category'][i] == 'Electronics':
        sample_data['sales_amount'][i] *= np.random.uniform(1.5, 3.0)
    
    # Promotions affect sales amount
    if sample_data['promotion_used'][i]:
        sample_data['sales_amount'][i] *= np.random.uniform(0.8, 1.2)  # Some randomness in promotion effect
    
    # Online sales have lower shipping costs on average
    if sample_data['sales_channel'][i] == 'Online':
        sample_data['shipping_cost'][i] *= np.random.uniform(0.5, 0.8)
    
    # Customer satisfaction correlates with age (loosely)
    age_factor = (sample_data['customer_age'][i] - 18) / 62  # Normalize age to 0-1
    sample_data['customer_satisfaction'][i] += np.random.normal(age_factor * 0.5, 0.2)
    sample_data['customer_satisfaction'][i] = np.clip(sample_data['customer_satisfaction'][i], 1, 5)

# Create DataFrame
df = pd.DataFrame(sample_data)

# Add some derived columns
df['profit_margin'] = np.random.uniform(0.1, 0.4, n_records).round(3)
df['total_profit'] = (df['sales_amount'] * df['profit_margin']).round(2)
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.day_name()
df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])

# Add some missing values to make it realistic
missing_indices = np.random.choice(n_records, size=int(n_records * 0.02), replace=False)
df.loc[missing_indices[:len(missing_indices)//2], 'customer_satisfaction'] = np.nan
df.loc[missing_indices[len(missing_indices)//2:], 'shipping_cost'] = np.nan

# Save to CSV
df.to_csv('data/sample_sales_data.csv', index=False)

# Create a smaller test dataset
small_df = df.sample(100).copy()
small_df.to_csv('data/test_data.csv', index=False)

# Create another dataset with different characteristics
np.random.seed(123)

# Employee dataset
employee_data = {
    'employee_id': range(1, 501),
    'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], 500),
    'salary': np.random.normal(75000, 20000, 500).astype(int).clip(40000, 150000),
    'years_experience': np.random.exponential(3, 500).astype(int).clip(0, 25),
    'performance_rating': np.random.normal(3.5, 0.7, 500).clip(1, 5).round(1),
    'training_hours': np.random.poisson(20, 500),
    'remote_work_days': np.random.choice([0, 1, 2, 3, 4, 5], 500, p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1]),
    'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 500, p=[0.1, 0.5, 0.35, 0.05]),
    'age': np.random.normal(32, 8, 500).astype(int).clip(22, 65),
    'job_satisfaction': np.random.normal(3.8, 0.9, 500).clip(1, 5).round(1)
}

# Add correlations
emp_df = pd.DataFrame(employee_data)

# Salary correlates with experience and education
for i in range(500):
    exp_bonus = emp_df.loc[i, 'years_experience'] * np.random.uniform(1000, 3000)
    
    if emp_df.loc[i, 'education_level'] == 'PhD':
        education_bonus = np.random.uniform(15000, 25000)
    elif emp_df.loc[i, 'education_level'] == 'Master':
        education_bonus = np.random.uniform(8000, 15000)
    elif emp_df.loc[i, 'education_level'] == 'Bachelor':
        education_bonus = np.random.uniform(2000, 8000)
    else:
        education_bonus = 0
    
    emp_df.loc[i, 'salary'] += int(exp_bonus + education_bonus)

# Performance rating correlates with training and satisfaction
emp_df['performance_rating'] += (emp_df['training_hours'] / 100) + (emp_df['job_satisfaction'] / 10)
emp_df['performance_rating'] = emp_df['performance_rating'].clip(1, 5).round(1)

emp_df.to_csv('data/employee_data.csv', index=False)

print("✅ Sample datasets created:")
print("   - data/sample_sales_data.csv (1000 records) - Comprehensive sales data")
print("   - data/test_data.csv (100 records) - Quick test dataset") 
print("   - data/employee_data.csv (500 records) - HR analytics data")
print()
print("📊 Dataset summaries:")
print("\n🛒 Sales Data:")
print(f"   - Records: {len(df)}")
print(f"   - Columns: {len(df.columns)}")
print(f"   - Categories: {df['product_category'].nunique()}")
print(f"   - Date range: {df['date'].min().date()} to {df['date'].max().date()}")

print("\n👥 Employee Data:")
print(f"   - Records: {len(emp_df)}")
print(f"   - Columns: {len(emp_df.columns)}")
print(f"   - Departments: {emp_df['department'].nunique()}")
print(f"   - Salary range: ${emp_df['salary'].min():,} - ${emp_df['salary'].max():,}")

print("\n💡 These datasets are perfect for testing:")
print("   - Multiple data types (numerical, categorical, datetime)")
print("   - Realistic correlations and patterns")
print("   - Some missing values for robustness testing")
print("   - Different sizes for performance testing")