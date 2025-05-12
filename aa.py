import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load your dataset (update path and file as needed)
# Example: data = pd.read_csv('your_dataset.csv')
data = pd.read_csv('cleaned_data.csv')  # Adjust the path if needed

# Define the column to analyze
column = 'price'  # Change this to the column you want to analyze

# Create directory if it doesn't exist
output_dir = 'static/images'
os.makedirs(output_dir, exist_ok=True)

# --- Plot 1: Before IQR (with outliers)
plt.figure(figsize=(10, 2))
sns.boxplot(x=data[column])
plt.title(f'{column.capitalize()} - Before IQR (With Outliers)')
plt.tight_layout()
plt.savefig(f'{output_dir}/before_iqr.png')
plt.close()

# --- IQR Calculation
Q1 = data[column].quantile(0.25)
Q3 = data[column].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter data to remove outliers
filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# --- Plot 2: After IQR (outliers removed)
plt.figure(figsize=(10, 2))
sns.boxplot(x=filtered_data[column])
plt.title(f'{column.capitalize()} - After IQR (Outliers Removed)')
plt.tight_layout()
plt.savefig(f'{output_dir}/after_iqr.png')
plt.close()

print("IQR plots saved successfully.")
