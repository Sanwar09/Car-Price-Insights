import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import kurtosis, skew
import os

def perform_analysis():
    # Create directory for images if not exists
    if not os.path.exists('static/images'):
        os.makedirs('static/images')

    # Load dataset
    dataset_path = "C:/Users/ASUS ROG/Downloads/archive/used_cars.csv"
    df = pd.read_csv(dataset_path)

    # Data Cleaning
    df['milage'] = df['milage'].astype(str).str.replace(',', '').str.extract(r'(\d+)').astype(float)
    df['price'] = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)

    # Handle missing values
    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)

    # IQR Outlier Removal
    def remove_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    df = remove_outliers_iqr(df, 'price')
    df = remove_outliers_iqr(df, 'milage')

    # Save cleaned data
    df.to_csv('cleaned_data.csv', index=False)

    # Univariate Analysis
    plt.figure(figsize=(10, 5))
    sns.histplot(df['price'], bins=30, kde=True)
    plt.title('Price Distribution')
    plt.savefig('static/images/univariate.png', bbox_inches='tight')
    plt.close()

    # Bivariate Analysis
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df['milage'], y=df['price'])
    plt.title('Mileage vs Price')
    plt.savefig('static/images/bivariate.png', bbox_inches='tight')
    plt.close()

    # Multivariate Analysis
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('static/images/multivariate.png', bbox_inches='tight')
    plt.close()

    # Outlier Detection (Boxplot After IOR)
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['price'])
    plt.title('Boxplot of Price After IQR')
    plt.savefig('static/images/iqr_boxplot.png', bbox_inches='tight')
    plt.close()

    # Time Series Analysis
    if 'model_year' in df.columns:
        df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')
        df.dropna(subset=['model_year'], inplace=True)
        df = df.sort_values(by='model_year')

        time_series = df.groupby('model_year')['price'].mean()

        plt.figure(figsize=(12, 6))
        plt.plot(time_series.index, time_series.values, marker='o', linestyle='-', alpha=0.7)
        plt.xlabel("Model Year")
        plt.ylabel("Average Price")
        plt.title("Price Over Model Year")
        plt.grid(True)
        plt.savefig('static/images/time_series.png', bbox_inches='tight')
        plt.close()

    # Correlation Matrix
    plt.figure(figsize=(10, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('static/images/correlation_matrix.png', bbox_inches='tight')
    plt.close()

    # Train Linear Regression Model
    X = df[['milage']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate accuracy
    r2 = r2_score(y_test, y_pred)
    accuracy_percentage = r2 * 100

    # Save metrics to a file (could be used by the Flask app)
    with open('regression_metrics.txt', 'w') as f:
        f.write(f"R2 Score: {r2}\n")
        f.write(f"MSE: {mean_squared_error(y_test, y_pred)}\n")
        f.write(f"Accuracy: {accuracy_percentage:.2f}%\n")

    # Skewness Analysis
    skewness_results = {}
    for col in numeric_df.columns:
        skewness_results[col] = skew(df[col], nan_policy='omit')

    with open('skewness_results.txt', 'w') as f:
        for col, val in skewness_results.items():
            f.write(f"Skewness of {col}: {val}\n")

    # Kurtosis and Skewness of price
    with open('price_stats.txt', 'w') as f:
        f.write(f"Kurtosis: {kurtosis(df['price'], nan_policy='omit')}\n")
        f.write(f"Skewness: {skew(df['price'], nan_policy='omit')}\n")
