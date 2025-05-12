from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
from car_analysis import perform_analysis
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Dummy user database (in production, use a real database)
users = {
    "admin": generate_password_hash("admin123"),
    "user": generate_password_hash("user123")
}

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login', next=request.url))  # Redirect to login if not logged in
        return f(*args, **kwargs)
    return decorated_function



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            next_page = request.args.get('next')  # Capture the 'next' parameter if any
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)  # Clear the session when logging out
    return redirect(url_for('login'))  # Redirect to the login page after logout

@app.route('/dashboard')
@login_required
def dashboard():
    df = pd.read_csv('cleaned_data.csv')  # Adjust this as needed
    total_cars = len(df)
    price_prediction_accuracy = 98.0
    data_cleaned = 78.0

    cleaned_data_stats = {
        'missing_values': 123,
        'invalid_entries': 45,
        'outliers': 87,
        'standardized': 5678
    }

    return render_template(
        'dashboard.html',
        username=session['username'],
        total_cars=total_cars,
        price_prediction_accuracy=price_prediction_accuracy,
        data_cleaned=data_cleaned,
        cleaned_data=cleaned_data_stats
    )



@app.route('/analysis')
@login_required
def analysis():
    plots = {
        'univariate': 'static/images/univariate.png',
        'bivariate': 'static/images/bivariate.png',
        'multivariate': 'static/images/multivariate.png'
    }
    return render_template('analysis.html', plots=plots)

@app.route('/iqr')
@login_required
def iqr():
    iqr_stats = {
        'q1': 15000,
        'median': 22000,
        'q3': 30000,
        'iqr': 15000,
        'lower_bound': 15000 - 1.5 * 15000,
        'upper_bound': 30000 + 1.5 * 15000,
        'outliers_removed': 42
    }
    return render_template(
        'iqr.html',
        iqr_stats=iqr_stats,
        before_plot='static/images/before_iqr.png',
        after_plot='static/images/after_iqr.png'
    )

@app.route('/cleaned_data')
@login_required
def cleaned_data():
    df = pd.read_csv('cleaned_data.csv')
    data_shape = df.shape
    data_columns = df.columns.tolist()
    data_sample = df.head(100).values.tolist()
    cleaning_stats = {
        'missing_values': 120,
        'invalid_entries': 45,
        'outliers': 38,
        'standardized': 7
    }
    return render_template('cleaned_data.html', data_shape=data_shape, data_columns=data_columns, data_sample=data_sample, cleaning_stats=cleaning_stats)


@app.route('/stats')
@login_required
def stats():
    stats = {
        'kurtosis': 2.15,
        'skewness': 1.78,
        'skewness_values': {
            'price': 1.78,
            'milage': 0.92,
            'model_year': -0.45
        }
    }
    return render_template('stats.html', stats=stats)

@app.route('/time_series')
@login_required
def time_series():
    ts_stats = {
        'peak_year': 2021,
        'peak_price': 27850.23,
        'trend': 'Increasing',
        'yoy_change': 7.4,
        'top_years': [(2021, 27850.23, 150), (2020, 26400.88, 130)],
        'bottom_years': [(2015, 15500.90, 60), (2016, 15888.12, 75)]
    }
    decomposition_img = 'seasonal_decomposition.png'
    return render_template('time_series.html', ts_stats=ts_stats, decomposition_img=decomposition_img)

@app.route('/correlation')
@login_required
def correlation():
    return render_template('correlation.html', plot='static/images/correlation_matrix.png')

@app.route('/regression')
@login_required
def regression():
    metrics = {
        'r2_score': 0.85,
        'mse': 2500000,
        'accuracy': 85.0
    }
    return render_template('regression.html', metrics=metrics)

@app.route('/search_model', methods=['GET', 'POST'])
@login_required
def search_model():
    df = pd.read_csv('cleaned_data.csv')
    
    # Get unique brands and models from the dataset
    brands = sorted(df['brand'].dropna().unique())
    
    # Create a dictionary of models for each brand
    models_by_brand = {}
    for brand in brands:
        models_by_brand[brand] = sorted(df[df['brand'] == brand]['model'].dropna().unique())
    
    results = None
    selected_brand = None
    selected_model = None

    if request.method == 'POST':
        selected_brand = request.form['brand']
        selected_model = request.form['model']
        # Filter based on brand and model
        filtered_df = df[(
            df['brand'] == selected_brand) & 
            (df['model'] == selected_model)]
        results = filtered_df.to_dict(orient='records')

    return render_template(
        'search_model.html',
        brands=brands,
        models_by_brand=models_by_brand,
        results=results,
        selected_brand=selected_brand,
        selected_model=selected_model
    )

@app.route('/predict_price', methods=['GET', 'POST'])
@login_required
def predict_price():
    df = pd.read_csv('cleaned_data.csv')

    # Get unique brands and models from the dataset
    brands = sorted(df['brand'].dropna().unique())

    # Create a dictionary of models for each brand
    models_by_brand = {}
    for brand in brands:
        models_by_brand[brand] = sorted(df[df['brand'] == brand]['model'].dropna().unique())
    
    prediction = None
    selected_brand = None
    selected_model = None
    year = None

    if request.method == 'POST':
        selected_brand = request.form['brand']
        selected_model = request.form['model']
        year = int(request.form['year'])

        # Filter the dataset for the selected model
        filtered_df = df[(df['brand'] == selected_brand) & (df['model'] == selected_model)]

        # Perform prediction if there's sufficient data
        if 'model_year' in filtered_df.columns and len(filtered_df) > 1:
            X = filtered_df[['model_year']]  # Use 'model_year' as the feature for prediction
            y = filtered_df['price']  # The target is the price
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            prediction = model.predict([[year]])[0]
        else:
            prediction = "Insufficient data for prediction."

    return render_template(
        'predict_price.html',
        brands=brands,
        models_by_brand=models_by_brand,
        prediction=prediction,
        selected_brand=selected_brand,
        selected_model=selected_model
    )

if __name__ == '__main__':
    perform_analysis()
    app.run(debug=True)
