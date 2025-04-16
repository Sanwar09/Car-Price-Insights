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

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session['username'])

@app.route('/analysis')
@login_required
def analysis():
    # Get paths to saved plots from your analysis
    plots = {
        'univariate': 'static/images/univariate.png',
        'bivariate': 'static/images/bivariate.png',
        'multivariate': 'static/images/multivariate.png'
    }
    return render_template('analysis.html', plots=plots)

@app.route('/iqr')
@login_required
def iqr():
    return render_template('iqr.html', plot='static/images/iqr_boxplot.png')

@app.route('/cleaned_data')
@login_required
def cleaned_data():
    # Load a sample of the cleaned data
    df = pd.read_csv('cleaned_data.csv')
    sample_data = df.head(100).to_html(classes='table table-striped', index=False)
    return render_template('cleaned_data.html', data_table=sample_data)

@app.route('/stats')
@login_required
def stats():
    stats = {
        'kurtosis': 2.15,  # Replace with actual values
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
    return render_template('time_series.html', plot='static/images/time_series.png')

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

if __name__ == '__main__':
    # Perform the analysis when starting the app
    perform_analysis()
    app.run(debug=True)
