# Car Data Analysis & Price Prediction System

A comprehensive, production-ready Flask-based web platform for exploratory car data analysis, statistical metric reporting, Interquartile Range (IQR) data cleaning, multivariate correlation visualizers, and machine learning-powered vehicle resale price prediction.

---

## Table of Contents

1. [Project Overview & Core Functionality](#1-project-overview--core-functionality)
2. [Technical Stack & Dependency Map](#2-technical-stack--dependency-map)
3. [Repository Directory Structure](#3-repository-directory-structure)
4. [Dataset Requirements](#4-dataset-requirements)
5. [Local Setup & Installation](#5-local-setup--installation)
   - [Prerequisites](#prerequisites)
   - [Step-by-Step Installation](#step-by-step-installation)
   - [Running the Application](#running-the-application)
6. [Troubleshooting & Edge Cases](#6-troubleshooting--edge-cases)
   - [Port Conflict Resolution](#port-conflict-resolution)
   - [Missing Input Data or Dynamic Output Files](#missing-input-data-or-dynamic-output-files)
   - [ARM / Apple Silicon Architecture Compilation](#arm--apple-silicon-architecture-compilation)
7. [Security & Environment Configuration](#7-security--environment-configuration)
   - [Critical Security Warnings](#critical-security-warnings)
   - [Environment Variables Reference](#environment-variables-reference)
   - [Generating Secure Secret Keys](#generating-secure-secret-keys)
8. [Production Deployment Guidance](#8-production-deployment-guidance)
   - [Running via Gunicorn](#running-via-gunicorn)
   - [Session Cookie & HTTPS Security](#session-cookie--https-security)
9. [Application Routes & UI Reference](#9-application-routes--ui-reference)
10. [License & Maintenance](#10-license--maintenance)

---

## 1. Project Overview & Core Functionality

The **Car Data Analysis & Price Prediction System** provides software developers, data analysts, system operators, and end-users with an integrated suite for analyzing automobile datasets, inspecting statistical parameters, stripping dataset noise/outliers via statistical bounds, and deploying machine learning models to estimate secondary market car values.

### Target Audience
* **Data Analysts:** Conduct exploratory data analysis (EDA), evaluate statistical skewness, generate correlation heatmaps, and inspect dataset distributions before and after IQR outlier filtering.
* **Software Developers:** Leverage a modular, extensible Flask architecture separating model training (`car_analysis.py`), application routing (`app.py`), dynamic outputs (`.txt` logs), and responsive user interface components (`templates/`).
* **System Operators:** Configure production WSGI application servers (Gunicorn), enforce environment-driven configuration management, and manage administrative user access.

### Core Functionality
* **User Authentication & Session Management:**
  * Role-based navigation, user registration (`/register`), secure login (`/login`), and persistent session tracking using Werkzeug password hashing.
  * Extensible OAuth 2.0 client architecture powered by `flask-dance` and `requests-oauthlib`.
* **Exploratory Data Analysis (EDA):**
  * **Summary Statistics (`/stats`):** Detailed mean, median, standard deviation, and quartile breakdowns parsed from auto-generated `price_stats.txt`.
  * **Skewness Analysis (`/skewness`):** Numerical column normality assessments and skewness distribution logs (`skewness_results.txt`).
  * **Time Series & Trend Analysis (`/time-series`):** Temporal visualizers charting price movement across vehicle manufacturing years.
  * **Multivariate Correlation (`/correlation`):** Feature matrix heatmaps illustrating cross-variable linear dependencies.
* **Data Cleaning & Interquartile Range (IQR) Processing:**
  * Interactive IQR parameter thresholding (`/iqr`) to identify low and high boundary outliers.
  * Cleaned dataset inspection (`/cleaned-data`) allowing analysts to evaluate dataset fidelity post-filtering.
* **Machine Learning Price Estimation:**
  * Multivariate regression model execution (`car_analysis.py`) evaluating feature attributes such as mileage, vehicle age, engine capacity, and horsepower.
  * Interactive inference form (`/predict`) providing real-time resale price predictions based on custom user inputs.
  * Model validation metric outputs (`/regression`) detailing $R^2$ score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) recorded in `regression_metrics.txt`.
* **User Administration:**
  * Admin user directory overview (`/users`) for monitoring registered platform user accounts.

---

## 2. Technical Stack & Dependency Map

The project relies on Python 3.10+ and standard analytical libraries pinned in `requirements.txt`:

| Category | Component / Dependency | Exact Version | Purpose & Operational Function |
| :--- | :--- | :--- | :--- |
| **Web Framework & Backend** | **Python** | `3.10+` | Core execution runtime environment |
| | **Flask** | `2.3.2` | Core WSGI web application framework & route controller engine |
| | **Werkzeug** | `2.3.7` | Standard WSGI web server utility & password hashing module |
| **Data Analytics & ML Engine** | **pandas** | `2.0.3` | High-performance dataframe manipulation & CSV parsing engine |
| | **numpy** | `1.24.3` | Multidimensional numerical computing & vectorized matrix math |
| | **scikit-learn** | `1.3.0` | Machine learning regression algorithms & model performance metrics |
| | **scipy** | `1.10.1` | Advanced statistical computations (Skewness, Kurtosis, IQR bounds) |
| **Data Visualization** | **matplotlib** | `3.7.2` | Programmatic chart generation & statistical graphic engine |
| | **seaborn** | `0.12.2` | High-level statistical dataset visualizer overlays |
| **Production Server** | **gunicorn** | `21.2.0` | Industrial-grade WSGI HTTP server for production deployment |
| **OAuth Authentication** | **flask-dance** | `6.2.0` | Flask extension for managing OAuth consumer connections |
| | **requests-oauthlib** | `1.3.1` | Transport layer support for OAuth 1.0 and OAuth 2.0 authentication |

---

## 3. Repository Directory Structure

The following tree maps the physical repository file layout, highlighting core controller logic, analytical scripts, output text artifacts, and UI templates:

```text
.
├── README.md                   # Primary project documentation and onboarding guide
├── app.py                      # Main Flask application controllers, authentication routes, and view handlers
├── car_analysis.py             # Core analytical script: dataset loading, IQR cleaning, stats generation, ML training
├── price_stats.txt             # Dynamic output file storing descriptive statistics for target vehicle prices
├── regression_metrics.txt      # Dynamic output file storing trained ML regression evaluation metrics (R², MAE, RMSE)
├── requirements.txt            # Explicit third-party Python dependency version manifest
├── skewness_results.txt        # Dynamic output file storing numerical feature skewness computations
├── static/                     # Static web server assets
│   ├── css/
│   │   └── styles.css          # Primary stylesheet for responsive grid, custom forms, and metric cards
│   └── js/
│       └── scripts.js          # Front-end interactive DOM manipulation and form validation scripts
├── templates/                  # Jinja2 HTML layout and page templates
│   ├── base.html               # Master layout containing navigation bar, header assets, and footer
│   ├── index.html              # Platform landing page presenting system features
│   ├── login.html              # User login interface
│   ├── register.html           # New user registration interface
│   ├── dashboard.html          # Main platform dashboard summarizing analytics and prediction routes
│   ├── analysis.html           # Deep-dive analytics overview landing page
│   ├── cleaned_data.html       # Dataset display showing filtered data post-IQR outlier removal
│   ├── correlation.html        # Multivariate feature correlation visualizers and heatmaps
│   ├── iqr.html                # Interquartile Range outlier parameters and statistical boundary reports
│   ├── predict_price.html      # Interactive interface to submit vehicle parameters for price inference
│   ├── regression.html         # ML performance metrics display (R², MAE, RMSE metrics)
│   ├── search_model.html       # Query page for searching specific vehicle models within the dataset
│   ├── skewness_results.html   # Detailed column skewness metrics report view
│   ├── stats.html              # General summary statistics report for the vehicle dataset
│   ├── time_series.html        # Vehicle pricing temporal trends across manufacturing years
│   └── view_users.html         # Administrative user management panel
└── [Scratchpad / Dev Artifacts]
    ├── aa.py                   # Sandbox / local developer scratchpad script (add to .gitignore)
    └── tempCodeRunnerFile.py   # Code runner temporary execution file (add to .gitignore)
```

### Dynamic Metric Log Files
* **`price_stats.txt`:** Generated during dataset processing by `car_analysis.py`. Contains mean, std dev, minimum, maximum, and quartile thresholds for car prices. Rendered in `/stats`.
* **`regression_metrics.txt`:** Updated whenever machine learning models undergo re-training. Contains key performance metrics ($R^2$, MAE, RMSE). Rendered in `/regression`.
* **`skewness_results.txt`:** Captures mathematical skew values for numerical features to identify log-transform requirements. Rendered in `/skewness`.

---

## 4. Dataset Requirements

The platform relies on tabular vehicle dataset files (e.g., `car_data.csv` or `cars.csv`) located in the project root directory when `car_analysis.py` executes.

### Expected Data Schema
For optimal performance, input CSV files should include the following standard fields:

| Column Name | Expected Type | Description |
| :--- | :--- | :--- |
| `Price` / `Selling_Price` | Float / Int | Resale value (Target column for regression and price stats) |
| `Year` | Int | Vehicle manufacturing year (used for age calculation and time-series plots) |
| `Present_Price` | Float | Original showroom or retail list price |
| `Kms_Driven` / `Mileage` | Int / Float | Total distance accumulated on vehicle odometer |
| `Fuel_Type` | Categorical | Fuel type (`Petrol`, `Diesel`, `CNG`, `Electric`) |
| `Seller_Type` | Categorical | Listing party context (`Dealer`, `Individual`) |
| `Transmission` | Categorical | Transmission mechanism (`Manual`, `Automatic`) |
| `Owner` | Int | Count of previous vehicle owners |

---

## 5. Local Setup & Installation

### Prerequisites
* **Python Runtime:** Python **3.10** or **3.11** installed. Verify via `python --version` or `python3 --version`.
* **Git:** Installed and configured in environment PATH.
* **Pip Package Manager:** Upgrade pip prior to installation:
  ```bash
  python -m pip install --upgrade pip
  ```

---

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/your-org/car-data-analysis.git
cd car-data-analysis
```

#### 2. Create and Activate Virtual Environment (`venv`)

* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

* **Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate.bat
  ```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### Running the Application

#### Environment Setup & Launch

* **macOS / Linux (Bash):**
  ```bash
  export FLASK_APP=app.py
  export FLASK_ENV=development
  export FLASK_DEBUG=1
  export FLASK_SECRET_KEY="dev_local_secret_key_change_in_production"
  flask run
  ```

* **Windows (PowerShell):**
  ```powershell
  $env:FLASK_APP = "app.py"
  $env:FLASK_ENV = "development"
  $env:FLASK_DEBUG = "1"
  $env:FLASK_SECRET_KEY = "dev_local_secret_key_change_in_production"
  flask run
  ```

* **Direct Execution via Python:**
  ```bash
  python app.py
  ```

Open your web browser and navigate to: **`http://127.0.0.1:5000`**

---

## 6. Troubleshooting & Edge Cases

### Port Conflict Resolution
If port `5000` is already in use by another service (such as macOS AirPlay Receiver or a background process):

* **Override Port via Flask CLI:**
  ```bash
  flask run --port 5001
  ```
* **Override Port via Python Scripting:**
  Modify `app.py` entry point:
  ```python
  if __name__ == "__main__":
      app.run(host="127.0.0.1", port=5001, debug=True)
  ```
* **Terminate Occupied Port (macOS/Linux):**
  ```bash
  lsof -ti:5000 | xargs kill -9
  ```

### Missing Input Data or Dynamic Output Files
If accessing `/stats`, `/regression`, or `/skewness` produces a `FileNotFoundError`:
1. Place a valid `car_data.csv` in the root folder.
2. Manually run `car_analysis.py` to regenerate the dynamic text logs:
   ```bash
   python car_analysis.py
   ```
3. Confirm that `price_stats.txt`, `regression_metrics.txt`, and `skewness_results.txt` are created in the working directory.

### ARM / Apple Silicon Architecture Compilation
On Apple Silicon (M1/M2/M3) or ARM64 Linux, compiling binary C-extensions for older versions of Scikit-Learn or SciPy can fail if wheel pre-builds are unavailable.
* **Recommended Fix:** Ensure Python **3.10** or **3.11** is used (where binary wheels are available).
* **Install Command for Wheel Resolution:**
  ```bash
  pip install --prefer-binary -r requirements.txt
  ```

---

## 7. Security & Environment Configuration

### Critical Security Warnings

> ⚠️ **HIGH RISK SECURITY NOTICE**
>
> 1. **Hardcoded Fallback Keys:** `app.py` contains fallback secret key declarations (e.g., `app.secret_key = 'your_secret_key_here'`) intended strictly for local development offline fallback. **Never run production servers with fallback keys.** Weak or public secret keys allow session forgery, cookie tampering, and unauthorized administrative access.
> 2. **Werkzeug Debug Console:** Setting `FLASK_DEBUG=1` or `debug=True` exposes an interactive web debugger capable of arbitrary Python code execution. Set `FLASK_DEBUG=0` in production.
> 3. **OAuth Insecure Transport:** Setting `OAUTHLIB_INSECURE_TRANSPORT=1` allows unencrypted HTTP transport for OAuth testing. This variable **MUST NOT** be set in production environments.

---

### Environment Variables Reference

Configure environment settings using a `.env` file in the root folder (ensure `.env` is added to `.gitignore`):

| Variable Name | Dev Default | Production Requirement | Description |
| :--- | :---: | :---: | :--- |
| `FLASK_APP` | `app.py` | `app.py` | Main application entry file |
| `FLASK_ENV` | `development` | `production` | Execution environment context |
| `FLASK_DEBUG` | `1` | `0` | Disables interactive debug console in production |
| `FLASK_SECRET_KEY` | *(Fallback)* | **REQUIRED** | Cryptographically strong random key for session signing |
| `OAUTHLIB_INSECURE_TRANSPORT` | `1` | **UNSET / `0`** | Must be disabled in production to enforce HTTPS for OAuth |
| `PORT` | `5000` | `8000` | Port bound by the web server |

---

### Generating Secure Secret Keys

Generate a production-ready 256-bit secret key using Python's native `secrets` module:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Copy the generated output string and set it as your environment variable:

```bash
export FLASK_SECRET_KEY="<paste_generated_32_byte_hex_string_here>"
```

---

## 8. Production Deployment Guidance

Production deployments must serve the Flask application using an industrial WSGI HTTP server such as **Gunicorn**, placed behind a reverse proxy (e.g., Nginx or AWS ALB) enforcing TLS/HTTPS termination.

### Running via Gunicorn

Do **not** use `python app.py` or `flask run` in production. Launch using Gunicorn with worker processes:

```bash
# 1. Export production variables
export FLASK_ENV=production
export FLASK_DEBUG=0
export FLASK_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export OAUTHLIB_INSECURE_TRANSPORT=0

# 2. Launch Gunicorn WSGI server
gunicorn --workers 4 --threads 2 --bind 0.0.0.0:8000 "app:app"
```

### Session Cookie & HTTPS Security

To prevent session hijacking and Cross-Site Scripting (XSS) cookie theft, ensure production session cookie security configurations are active in Flask:

```python
# Security configuration block for production app initialization
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,     # Requires active HTTPS connection
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=3600 # 1 hour session expiration
)
```

---

## 9. Application Routes & UI Reference

Below is the complete map of active application routes, associated HTTP methods, target Jinja2 templates, and operational descriptions:

| Route Path | HTTP Method(s) | Template Rendered | Functional Purpose |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | `index.html` | Application landing homepage |
| `/login` | `GET`, `POST` | `login.html` | User authentication & credential submission |
| `/register` | `GET`, `POST` | `register.html` | New user registration form |
| `/logout` | `GET` | *Redirect to `/`* | Session invalidation and logout handler |
| `/dashboard` | `GET` | `dashboard.html` | Analytics dashboard and core feature navigation hub |
| `/analysis` | `GET` | `analysis.html` | Deep-dive analytical tools overview |
| `/stats` | `GET` | `stats.html` | Renders price summary stats parsed from `price_stats.txt` |
| `/skewness` | `GET` | `skewness_results.html` | Visualizes feature skewness metrics from `skewness_results.txt` |
| `/iqr` | `GET`, `POST` | `iqr.html` | Interquartile Range parameters input & boundary evaluation |
| `/cleaned-data` | `GET` | `cleaned_data.html` | Displays dataset view with IQR outliers stripped |
| `/correlation` | `GET` | `correlation.html` | Generates feature correlation matrix and interaction maps |
| `/time-series` | `GET` | `time_series.html` | Renders temporal vehicle price trends across manufacturing years |
| `/predict` | `GET`, `POST` | `predict_price.html` | ML form for submitting vehicle attributes & receiving price predictions |
| `/regression` | `GET` | `regression.html` | ML regression validation metrics parsed from `regression_metrics.txt` |
| `/search` | `GET`, `POST` | `search_model.html` | Vehicle model search and filtering interface |
| `/users` | `GET` | `view_users.html` | Administrative user directory and account inspection view |

---

## 10. License & Maintenance

* **System Architecture:** Flask WSGI Application with Data Science & ML Engine.
* **License:** Proprietary / Educational Reference (Consult project repository root for repository-specific licensing terms).
* **Technical Maintenance:** For issues, bug reports, or enhancement suggestions, please open a ticket in the project repository issue tracker.