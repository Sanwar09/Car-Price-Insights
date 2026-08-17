# Car Data Analysis & Price Prediction System

A comprehensive Flask-based web platform for exploratory car data analysis, statistical metric reporting, Interquartile Range (IQR) data cleaning, and machine learning-powered vehicle price prediction.

---

## Table of Contents

1. [Project Overview & Key Features](#1-project-overview--key-features)
2. [Technical Stack](#2-technical-stack)
3. [Repository Directory Structure](#3-repository-directory-structure)
4. [Dataset Requirements](#4-dataset-requirements)
5. [Local Setup & Installation](#5-local-setup--installation)
   - [Prerequisites](#prerequisites)
   - [Step-by-Step Installation](#step-by-step-installation)
   - [Running the Application](#running-the-application)
   - [Port Conflict Mitigation](#port-conflict-mitigation)
6. [Security & Environment Configuration](#6-security--environment-configuration)
   - [Critical Security Warning](#critical-security-warning)
   - [Environment Variables Guide](#environment-variables-guide)
   - [Production Deployment Guidance](#production-deployment-guidance)
7. [Application Routes & Navigation](#7-application-routes--navigation)
8. [Troubleshooting & FAQ](#8-troubleshooting--faq)
9. [License & Maintenance](#9-license--maintenance)

---

## 1. Project Overview & Key Features

The **Car Data Analysis & Price Prediction System** provides automotive data analysts, consumers, and platform administrators with an end-to-end suite for exploring vehicle statistics, identifying market trends, filtering data outliers, and training linear/multivariate regression models to estimate vehicle resale values.

### Key Functional Features

* **User Authentication & Session Management:**
  * User registration, secure login, role-aware routing, and session state persistence.
  * OAuth authentication extension support via Flask-Dance.
* **Exploratory Data Analysis (EDA):**
  * **Price Statistics & Distribution:** General descriptive statistics including mean, median, standard deviation, and variance across price distributions.
  * **Skewness Analysis:** Statistical skewness calculations and normality assessments for numerical dataset attributes.
  * **Time Series Analysis:** Temporal trend plotting to observe pricing shifts across manufacturing years.
  * **Correlation Matrix:** Multivariate correlation analysis and feature interaction charts.
* **Data Cleaning & IQR Outlier Processing:**
  * Interquartile Range (IQR) statistical filtering to detect and strip price outliers.
  * Side-by-side inspection of raw versus cleaned datasets.
* **Machine Learning Price Prediction Engine:**
  * Regression model training utilizing feature sets (e.g., mileage, vehicle age, engine size, horsepower).
  * Interactive web interface for entering custom car parameters to receive real-time price predictions.
  * Model evaluation metric tracking exported to text reports (e.g., $R^2$ score, Mean Absolute Error, Root Mean Squared Error).
* **Admin & User Management Views:**
  * Administrative dashboard for managing registered system accounts and monitoring data pipeline output files.

---

## 2. Technical Stack

The system relies on Python 3.x and the exact dependency versions specified in `requirements.txt`:

| Component | Library / Tool | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Web Backend** | Python | `3.x` | Core execution runtime |
| | Flask | `2.3.2` | Web framework & route routing engine |
| | Werkzeug | `2.3.7` | WSGI utility library & secure password hashing |
| **Data Processing & ML** | Pandas | `2.0.3` | Dataframe manipulation & CSV parsing |
| | NumPy | `1.24.3` | Vectorized numerical computing & matrix operations |
| | Scikit-Learn | `1.3.0` | Machine learning regression pipelines & metrics |
| | SciPy | `1.10.1` | Advanced statistical computations (Skewness, IQR) |
| **Data Visualization** | Matplotlib | `3.7.2` | Static chart generation & statistical plotting |
| | Seaborn | `0.12.2` | High-level data visualization overlays |
| **Deployment / Server** | Gunicorn | `21.2.0` | Production-grade WSGI HTTP server |
| **OAuth Integration** | Flask-Dance | `6.2.0` | OAuth consumer extension for Flask |
| | Requests-OAuthlib | `1.3.1` | OAuth library transport helper |

---

## 3. Repository Directory Structure

```text
.
├── app.py                      # Flask entry point, routes, session management, and view handlers
├── car_analysis.py             # Data cleaning logic, statistical calculations, and ML model routines
├── price_stats.txt             # Auto-generated summary file containing vehicle price descriptive stats
├── regression_metrics.txt      # Auto-generated summary file containing ML regression performance metrics
├── skewness_results.txt        # Auto-generated summary file containing numerical column skewness metrics
├── requirements.txt            # Explicit project dependencies and version pin list
│
├── static/                     # Web assets served directly by Flask / web server
│   ├── css/
│   │   └── styles.css          # Primary stylesheet for dashboard and responsive layouts
│   └── js/
│       └── scripts.js          # Front-end interactivity scripts and client-side form validation
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Parent HTML template with standard navbar, head, and footer
│   ├── index.html              # Application homepage landing page
│   ├── login.html              # User authentication login interface
│   ├── register.html           # New account registration view
│   ├── dashboard.html          # Main platform dashboard summarizing analytics options
│   ├── cleaned_data.html       # Visual display of data post-IQR outlier removal
│   ├── iqr.html                # Interquartile Range outlier detection parameters and results
│   ├── stats.html              # Summary statistics view for vehicle dataset parameters
│   ├── skewness_results.html   # Detailed metric reports for numerical skewness
│   ├── correlation.html        # Multivariate feature correlation visualizers
│   ├── time_series.html        # Temporal trends and pricing over time visualization
│   ├── predict_price.html      # Interactive form to submit vehicle specs for price estimation
│   ├── regression.html         # Detailed metrics view for model evaluation ($R^2$, MAE, RMSE)
│   ├── search_model.html       # Vehicle model filter and query lookup tool
│   ├── view_users.html         # Administrative user management panel
│   └── analysis.html           # Comprehensive deep-dive analytics landing page
│
└── [Workspace Noise / Temporary Files]
    ├── aa.py                   # (Optional dev scratchpad / sandbox script — ignorable)
    └── tempCodeRunnerFile.py   # (VS Code execution temporary artifact — ignorable)
```

> **Note on Temporary Files:** Files such as `tempCodeRunnerFile.py` and `aa.py` are transient local development artifacts and should be added to `.gitignore`.

---

## 4. Dataset Requirements

`car_analysis.py` expects a dataset CSV file in the root directory at runtime (e.g., `car_data.csv` or `cars.csv`).

* **Default Expected CSV Schema:**
  * `Price` (Numeric: target variable for regression and descriptive stats)
  * `Year` / `Age` (Numeric: manufacturing year or age of vehicle)
  * `Mileage` / `Kms_Driven` (Numeric: distance driven)
  * `Engine` / `Present_Price` / `HP` (Numeric: engine specs/horsepower)
  * `Transmission` / `Fuel_Type` / `Seller_Type` (Categorical: transmission type, fuel source, seller context)

If no dataset is present when initializing `car_analysis.py`, ensure your local CSV dataset file is placed in the project root directory and matching environment paths before launching analytical endpoints.

---

## 5. Local Setup & Installation

### Prerequisites

* **Python:** Python 3.8+ installed locally.
* **Git:** Installed and configured in system PATH.
* **Package Manager:** `pip` updated to latest version (`python -m pip install --upgrade pip`).

---

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/your-org/car-data-analysis.git
cd car-data-analysis
```

#### 2. Create and Activate Virtual Environment (`venv`)

* **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

* **Windows PowerShell:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

* **Windows Command Prompt (cmd):**
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

#### Development Mode (Local Testing)

Set your environment variables before starting the server.

* **Linux / macOS:**
  ```bash
  export FLASK_APP=app.py
  export FLASK_ENV=development
  export FLASK_DEBUG=1
  export FLASK_SECRET_KEY="dev_only_secret_key_change_in_production"
  flask run
  ```

* **Windows PowerShell:**
  ```powershell
  $env:FLASK_APP = "app.py"
  $env:FLASK_ENV = "development"
  $env:FLASK_DEBUG = "1"
  $env:FLASK_SECRET_KEY = "dev_only_secret_key_change_in_production"
  flask run
  ```

Alternatively, run directly via Python:
```bash
python app.py
```

Access the application in your browser at: `http://127.0.0.1:5000`

---

### Port Conflict Mitigation

By default, Flask binds to port `5000`. If port `5000` is occupied (e.g., by AirPlay on macOS or another local service), specify an alternate port:

* **Using Flask CLI:**
  ```bash
  flask run --port 5001
  ```

* **Modifying Direct Python Execution:**
  Update the bottom of `app.py`:
  ```python
  if __name__ == "__main__":
      app.run(host="127.0.0.1", port=5001, debug=True)
  ```

---

## 6. Security & Environment Configuration

### Critical Security Warning

> ⚠️ **IMPORTANT SECURITY NOTICE FOR SYSTEM ADMINISTRATORS & DEVELOPERS**
>
> 1. **Default Secret Keys:** The codebase contains fallback secret keys in `app.py` for local developer convenience. **Never deploy to production with hardcoded or default secret keys.** Hardcoded secret keys allow attackers to forge signed Flask session cookies and hijack user privileges.
> 2. **Debug Mode:** `FLASK_DEBUG=1` or `debug=True` MUST be set to `0`/`False` in production. Leaving debug mode active exposes Werkzeug's interactive web console, allowing arbitrary Python code execution.
> 3. **OAuth Insecure Transport:** Setting `OAUTHLIB_INSECURE_TRANSPORT=1` permits HTTP traffic for OAuth testing. This flag is strictly intended for local offline testing and **MUST NOT** be enabled in production environments.

---

### Environment Variables Guide

Generate a cryptographically secure random key for production using Python:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Configure environment variables using a secure `.env` file (ensure `.env` is listed in `.gitignore`):

| Variable | Required in Dev | Required in Prod | Description / Recommended Value |
| :--- | :---: | :---: | :--- |
| `FLASK_APP` | Yes | Yes | `app.py` |
| `FLASK_ENV` | Optional | Yes | `development` (Local) / `production` (Live) |
| `FLASK_DEBUG` | Optional | Yes | `1` in Dev / `0` in Production |
| `FLASK_SECRET_KEY` | Optional | **CRITICAL** | Cryptographically random 64-character hex string |
| `OAUTHLIB_INSECURE_TRANSPORT` | Dev Only | **NEVER** | `1` during local OAuth HTTP testing / `0` (or unset) in live environments |
| `PORT` | Optional | Optional | Server port (e.g., `5000` or `8000`) |

---

### Production Deployment Guidance

Deploy the platform behind a WSGI HTTP server such as **Gunicorn** reverse-proxied by Nginx.

#### 1. Execute via Gunicorn

```bash
export FLASK_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export OAUTHLIB_INSECURE_TRANSPORT=0

gunicorn --workers 4 --bind 0.0.0.0:8000 "app:app"
```

#### 2. Enforce Session Security Cookies

Ensure session cookie flags are declared in `app.py` or configured prior to live launch:

```python
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,    # Requires HTTPS proxy termination
    SESSION_COOKIE_SAMESITE='Lax',
)
```

---

## 7. Application Routes & Navigation

| Route Path | Method(s) | Functionality & Rendered View |
| :--- | :--- | :--- |
| `/` | `GET` | Homepage landing view (`index.html`) |
| `/login` | `GET`, `POST` | User login form processing (`login.html`) |
| `/register` | `GET`, `POST` | New user account creation (`register.html`) |
| `/dashboard` | `GET` | Main analytics control center (`dashboard.html`) |
| `/stats` | `GET` | Vehicle dataset summary metrics (`stats.html`, reads `price_stats.txt`) |
| `/skewness` | `GET` | Column skewness metrics (`skewness_results.html`) |
| `/iqr` | `GET`, `POST` | IQR outlier calculation parameters (`iqr.html`) |
| `/cleaned-data` | `GET` | View dataset with outliers removed (`cleaned_data.html`) |
| `/correlation` | `GET` | Multivariate correlation heatmaps (`correlation.html`) |
| `/time-series` | `GET` | Temporal price trend chart view (`time_series.html`) |
| `/predict` | `GET`, `POST` | Vehicle price estimation model inference form (`predict_price.html`) |
| `/regression` | `GET` | Regression performance metrics display (`regression.html`) |
| `/search` | `GET`, `POST` | Vehicle make/model query lookup (`search_model.html`) |
| `/users` | `GET` | Administrative user directory management (`view_users.html`) |
| `/logout` | `GET` | Session termination and redirect |

---

## 8. Troubleshooting & FAQ

* **ModuleNotFoundError: No module named 'flask'**
  * *Solution:* Verify your virtual environment is active (`source venv/bin/activate` or `.\venv\Scripts\Activate.ps1`) and run `pip install -r requirements.txt`.
* **FileNotFoundError for `price_stats.txt` or `regression_metrics.txt`**
  * *Solution:* Ensure `car_analysis.py` has executed at least once to process the underlying CSV dataset and generate summary output metric text files.
* **Port 5000 in use error (`Address already in use`)**
  * *Solution:* Pass the custom port flag: `flask run --port 5001` or terminate the process using port 5000 (`lsof -ti:5000 | xargs kill -9` on Unix).
* **OAuth authentication error on HTTP**
  * *Solution:* For local dev environments only, ensure `export OAUTHLIB_INSECURE_TRANSPORT=1` is set in your shell session.

---

## 9. License & Maintenance

* **License:** Internal proprietary software / Standard MIT License (refer to repository root LICENSE file if applicable).
* **Maintenance & Support:** For bugs, feature requests, or technical support, contact the project maintainers or submit an issue ticket in the project repository.