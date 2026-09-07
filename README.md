# 🚗 Car Market Data Analysis & Price Prediction Platform

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Framework](https://img.shields.io/badge/framework-Flask%20%2F%20FastAPI-green.svg)
![Data Science](https://img.shields.io/badge/libraries-Pandas%20%7C%20NumPy%20%7C%20Scikit--Learn-orange.svg)
![Docker](https://img.shields.io/badge/containerization-Docker%20%7C%20Docker%20Compose-blue.svg)
![CI/CD](https://img.shields.io/badge/build-GitHub%20Actions-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

An end-to-end web platform and computational backend designed for exploratory automobile market data analysis, statistical metric reporting, Interquartile Range (IQR) outlier detection and cleaning, multivariate correlation modeling, time series price tracking, and machine learning price estimation.

---

## 📋 Table of Contents

1. [Project Overview & Core Business Logic](#1-project-overview--core-business-logic)
2. [Key Features](#2-key-features)
3. [Architecture & Technology Stack](#3-architecture--technology-stack)
4. [Annotated Repository Directory Structure](#4-annotated-repository-directory-structure)
5. [Dataset Requirements](#5-dataset-requirements)
6. [Getting Started & Installation](#6-getting-started--installation)
   - [Prerequisites](#prerequisites)
   - [Local Setup Guide (Flask Workflow)](#local-setup-guide-flask-workflow)
   - [Environment Configuration (`.env`)](#environment-configuration-env)
   - [Running with Docker & Docker Compose](#running-with-docker--docker-compose)
7. [Analytics & ML Processing Workflow (`car_analysis.py`)](#7-analytics--ml-processing-workflow-car_analysispy)
   - [Workflow Pipeline](#workflow-pipeline)
   - [Generated Report Artifacts](#generated-report-artifacts)
8. [CI/CD Pipeline](#8-cicd-pipeline)
9. [Troubleshooting & Edge Cases](#9-troubleshooting--edge-cases)
   - [Dockerfile Entry Point vs. Local Flask Mismatch](#dockerfile-entry-point-vs-local-flask-mismatch)
   - [Missing Input Data File Handling](#missing-input-data-file-handling)
   - [Port Conflicts & Architecture Compilation](#port-conflicts--architecture-compilation)
10. [Security & Production Hardening Guidelines](#10-security--production-hardening-guidelines)
11. [Application Routes & UI Reference](#11-application-routes--ui-reference)
12. [License & Maintenance](#12-license--maintenance)

---

## 1. Project Overview & Core Business Logic

The **Car Market Data Analysis & Price Prediction Platform** combines computational statistical analytics with interactive web instrumentation. The application is engineered to assist data analysts, developers, and automobile industry specialists in converting raw vehicular market data into actionable pricing and evaluation metrics.

### Core Business Objectives
* **Data Cleaning & Noise Reduction:** Filter skewed vehicle valuations and statistical anomalies using custom Interquartile Range (IQR) thresholds.
* **Exploratory Data Analysis (EDA):** Compute summary statistics, skewness distributions, temporal pricing trends, and multivariate linear correlations.
* **Valuation Estimation:** Deploy machine learning regression models (e.g., Random Forest, Linear Regression) to estimate current vehicle resale market values based on key attributes such as mileage, engine capacity, vehicle age, and brand.
* **Operational Web Interface:** Provide user authentication, administration controls, search filters, and real-time visualization dashboards.

---

## 2. Key Features

* **User Authentication & Session Management:** Secure registration (`/register`), login (`/login`), user listing (`/view_users`), and role-aware navigation protected with hashed passwords.
* **Interactive Dashboards (`/dashboard`):** Real-time aggregation of statistical parameters, model lookups, and direct access to dataset inspection toolkits.
* **Vehicle Price Prediction (`/predict_price`):** Dynamic input forms feeding regression engines to estimate market values for specific vehicle criteria.
* **Statistical Metrics Reporting (`/stats`):** Automatic generation and interactive viewing of mean, median, standard deviation, variance, and percentiles (`price_stats.txt`).
* **Skewness & Normality Analysis (`/skewness`):** Identification of distribution tailing across pricing and operational vehicle metrics (`skewness_results.txt`).
* **Interquartile Range Outlier Filtering (`/iqr` & `/cleaned-data`):** Parameterized statistical boundary definition ($Q1 - 1.5 \times IQR$ to $Q3 + 1.5 \times IQR$) to strip spurious market outliers.
* **Multivariate Correlation Visualizer (`/correlation`):** Matrix charts revealing cross-variable feature collinearity.
* **Time Series Trend Analysis (`/time-series`):** Temporal price progression tracking mapped against manufacturing years.
* **Model Search Engine (`/search_model`):** Filterable search interface for exploring historic vehicle listings by make, model, and year.

---

## 3. Architecture & Technology Stack

```text
  +-----------------------------------------------------------------------+
  |                          Web Frontend UI                              |
  |             (HTML5 / CSS3 / JavaScript / Jinja2 Templates)             |
  +-----------------------------------+-----------------------------------+
                                      |
                                      v
  +-----------------------------------------------------------------------+
  |                     Backend Application Engine                        |
  |            Flask (app.py) / FastAPI App Entrypoint (app.main)        |
  +-----------------------------------+-----------------------------------+
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
  +----------------------------------+   +--------------------------------+
  |  Data Science & Processing Engine|   | Dynamic Analytics Report Files |
  |       (car_analysis.py)          |   |  - price_stats.txt             |
  |   Pandas | NumPy | Scikit-Learn  |   |  - regression_metrics.txt      |
  +----------------------------------+   |  - skewness_results.txt        |
                                         +--------------------------------+
```

### Technology Breakdown

| Layer | Technology / Library | Purpose |
|---|---|---|
| **Language** | Python 3.11+ | Primary application runtime |
| **Web Framework** | Flask 3.x / FastAPI | Routing, controller execution, and session management |
| **Templating Engine** | Jinja2 | Dynamic HTML UI rendering |
| **Frontend Utilities** | HTML5, CSS3 (`static/css/styles.css`), Vanilla JS (`static/js/scripts.js`) | User interaction and visual dashboard styling |
| **Data Processing** | Pandas, NumPy | Data manipulation, matrix calculations, and dataset transformations |
| **Machine Learning & Stats** | Scikit-Learn, Statsmodels | Regression modeling, IQR calculations, skewness assessment |
| **Containerization** | Docker, Docker Compose | Application containerization and service orchestration |
| **CI/CD Automation** | GitHub Actions (`.github/workflows/ci.yml`) | Continuous Integration, automated testing, and linting |

---

## 4. Annotated Repository Directory Structure

```text
car-analysis-platform/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions pipeline for linting, testing, and Docker verification
├── static/
│   ├── css/
│   │   └── styles.css             # Main stylesheet for dynamic web pages and responsive layout
│   └── js/
│       └── scripts.js             # Client-side user interactions, form validation, and dashboard logic
├── templates/                     # Jinja2 HTML layout templates
│   ├── analysis.html              # High-level data analysis portal
│   ├── base.html                  # Core layout frame, header, navigation bar, and footer
│   ├── cleaned_data.html          # View raw dataset table post-IQR outlier removal
│   ├── correlation.html           # Feature correlation heatmaps and matrix display
│   ├── dashboard.html             # Central analytics overview dashboard
│   ├── index.html                 # Public landing page
│   ├── iqr.html                   # Interquartile range threshold customization UI
│   ├── login.html                 # User authentication login form
│   ├── predict_price.html         # Interactive form for machine learning vehicle price inference
│   ├── register.html              # Account creation interface
│   ├── regression.html            # Model training outputs and evaluation metrics ($R^2$, RMSE, MAE)
│   ├── search_model.html          # Vehicle make/model search tool
│   ├── stats.html                 # Rendered view of general summary statistics
│   ├── time_series.html           # Price trends across manufacturing years
│   └── view_users.html            # User account management view
├── .env.example                   # Standard environment variable template
├── Dockerfile                     # Container construction instructions
├── docker-compose.yml             # Service orchestration configuration
├── app.py                         # Primary Flask web application and HTTP route handlers
├── car_analysis.py                # Standalone data analysis, feature engineering, and ML script
├── aa.py                          # Auxiliary operational script / workspace helper
├── requirements.txt               # Locked Python dependencies list
├── price_stats.txt                # Dynamic output: Generated price summary statistics
├── regression_metrics.txt         # Dynamic output: Generated model accuracy & error metrics
├── skewness_results.txt           # Dynamic output: Column-wise distribution skewness calculations
└── README.md                      # System documentation (this file)
```

---

## 5. Dataset Requirements

To enable `car_analysis.py` and the application's backend statistical processors to operate correctly, a raw vehicle dataset in CSV format (e.g., `car_data.csv` or `cars.csv`) must be present in the project root directory or referenced path.

### Expected Schema Specifications

| Column Name | Data Type | Description | Example |
|---|---|---|---|
| `Year` | Integer | Manufacturing year of the vehicle | `2018` |
| `Selling_Price` | Float / Int | Price at which the car is listed/sold ($USD) | `12500.00` |
| `Present_Price` | Float / Int | Current showroom/new retail price ($USD) | `20000.00` |
| `Kms_Driven` | Integer | Total distance odometer reading in kilometers | `45000` |
| `Fuel_Type` | String / Categorical | Fuel system type (`Petrol`, `Diesel`, `CNG`) | `Petrol` |
| `Seller_Type` | String / Categorical | Sales channel (`Dealer`, `Individual`) | `Dealer` |
| `Transmission` | String / Categorical | Gearbox type (`Manual`, `Automatic`) | `Manual` |
| `Owner` | Integer | Number of previous owners | `0` |

*Note: If no dataset is present upon startup, `car_analysis.py` will attempt to fall back to a mock sample generator or output missing-file alerts in the generated logs.*

---

## 6. Getting Started & Installation

### Prerequisites
* **Python:** Version 3.11 or higher installed on host machine.
* **Git:** Version control system.
* **Docker & Docker Compose:** Required only for containerized deployment.

---

### Local Setup Guide (Flask Workflow)

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-org/car-analysis-platform.git
   cd car-analysis-platform
   ```

2. **Create and Activate a Virtual Environment:**
   * **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   * **Windows (Command Prompt / PowerShell):**
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Initialize Environment Variables:**
   Copy the example file to `.env`:
   ```bash
   cp .env.example .env
   ```
   *(On Windows CMD: `copy .env.example .env`)*

5. **Execute Analytics Pre-Processing Script:**
   Generate initial reports (`price_stats.txt`, `regression_metrics.txt`, `skewness_results.txt`):
   ```bash
   python car_analysis.py
   ```

6. **Start the Flask Application:**
   ```bash
   python app.py
   ```
   Access the web interface in your browser at `http://127.0.0.1:5000/`.

---

### Environment Configuration (`.env`)

Configure operational settings in your local `.env` file:

```env
# Flask Application Configuration
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=c3a9f8b42e7d101569a4e82b7f3d90e14a1c5b8d9e2f3a4b5c6d7e8f9a0b1c2d

# Database & Path Configurations
DATABASE_URL=sqlite:///car_platform.db
DATASET_PATH=car_data.csv

# Server Port
PORT=5000
```

---

### Running with Docker & Docker Compose

#### Option 1: Multi-Container Orchestration (Docker Compose)
To spin up the containerized environment:

```bash
docker-compose up --build -d
```
The application will be accessible at `http://localhost:8000` (or `http://localhost:5000` based on `docker-compose.yml` port mappings).

To stop the containers:
```bash
docker-compose down
```

#### Option 2: Single Container Execution via Dockerfile

1. **Build the Image:**
   ```bash
   docker build -t car-analysis-app .
   ```

2. **Run the Container:**
   ```bash
   docker run -d -p 5000:5000 --name car-analysis-container --env-file .env car-analysis-app
   ```

---

## 7. Analytics & ML Processing Workflow (`car_analysis.py`)

The `car_analysis.py` module acts as the core statistical engine. It runs independently or as a background module during web service initialization.

```text
                   +---------------------------+
                   |   Raw Input Data (CSV)    |
                   +-------------+-------------+
                                 |
                                 v
                   +---------------------------+
                   |  Exploratory Data Analysis|
                   +-------------+-------------+
                                 |
         +-----------------------+-----------------------+
         |                       |                       |
         v                       v                       v
+------------------+   +-------------------+   +--------------------+
| Price Statistics |   | Skewness Assessment|   | Outlier Removal    |
| Mean, Median, Std|   | Tail Distribution |   | Bounds (IQR Method)|
+--------+---------+   +---------+---------+   +---------+----------+
         |                       |                       |
         v                       v                       v
  price_stats.txt       skewness_results.txt     Cleaned Dataset
                                                         |
                                                         v
                                               +--------------------+
                                               | Regression Model   |
                                               | Feature Engineering|
                                               +---------+----------+
                                                         |
                                                         v
                                             regression_metrics.txt
```

### Generated Report Artifacts

* **`price_stats.txt`:** Contains detailed mathematical breakdowns for target variables:
  * Mean, Median, Variance, Standard Deviation
  * Quartiles ($25\%$, $50\%$, $75\%$) and Min/Max ranges
* **`skewness_results.txt`:** Reports skewness coefficients for numeric columns:
  * Positive/Right-skewed vs. Negative/Left-skewed feature distribution indicators
  * Log-transform recommendations for machine learning preprocessing
* **`regression_metrics.txt`:** Summarizes evaluation scores for price prediction models:
  * Coefficient of Determination ($R^2$ Score)
  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)

---

## 8. CI/CD Pipeline

Continuous Integration is powered by GitHub Actions in `.github/workflows/ci.yml`.

### Workflow Workflow Pipeline Jobs:
1. **Lint & Code Quality Check:** Runs `flake8` or `black` to enforce Python standard style compliance.
2. **Dependency Verification:** Validates `requirements.txt` compatibility under Python 3.11+.
3. **Automated Analytics Execution:** Executes `python car_analysis.py` in test mode to verify output file generation (`price_stats.txt`, `regression_metrics.txt`, `skewness_results.txt`).
4. **Docker Image Build Verification:** Executes `docker build` to guarantee container compilation validity prior to main branch merges.

---

## 9. Troubleshooting & Edge Cases

### Dockerfile Entry Point vs. Local Flask Mismatch
* **Issue:** The repository Dockerfile specifies an entry command referencing `app.main:app` (FastAPI/Uvicorn runtime structure), whereas running locally uses `python app.py` (Flask execution).
* **Resolution / Standard Operating Procedure:**
  * **For Local Flask Testing:** Always execute `python app.py` within your active virtual environment.
  * **For Docker Deployment:** If using `docker-compose.yml`, verify whether Uvicorn or Flask Gunicorn is selected as the primary process manager. If deployment errors occur in single-container mode, align the `Dockerfile` `CMD` command with `CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]` or `CMD ["python", "app.py"]`.

### Missing Input Data File Handling
* **Issue:** Executing `car_analysis.py` without a dataset present results in `FileNotFoundError`.
* **Resolution:** Ensure `car_data.csv` is placed in the project root directory. Alternatively, update the dataset path variable in `.env`:
  ```bash
  DATASET_PATH=/path/to/your/dataset.csv
  ```

### Port Conflicts & Architecture Compilation
* **Port 5000 in Use (macOS AirPlay / Control Center Conflict):**
  On macOS Monterey or newer, system services may occupy port `5000`. You can change the port in `app.py` or run:
  ```bash
  flask run --port 5001
  ```
* **ARM64 / Apple Silicon Compilation Issues:**
  If Scikit-Learn or NumPy fails to compile under Docker on Apple Silicon ($M1/M2/M3$), ensure your `Dockerfile` uses an explicit platform specifier:
  ```dockerfile
  FROM --platform=linux/amd64 python:3.11-slim
  ```

---

## 10. Security & Production Hardening Guidelines

1. **Secrets Management:**
   * **NEVER** commit production `.env` files or hardcode API keys into templates/scripts.
   * Generate secure session keys for production:
     ```bash
     python -c 'import secrets; print(secrets.token_hex(32))'
     ```
2. **Session Security & Cookies:**
   * Enable `SESSION_COOKIE_HTTPONLY = True`, `SESSION_COOKIE_SECURE = True`, and `SESSION_COOKIE_SAMESITE = 'Lax'` when serving over HTTPS.
3. **Machine Learning Model Deserialization Safety:**
   * Avoid loading untrusted `.pkl` or `.joblib` model binaries from public or unverified remote sources to prevent Arbitrary Code Execution vulnerabilities. Verify SHA-256 hashes for serialized weights.
4. **Non-Root Docker Container Execution:**
   * Ensure container processes run under an unprivileged user inside the `Dockerfile`:
     ```dockerfile
     RUN useradd -m appuser
     USER appuser
     ```

---

## 11. Application Routes & UI Reference

| Route Path | HTTP Method | Associated Template | Purpose / Description |
|---|---|---|---|
| `/` | GET | `index.html` | Public landing page and platform introduction |
| `/login` | GET, POST | `login.html` | User login authentication interface |
| `/register` | GET, POST | `register.html` | User registration and account creation |
| `/dashboard` | GET | `dashboard.html` | Core metrics summary dashboard |
| `/predict_price` | GET, POST | `predict_price.html` | Price estimation ML model inference form |
| `/stats` | GET | `stats.html` | Displays dataset summary statistics from `price_stats.txt` |
| `/skewness` | GET | `analysis.html` | Displays numerical distribution skewness metrics |
| `/iqr` | GET, POST | `iqr.html` | Configures Interquartile Range outlier filtering |
| `/cleaned-data` | GET | `cleaned_data.html` | Inspects cleaned dataset post-IQR processing |
| `/correlation` | GET | `correlation.html` | Feature correlation matrix visualizer |
| `/time-series` | GET | `time_series.html` | Temporal price trend tracker across manufacturing years |
| `/regression` | GET | `regression.html` | Machine learning model evaluation metrics ($R^2$, MAE, RMSE) |
| `/search_model` | GET, POST | `search_model.html` | Search interface for filtering car models |
| `/view_users` | GET | `view_users.html` | Administrative view for account auditing |

---

## 12. License & Maintenance

This project is distributed under the **MIT License**.

**Maintainer:** Development & Analytics Engineering Team  
**Issue Tracking & Support:** Please log technical questions or bug reports in the repository's GitHub Issues tracker.