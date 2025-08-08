# 📈 Expense Forecasting App

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B)](https://streamlit.io/)
[![AWS S3](https://img.shields.io/badge/AWS-S3-orange)](https://aws.amazon.com/s3/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A simple **Streamlit** app for forecasting expenses using **Prophet** and **ARIMA** models,  
with support for **AWS S3** storage of forecasts and trained models.

---

## 🚀 Features

- **Two Forecasting Models**
  - [Prophet](https://facebook.github.io/prophet/) — seasonality-aware forecasting
  - [ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) — classic time series modeling  
- **Flexible Input Options**
  - Upload a CSV (`Date`, `Expense`)
  - Enter expense history manually
- **AWS S3 Integration**
  - Save & load trained models
  - Store forecast outputs as CSV
- **Interactive Visualizations**
  - Historical vs forecast plots
  - Prophet component breakdown

---

## 📂 CSV Format

Your input CSV **must** have two columns:  
- **Date** in `YYYY-MM-DD` format  
- **Expense** as a numeric value  

Example:

Date,Expense  
2024-01-31,1500  
2024-02-29,1750  
2024-03-31,1620  

---

## 🛠 Installation

1. Clone the repository:  
   `git clone git@github.com:Kumar-gaurav-rvce/expense-forecasting.git`  
   `cd expense-forecasting`  

2. Install Python dependencies:  
   `pip install -r requirements.txt`  

---

## ▶ Run Locally

Run the app with:  
`streamlit run app.py`

---

## 📊 Quick Start with Example CSV

Example data you can use:  

Date,Expense  
2024-01-01,1200  
2024-02-01,1500  
2024-03-01,1000  
2024-04-01,1800  

Upload it via the **📂 Upload CSV** option in the app.

---

## ☁ Deployment on Streamlit Cloud

1. Push the code to your GitHub repository.  
2. Go to [Streamlit Cloud](https://share.streamlit.io/) and create a **New App** from your repo.  
3. In **Settings > Secrets**, add:  

AWS_ACCESS_KEY_ID="your-access-key"  
AWS_SECRET_ACCESS_KEY="your-secret-key"  
AWS_DEFAULT_REGION="your-region"  
S3_BUCKET="your-bucket-name"  
S3_PREFIX="your/prefix/path"  

4. Deploy 🚀

---

## 📦 S3 Structure

my-bucket/  
└── expense-forecast/  
  ├── prophet_model_YYYYMMDDTHHMMSSZ.pkl  
  ├── prophet_forecast_YYYYMMDDTHHMMSSZ.csv  
  ├── arima_model_YYYYMMDDTHHMMSSZ.pkl  
  └── arima_forecast_YYYYMMDDTHHMMSSZ.csv  

---

## 🔄 Workflow Diagram (ASCII)

+-----------------------+  
| CSV or Manual Input   |  
+----------+------------+  
           |  
           v  
+-----------------------+  
| Model Training        |  
| (Prophet / ARIMA)     |  
+----------+------------+  
           |  
           v  
+-----------------------+  
| Forecast Results      |  
+----------+------------+  
           |  
           v  
+-----------------------+  
| Save Forecast + Model |  
| to AWS S3             |  
+----------+------------+  
           |  
           v  
+-----------------------+  
| Reload from S3        |  
| for Future Forecasts  |  
+-----------------------+  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
