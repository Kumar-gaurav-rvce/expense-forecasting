from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from botocore.exceptions import BotoCoreError, ClientError
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import boto3
import pickle
import os

st.set_page_config(page_title="Expense Forecasting App", layout="centered")
st.title("📈 Expense Forecasting with Model History (AWS S3)")

# --------------------------
# AWS from Streamlit Secrets
# --------------------------
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = st.secrets["AWS_DEFAULT_REGION"]
S3_BUCKET = st.secrets["S3_BUCKET"]
S3_PREFIX = st.secrets["S3_PREFIX"]

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )

# --------------------------
# S3 helpers
# --------------------------
def list_s3_files(suffix=None):
    try:
        s3 = get_s3_client()
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        files = [obj["Key"] for obj in resp.get("Contents", [])]
        if suffix:
            files = [f for f in files if f.endswith(suffix)]
        return sorted(files)
    except Exception as e:
        st.error(f"S3 listing failed: {e}")
        return []

def upload_bytes_to_s3(data_bytes, key, content_type):
    try:
        s3 = get_s3_client()
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data_bytes, ContentType=content_type)
        return True
    except (BotoCoreError, ClientError) as e:
        st.error(f"S3 upload failed: {e}")
        return False

def download_s3_object(key):
    try:
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return obj["Body"].read()
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None

# --------------------------
# Sidebar — Model History
# --------------------------
st.sidebar.header("📂 Model History from S3")
model_files = list_s3_files(".pkl")
selected_model_key = st.sidebar.selectbox("Choose saved model to load", ["-- None --"] + model_files)

if selected_model_key != "-- None --":
    model_bytes = download_s3_object(selected_model_key)
    loaded_model = pickle.loads(model_bytes)
    st.sidebar.success(f"Loaded model: {selected_model_key.split('/')[-1]}")
else:
    loaded_model = None

# --------------------------
# Input Method
# --------------------------
input_method = st.radio("Select Input Method:", ["📂 Upload CSV", "✏️ Enter Manually"])

if input_method == "📂 Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV (Date, Expense)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" not in df.columns or "expense" not in df.columns:
            st.error("CSV must have 'Date' and 'Expense'")
            st.stop()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").dropna()

elif input_method == "✏️ Enter Manually":
    months_back = st.slider("Months of past data:", 6, 24, 12)
    today = datetime.date.today()
    dates = pd.date_range(end=today, periods=months_back, freq="M")
    expenses = [st.number_input(f"{d.strftime('%B %Y')}", 0.0, 100000.0, 500.0, step=10.0) for d in dates]
    df = pd.DataFrame({"date": dates, "expense": expenses})

# --------------------------
# Forecast Section
# --------------------------
if "df" in locals() and not df.empty:
    periods = st.slider("Forecast months ahead:", 1, 24, 12)
    timestamp_str = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    if loaded_model:
        if isinstance(loaded_model, Prophet):
            prophet_df = df.rename(columns={"date": "ds", "expense": "y"})
            future = loaded_model.make_future_dataframe(periods=periods, freq="M")
            forecast = loaded_model.predict(future)
            st.pyplot(loaded_model.plot(forecast))
            st.pyplot(loaded_model.plot_components(forecast))
            save_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
                columns={"ds": "date", "yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"}
            )
        else:
            df_arima = df.set_index("date")
            forecast_values = loaded_model.forecast(steps=periods)
            forecast_index = pd.date_range(df_arima.index[-1] + pd.Timedelta(days=30), periods=periods, freq='M')
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_arima.index, df_arima['expense'], label="Historical")
            ax.plot(forecast_index, forecast_values, color="red", label="Forecast")
            ax.legend()
            st.pyplot(fig)
            save_df = pd.DataFrame({"date": forecast_index, "forecast": forecast_values.astype(float)})

        st.subheader("Forecast Preview (Loaded Model)")
        st.write(save_df.head())

    else:
        model_choice = st.radio("Choose Model:", ["Prophet", "ARIMA"])
        if model_choice == "Prophet":
            prophet_df = df.rename(columns={"date": "ds", "expense": "y"})
            model = Prophet(yearly_seasonality=True, daily_seasonality=False)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=periods, freq="M")
            forecast = model.predict(future)
            st.pyplot(model.plot(forecast))
            st.pyplot(model.plot_components(forecast))
            save_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
                columns={"ds": "date", "yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"}
            )
            model_bytes = pickle.dumps(model)
            pkl_key = f"{S3_PREFIX.rstrip('/')}/{model_choice.lower()}_model_{timestamp_str}.pkl"

        else:
            df_arima = df.set_index("date")
            # Handle small dataset by reducing ARIMA order
            if len(df_arima) < 15:
                order = (1, 1, 1)
            else:
                order = (2, 1, 2)

            try:
                model = ARIMA(df_arima['expense'], order=order)
                model_fit = model.fit()
                forecast_values = model_fit.forecast(steps=periods)
            except Exception as e:
                st.error(f"ARIMA failed: {e}")
                st.stop()

            forecast_index = pd.date_range(df_arima.index[-1] + pd.Timedelta(days=30), periods=periods, freq='M')
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_arima.index, df_arima['expense'], label="Historical")
            ax.plot(forecast_index, forecast_values, color="red", label="Forecast")
            ax.legend()
            st.pyplot(fig)
            save_df = pd.DataFrame({"date": forecast_index, "forecast": forecast_values.astype(float)})
            model_bytes = pickle.dumps(model_fit)
            pkl_key = f"{S3_PREFIX.rstrip('/')}/{model_choice.lower()}_model_{timestamp_str}.pkl"

        csv_bytes = save_df.to_csv(index=False).encode()
        csv_key = f"{S3_PREFIX.rstrip('/')}/{model_choice.lower()}_forecast_{timestamp_str}.csv"
        upload_bytes_to_s3(csv_bytes, csv_key, "text/csv")
        upload_bytes_to_s3(model_bytes, pkl_key, "application/octet-stream")
        st.success(f"Saved forecast + model to S3 at {S3_PREFIX}")
