# Prophet model for time series forecasting (seasonality, trend, holidays)
from prophet import Prophet  

# ARIMA model from statsmodels for classical statistical time series forecasting
from statsmodels.tsa.arima.model import ARIMA  

# AWS Boto3-specific exceptions for error handling in S3 operations
from botocore.exceptions import BotoCoreError, ClientError  

# Streamlit library for building interactive web apps
import streamlit as st  

# Pandas for data manipulation and analysis (CSV reading, dataframes)
import pandas as pd  

# NumPy for numerical operations (arrays, math functions)
import numpy as np  

# Matplotlib for plotting charts and visualizations
import matplotlib.pyplot as plt  


# Boto3 SDK for interacting with AWS services (S3 in our case)
import boto3  

# pickle for saving and loading Python objects (models, forecasts)
import pickle  

# os module for file system and environment variable operations
import os  

# Importing 'datetime' and 'timezone' classes from the built-in datetime module.
# - datetime: used to work with dates and times (e.g., current time, formatting timestamps).
# - timezone: used to create timezone-aware datetime objects (important for UTC and avoiding deprecation warnings).
from datetime import date, datetime, timezone

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
    # Creates a slider widget in the Streamlit UI for selecting how many months of past data to use.
    # Arguments:
    #   "Months of past data:" → Label shown in the app.
    #   6   → Minimum selectable value (6 months).
    #   24  → Maximum selectable value (24 months).
    #   12  → Default value when the app loads (12 months).
    months_back = st.slider("Months of past data:", 6, 24, 12)

    today = date.today()
    dates = pd.date_range(end=today, periods=months_back, freq="M")
    expenses = [
        st.number_input(
            f"{d.strftime('%B %Y')}",  # Label for input (e.g., "January 2025")
            0.0,                       # Minimum value allowed
            1000000.0,                  # Maximum value allowed
            500.0,                     # Default value shown
            step=10.0                  # Increment/decrement step
        )
        for d in dates                 # Loop over every date in the 'dates' list
    ]

    df = pd.DataFrame({"date": dates, "expense": expenses})

# --------------------------
# Forecast Section
# --------------------------
if "df" in locals() and not df.empty:
    periods = st.slider("Forecast months ahead:", 1, 24, 12)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if loaded_model:
        if isinstance(loaded_model, Prophet):
            # 1. Prophet expects the training data to have:
            #    - 'ds' column for dates
            #    - 'y' column for the target values
            # So here we rename from our CSV-friendly column names ("date", "expense") to Prophet's format.
            prophet_df = df.rename(columns={"date": "ds", "expense": "y"})

            # 2. Create a future dataframe for prediction.
            #    - `periods` = how many months ahead you want to predict
            #    - `freq="M"` means monthly frequency
            future = loaded_model.make_future_dataframe(periods=periods, freq="M")

            # 3. Generate the forecast using the loaded Prophet model.
            #    - This will return a DataFrame with predictions (`yhat`), uncertainty intervals (`yhat_lower`, `yhat_upper`), and the dates (`ds`)
            forecast = loaded_model.predict(future)

            # 4. Plot the forecasted time series (historical + future) using Prophet's built-in plotting function.
            st.pyplot(loaded_model.plot(forecast))

            # 5. Plot the forecast components (trend, yearly seasonality, weekly seasonality, etc.).
            st.pyplot(loaded_model.plot_components(forecast))

            # 6. Prepare a smaller DataFrame with only the essential forecast results
            #    and rename columns back to your app-friendly format:
            #    - `ds` → `date`
            #    - `yhat` → `forecast`
            #    - `yhat_lower` → `lower`
            #    - `yhat_upper` → `upper`
            save_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
                columns={"ds": "date", "yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"}
            )

        else:
            # 1. ARIMA models expect a time series with the datetime column as the index.
            #    Here, we take our DataFrame 'df' and set "date" as the index.
            df_arima = df.set_index("date")

            # 2. Use the loaded ARIMA model to forecast into the future.
            #    - `steps=periods` means predict 'periods' months ahead.
            forecast_values = loaded_model.forecast(steps=periods)

            # 3. Create a date index for the forecast period.
            #    - Starts 30 days after the last known date in the dataset.
            #    - Has 'periods' months, spaced monthly (freq='M').
            forecast_index = pd.date_range(
                df_arima.index[-1] + pd.Timedelta(days=30),
                periods=periods,
                freq='M'
            )

            # 4. Create a Matplotlib figure to visualize:
            #    - Historical expense data in blue (default)
            #    - Forecasted expense data in red
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_arima.index, df_arima['expense'], label="Historical")
            ax.plot(forecast_index, forecast_values, color="red", label="Forecast")
            ax.legend()

            # 5. Display the plot in Streamlit.
            st.pyplot(fig)

            # 6. Save the forecast results in a new DataFrame for later storage (e.g., AWS S3).
            #    - Converts forecast values to float for compatibility.
            save_df = pd.DataFrame({
                "date": forecast_index,
                "forecast": forecast_values.astype(float)
            })


        st.subheader("Forecast Preview (Loaded Model)")
        st.write(save_df.head())

    else:
        model_choice = st.radio("Choose Model:", ["Prophet", "ARIMA"])
        if model_choice == "Prophet":
            # 1. Prophet requires columns named 'ds' (date) and 'y' (value to forecast).
            #    Here, we rename our DataFrame's columns accordingly.
            prophet_df = df.rename(columns={"date": "ds", "expense": "y"})

            # 2. Create a new Prophet model instance.
            #    - yearly_seasonality=True: detects and models yearly patterns.
            #    - daily_seasonality=False: daily patterns disabled (monthly data won't benefit from this).
            model = Prophet(yearly_seasonality=True, daily_seasonality=False)

            # 3. Fit the Prophet model on the prepared dataset.
            model.fit(prophet_df)

            # 4. Create a future dataframe for predictions.
            #    - periods=periods: number of months ahead to forecast.
            #    - freq="M": monthly intervals.
            future = model.make_future_dataframe(periods=periods, freq="M")

            # 5. Generate forecasts for historical + future dates.
            forecast = model.predict(future)

            # 6. Plot the full forecast (historical + predicted values) in Streamlit.
            st.pyplot(model.plot(forecast))

            # 7. Plot the model's components in Streamlit.
            #    - Trends, seasonality, and holiday effects.
            st.pyplot(model.plot_components(forecast))

            # 8. Create a clean DataFrame with only key forecast values:
            #    - forecast: predicted value
            #    - lower: lower confidence bound
            #    - upper: upper confidence bound
            save_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
                columns={
                    "ds": "date",
                    "yhat": "forecast",
                    "yhat_lower": "lower",
                    "yhat_upper": "upper"
                }
            )

            # 9. Serialize (pickle) the trained model for storage.
            model_bytes = pickle.dumps(model)

            # 10. Create an S3 key (file path in bucket) for saving the model.
            #     - Includes model type and a timestamp for uniqueness.
            pkl_key = f"{S3_PREFIX.rstrip('/')}/{model_choice.lower()}_model_{timestamp_str}.pkl"


        else:
            # 1. Set 'date' as the DataFrame index (time series models require a time-based index).
            df_arima = df.set_index("date")

            # 2. Dynamically choose ARIMA order based on dataset size:
            #    - Small datasets (< 15 points) risk overfitting with large order values,
            #      so use a simpler (1, 1, 1) configuration.
            #    - Larger datasets can handle more complexity (2, 1, 2).
             #   - In ARIMA, the order parameter is a tuple (p, d, q)
            if len(df_arima) < 15:
                # p = 1 means the model will look one previous month’s expense to predict the next month.
                # d = 1 means the model will use first-order differencing.
                # q = 1 means the model uses the previous month’s prediction error to improve the next prediction.
                order = (1, 1, 1)
            else:
                order = (2, 1, 2)

            try:
                # 3. Initialize ARIMA model:
                #    - order=(p, d, q):
                #      p = autoregressive terms
                #      d = differences
                #      q = moving average terms
                model = ARIMA(df_arima['expense'], order=order)

                # 4. Fit ARIMA to historical data.
                model_fit = model.fit()

                # 5. Forecast `periods` months ahead.
                forecast_values = model_fit.forecast(steps=periods)

            except Exception as e:
                # 6. Handle ARIMA errors gracefully and stop execution.
                st.error(f"ARIMA failed: {e}")
                st.stop()

            # 7. Create future date range for forecast values:
            #    - Starts roughly one month after last date in data.
            forecast_index = pd.date_range(
                df_arima.index[-1] + pd.Timedelta(days=30),
                periods=periods,
                freq='M'
            )

            # 8. Plot historical and forecasted data.
            # Create a figure and axis for the plot, setting the figure size to 10 inches wide by 5 inches tall
            fig, ax = plt.subplots(figsize=(10, 5))

            # Plot the historical expense data
            # x-axis: dates from df_arima's index
            # y-axis: actual expense values
            # label: used for the legend to identify this line as "Historical"
            ax.plot(df_arima.index, df_arima['expense'], label="Historical")

            # Plot the forecasted expense values
            # x-axis: future dates generated in forecast_index
            # y-axis: predicted expense values from ARIMA model
            # color: red to visually distinguish from historical data
            # label: used for the legend to identify this line as "Forecast"
            ax.plot(forecast_index, forecast_values, color="red", label="Forecast")

            # Display the legend so users can differentiate between historical and forecast lines
            ax.legend()


            # 9. Render the plot in the Streamlit app.
            st.pyplot(fig)

            # 10. Create DataFrame with forecast results.
            save_df = pd.DataFrame({
                "date": forecast_index,
                "forecast": forecast_values.astype(float)
            })

            # 11. Serialize (pickle) the trained ARIMA model for saving to S3.
            model_bytes = pickle.dumps(model_fit)

            # 12. Build the S3 key path with model type and timestamp.
            pkl_key = f"{S3_PREFIX.rstrip('/')}/{model_choice.lower()}_model_{timestamp_str}.pkl"


        # Convert the forecast DataFrame (save_df) to CSV format without the index column
        csv_bytes = save_df.to_csv(index=False).encode()

        # Create the S3 key (file path) for the forecast CSV, including the model type and timestamp
        csv_key = f"{S3_PREFIX.rstrip('/')}/{model_choice.lower()}_forecast_{timestamp_str}.csv"

        # Upload the forecast CSV bytes to the specified S3 bucket and key, with content type set to "text/csv"
        upload_bytes_to_s3(csv_bytes, csv_key, "text/csv")

        # Upload the serialized (pickled) model bytes to S3 with content type "application/octet-stream"
        upload_bytes_to_s3(model_bytes, pkl_key, "application/octet-stream")

        # Show a success message in the Streamlit app confirming upload location
        st.success(f"Saved forecast + model to S3 at {S3_PREFIX}")
