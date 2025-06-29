## Import Packages and Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(42) # set random seed for repeatability
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from scipy import stats
import uncertainties as u
from uncertainties.umath import __all__
from uncertainties.umath import *
from scipy.special import erf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf

## Import Data
df = pd.read_csv('BIRD PROP.csv')
df['ds'] = pd.to_datetime(df['ds'])
df.columns = ['ds', 'y']

## Descriptive Data Analysis
df['y'].describe() #descriptive statistics of data
# Data Visualization
plt.figure(figsize=(10,5))
plt.plot(df['ds'], df['y'])
plt.title('Stock Price Movements')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.grid()
plt.show()

df.isnull().sum() #check missing values

## Setting-up The Dataset
data_train = df.iloc[0:190] #to build the model
data_test = df.iloc[191:237] #to check the accuracy of the model

## Building the Prophet (basic) Model
model = Prophet()
model.fit(data_train)
future = model.make_future_dataframe(periods=47)
forecast = model.predict(future)

## Plot of Prediction Results
fig = model.plot(forecast)
plt.title("Stock Price Prediction Using Prophet", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.show()

## Plot of Trend and Seasonal Components
fig2 = model.plot_components(forecast)
plt.show()

## Colored Visualization Plot
plt.figure(figsize=(10, 6))
# Plot Actual Data (only up to the last point of historical data)
plt.plot(df['ds'], df['y'], label="Actual Data (Stock Prices)", color="blue")
# Determine the starting point for predictions
cutoff = data_train['ds'].iloc[-1]  #  Take the latest date of historical data
# Filter prediction results only for the period after the cutoff
forecast_future = forecast[forecast['ds'] > cutoff]
# Prediction Plot
plt.plot(forecast_future['ds'], forecast_future['yhat'], label="Stock Price Predictions", color="red")
# Plot Confidence Interval for the prediction period
plt.fill_between(forecast_future['ds'], forecast_future['yhat_lower'], forecast_future['yhat_upper'], 
                 color='grey', alpha=0.3, label="Confidence Interval (95%)")
# Add a vertical line as a separator between historical data and predictions.
plt.axvline(x=cutoff, color="black", linestyle="--", linewidth=1.5, label="Initial Prediction")
# Add Labels and Titles
plt.title("Stock Price Prediction Using Prophet", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
# Show Plot
plt.show()

## Modification of the Prophet Model
# Viewing prediction results using the Prophet (basic) model
prediksi_test = forecast[['ds', 'yhat']]
print(prediksi_test)
testing = prediksi_test.iloc[191:237]
print(testing)
# Calculating the error value of the Prophet model (basic)
error_data = data_test['y'] - testing['yhat']
print(error_data)
std_error = np.std(error_data) 
# Generating new errors that spread normally
num_samples = 46  
generated_error = np.random.normal(loc=0, scale=std_error, size=num_samples)
print("generated error:", generated_error)
# Prediction results using the Prophet model (modified)
adjusted_predictions = testing['yhat'] + generated_error
print(adjusted_predictions)
# Calculating evaluation metrics
mape = np.mean(np.abs((data_test['y'] - adjusted_predictions) / data_test['y'])) * 100
print(f"MAPE: {mape:.2f}%")
## Colored Visualization Plot
plt.figure(figsize=(10, 6))
# Plot Actual Data (only up to the last point of historical data)
plt.plot(df['ds'], df['y'], label="Actual Data (Stock Prices))", color="blue")
# Determine the starting point for predictions
cutoff = data_train['ds'].iloc[-1]  # Ambil tanggal terakhir data historis
# Filter prediction results only for the period after the cutoff
forecast_future = forecast[forecast['ds'] > cutoff]
# Prediction Plot
plt.plot(testing['ds'], adjusted_predictions, label="Stock Price Predictions", color="red")
# Plot Confidence Interval for the prediction period
plt.fill_between(forecast_future['ds'], forecast_future['yhat_lower'], forecast_future['yhat_upper'], 
                 color='grey', alpha=0.3, label="Confidence Interval (95%)")
# Add a vertical line as a separator between historical data and predictions.
plt.axvline(x=cutoff, color="black", linestyle="--", linewidth=1.5, label="Awal Prediksi")
# Add Labels and Titles
plt.title("Stock Price Prediction Using Prophet with Error", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
# Show Plot
plt.show()
