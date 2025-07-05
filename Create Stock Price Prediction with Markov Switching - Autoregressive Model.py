## Import Packages and Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from scipy.stats import lognorm
from scipy.stats import kstest
from sklearn.cluster import KMeans
from scipy.linalg import eig
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

## Import Data 
data = pd.read_csv('BIRD FIX.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
df = pd.DataFrame(data)

# Dividing data sequentially: 191 data for training and 46 data for testing
df_train = df[:191]
df_test = df[191:191+46]

# Calculating log return
log_returns = np.log(df_train['Close'] / df_train['Close'].shift(1))

# Filter out NaN values that may appear in the first row
log_returns = log_returns.dropna()

# Showing the first few lines of the log return
print(log_returns.head())

## Fit model MS-AR dengan 2 regime dan order 3
order = 3 
model = MarkovAutoregression(log_returns, k_regimes=2, order=order, switching_ar=True)
result = model.fit()

# Displaying the Model Summary
print(result.summary())

# Taking the probability of each state
state_probs = result.smoothed_marginal_probabilities
print(state_probs.head())
predicted_states = state_probs.idxmax(axis=1)
resultp_df = pd.DataFrame({
    "log_return": log_returns,
    "predicted_state": predicted_states
})
print(resultp_df.head())

# Take the transition matrix between states
transition_matrix = result.regime_transition

# Take the probability of the last state
last_state_probs = result.smoothed_marginal_probabilities.iloc[-1]

# Determine the initial state based on the probability of the last state
current_state = np.random.choice([0, 1], p=last_state_probs)

print(f"Current State: {current_state}")
transition_matrix = np.squeeze(result.regime_transition)
print(transition_matrix)
print(transition_matrix.shape)
if np.allclose(transition_matrix.sum(axis=0), 1):
    transition_matrix = transition_matrix.T
print("Matrix After Correction:")
print(transition_matrix)

simulated_states = []
n_steps = 46
last_state_probs = result.smoothed_marginal_probabilities.iloc[-1]
current_state = np.random.choice([0, 1], p=last_state_probs)
for _ in range(n_steps):
    simulated_states.append(current_state)
    probs = transition_matrix[current_state]
    probs = np.squeeze(probs)
    probs = probs / probs.sum()
    current_state = np.random.choice([0, 1], p=probs)
print("State Route 46 Period:", simulated_states)
y_t_1 = log_returns.iloc[-1]
y_t_2 = log_returns.iloc[-2]
y_t_3 = log_returns.iloc[-3]
forecast_values = []
for state in simulated_states:
    if state == 0:
        const = result.params['const[0]']
        ar1 = result.params['ar.L1[0]']
        ar2 = result.params['ar.L2[0]']
        ar3 = result.params['ar.L3[0]']
    else:
        const = result.params['const[1]']
        ar1 = result.params['ar.L1[1]']
        ar2 = result.params['ar.L2[1]']
        ar3 = result.params['ar.L3[1]']
    
    y_forecast = const + ar1 * y_t_1 + ar2 * y_t_2 + ar3 * y_t_3
    forecast_values.append(y_forecast)
    
    y_t_3 = y_t_2
    y_t_2 = y_t_1
    y_t_1 = y_forecast
  import pandas as pd

result_forecast_df = pd.DataFrame({
    "forecast_log_return": forecast_values,
    "predicted_state": simulated_states
})

print(result_forecast_df)

plt.figure(figsize=(12,6))
plt.plot(result_forecast_df["forecast_log_return"], marker='o', label='Forecast Log Return', color='black')

state_1 = result_forecast_df.index[result_forecast_df['predicted_state'] == 0]
state_2 = result_forecast_df.index[result_forecast_df['predicted_state'] == 1]

plt.scatter(state_1, result_forecast_df.loc[state_1, 'forecast_log_return'], color='green', label='State 1', s=50)
plt.scatter(state_2, result_forecast_df.loc[state_2, 'forecast_log_return'], color='red', label='State 2', s=50)

plt.title("Forecast Log Return for the Next 46 Periods with State Change")
plt.xlabel("Period")
plt.ylabel("Forecast Log Return")
plt.legend()
plt.grid()
plt.show()

last_price = 1895

forecast_prices = [last_price]

# Converting log returns to stock prices
for ret in forecast_values:
    next_price = forecast_prices[-1] * np.exp(ret)
    forecast_prices.append(next_price)

forecast_prices = forecast_prices[1:]

result_forecast_price_df = pd.DataFrame({
    "forecast_price": forecast_prices,
    "predicted_state": simulated_states
})

print(result_forecast_price_df.head())

# Calculating MAPE
mape = np.mean(np.abs((df_test['Close'] - forecast_prices) / df_test['Close'])) * 100

print(f"MAPE: {mape:.2f}%")

plt.figure(figsize=(12, 6))

## Create Plot 
plt.plot(df.index, df['Close'], label="Actual Data (Stock Prices)", color="blue")
# Starting point of prediction
cutoff = df_test.index[0]
# Prediction Results DataFrame + State
predicted_prices_df = pd.DataFrame({
    'Date': df_test.index,
    'forecast_price': forecast_prices,
    'State': simulated_states
})
plt.plot(predicted_prices_df['Date'], predicted_prices_df['forecast_price'], color="gray", linestyle="--", label="Prediksi Harga (Semua State)")
# Highlight State 1
state_1 = predicted_prices_df[predicted_prices_df['State'] == 0]
plt.scatter(state_1['Date'], state_1['forecast_price'], color="#39FF14", label="State 1", s=50)
# Highlight State 2
state_2 = predicted_prices_df[predicted_prices_df['State'] == 1]
plt.scatter(state_2['Date'], state_2['forecast_price'], color="#FF073A", label="State 2", s=50)
plt.axvline(x=cutoff, color="black", linestyle="--", linewidth=1.5, label="Awal Prediksi")
plt.title("Stock Price Prediction Based on Log Return with Highlight State", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
