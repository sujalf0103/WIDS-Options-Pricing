import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from data_loader_lstm import get_lstm_data

#  The Black-Scholes Formula
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return call_price


print("Loading Test Data...")
(_, _, _, X_seq_test, X_static_test_scaled, y_test_scaled, 
 scaler_y, X_static_test_raw, sigma_test) = get_lstm_data()


print("Loading LSTM Model...")
model = tf.keras.models.load_model('lstm_model.h5', compile=False)

#  Predict with (LSTM)
print("Running LSTM Predictions...")

lstm_pred_scaled = model.predict([X_seq_test, X_static_test_scaled])
lstm_pred_rupees = scaler_y.inverse_transform(lstm_pred_scaled).flatten()

#  Predict with Black-Scholes (Benchmark)
print("Running Black-Scholes Formula...")
bs_prices = []
actual_prices = scaler_y.inverse_transform(y_test_scaled).flatten()

for i in range(len(actual_prices)):
    K = X_static_test_raw[i][0]
    T = X_static_test_raw[i][1]
    r = X_static_test_raw[i][2]
    S = X_static_test_raw[i][3]
    sig = sigma_test[i]
    
    bs_price = black_scholes_call(S, K, T, r, sig)
    bs_prices.append(bs_price)

bs_prices = np.array(bs_prices)

#  Final Scoreboard
mse_ai = np.mean((actual_prices - lstm_pred_rupees) ** 2)
mse_bs = np.mean((actual_prices - bs_prices) ** 2)

print("\n" + "="*40)
print("     FINAL RESULTS")
print("="*40)
print(f"Black-Scholes MSE:  {mse_bs:.2f}")
print(f"Your LSTM AI MSE:   {mse_ai:.2f}")
print("="*40)

if mse_ai < mse_bs:
    improvement = ((mse_bs - mse_ai) / mse_bs) * 100
    print(f"SUCCESS: You beat Black-Scholes by {improvement:.1f}%")
else:
    print("Keep optimizing! BS is still winning.")