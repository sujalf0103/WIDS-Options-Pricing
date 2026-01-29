import pandas as pd
import numpy as np
import tensorflow as tf
from data_loader import get_data


print("Loading data...")
X_train, y_train, X_test, y_test, scaler_y = get_data()

print("Loading model...")
model = tf.keras.models.load_model('mlp1_model.h5', compile=False)


model.compile(optimizer='adam', loss='mse')


print("Predicting...")
predictions_scaled = model.predict(X_test)

predictions_rupees = scaler_y.inverse_transform(predictions_scaled)
actual_rupees = scaler_y.inverse_transform(y_test)

results = pd.DataFrame({
    'Actual_Price': actual_rupees.flatten(),
    'Predicted_Price': predictions_rupees.flatten()
})

results['Error'] = results['Predicted_Price'] - results['Actual_Price']
results['Abs_Error'] = results['Error'].abs()

print("\n--- Prediction Sample (First 20) ---")
print(results.head(20))

print("\n--- Model Failures (Largest Errors) ---")
print(results.sort_values(by='Abs_Error', ascending=False).head(5))

print("\n--- Model Success (Smallest Errors) ---")
print(results.sort_values(by='Abs_Error', ascending=True).head(5))