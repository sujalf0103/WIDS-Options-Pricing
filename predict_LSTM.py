import pandas as pd
import tensorflow as tf
from data_loader_lstm import get_lstm_data

print("Loading Data...")
(_, _, _, X_seq_test, X_static_test_scaled, y_test_scaled, 
 scaler_y, _, _) = get_lstm_data()

print("Loading Model...")
model = tf.keras.models.load_model('lstm_model.h5', compile=False)


print("Predicting...")
predictions_scaled = model.predict([X_seq_test, X_static_test_scaled])

predictions_rupees = scaler_y.inverse_transform(predictions_scaled)
actual_rupees = scaler_y.inverse_transform(y_test_scaled)


results = pd.DataFrame({
    'Actual_Price': actual_rupees.flatten(),
    'Predicted_Price': predictions_rupees.flatten()
})

results['Error'] = results['Predicted_Price'] - results['Actual_Price']
results['Abs_Error'] = results['Error'].abs()

print("\n--- Prediction Sample (First 20) ---")
print(results.head(20))

print("\n--- Largest Errors (Where did it fail?) ---")
print(results.sort_values(by='Abs_Error', ascending=False).head(5))