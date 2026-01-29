import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from data_loader_lstm import get_lstm_data

print("Loading Data...")
(X_seq_train, X_static_train, y_train, 
 X_seq_test, X_static_test, y_test, 
 scaler_y, _, _) = get_lstm_data()

def build_lstm_model():
    
    input_seq = Input(shape=(20, 1))
    x1 = LSTM(64, return_sequences=True)(input_seq)
    x1 = LSTM(32)(x1) 
    
   
    input_static = Input(shape=(4,)) 
    x2 = Dense(32, activation='relu')(input_static)
    
  
    merged = Concatenate()([x1, x2])
    
  
    z = Dense(400, activation='relu')(merged)
    z = BatchNormalization()(z)
    z = Dense(400, activation='relu')(z)
    z = BatchNormalization()(z)
    z = Dense(400, activation='relu')(z)
    

    output = Dense(1)(z)
    
    model = Model(inputs=[input_seq, input_static], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

model = build_lstm_model()

# 2. Train
print("Training LSTM...")
model.fit(
    [X_seq_train, X_static_train], y_train,
    validation_split=0.2,
    epochs=20, 
    batch_size=256
)

model.save("lstm_model.h5")
print("LSTM Model Trained & Saved.")