import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_loader import get_data


print("Loading Data...")
X_train, y_train, X_test, y_test, scaler_y = get_data()


def build_model(input_dim):
    model = Sequential()
    

    model.add(Dense(400, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    
    model.add(Dense(400))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    
    model.add(Dense(400))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    
    model.add(Dense(400))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    
  
    model.add(Dense(1)) 
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

model = build_model(X_train.shape[1])


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

print("Starting Training (50 Epochs)...")
history = model.fit(
    X_train, y_train, 
    validation_split=0.2, 
    epochs=50, 
    batch_size=256, 
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


model.save("mlp1_model.h5")
print("Model Saved Successfully.")