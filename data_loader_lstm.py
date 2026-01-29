import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_lstm_data():
    file_path = "ASIANPAINT_Dataset.xlsx"
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    

    ce_sheets = [df for name, df in all_sheets.items() if 'CE' in name]
    data = pd.concat(ce_sheets, ignore_index=True)

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    

    daily_data = data[['Date', 'underlying_value']].drop_duplicates().set_index('Date')
    all_prices = daily_data['underlying_value'].values
    all_dates = daily_data.index
    
    date_to_seq = {}
    for i in range(20, len(all_prices)):
       
        seq = all_prices[i-20 : i]
        date_to_seq[all_dates[i]] = seq

    data['T_years'] = data['t'] / 365.0
    
    X_seq_list = []
    X_static_list = []
    y_list = []
    sigma_list = [] # We need this for Black-Scholes comparison later!
    
    subset = data[['Date', 'strike_price', 'T_years', 'r', 'close', 'sigma', 'underlying_value']]
    
    for row in subset.itertuples():
        if row.Date in date_to_seq:
            
            X_seq_list.append(date_to_seq[row.Date])
          
            X_static_list.append([row.strike_price, row.T_years, row.r, row.underlying_value])
            y_list.append(row.close)
            sigma_list.append(row.sigma)

    X_seq = np.array(X_seq_list)
    X_static = np.array(X_static_list) 
    y = np.array(y_list)
    sigma = np.array(sigma_list)

    # 3. Split Data (Last 20% is Test)
    split = int(len(y) * 0.8)
    
    # Train
    X_seq_train = X_seq[:split]
    X_static_train = X_static[:split] 
    y_train = y[:split]
    
    # Test
    X_seq_test = X_seq[split:]
    X_static_test = X_static[split:]
    y_test = y[split:]
    sigma_test = sigma[split:] 
    scaler_static = StandardScaler()
    X_static_train_scaled = scaler_static.fit_transform(X_static_train)
    X_static_test_scaled = scaler_static.transform(X_static_test)
    
   
    scaler_seq = StandardScaler()
    flat_train = X_seq_train.reshape(-1, 1)
    scaler_seq.fit(flat_train)
    X_seq_train_scaled = scaler_seq.transform(flat_train).reshape(X_seq_train.shape[0], 20, 1)
    X_seq_test_scaled = scaler_seq.transform(X_seq_test.reshape(-1, 1)).reshape(X_seq_test.shape[0], 20, 1)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    
    return (X_seq_train_scaled, X_static_train_scaled, y_train_scaled, 
            X_seq_test_scaled, X_static_test_scaled, y_test_scaled, 
            scaler_y, X_static_test, sigma_test)