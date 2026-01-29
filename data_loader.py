import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_data():
    file_path = "ASIANPAINT_Dataset.xlsx"

    all_sheets = pd.read_excel(file_path, sheet_name=None)
    

    ce_sheets = [df for name, df in all_sheets.items() if 'CE' in name]
    
    if not ce_sheets:
        raise ValueError("No 'CE' sheets found! Check Excel file.")
        
    data = pd.concat(ce_sheets, ignore_index=True)

  
    data['T_years'] = data['t'] / 365.0
    data['Date'] = pd.to_datetime(data['Date'])

    features = ['underlying_value', 'strike_price', 'T_years', 'r', 'sigma']
    target = 'close'

    data = data.dropna(subset=features + [target])

    train_data = data[data['Date'].dt.year < 2020].copy()
    test_data = data[data['Date'].dt.year == 2020].copy()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(train_data[features])
    y_train = scaler_y.fit_transform(train_data[[target]])

    X_test = scaler_X.transform(test_data[features])
    y_test = scaler_y.transform(test_data[[target]])

    return X_train, y_train, X_test, y_test, scaler_y

if __name__ == "__main__":
    X_train, _, _, _, _ = get_data()
    print(f"Data Loaded. Training samples (Calls Only): {X_train.shape[0]}")