import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    """Handles data transformation and scaling."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def transform(self, data):
        print("Preprocessing data...")
        processed_data = np.zeros(data.shape, dtype=float)
        
        for col in range(data.shape[1]):
            col_data = data[:, col]
            try:
                processed_data[:, col] = col_data.astype(float)
            except ValueError:
                le = LabelEncoder()
                processed_data[:, col] = le.fit_transform(col_data)
                self.label_encoders[col] = le
            
        X = processed_data[:, :-1]
        y = processed_data[:, -1]
        
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
