import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

class DataPreprocessor:
    def __init__(self, file_path='data/stock_data.csv'):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Dosya {file_path} bulunamadı.")
        self.data = pd.read_csv(file_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess(self):
        self.data.fillna(method='ffill', inplace=True)
        self.scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))
        return self.data, self.scaled_data

    def create_dataset(self, dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)
