from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from arch import arch_model
import numpy as np

class ModelTrainer:
    def __init__(self, data, scaled_data):
        self.data = data
        self.scaled_data = scaled_data

    def train_arima(self):
        arima_model = ARIMA(self.data['Close'], order=(5,1,0))
        self.arima_fit = arima_model.fit()

    def train_lstm(self, time_step=50):  # time_step değerini uygun hale getirin
        try:
            X, y = self.create_dataset(self.scaled_data, time_step)
            if X.shape[0] == 0 or y.shape[0] == 0:
                raise ValueError(f"Veri kümesi oluşturulamadı. time_step ({time_step}) ve veri kümesi uzunluğu ({len(self.scaled_data)}) kontrol edin.")
            X = X.reshape(X.shape[0], X.shape[1], 1)

            lstm_model = Sequential()
            lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
            lstm_model.add(LSTM(50, return_sequences=False))
            lstm_model.add(Dense(25))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            lstm_model.fit(X, y, batch_size=1, epochs=1)
            self.lstm_model = lstm_model
        except Exception as e:
            print(f"LSTM modeli eğitiminde hata: {e}")

    def train_xgboost(self):
        X_train, y_train = self.data.drop(columns=['Close']), self.data['Close']
        self.xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.xgb_model.fit(X_train, y_train)

    def train_lightgbm(self):
        X_train, y_train = self.data.drop(columns=['Close']), self.data['Close']
        self.lgb_model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
        self.lgb_model.fit(X_train, y_train)

    def train_catboost(self):
        X_train, y_train = self.data.drop(columns=['Close']), self.data['Close']
        self.catboost_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6)
        self.catboost_model.fit(X_train, y_train)

    def train_garch(self):
        self.garch_model = arch_model(self.data['Close'], vol='Garch', p=1, q=1)
        self.garch_fit = self.garch_model.fit()

    def create_dataset(self, dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)
