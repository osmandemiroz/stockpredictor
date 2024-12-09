import numpy as np

class Predictor:
    def __init__(self, models, scaler):
        self.models = models
        self.scaler = scaler

    def predict_arima(self, steps=10):
        return self.models['arima_fit'].forecast(steps=steps)

    def predict_lstm(self, current_data, time_step=50):  # time_step değerini uygun hale getirin
        scaled_current_data = self.scaler.transform(current_data.values.reshape(-1, 1))
        lstm_pred = self.models['lstm_model'].predict(scaled_current_data.reshape(1, time_step, 1))
        return self.scaler.inverse_transform(lstm_pred)

    def predict_xgboost(self, current_data):
        return self.models['xgb_model'].predict(current_data)

    def predict_lightgbm(self, current_data):
        return self.models['lgb_model'].predict(current_data)

    def predict_catboost(self, current_data):
        return self.models['catboost_model'].predict(current_data)

    def predict_garch(self, horizon=10):
        return self.models['garch_fit'].forecast(horizon=horizon).mean.iloc[-1]
