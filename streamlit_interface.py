import streamlit as st
import pandas as pd
from models.data_preprocessing import DataPreprocessor
from models.model_training import ModelTrainer
from models.prediction import Predictor

# Veri yükleme ve ön işleme
data_preprocessor = DataPreprocessor('data/thy_stock_data.csv')
data, scaled_data = data_preprocessor.preprocess()

# Modelleri eğitme
model_trainer = ModelTrainer(data, scaled_data)
model_trainer.train_arima()
try:
    model_trainer.train_lstm()
except ValueError as e:
    st.error(str(e))
model_trainer.train_xgboost()
model_trainer.train_lightgbm()
model_trainer.train_catboost()
model_trainer.train_garch()

# Modelleri tahmin için hazırlama
models = {
    'arima_fit': model_trainer.arima_fit,
    'lstm_model': model_trainer.lstm_model,
    'xgb_model': model_trainer.xgb_model,
    'lgb_model': model_trainer.lgb_model,
    'catboost_model': model_trainer.catboost_model,
    'garch_fit': model_trainer.garch_fit
}

predictor = Predictor(models, data_preprocessor.scaler)

# Streamlit Arayüzü
st.title('Hisse Senedi Tahmini')
st.write('Mevcut hisse senedi verilerini ve diğer değişkenleri girerek gelecekteki hisse senedi verilerini tahmin edin.')

# Örnek giriş değerleri
example_close_prices = "29.0,29.1,29.2,29.3,29.4,29.5,29.6,29.7,29.8,29.9,30.0"
example_volume = 2000000

# Giriş alanları
current_data = st.text_input('Mevcut hisse senedi verilerini virgülle ayırarak girin:', example_close_prices)
volume = st.number_input('Hacim değerini girin:', value=example_volume)

# Tahmin yapma ve sonuçları gösterme
if st.button('Tahmin Et'):
    # Giriş verilerini DataFrame'e dönüştürme
    current_data_list = [float(x) for x in current_data.split(',')]
    current_data_df = pd.DataFrame({'Close': current_data_list, 'Volume': [volume] * len(current_data_list)})

    # ARIMA Tahmini
    arima_pred = predictor.predict_arima()

    # LSTM Tahmini
    try:
        lstm_pred = predictor.predict_lstm(current_data_df[['Close']])
    except Exception as e:
        lstm_pred = "LSTM tahmininde hata: " + str(e)

    # XGBoost Tahmini
    xgb_pred = predictor.predict_xgboost(current_data_df.drop(columns=['Close']))

    # LightGBM Tahmini
    lgb_pred = predictor.predict_lightgbm(current_data_df.drop(columns=['Close']))

    # CatBoost Tahmini
    catboost_pred = predictor.predict_catboost(current_data_df.drop(columns=['Close']))

    # GARCH Tahmini
    garch_pred = predictor.predict_garch()

    st.write('ARIMA Tahmini:', arima_pred)
    st.write('LSTM Tahmini:', lstm_pred)
    st.write('XGBoost Tahmini:', xgb_pred)
    st.write('LightGBM Tahmini:', lgb_pred)
    st.write('CatBoost Tahmini:', catboost_pred)
    st.write('GARCH Tahmini:', garch_pred)
